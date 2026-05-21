from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError

from capture import (
    LIVE_LAG_TOLERANCE_SECONDS,
    LIVE_SAMPLE_RATE,
    LIVE_TICK_SECONDS,
    LIVE_WINDOW_SECONDS,
    encode_wav,
    get_loopback_mic,
)
from llm import TRANSLATE_HISTORY_LEN, translate

log = logging.getLogger("subsvibe.pipeline")


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


def _fmt_ts_filename(seconds: float) -> str:
    return _fmt_ts(seconds).replace(":", "-").replace(".", "-")


SILENCE_PEAK_THRESHOLD = 0.01  # float32 abs-peak below this == silent window, skip ASR


@dataclass
class _Window:
    wav_bytes: bytes
    filename: str
    win_start: float
    win_end: float
    enqueued_at: float  # time.monotonic() at enqueue
    is_silent: bool = False


@dataclass
class _TranslateJob:
    win: _Window
    text: str
    queue_wait: float       # window_q wait
    transcript_elapsed: float
    asr_done_at: float      # time.monotonic() when ASR returned


def live_capture(
    *,
    asr_client: OpenAI,
    asr_base_url: str,
    model: str,
    language: str | None,
    prompt: str | None,
    translate_target: str | None,
    window: int = LIVE_WINDOW_SECONDS,
    tick: int = LIVE_TICK_SECONDS,
) -> None:
    mic = get_loopback_mic()

    samples_per_tick = LIVE_SAMPLE_RATE * tick
    samples_per_window = LIVE_SAMPLE_RATE * window
    chunk_q: queue.Queue[np.ndarray] = queue.Queue()
    window_q: queue.Queue[_Window] = queue.Queue()
    translate_q: queue.Queue[_TranslateJob] = queue.Queue()
    stop_event = threading.Event()
    # If a dequeued item is older than this, drop it and drain to the freshest one
    # whose age is still under the threshold. Bounds end-to-end lag stage-by-stage.
    max_item_age = float(LIVE_LAG_TOLERANCE_SECONDS)
    # Wall-clock anchor for the first captured sample. Used to compute lag of subtitles
    # against real-time audio: lag = monotonic() - (capture_start_monotonic + win.win_end).
    capture_start_monotonic = time.monotonic()
    # warn when audio-vs-wallclock lag exceeds this many seconds; one tick over window
    # is the natural pipeline floor (window must fill before first transcribe).
    lag_warn_threshold = window + tick

    # --- recording thread: captures audio chunks at real time ---
    def _record_worker() -> None:
        with mic.recorder(samplerate=LIVE_SAMPLE_RATE, channels=1) as recorder:
            while not stop_event.is_set():
                chunk = recorder.record(numframes=samples_per_tick).reshape(-1).astype(np.float32)
                chunk_q.put(chunk)

    def _log_lag(label: str, lag: float) -> None:
        if lag > lag_warn_threshold:
            log.warning("%s lag behind real-time: %.2fs (threshold %.2fs)", label, lag, lag_warn_threshold)

    # --- ASR thread: consumes assembled windows, calls transcription API ---
    def _transcribe_worker() -> None:
        while True:
            win = window_q.get()
            if win is None:  # sentinel
                if translate_target:
                    translate_q.put(None)  # propagate shutdown to translate thread
                break

            # Drop stale windows: if this one already exceeds max_item_age, drain forward
            # until we find one whose age is under the threshold (or the queue is empty).
            dropped = 0
            while time.monotonic() - win.enqueued_at > max_item_age:
                try:
                    newer = window_q.get_nowait()
                except queue.Empty:
                    break
                if newer is None:
                    window_q.put(None)  # preserve sentinel for shutdown
                    break
                dropped += 1
                win = newer
            if dropped:
                log.warning(
                    "window stale > %.1fs - dropped %d window(s), jumped to %s-%s",
                    max_item_age, dropped,
                    _fmt_ts(win.win_start), _fmt_ts(win.win_end),
                )

            queue_wait = time.monotonic() - win.enqueued_at

            if win.is_silent:
                log.info("silent window %s-%s - skipping ASR", _fmt_ts(win.win_start), _fmt_ts(win.win_end))
                continue

            t0 = time.monotonic()
            try:
                result = asr_client.audio.transcriptions.create(
                    model=model,
                    file=(win.filename, win.wav_bytes, "audio/wav"),
                    response_format="json",
                    timeout=LIVE_LAG_TOLERANCE_SECONDS,
                    **({"language": language} if language else {}),
                    **({"prompt": prompt} if prompt else {}),
                )
            except APITimeoutError:
                log.error(
                    "ASR call exceeded %.1fs timeout for window %s-%s - dropping",
                    float(LIVE_LAG_TOLERANCE_SECONDS),
                    _fmt_ts(win.win_start), _fmt_ts(win.win_end),
                )
                continue
            except APIConnectionError:
                log.error("could not connect to transcription backend at %s", asr_base_url)
                continue
            except APIStatusError as exc:
                log.error("server error %s: %s", exc.status_code, exc.message)
                continue
            elapsed = time.monotonic() - t0

            text = result if isinstance(result, str) else result.text
            if not text:
                continue

            if not translate_target:
                now = time.monotonic()
                staleness = now - win.enqueued_at
                lag = now - (capture_start_monotonic + win.win_end)
                print(text)
                log.info(
                    "wait=%.2fs transcript=%.2fs stale=%.2fs lag=%.2fs",
                    queue_wait, elapsed, staleness, lag,
                )
                _log_lag("subtitle", lag)
                continue

            translate_q.put(_TranslateJob(
                win=win,
                text=text,
                queue_wait=queue_wait,
                transcript_elapsed=elapsed,
                asr_done_at=time.monotonic(),
            ))

    # --- translate thread: consumes ASR results, calls LLM ---
    def _translate_worker() -> None:
        history: list[tuple[str, str]] = []
        while True:
            job = translate_q.get()
            if job is None:
                break

            # Drop stale jobs: same age-based policy as the ASR stage.
            dropped = 0
            while time.monotonic() - job.win.enqueued_at > max_item_age:
                try:
                    newer = translate_q.get_nowait()
                except queue.Empty:
                    break
                if newer is None:
                    translate_q.put(None)
                    break
                dropped += 1
                job = newer
            if dropped:
                log.warning(
                    "translate stale > %.1fs - dropped %d job(s), jumped to %s-%s",
                    max_item_age, dropped,
                    _fmt_ts(job.win.win_start), _fmt_ts(job.win.win_end),
                )

            translate_wait = time.monotonic() - job.asr_done_at

            t_tx0 = time.monotonic()
            try:
                translation = translate(
                    job.text, history,
                    target=translate_target,
                    timeout=float(LIVE_LAG_TOLERANCE_SECONDS),
                )
            except APITimeoutError:
                log.error(
                    "translate call exceeded %.1fs timeout for window %s-%s - dropping",
                    float(LIVE_LAG_TOLERANCE_SECONDS),
                    _fmt_ts(job.win.win_start), _fmt_ts(job.win.win_end),
                )
                continue
            t_translate = time.monotonic() - t_tx0
            history.append((job.text, translation))
            if len(history) > TRANSLATE_HISTORY_LEN:
                history.pop(0)
            now = time.monotonic()
            staleness = now - job.win.enqueued_at
            lag = now - (capture_start_monotonic + job.win.win_end)
            print(job.text)
            print(f"  -> {translation}")
            log.info(
                "wait=%.2fs transcript=%.2fs tr_wait=%.2fs translate=%.2fs stale=%.2fs lag=%.2fs",
                job.queue_wait, job.transcript_elapsed, translate_wait, t_translate, staleness, lag,
            )
            _log_lag("subtitle", lag)

    record_thread = threading.Thread(target=_record_worker, daemon=True)
    transcribe_thread = threading.Thread(target=_transcribe_worker, daemon=True)
    translate_thread: threading.Thread | None = None
    record_thread.start()
    transcribe_thread.start()
    if translate_target:
        translate_thread = threading.Thread(target=_translate_worker, daemon=True)
        translate_thread.start()

    ring: deque[np.ndarray] = deque()
    ring_len = 0
    ticks_elapsed = 0
    lag_tolerance_ticks = max(1, int(LIVE_LAG_TOLERANCE_SECONDS / tick))

    log.info(
        "starting live capture - window=%ds tick=%ds tolerance=%ds (Ctrl+C to stop)",
        window, tick, LIVE_LAG_TOLERANCE_SECONDS,
    )

    try:
        while True:
            chunk = chunk_q.get(timeout=tick + 2)

            backlog = chunk_q.qsize()
            if backlog > lag_tolerance_ticks:
                drop = backlog - lag_tolerance_ticks
                for _ in range(drop):
                    try:
                        chunk_q.get_nowait()
                    except queue.Empty:
                        break
                ticks_elapsed += drop
                log.warning("capture lagging - dropped %d tick(s) (backlog was %d)", drop, backlog)

            ring.append(chunk)
            ring_len += len(chunk)
            ticks_elapsed += 1

            if ring_len < samples_per_window:
                continue

            while ring_len > samples_per_window:
                oldest = ring[0]
                excess = ring_len - samples_per_window
                if excess >= len(oldest):
                    ring.popleft()
                    ring_len -= len(oldest)
                else:
                    ring[0] = oldest[excess:]
                    ring_len -= excess

            pcm = np.concatenate(list(ring))
            peak = float(np.abs(pcm).max()) if pcm.size else 0.0
            is_silent = peak < SILENCE_PEAK_THRESHOLD
            wav_bytes = b"" if is_silent else encode_wav(pcm)
            win_end = ticks_elapsed * tick
            win_start = max(0, win_end - window)
            filename = f"{_fmt_ts_filename(win_start)}-{_fmt_ts_filename(win_end)}.wav"

            window_q.put(_Window(
                wav_bytes=wav_bytes,
                filename=filename,
                win_start=win_start,
                win_end=win_end,
                enqueued_at=time.monotonic(),
                is_silent=is_silent,
            ))
    finally:
        stop_event.set()
        window_q.put(None)  # wake transcribe thread; it forwards a sentinel to translate_q on exit
        record_thread.join(timeout=tick + 2)
        transcribe_thread.join(timeout=10)
        if translate_thread is not None:
            translate_thread.join(timeout=10)
