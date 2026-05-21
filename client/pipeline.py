from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from openai import OpenAI, APIConnectionError, APIStatusError

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
    stop_event = threading.Event()
    # window_q backlog above this means transcribe+translate is falling behind real-time;
    # drain to the newest window so subtitles don't drift further behind audio.
    window_backlog_tolerance = max(1, LIVE_LAG_TOLERANCE_SECONDS // tick)
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

    # --- transcription+translation thread: consumes assembled windows ---
    def _transcribe_worker() -> None:
        history: list[tuple[str, str]] = []
        while True:
            win = window_q.get()
            if win is None:  # sentinel
                break

            # If we're behind real-time, drain to the newest window. Intermediate windows
            # are dropped on purpose - showing the freshest subtitle beats catching up slowly.
            backlog = window_q.qsize()
            if backlog > window_backlog_tolerance:
                dropped = 0
                while True:
                    try:
                        newer = window_q.get_nowait()
                    except queue.Empty:
                        break
                    if newer is None:
                        window_q.put(None)  # preserve sentinel for shutdown
                        break
                    win = newer
                    dropped += 1
                log.warning(
                    "window backlog %d > %d - dropped %d window(s), jumped to %s-%s",
                    backlog, window_backlog_tolerance, dropped,
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
                    **({"language": language} if language else {}),
                    **({"prompt": prompt} if prompt else {}),
                )
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

            if translate_target:
                t_tx0 = time.monotonic()
                translation = translate(text, history, target=translate_target)
                t_translate = time.monotonic() - t_tx0
                history.append((text, translation))
                if len(history) > TRANSLATE_HISTORY_LEN:
                    history.pop(0)
                now = time.monotonic()
                staleness = now - win.enqueued_at
                lag = now - (capture_start_monotonic + win.win_end)
                print(text)
                print(f"  -> {translation}")
                log.info(
                    "wait=%.2fs transcript=%.2fs translate=%.2fs stale=%.2fs lag=%.2fs",
                    queue_wait, elapsed, t_translate, staleness, lag,
                )
                if lag > lag_warn_threshold:
                    log.warning("subtitle lag behind real-time: %.2fs (threshold %.2fs)", lag, lag_warn_threshold)
            else:
                now = time.monotonic()
                staleness = now - win.enqueued_at
                lag = now - (capture_start_monotonic + win.win_end)
                print(text)
                log.info(
                    "wait=%.2fs transcript=%.2fs stale=%.2fs lag=%.2fs",
                    queue_wait, elapsed, staleness, lag,
                )
                if lag > lag_warn_threshold:
                    log.warning("subtitle lag behind real-time: %.2fs (threshold %.2fs)", lag, lag_warn_threshold)

    record_thread = threading.Thread(target=_record_worker, daemon=True)
    transcribe_thread = threading.Thread(target=_transcribe_worker, daemon=True)
    record_thread.start()
    transcribe_thread.start()

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
        window_q.put(None)  # wake transcribe thread so it can exit
        record_thread.join(timeout=tick + 2)
        transcribe_thread.join(timeout=10)
