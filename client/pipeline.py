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


@dataclass
class _Window:
    wav_bytes: bytes
    filename: str
    win_start: float
    win_end: float


def live_capture(
    *,
    asr_client: OpenAI,
    asr_base_url: str,
    model: str,
    language: str | None,
    prompt: str | None,
    do_translate: bool,
    window: int = LIVE_WINDOW_SECONDS,
    tick: int = LIVE_TICK_SECONDS,
) -> None:
    mic = get_loopback_mic()

    samples_per_tick = LIVE_SAMPLE_RATE * tick
    samples_per_window = LIVE_SAMPLE_RATE * window
    chunk_q: queue.Queue[np.ndarray] = queue.Queue()
    window_q: queue.Queue[_Window] = queue.Queue()
    stop_event = threading.Event()

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

            if do_translate:
                t_tx0 = time.monotonic()
                translation = translate(text, history)
                t_translate = time.monotonic() - t_tx0
                history.append((text, translation))
                if len(history) > TRANSLATE_HISTORY_LEN:
                    history.pop(0)
                print(text)
                print(f"  -> {translation}")
                log.info("transcript=%.2fs translate=%.2fs", elapsed, t_translate)
            else:
                print(text)
                log.info("transcript=%.2fs", elapsed)

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

            wav_bytes = encode_wav(np.concatenate(list(ring)))
            win_end = ticks_elapsed * tick
            win_start = max(0, win_end - window)
            filename = f"{_fmt_ts_filename(win_start)}-{_fmt_ts_filename(win_end)}.wav"

            window_q.put(_Window(wav_bytes=wav_bytes, filename=filename, win_start=win_start, win_end=win_end))
    finally:
        stop_event.set()
        window_q.put(None)  # wake transcribe thread so it can exit
        record_thread.join(timeout=tick + 2)
        transcribe_thread.join(timeout=10)
