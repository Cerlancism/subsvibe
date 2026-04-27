from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import av
import numpy as np

if TYPE_CHECKING:
    pass

log = logging.getLogger("subsvibe.vad")

SPEECH_THRESHOLD = 0.2
MIN_SILENCE_MS = 1000
MAX_SEGMENT_SECONDS = 120.0


def _decode_audio_mono_16k(path: Path) -> np.ndarray:
    """Decode any audio file to mono float32 PCM at 16 kHz."""
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for packet in container.demux(container.streams.audio[0]):
            for frame in packet.decode():
                for resampled in resampler.resample(frame):
                    frames.append(resampled.to_ndarray()[0])
        for resampled in resampler.resample(None):
            frames.append(resampled.to_ndarray()[0])
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames).astype(np.float32)


def get_speech_segments(path: Path) -> list[dict]:
    """
    Run Silero VAD on an audio file and return speech intervals.

    Returns a list of {start: float, end: float} dicts in seconds
    """
    from silero_vad import get_speech_timestamps, load_silero_vad

    log.info("decoding audio for VAD: %s", path.name)
    audio = _decode_audio_mono_16k(path)
    if len(audio) == 0:
        return []

    model = load_silero_vad(onnx=True)

    log.info(
        "running Silero VAD (threshold=%.2f, min_silence=%dms)…",
        SPEECH_THRESHOLD, MIN_SILENCE_MS,
    )
    raw = get_speech_timestamps(
        audio,
        model,
        sampling_rate=16000,
        threshold=SPEECH_THRESHOLD,
        min_silence_duration_ms=MIN_SILENCE_MS,
        return_seconds=True,
    )

    segments: list[dict] = []
    for seg in raw:
        start, end = float(seg["start"]), float(seg["end"])
        while end - start > MAX_SEGMENT_SECONDS:
            segments.append({"start": start, "end": start + MAX_SEGMENT_SECONDS})
            start += MAX_SEGMENT_SECONDS
        segments.append({"start": start, "end": end})

    total_duration = len(audio) / 16000
    if segments:
        durations = [s["end"] - s["start"] for s in segments]
        speech = sum(durations)
        skipped = max(0.0, total_duration - speech)
        skipped_pct = (skipped / total_duration * 100) if total_duration else 0.0
        durations_sorted = sorted(durations)
        n = len(durations_sorted)
        median = durations_sorted[n // 2] if n % 2 else (durations_sorted[n // 2 - 1] + durations_sorted[n // 2]) / 2
        log.info(
            "VAD found %d segment(s) — speech=%.1fs skipped=%.1fs (%.1f%%) — avg=%.2fs median=%.2fs min=%.2fs max=%.2fs",
            n, speech, skipped, skipped_pct,
            sum(durations) / n, median, durations_sorted[0], durations_sorted[-1],
        )
    else:
        log.info("VAD found 0 speech segment(s) — skipped=%.1fs (100%%)", total_duration)
    return segments
