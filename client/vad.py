from __future__ import annotations

import logging
from pathlib import Path

import av
import numpy as np

log = logging.getLogger("subsvibe.vad")

SPEECH_THRESHOLD = 0.2
MAX_SEGMENT_SECONDS = 120.0
TARGET_SEGMENT_SECONDS = 5.0


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

    log.info("running Silero VAD (threshold=%.2f)…", SPEECH_THRESHOLD)
    raw = get_speech_timestamps(
        audio,
        model,
        sampling_rate=16000,
        threshold=SPEECH_THRESHOLD,
        return_seconds=True,
    )

    # VAD is used only to choose chunk boundaries — we never drop silence.
    # Tile the whole audio [0, total_duration] with pieces split at each
    # speech-start boundary.
    total_duration = len(audio) / 16000
    boundaries = [0.0, *(float(seg["start"]) for seg in raw), total_duration]
    pieces: list[dict] = []
    for start, end in zip(boundaries, boundaries[1:]):
        while end - start > MAX_SEGMENT_SECONDS:
            pieces.append({"start": start, "end": start + MAX_SEGMENT_SECONDS})
            start += MAX_SEGMENT_SECONDS
        if end > start:
            pieces.append({"start": start, "end": end})

    # Bundle consecutive pieces toward TARGET_SEGMENT_SECONDS to give the ASR
    # more context (very short clips tend to hallucinate); never exceed
    # MAX_SEGMENT_SECONDS for a single bundle.
    segments: list[dict] = []
    for p in pieces:
        if segments:
            cur = segments[-1]
            cur_dur = cur["end"] - cur["start"]
            merged_span = p["end"] - cur["start"]
            if cur_dur < TARGET_SEGMENT_SECONDS and merged_span <= MAX_SEGMENT_SECONDS:
                cur["end"] = p["end"]
                continue
        segments.append({"start": p["start"], "end": p["end"]})

    if segments:
        durations = sorted(s["end"] - s["start"] for s in segments)
        n = len(durations)
        median = durations[n // 2] if n % 2 else (durations[n // 2 - 1] + durations[n // 2]) / 2
        log.info(
            "VAD produced %d segment(s) over %.1fs — avg=%.2fs median=%.2fs min=%.2fs max=%.2fs",
            n, total_duration,
            sum(durations) / n, median, durations[0], durations[-1],
        )
    else:
        log.info("VAD produced 0 segment(s) — audio=%.1fs not transcribed", total_duration)
    return segments
