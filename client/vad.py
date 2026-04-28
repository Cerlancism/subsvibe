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
MAX_SEGMENT_SECONDS = 120.0
TARGET_SEGMENT_SECONDS = 30.0


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

    # VAD is used only to choose chunk boundaries — we never drop the silence
    # between speech regions. Extend each speech piece's end up to the next
    # piece's start so the union of segments covers the whole audio from the
    # first speech start to the last speech end.
    pieces: list[dict] = []
    for i, seg in enumerate(raw):
        start = float(seg["start"])
        end = float(raw[i + 1]["start"]) if i + 1 < len(raw) else float(seg["end"])
        while end - start > MAX_SEGMENT_SECONDS:
            pieces.append({"start": start, "end": start + MAX_SEGMENT_SECONDS})
            start += MAX_SEGMENT_SECONDS
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

    total_duration = len(audio) / 16000
    if segments:
        durations = [s["end"] - s["start"] for s in segments]
        covered = sum(durations)
        durations_sorted = sorted(durations)
        n = len(durations_sorted)
        median = durations_sorted[n // 2] if n % 2 else (durations_sorted[n // 2 - 1] + durations_sorted[n // 2]) / 2
        log.info(
            "VAD found %d segment(s) — covered=%.1fs/%.1fs — avg=%.2fs median=%.2fs min=%.2fs max=%.2fs",
            n, covered, total_duration,
            sum(durations) / n, median, durations_sorted[0], durations_sorted[-1],
        )
    else:
        log.info("VAD found 0 speech segment(s) — audio=%.1fs not transcribed", total_duration)
    return segments
