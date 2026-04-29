from __future__ import annotations

import logging
from pathlib import Path

import av
import numpy as np

log = logging.getLogger("subsvibe.vad")

SPEECH_THRESHOLD = 0.2
SUBSLICE_PASSES = (
    {"threshold": 0.5, "min_silence_duration_ms": 100},
    {"threshold": 0.8, "min_silence_duration_ms": 50},
)
MAX_SEGMENT_SECONDS = 45.0
HARD_SLICE_SECONDS = 60.0
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
    def _silero_max_split(s: float, e: float) -> list[dict]:
        """Last resort: ask silero itself to honour max_speech_duration_s,
        which cuts at the best internal silence rather than blindly."""
        log.warning("forcing silero max-speech split on %.1fs piece [%.2f–%.2f] (max=%.1fs)",
                    e - s, s, e, MAX_SEGMENT_SECONDS)
        sub_audio = audio[int(s * 16000):int(e * 16000)]
        sub_raw = get_speech_timestamps(
            sub_audio,
            model,
            sampling_rate=16000,
            threshold=0.8,
            min_silence_duration_ms=50,
            max_speech_duration_s=MAX_SEGMENT_SECONDS,
            return_seconds=True,
        )
        sub_boundaries = [s, *(s + float(x["start"]) for x in sub_raw), e]
        out: list[dict] = []
        cur_s = sub_boundaries[0]
        for boundary in sub_boundaries[1:]:
            cur_e = boundary
            while cur_e - cur_s > HARD_SLICE_SECONDS:
                log.warning("hard-slicing %.1fs piece [%.2f–%.2f] at %.1fs (silero couldn't split further)",
                            cur_e - cur_s, cur_s, cur_e, HARD_SLICE_SECONDS)
                out.append({"start": cur_s, "end": cur_s + HARD_SLICE_SECONDS})
                cur_s += HARD_SLICE_SECONDS
            if cur_e > cur_s:
                if cur_e - cur_s > MAX_SEGMENT_SECONDS:
                    log.warning("piece [%.2f–%.2f] is %.1fs, over target max=%.1fs but under hard-slice=%.1fs — keeping as-is",
                                cur_s, cur_e, cur_e - cur_s, MAX_SEGMENT_SECONDS, HARD_SLICE_SECONDS)
                out.append({"start": cur_s, "end": cur_e})
            cur_s = boundary
        return out

    def _split_oversized(s: float, e: float, passes: tuple[dict, ...]) -> list[dict]:
        if e - s <= MAX_SEGMENT_SECONDS:
            return [{"start": s, "end": e}] if e > s else []
        if not passes:
            return _silero_max_split(s, e)

        params, *rest = passes
        log.info("subslicing %.1fs piece [%.2f–%.2f] with sensitive VAD (%s)",
                 e - s, s, e, ", ".join(f"{k}={v}" for k, v in params.items()))
        sub_audio = audio[int(s * 16000):int(e * 16000)]
        sub_raw = get_speech_timestamps(
            sub_audio,
            model,
            sampling_rate=16000,
            return_seconds=True,
            **params,
        )
        sub_boundaries = [s, *(s + float(x["start"]) for x in sub_raw), e]
        out: list[dict] = []
        for ss, ee in zip(sub_boundaries, sub_boundaries[1:]):
            out.extend(_split_oversized(ss, ee, tuple(rest)))
        return out

    pieces: list[dict] = []
    for start, end in zip(boundaries, boundaries[1:]):
        pieces.extend(_split_oversized(start, end, SUBSLICE_PASSES))

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
