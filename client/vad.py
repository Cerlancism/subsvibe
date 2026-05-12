from __future__ import annotations

import logging
from pathlib import Path

import av
import numpy as np

log = logging.getLogger("subsvibe.vad")

SAMPLE_RATE = 16000
SPEECH_THRESHOLD = 0.2
SUBSLICE_PASSES = (
    {"threshold": 0.5, "min_silence_duration_ms": 100},
    {"threshold": 0.8, "min_silence_duration_ms": 50},
)
MAX_SPLIT_THRESHOLD = 0.8
MAX_SPLIT_MIN_SILENCE_MS = 50
MAX_SEGMENT_SECONDS = 19.0
HARD_SLICE_SECONDS = 29.0
TARGET_SEGMENT_SECONDS = 5.0


def _decode_audio_mono_16k(path: Path) -> np.ndarray:
    """Decode any audio file to mono float32 PCM at 16 kHz."""
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
        for packet in container.demux(container.streams.audio[0]):
            for frame in packet.decode():
                for resampled in resampler.resample(frame):
                    frames.append(resampled.to_ndarray()[0])
        for resampled in resampler.resample(None):
            frames.append(resampled.to_ndarray()[0])
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames).astype(np.float32)


def _even_split(s: float, e: float, max_len: float) -> list[dict]:
    """Subdivide [s, e] into the fewest equal parts each shorter than max_len."""
    dur = e - s
    if dur <= max_len:
        return [{"start": s, "end": e}]
    n = int(np.ceil(dur / max_len))
    step = dur / n
    return [{"start": s + i * step, "end": s + (i + 1) * step} for i in range(n)]


def _vad_boundaries(audio: np.ndarray, model, s: float, e: float, **params) -> list[float]:
    """Run silero on the [s, e] sub-range and return boundaries in original timeline."""
    from silero_vad import get_speech_timestamps

    sub_audio = audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]
    sub_raw = get_speech_timestamps(
        sub_audio,
        model,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
        **params,
    )
    return [s, *(s + float(x["start"]) for x in sub_raw), e]


def _bundle_to_target(pieces: list[dict]) -> list[dict]:
    """Merge consecutive short pieces toward TARGET_SEGMENT_SECONDS without
    exceeding MAX_SEGMENT_SECONDS — gives the ASR more context (very short
    clips tend to hallucinate)."""
    segments: list[dict] = []
    for p in pieces:
        if segments:
            cur = segments[-1]
            cur_dur = cur["end"] - cur["start"]
            p_dur = p["end"] - p["start"]
            merged_span = p["end"] - cur["start"]
            either_short = cur_dur < TARGET_SEGMENT_SECONDS or p_dur < TARGET_SEGMENT_SECONDS
            if either_short and merged_span <= MAX_SEGMENT_SECONDS:
                cur["end"] = p["end"]
                continue
        segments.append({"start": p["start"], "end": p["end"]})
    return segments


def _log_segment_stats(segments: list[dict], total_duration: float) -> None:
    if not segments:
        log.info("VAD produced 0 segment(s) - audio=%.1fs not transcribed", total_duration)
        return
    durations = sorted(s["end"] - s["start"] for s in segments)
    n = len(durations)
    median = durations[n // 2] if n % 2 else (durations[n // 2 - 1] + durations[n // 2]) / 2
    log.info(
        "VAD produced %d segment(s) over %.1fs - avg=%.2fs median=%.2fs min=%.2fs max=%.2fs",
        n, total_duration,
        sum(durations) / n, median, durations[0], durations[-1],
    )


def _seed_boundaries(
    audio: np.ndarray,
    model,
    total_duration: float,
    reference_entries: list[dict] | None,
) -> list[float]:
    if reference_entries:
        log.info("seeding boundaries from %d reference entry(ies) (skipping first VAD pass)", len(reference_entries))
        seeds = sorted(
            float(e["start"]) for e in reference_entries
            if 0.0 < float(e["start"]) < total_duration
        )
        return [0.0, *seeds, total_duration]
    log.info("running Silero VAD (threshold=%.2f)…", SPEECH_THRESHOLD)
    # VAD is used only to choose chunk boundaries - we never drop silence.
    # Tile the whole audio [0, total_duration] with pieces split at each
    # speech-start boundary.
    return _vad_boundaries(audio, model, 0.0, total_duration, threshold=SPEECH_THRESHOLD)


def _enforce_hard_slice(pieces: list[dict], *, reason: str) -> list[dict]:
    """Even-split any piece longer than HARD_SLICE_SECONDS; drop empty pieces."""
    out: list[dict] = []
    for p in pieces:
        cur_s, cur_e = p["start"], p["end"]
        dur = cur_e - cur_s
        if dur <= 0:
            continue
        if dur > HARD_SLICE_SECONDS:
            parts = _even_split(cur_s, cur_e, HARD_SLICE_SECONDS)
            log.warning("%s: hard-slicing %.1fs [%.2f-%.2f] into %d evenly-sized parts of %.2fs",
                        reason, dur, cur_s, cur_e, len(parts), dur / len(parts))
            out.extend(parts)
        else:
            out.append({"start": cur_s, "end": cur_e})
    return out


def _silero_max_split(audio: np.ndarray, model, s: float, e: float) -> list[dict]:
    """Last resort: ask silero itself to honour max_speech_duration_s, which
    cuts at the best internal silence rather than blindly. Anything still over
    HARD_SLICE_SECONDS afterwards is divided into even parts."""
    log.warning("forcing silero max-speech split on %.1fs piece [%.2f–%.2f] (max=%.1fs)",
                e - s, s, e, MAX_SEGMENT_SECONDS)
    sub_boundaries = _vad_boundaries(
        audio, model, s, e,
        threshold=MAX_SPLIT_THRESHOLD,
        min_silence_duration_ms=MAX_SPLIT_MIN_SILENCE_MS,
        max_speech_duration_s=MAX_SEGMENT_SECONDS,
    )
    pieces = [{"start": cs, "end": ce} for cs, ce in zip(sub_boundaries, sub_boundaries[1:])]
    for p in pieces:
        dur = p["end"] - p["start"]
        if MAX_SEGMENT_SECONDS < dur <= HARD_SLICE_SECONDS:
            log.warning("piece [%.2f–%.2f] is %.1fs, over target max=%.1fs but under hard-slice=%.1fs - keeping as-is",
                        p["start"], p["end"], dur, MAX_SEGMENT_SECONDS, HARD_SLICE_SECONDS)
    return _enforce_hard_slice(pieces, reason="silero couldn't split further")


def _split_oversized(
    audio: np.ndarray,
    model,
    s: float,
    e: float,
    passes: tuple[dict, ...],
) -> list[dict]:
    if e - s <= MAX_SEGMENT_SECONDS:
        return [{"start": s, "end": e}] if e > s else []
    if not passes:
        return _silero_max_split(audio, model, s, e)

    params, *rest = passes
    log.info("subslicing %.1fs piece [%.2f–%.2f] with sensitive VAD (%s)",
             e - s, s, e, ", ".join(f"{k}={v}" for k, v in params.items()))
    sub_boundaries = _vad_boundaries(audio, model, s, e, **params)
    out: list[dict] = []
    for ss, ee in zip(sub_boundaries, sub_boundaries[1:]):
        out.extend(_split_oversized(audio, model, ss, ee, tuple(rest)))
    return out


def get_speech_segments(path: Path, *, reference_entries: list[dict] | None = None) -> list[dict]:
    """
    Run Silero VAD on an audio file and return speech intervals.

    When `reference_entries` is provided, its entry start times replace the
    first Silero pass; subslicing and bundling still run so segments respect
    TARGET/MAX/HARD constants.

    Returns a list of {start: float, end: float} dicts in seconds
    """
    from silero_vad import load_silero_vad

    log.info("decoding audio for VAD: %s", path.name)
    audio = _decode_audio_mono_16k(path)
    if len(audio) == 0:
        return []

    model = load_silero_vad(onnx=True)
    total_duration = len(audio) / SAMPLE_RATE

    boundaries = _seed_boundaries(audio, model, total_duration, reference_entries)

    pieces: list[dict] = []
    for start, end in zip(boundaries, boundaries[1:]):
        pieces.extend(_split_oversized(audio, model, start, end, SUBSLICE_PASSES))

    segments = _bundle_to_target(pieces)
    # Bundling caps at MAX_SEGMENT_SECONDS, so this only fires on pre-merge
    # pieces that survived (e.g. silero couldn't split a long monologue).
    segments = _enforce_hard_slice(segments, reason="final pass")

    _log_segment_stats(segments, total_duration)
    return segments
