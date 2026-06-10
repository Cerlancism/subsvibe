from __future__ import annotations

import logging
import os
from pathlib import Path

import av
import numpy as np

log = logging.getLogger("subsvibe.vad")

SAMPLE_RATE = 16000
SPEECH_THRESHOLD = 0.2
# Second-chance passes over any piece still longer than MAX_SEGMENT_SECONDS
# after the seed pass, tried in order; each recursion level consumes one pass,
# and quiet-split remains the final fallback when all passes are exhausted.
# 1. Silero at a stricter threshold + tiny min-silence: splits on phrase-level
#    pauses the permissive seed pass rode through.
# 2. webrtcvad (energy/GMM, per-frame): a different detector entirely, for
#    spans where Silero finds no boundaries at any threshold. Mirrors the
#    live recovery VAD in ./client/live_vad.py — the sub-range is
#    peak-normalised first since an energy-based detector cannot see locally
#    quiet audio, and aggressiveness 3 marks marginal frames unvoiced, giving
#    the most de-trigger (split) opportunities while the 300ms hysteresis
#    keeps cuts off mid-word dips.
SUBSLICE_PASSES = (
    ("silero", {"threshold": 0.8, "min_silence_duration_ms": 50}),
    ("webrtcvad", {"aggressiveness": 3}),
)
QUIET_SPLIT_WINDOW_MS = 20
QUIET_SPLIT_EDGE_MARGIN = 0.2
QUIET_SPLIT_MIN_WINDOWS = 3
MAX_SEGMENT_SECONDS = float(os.environ.get("MAX_SEGMENT_SECONDS", "30"))
HARD_SLICE_SECONDS = float(os.environ.get("HARD_SLICE_SECONDS", "30"))
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


def _webrtcvad_boundaries(audio: np.ndarray, s: float, e: float, *, aggressiveness: int) -> list[float]:
    """Run webrtcvad on the [s, e] sub-range and return boundaries in the
    original timeline. Same contract as _vad_boundaries: boundaries are
    speech-start times, so leading silence attaches to the preceding piece.

    The sub-range is re-normalised before classification: the file-level
    pass in get_speech_segments targets the global peak, so a locally quiet
    span can still sit far below it — invisible to an energy-based detector.
    """
    import webrtcvad

    from capture import peak_normalize
    from live_vad import webrtcvad_speech_timestamps

    sub_audio = audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]
    normalised, gain_db = peak_normalize(sub_audio)
    log.info("webrtcvad subslice: %+.1fdB applied to [%.2f-%.2f] before classification", gain_db, s, e)
    spans = webrtcvad_speech_timestamps(normalised, webrtcvad.Vad(aggressiveness))
    return [s, *(s + span["start"] / SAMPLE_RATE for span in spans), e]


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


def _quiet_split(audio: np.ndarray, s: float, e: float) -> list[dict]:
    """Last resort: split [s, e] at the quietest short window in its middle
    band. Recurses until every piece is under MAX_SEGMENT_SECONDS."""
    if e - s <= MAX_SEGMENT_SECONDS:
        return [{"start": s, "end": e}] if e > s else []
    sub = audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]
    win = int(SAMPLE_RATE * QUIET_SPLIT_WINDOW_MS / 1000)
    n_windows = len(sub) // win
    if n_windows < QUIET_SPLIT_MIN_WINDOWS:
        log.warning("quiet-split fell back to even-split on %.1fs piece [%.2f-%.2f] (too short to scan)",
                    e - s, s, e)
        return _even_split(s, e, MAX_SEGMENT_SECONDS)
    energy = np.abs(sub[:n_windows * win].reshape(n_windows, win)).mean(axis=1)
    lo = int(n_windows * QUIET_SPLIT_EDGE_MARGIN)
    hi = n_windows - lo
    cut_window = lo + int(np.argmin(energy[lo:hi]))
    cut_time = s + (cut_window + 0.5) * win / SAMPLE_RATE
    log.warning("quiet-split %.1fs piece [%.2f-%.2f] at %.2fs (energy=%.4f vs median=%.4f)",
                e - s, s, e, cut_time, float(energy[cut_window]), float(np.median(energy)))
    return [*_quiet_split(audio, s, cut_time), *_quiet_split(audio, cut_time, e)]


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
        return _quiet_split(audio, s, e)

    (engine, params), *rest = passes
    log.info("subslicing %.1fs piece [%.2f–%.2f] with %s VAD (%s)",
             e - s, s, e, engine, ", ".join(f"{k}={v}" for k, v in params.items()))
    if engine == "webrtcvad":
        sub_boundaries = _webrtcvad_boundaries(audio, s, e, **params)
    else:
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

    from capture import peak_normalize

    log.info("decoding audio for VAD: %s", path.name)
    audio = _decode_audio_mono_16k(path)
    if len(audio) == 0:
        return []

    # Peak-normalise the whole file before VAD so quieter content still
    # crosses the speech-probability threshold. ASR gets its own per-segment
    # normalisation later — this pass exists for VAD sensitivity only.
    audio, gain_db = peak_normalize(audio)
    log.info("normalised file audio: %+.1fdB applied (pre-VAD)", gain_db)

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
