from __future__ import annotations

import io
import logging
import os
import wave

import numpy as np

log = logging.getLogger("subsvibe.capture")

LIVE_SAMPLE_RATE = 16000
# Silero VAD requires exactly this chunk size at 16 kHz.
LIVE_VAD_CHUNK_FRAMES = 512
# How often the recorder pulls from the OS (one VAD chunk = ~32 ms).
LIVE_CAPTURE_TICK_FRAMES = LIVE_VAD_CHUNK_FRAMES

# Provisional-update floor: while a speech segment is open, re-transcribe the
# in-progress audio no more often than this so the user sees a mid-sentence
# preview. It is a *minimum* spacing, not a fixed tick: when ASR keeps up this
# is the effective cadence (cap), but when ASR runs slower than this the next
# provisional fires the instant ASR frees up, over all audio accumulated so far
# (up to the VAD slice or LIVE_MAX_SEGMENT_SECONDS). The emit gate (see
# LiveVAD.feed) only emits once both this interval has elapsed AND ASR is idle.
LIVE_PROVISIONAL_MIN_INTERVAL_SECONDS = 1.5
# Silence duration that finalises a speech segment (passed to Silero VADIterator).
# Lower = splits more aggressively on phrase-level pauses (fillers, breaths),
# producing shorter, lower-latency subtitles. Too low can split mid-thought
# before a postposition / closing particle lands.
LIVE_MIN_SILENCE_MS = 400
# Hard cap on an in-progress segment. If exceeded, the segment is force-finalised
# so the LLM/ASR never sits on a runaway monologue. Boundary may chop mid-word.
# On force-flush, live_vad stashes the just-flushed PCM in a one-slot buffer so
# the pipeline can ask it to splice a precise audio range (the dropped trailing
# entry) into the next utterance — see LiveVAD.request_splice and the force-
# flush handling in the live pipeline.
LIVE_MAX_SEGMENT_SECONDS = 16.0
# Stage-by-stage drop threshold: a queued item older than this is dropped in
# favour of a fresher one.
LIVE_LAG_TOLERANCE_SECONDS = 24.0
# When a provisional sitting in the ASR or translate queue is older than this,
# the enqueue path collapses it into the newer provisional (same open segment,
# strictly more audio) instead of piling on. Keeps ASR and the LLM from chasing
# stale work on slow backends while preserving final accuracy — the next
# provisional cycle covers the same audio range plus the newly captured tail.
# Expressed as 3 emit cycles: tolerate a short burst of queued provs before
# declaring backlog. Scales with the interval so changing one knob preserves
# the policy.
LIVE_PROVISIONAL_BACKOFF_SECONDS = LIVE_PROVISIONAL_MIN_INTERVAL_SECONDS * 3


# Maximum gain applied by peak_normalize. Even on near-silent audio we cap
# amplification here so static/hiss can't explode without bound.
PEAK_NORMALIZE_MAX_DB = 20.0


def peak_normalize(pcm: np.ndarray) -> tuple[np.ndarray, float]:
    """Peak-normalise float32 mono PCM up to PEAK_NORMALIZE_MAX_DB.

    Returns (normalised_pcm, gain_db). Gain is the actual dB applied, which
    is min(headroom_to_target_peak, PEAK_NORMALIZE_MAX_DB). Target peak is
    0.99 to leave a hair of headroom against int16 clipping after encode_wav.

    No noise-floor gate: pure silence or static still gets amplified up to
    the cap so downstream VAD sees a consistent loudness floor. The cap is
    what prevents arbitrary explosion."""
    if pcm.size == 0:
        return pcm, 0.0
    peak = float(np.abs(pcm).max())
    if peak <= 0.0:
        return pcm, 0.0
    target = 0.99
    headroom = target / peak
    max_linear = 10.0 ** (PEAK_NORMALIZE_MAX_DB / 20.0)
    gain = min(headroom, max_linear)
    if gain <= 1.0:
        return pcm, 0.0
    out = np.clip(pcm * gain, -1.0, 1.0).astype(np.float32)
    gain_db = 20.0 * float(np.log10(gain))
    return out, gain_db


def encode_wav(pcm_float32: np.ndarray, sample_rate: int = LIVE_SAMPLE_RATE) -> bytes:
    """Encode a float32 mono PCM array as WAV bytes (int16)."""
    pcm_int16 = (np.clip(pcm_float32, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def get_loopback_mic():
    import sys
    import warnings

    import soundcard as sc

    if sys.platform == "win32":
        from soundcard.mediafoundation import SoundcardRuntimeWarning

        warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)
    elif sys.platform == "darwin":
        warnings.filterwarnings(
            "ignore",
            message="macOS does not support loopback recording functionality",
        )

    override = os.environ.get("LOOPBACK_DEVICE", "").strip()
    if override:
        mic = sc.get_microphone(id=override, include_loopback=True)
    else:
        mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
    log.info("capturing loopback from: %s", mic.name)
    return mic
