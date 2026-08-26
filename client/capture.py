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
LIVE_PROVISIONAL_MIN_INTERVAL_SECONDS = 0.1
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
# Early-split target. Silero finalises only on LIVE_MIN_SILENCE_MS of silence,
# so a continuous talker who never pauses that long is held until the hard
# LIVE_MAX_SEGMENT_SECONDS chop (which lands mid-word). Once an open primary
# segment grows past this target — and only when Silero has NOT already closed
# it on silence — a fallback split scan looks for a natural seam and finalises
# there instead, yielding a boundary somewhere in (TARGET, MAX) rather than
# waiting for the hard cap. The scan is a two-pass chain mirroring the tail of
# the offline file-mode ladder (DETECTOR_LADDER in ./client/vad.py): a
# webrtcvad span-gap pass first, then an energy quiet-window pass as last
# resort. If neither finds a seam, the existing 16 s force-flush still applies.
# The target value itself mirrors the file mode's CHUNK_MIN_SECONDS floor:
# much below it ASR starves for context and hallucinates; far above it one bad
# transcription poisons a long stretch.
LIVE_SPLIT_TARGET_SECONDS = 5.0
# A webrtcvad-detected inter-speech gap must be at least this long to count as
# an early-split point. Below LIVE_MIN_SILENCE_MS by design: the whole purpose
# is to cut on a dip too brief for Silero's silence threshold to finalise on.
# webrtcvad's own hysteresis means it only resolves gaps from ~500 ms up, so in
# practice this floor is reached only by the genuinely clear pauses; shorter
# dips fall through to the energy pass below.
LIVE_SPLIT_MIN_GAP_MS = 200
# Energy-based second-fallback split (mirrors _energy_seam in ./client/vad.py).
# Tried only when the webrtcvad pass finds no usable gap: scan the open
# segment's middle band for the quietest short window and cut there. This is
# what catches the sub-webrtcvad-resolution dips — a brief amplitude trough
# that is not a full silence but is the best available seam before the hard
# cap. Same scan-window size as the file-mode energy seam.
LIVE_SPLIT_QUIET_WINDOW_MS = 20
# Keep the energy cut away from the segment edges (fraction of the scanned
# band trimmed at each end) so we don't shave a sliver off the start/end.
LIVE_SPLIT_QUIET_EDGE_MARGIN = 0.2
# Need at least this many scan windows to bother with an energy cut; below it
# the band is too short to locate a meaningful trough.
LIVE_SPLIT_QUIET_MIN_WINDOWS = 3
# The quietest window only counts as a real seam if it is at most this fraction
# of the band's median energy — i.e. a genuine amplitude trough, not merely the
# least-loud moment of unbroken speech. Without this gate the energy pass would
# cut seamless continuous speech at an arbitrary ~5 s point; with it, truly
# gapless audio falls through to the 16 s hard cap. (File mode has no such gate
# because its energy seam only runs once every detector has failed on a window
# that must be cut anyway, so an always-cut there is correct; the live pass
# runs on every over-target scan.)
LIVE_SPLIT_QUIET_MAX_RATIO = 0.5
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
