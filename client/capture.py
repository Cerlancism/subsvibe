from __future__ import annotations

import io
import logging
import wave

import numpy as np

log = logging.getLogger("subsvibe.capture")

LIVE_SAMPLE_RATE = 16000
# Silero VAD requires exactly this chunk size at 16 kHz.
LIVE_VAD_CHUNK_FRAMES = 512
# How often the recorder pulls from the OS (one VAD chunk = ~32 ms).
LIVE_CAPTURE_TICK_FRAMES = LIVE_VAD_CHUNK_FRAMES

# Provisional-update cadence: while a speech segment is open, re-transcribe the
# in-progress audio at most this often so the user sees a mid-sentence preview.
LIVE_PROVISIONAL_INTERVAL_SECONDS = 1.0
# Silence duration that finalises a speech segment (passed to Silero VADIterator).
# Lower = splits more aggressively on phrase-level pauses (fillers, breaths),
# producing shorter, lower-latency subtitles. Too low can split mid-thought
# before a postposition / closing particle lands.
LIVE_MIN_SILENCE_MS = 400
# Hard cap on an in-progress segment. If exceeded, the segment is force-finalised
# so the LLM/ASR never sits on a runaway monologue. Boundary may chop mid-word.
LIVE_MAX_SEGMENT_SECONDS = 10.0
# Stage-by-stage drop threshold: a queued item older than this is dropped in
# favour of a fresher one.
LIVE_LAG_TOLERANCE_SECONDS = 8.0
# When a provisional sitting in the ASR queue is older than this, the capture
# worker collapses it into the newer provisional (same open segment, strictly
# more audio) instead of piling on. Keeps ASR from chasing stale work on slow
# backends while preserving final accuracy — the next provisional cycle covers
# the same audio range plus the newly captured tail.
LIVE_PROVISIONAL_BACKOFF_SECONDS = 3.0


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
    import warnings

    import soundcard as sc
    from soundcard import SoundcardRuntimeWarning

    warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)
    mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
    log.info("capturing loopback from: %s", mic.name)
    return mic
