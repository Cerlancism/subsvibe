from __future__ import annotations

import io
import logging
import wave

import numpy as np

log = logging.getLogger("subsvibe.capture")

LIVE_SAMPLE_RATE = 16000
LIVE_WINDOW_SECONDS = 5
LIVE_TICK_SECONDS = 1
LIVE_LAG_TOLERANCE_SECONDS = 5  # drop windows when capture is this far ahead of transcription


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
