"""ASR backend configuration: constants first, environment overrides second.

Every tunable has a module-level constant holding its built-in value. Values
shared by more than one backend keep the ``TRANSCRIPT_`` prefix; values that
only one backend reads are prefixed with that backend's name
(``FASTER_WHISPER_``, ``QWEN_``, ``ANIME_WHISPER_``), so an env.sh can carry
the tuning for every backend side by side without one clobbering another.

Model workers run in spawn'd child processes that re-import this module, so
the module-level values reflect the environment at spawn time. The active
model id is not configured here: model.py owns it (it changes on every
model/backend switch) and hands it to the backend, which passes it to its
worker child at spawn.
"""
from __future__ import annotations

from utils.env import env_float, env_int, env_str


# ---------------------------------------------------------------------------
# Shared across backends (TRANSCRIPT_*)
# ---------------------------------------------------------------------------

# Reject audio longer than this; the client splits. Reported on GET
# /v1/backend so the client adopts the live value rather than mirroring it.
DEFAULT_MAX_INPUT_SECONDS = 180.0
MAX_INPUT_SECONDS = env_float("TRANSCRIPT_MAX_INPUT_SECONDS", DEFAULT_MAX_INPUT_SECONDS)

# Forced aligner for word/segment timestamps. Shared by qwen and anime-whisper
# (faster-whisper aligns natively and never loads it).
DEFAULT_ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
ALIGNER_ID = env_str("TRANSCRIPT_ALIGNER_ID", DEFAULT_ALIGNER_ID)

# Model each backend starts on, overridable per backend so env.sh pins a model
# for every backend at once. A transcription request naming another model
# switches the server to it at runtime (see model.py).
DEFAULT_FASTER_WHISPER_MODEL_ID = "Systran/faster-whisper-large-v3"
DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ANIME_WHISPER_MODEL_ID = "litagin/anime-whisper"

FASTER_WHISPER_MODEL_ID = env_str("FASTER_WHISPER_MODEL_ID", DEFAULT_FASTER_WHISPER_MODEL_ID)
QWEN_MODEL_ID = env_str("QWEN_MODEL_ID", DEFAULT_QWEN_MODEL_ID)
ANIME_WHISPER_MODEL_ID = env_str("ANIME_WHISPER_MODEL_ID", DEFAULT_ANIME_WHISPER_MODEL_ID)

# backend name -> model id a freshly started server uses on that backend.
# model.py derives SUPPORTED_BACKENDS from the keys.
DEFAULT_MODEL_IDS: dict[str, str] = {
    "faster-whisper": FASTER_WHISPER_MODEL_ID,
    "qwen": QWEN_MODEL_ID,
    "anime-whisper": ANIME_WHISPER_MODEL_ID,
}


# ---------------------------------------------------------------------------
# faster-whisper (FASTER_WHISPER_*)
# ---------------------------------------------------------------------------

# "cuda", "cpu", or empty for auto-detect.
DEFAULT_FASTER_WHISPER_DEVICE = ""
FASTER_WHISPER_DEVICE = env_str("FASTER_WHISPER_DEVICE", DEFAULT_FASTER_WHISPER_DEVICE)

# CTranslate2 compute type; empty derives float16 on CUDA, int8 on CPU.
DEFAULT_FASTER_WHISPER_COMPUTE_TYPE = ""
FASTER_WHISPER_COMPUTE_TYPE = env_str(
    "FASTER_WHISPER_COMPUTE_TYPE", DEFAULT_FASTER_WHISPER_COMPUTE_TYPE
)

DEFAULT_FASTER_WHISPER_BEAM_SIZE = 5
FASTER_WHISPER_BEAM_SIZE = env_int("FASTER_WHISPER_BEAM_SIZE", DEFAULT_FASTER_WHISPER_BEAM_SIZE)

# CPU-only intra-op threads. 0 = estimate physical cores (logical // 2).
DEFAULT_FASTER_WHISPER_CPU_THREADS = 0
FASTER_WHISPER_CPU_THREADS = env_int(
    "FASTER_WHISPER_CPU_THREADS", DEFAULT_FASTER_WHISPER_CPU_THREADS
)


# ---------------------------------------------------------------------------
# qwen (QWEN_*)
# ---------------------------------------------------------------------------

# Generation cap per request; 512 covers MAX_INPUT_SECONDS of speech.
DEFAULT_QWEN_MAX_NEW_TOKENS = 512
QWEN_MAX_NEW_TOKENS = env_int("QWEN_MAX_NEW_TOKENS", DEFAULT_QWEN_MAX_NEW_TOKENS)


# ---------------------------------------------------------------------------
# anime-whisper (ANIME_WHISPER_*)
# ---------------------------------------------------------------------------

# Generation defaults follow the model card's recommended setup.
DEFAULT_ANIME_WHISPER_NO_REPEAT_NGRAM_SIZE = 5
ANIME_WHISPER_NO_REPEAT_NGRAM_SIZE = env_int(
    "ANIME_WHISPER_NO_REPEAT_NGRAM_SIZE", DEFAULT_ANIME_WHISPER_NO_REPEAT_NGRAM_SIZE
)

DEFAULT_ANIME_WHISPER_REPETITION_PENALTY = 1.0
ANIME_WHISPER_REPETITION_PENALTY = env_float(
    "ANIME_WHISPER_REPETITION_PENALTY", DEFAULT_ANIME_WHISPER_REPETITION_PENALTY
)

DEFAULT_ANIME_WHISPER_CHUNK_LENGTH_S = 30.0
ANIME_WHISPER_CHUNK_LENGTH_S = env_float(
    "ANIME_WHISPER_CHUNK_LENGTH_S", DEFAULT_ANIME_WHISPER_CHUNK_LENGTH_S
)

DEFAULT_ANIME_WHISPER_BATCH_SIZE = 16
ANIME_WHISPER_BATCH_SIZE = env_int("ANIME_WHISPER_BATCH_SIZE", DEFAULT_ANIME_WHISPER_BATCH_SIZE)
