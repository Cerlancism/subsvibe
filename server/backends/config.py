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

# Reject audio longer than this; the client splits. The local backends share
# this bound (Whisper-family models work in 30 s windows and the aligner in
# minutes); a remote long-form backend carries its own, see
# OPENROUTER_MAX_INPUT_SECONDS. Each backend reports its bound through
# Backend.profile(), and GET /v1/backend hands the active one to the client.
DEFAULT_MAX_INPUT_SECONDS = 180.0
MAX_INPUT_SECONDS = env_float("TRANSCRIPT_MAX_INPUT_SECONDS", DEFAULT_MAX_INPUT_SECONDS)

# Transcriptions a local backend runs at once on its resident model; the
# server queues the rest in arrival order. Shared by the local backends (the
# remote backend has OPENROUTER_MAX_CONCURRENT). Two lets one request's
# parent-side work (decode, pickling across to the worker, result handling)
# overlap another's inference; faster-whisper's CTranslate2 model also runs
# the two inferences together, while qwen and anime-whisper serialise their
# GPU work behind their own lock. A request naming a different model waits
# until nothing is in flight, then switches.
DEFAULT_MAX_CONCURRENT = 2
MAX_CONCURRENT = max(1, env_int("TRANSCRIPT_MAX_CONCURRENT", DEFAULT_MAX_CONCURRENT))

# Forced aligner for word/segment timestamps. Shared by qwen and anime-whisper
# (faster-whisper aligns natively and never loads it).
DEFAULT_ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
ALIGNER_ID = env_str("TRANSCRIPT_ALIGNER_ID", DEFAULT_ALIGNER_ID)

# Model each backend uses for requests that name none, overridable per
# backend so env.sh pins a model for every backend at once. On a local
# backend a request naming another model switches that backend to it (see
# model.py); the remote backend takes the model per request.
DEFAULT_FASTER_WHISPER_MODEL_ID = "Systran/faster-whisper-large-v3"
DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ANIME_WHISPER_MODEL_ID = "litagin/anime-whisper"
DEFAULT_OPENROUTER_MODEL_ID = "microsoft/mai-transcribe-2"

FASTER_WHISPER_MODEL_ID = env_str("FASTER_WHISPER_MODEL_ID", DEFAULT_FASTER_WHISPER_MODEL_ID)
QWEN_MODEL_ID = env_str("QWEN_MODEL_ID", DEFAULT_QWEN_MODEL_ID)
ANIME_WHISPER_MODEL_ID = env_str("ANIME_WHISPER_MODEL_ID", DEFAULT_ANIME_WHISPER_MODEL_ID)
OPENROUTER_MODEL_ID = env_str("OPENROUTER_MODEL_ID", DEFAULT_OPENROUTER_MODEL_ID)

# backend name -> model id a freshly started server uses on that backend.
# model.py derives SUPPORTED_BACKENDS from the keys.
DEFAULT_MODEL_IDS: dict[str, str] = {
    "faster-whisper": FASTER_WHISPER_MODEL_ID,
    "qwen": QWEN_MODEL_ID,
    "anime-whisper": ANIME_WHISPER_MODEL_ID,
    "openrouter": OPENROUTER_MODEL_ID,
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


# ---------------------------------------------------------------------------
# openrouter (OPENROUTER_*): remote, forwarded over HTTP (backends/remote.py)
# ---------------------------------------------------------------------------

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_BASE_URL = env_str("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)

# No default: the key lives in the environment only (scripts/backends/
# env.openrouter.sh maps it from wherever the machine keeps it).
OPENROUTER_API_KEY = env_str("OPENROUTER_API_KEY", "")

# Longest audio one request may carry, in seconds. mai-transcribe-2 answered
# 100 min in one request; the binding limit is the 50 MB JSON body, which an
# hour of Opus at the bitrate below fits with a little room.
DEFAULT_OPENROUTER_MAX_INPUT_SECONDS = 3600.0
OPENROUTER_MAX_INPUT_SECONDS = env_float(
    "OPENROUTER_MAX_INPUT_SECONDS", DEFAULT_OPENROUTER_MAX_INPUT_SECONDS
)

# Outbound timeouts, one per phase of a call. The upload bound covers the
# whole request body (`sendall` counts its timeout over the entire send): a
# near-cap FLAC body is ~50 MB, which 150 s only fits above ~2.7 Mbit/s up.
# The response bound is the wait from the end of the upload to the reply,
# then per read. The long-form models answer an hour chunk in about a
# minute (mai 100 min in 92 s, Scribe in 117 s); five minutes holds slow
# outliers while a hung provider is still retried rather than waited on.
# The client's file-mode timeout must stay above every attempt together.
DEFAULT_OPENROUTER_UPLOAD_TIMEOUT_SECONDS = 300.0
OPENROUTER_UPLOAD_TIMEOUT_SECONDS = env_float(
    "OPENROUTER_UPLOAD_TIMEOUT_SECONDS", DEFAULT_OPENROUTER_UPLOAD_TIMEOUT_SECONDS
)
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = 300.0
OPENROUTER_TIMEOUT_SECONDS = env_float(
    "OPENROUTER_TIMEOUT_SECONDS", DEFAULT_OPENROUTER_TIMEOUT_SECONDS
)

# Retries on timeouts, connection errors and 5xx only; a 4xx is final. Each
# retry re-uploads the chunk, so this stays small.
DEFAULT_OPENROUTER_MAX_RETRIES = 2
OPENROUTER_MAX_RETRIES = env_int("OPENROUTER_MAX_RETRIES", DEFAULT_OPENROUTER_MAX_RETRIES)

# Requests forwarded at once. Nothing runs locally, so the backend serves
# requests in parallel, each naming its own model; the server queues what
# arrives beyond this many until a slot frees. Bounds the open connections
# and in-flight re-encoded audio (an hour chunk is 115 MB of PCM plus its
# Opus body per request), not the provider's own rate limits.
DEFAULT_OPENROUTER_MAX_CONCURRENT = 8
OPENROUTER_MAX_CONCURRENT = max(
    1, env_int("OPENROUTER_MAX_CONCURRENT", DEFAULT_OPENROUTER_MAX_CONCURRENT)
)

# Opus bitrate the audio is re-encoded at before forwarding, in bits per
# second. Measured against lossless on mai-transcribe-2: 32k loses clauses,
# 64k 97.5% of characters, 80k 98.6% with nothing known missing, 96k no
# better - so 80k, which also keeps an hour under the body cap.
DEFAULT_OPENROUTER_OPUS_BITRATE = 80_000
OPENROUTER_OPUS_BITRATE = env_int("OPENROUTER_OPUS_BITRATE", DEFAULT_OPENROUTER_OPUS_BITRATE)

# Request body caps OpenRouter enforces, in bytes: 50 MB for the JSON
# transport (base64 `input_audio`), 25 MB for multipart. Audio is sent as
# lossless FLAC while the request fits, else as Opus at the bitrate above;
# lower the cap to force Opus sooner.
DEFAULT_OPENROUTER_JSON_CAP_BYTES = 50_000_000
OPENROUTER_JSON_CAP_BYTES = env_int("OPENROUTER_JSON_CAP_BYTES", DEFAULT_OPENROUTER_JSON_CAP_BYTES)
DEFAULT_OPENROUTER_MULTIPART_CAP_BYTES = 25_000_000
OPENROUTER_MULTIPART_CAP_BYTES = env_int(
    "OPENROUTER_MULTIPART_CAP_BYTES", DEFAULT_OPENROUTER_MULTIPART_CAP_BYTES
)
