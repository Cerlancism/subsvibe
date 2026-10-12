#!/usr/bin/env bash
# SubsVibe server environment configuration
# Copy to env.sh and customize for your setup

# ============================================================
# FastAPI Server Binding
# ============================================================
export TRANSCRIPT_HOST="0.0.0.0"
export TRANSCRIPT_PORT="8000"

# ============================================================
# Transcription (ASR) Model Backend
# ============================================================
# Which backend to load AT STARTUP. Supported: "faster-whisper", "qwen",
# "anime-whisper", "openrouter". Swap it later without restarting via POST /v1/backend, or
# from the client with `scripts/client.sh --backend <name>` - a step to run
# between sessions, not a flag to add to one.
# Read by BOTH server and client:
#   - server: selects which model implementation to load first
#   - client: the fallback SRT generation path when the server cannot be
#     reached. A session normally asks the server which backend is active and
#     follows that instead, so the two can never silently disagree:
#       * "faster-whisper": request segment timestamps and map Whisper's
#         own segments directly to SRT entries (no word-level pass)
#       * "qwen": request word timestamps, run punctuation attachment +
#         entries_from_words to build SRT entries from aligned words
#     The server also reports, per active model, how long a request may be
#     and whether it is long-form (an hour per request, even split) or
#     Whisper-shaped (30 s windows); the session follows that too.
export TRANSCRIPT_BACKEND="faster-whisper"

# There is no global model id: each backend starts on its own
# <BACKEND>_MODEL_ID below, which also names the model on /v1/models. A request
# naming another model switches the server to it at runtime; switching backends
# selects the model last used on the incoming backend this run, else its default.

# Reject audio longer than this many seconds; the client splits before POST.
# Shared by the local backends (the remote openrouter backend has its own,
# OPENROUTER_MAX_INPUT_SECONDS); the client adopts the active one from
# GET /v1/backend at session start, so it is a server-side setting only.
#export TRANSCRIPT_MAX_INPUT_SECONDS="180"

# Forced aligner for word/segment timestamps. Shared by the qwen and
# anime-whisper backends; faster-whisper aligns natively and never loads it.
#export TRANSCRIPT_ALIGNER_ID="Qwen/Qwen3-ForcedAligner-0.6B"

# Silence/noise hallucination filters: blank transcriptions whose whole text
# matches a known hallucination of the active backend/model/language. Two
# separately-toggleable datasets: pure silence
# (server/data/silence_hallucinations.json) and background noise/music
# (server/data/noise_hallucinations.json). Set either to "0" to disable.
export TRANSCRIPT_SILENCE_FILTER="1"
export TRANSCRIPT_NOISE_FILTER="1"

# Backend-specific settings are prefixed with the backend name so every
# backend's tuning can sit here side by side. Each has a built-in default
# (see server/backends/config.py); an unset or empty variable keeps it.

# --- Faster Whisper backend (TRANSCRIPT_BACKEND=faster-whisper) ---
# CTranslate2-converted Whisper. Available repos:
#   Systran/faster-whisper-large-v3, ...-large-v2, ...-medium,
#   Systran/faster-whisper-small, ...-base, ...-tiny.
export FASTER_WHISPER_MODEL_ID="Systran/faster-whisper-medium"
export FASTER_WHISPER_DEVICE=""         # "cuda", "cpu", or empty (auto)
export FASTER_WHISPER_COMPUTE_TYPE=""   # "float16", "int8_float16", "int8", or empty (auto)
export FASTER_WHISPER_BEAM_SIZE="5"
export FASTER_WHISPER_CPU_THREADS=""    # CPU-only: intra-op threads. Empty = physical-core estimate (logical/2)

# --- Qwen3-ASR backend (TRANSCRIPT_BACKEND=qwen) ---
export QWEN_MODEL_ID="Qwen/Qwen3-ASR-1.7B"
export QWEN_MAX_NEW_TOKENS="512"        # generation cap per request

# --- Anime Whisper backend (TRANSCRIPT_BACKEND=anime-whisper) ---
# Japanese-only ASR fine-tuned on anime/galgame speech (litagin/anime-whisper).
# Word/segment timestamps reuse TRANSCRIPT_ALIGNER_ID (Qwen ForcedAligner above).
export ANIME_WHISPER_MODEL_ID="litagin/anime-whisper"
# Generation tuning (defaults match the model card's recommended setup):
export ANIME_WHISPER_NO_REPEAT_NGRAM_SIZE="5"   # raise toward 10 if repetition appears
export ANIME_WHISPER_REPETITION_PENALTY="1.0"
export ANIME_WHISPER_CHUNK_LENGTH_S="30.0"
export ANIME_WHISPER_BATCH_SIZE="16"            # lower if you hit OOM

# --- OpenRouter backend (TRANSCRIPT_BACKEND=openrouter) ---
# Remote: the server re-encodes each chunk to Opus and forwards it to
# OpenRouter. microsoft/mai-transcribe-2 is long-form (up to an hour per
# request, verbatim, language auto-detected); any other id, such as the
# whisper ones, is fed in 30 s windows like a local Whisper. The key is read
# from the environment only; scripts/backends/env.openrouter.sh has every knob.
#export OPENROUTER_API_KEY=""
#export OPENROUTER_MODEL_ID="microsoft/mai-transcribe-2"
#export OPENROUTER_MODEL_ID="x-ai/grok-stt-1.0"
#export OPENROUTER_MODEL_ID="elevenlabs/scribe-v2"
#export OPENROUTER_MODEL_ID="openai/whisper-large-v3"

# Client connects here to reach the transcription server
# Use 127.0.0.1 instead of localhost on Windows - avoids IPv6 loopback delay
export TRANSCRIPT_BASE_URL="http://127.0.0.1:${TRANSCRIPT_PORT}/v1"
export TRANSCRIPT_API_KEY="not-needed-locally"  # Set to a real key for secured/remote backends

# ============================================================
# LLM Model Backend: Ollama (OpenAI-compatible)
# ============================================================
export LLM_BASE_URL="http://127.0.0.1:11434/v1"
export LLM_MODEL_ID="qwen3.5:4b"
export LLM_API_KEY="ollama"  # Ollama ignores this but the OpenAI client requires a value

# Optional: model used for transcription when the client is run with --llm-asr.
# Routes audio through the LLM backend (Ollama) instead of the FastAPI server.
export LLM_ASR_MODEL_ID="gemma4:e4b"

# ============================================================
# Client Audio Capture (loopback device override)
# ============================================================
# Optional. Name of the input device to capture from (matched against
# sc.all_microphones(include_loopback=True)). Empty = use default_speaker()
# loopback (WASAPI on Windows). Required on macOS — install BlackHole and
# set this to "BlackHole 2ch".
#export LOOPBACK_DEVICE=""

# ============================================================
# Client VAD Segmentation (file mode)
# ============================================================
# File mode chunks audio on the fly: coarse VAD picks one chunk per ASR call,
# the ASR's own segments become the subtitles, and each chunk's last entry is
# discarded as provisional so the next chunk starts where that utterance began.
# Hard upper/lower bound (seconds) on one coarse chunk — i.e. on how much audio
# goes into a single ASR request. Chunks average well above the minimum since
# the chunker takes the latest boundary that fits. The bounds belong to the
# model serving the request, so they are per backend, keyed by the backend the
# server reports at session start (defaults 30 / 5 for every backend; see
# client/chunking.py). --llm-asr and unknown backends use the defaults.
#export FASTER_WHISPER_CHUNK_MAX_SECONDS="30"
#export FASTER_WHISPER_CHUNK_MIN_SECONDS="5"
#export QWEN_CHUNK_MAX_SECONDS="30"
#export QWEN_CHUNK_MIN_SECONDS="5"
#export ANIME_WHISPER_CHUNK_MAX_SECONDS="30"
#export ANIME_WHISPER_CHUNK_MIN_SECONDS="5"
# The snap-back to a discarded provisional entry is deliberately uncapped: an
# overly long trailing entry signals degraded inference on the chunk's ragged
# edge, so it is re-run as the next chunk rather than kept.
# Passing --context-src <file>.srt skips coarse VAD altogether wherever that
# reference reaches: its entry starts are already speech onsets, so they are
# taken as the chunk boundaries directly (uncovered stretches still detect).
# A chunk's first entry start is moved forward onto the VAD speech onset,
# ANCHOR_LEAD_IN_SECONDS early (0 = exactly on it). The move is refused if it
# would leave the entry denser than its script's cap, in letters/digits per
# second: ANCHOR_MAX_CPS_CJK for entries containing CJK (tuned on Japanese),
# ANCHOR_MAX_CPS_LATIN otherwise. Defaults 0.12 / 11 / 20.
#export ANCHOR_LEAD_IN_SECONDS="0.12"
#export ANCHOR_MAX_CPS_CJK="11"
#export ANCHOR_MAX_CPS_LATIN="20"

# ============================================================
# Model Lifecycle: Idle Unload
# ============================================================
# After IDLE_UNLOAD_SECONDS without requests, models are unloaded to free VRAM
export IDLE_UNLOAD_SECONDS="120"
export IDLE_CHECK_SECONDS="10"

# ============================================================
# File Output Line Endings
# ============================================================
# Line ending for files SubsVibe writes (.srt subtitles, --log-file logs)
# and for its console output.
# "lf", "crlf", or "auto". Unset (the default) means auto: the platform
# default, i.e. CRLF on Windows and LF elsewhere.
#export SUBSVIBE_NEWLINE="auto"

# ============================================================
# PyTorch Installation
# ============================================================
# Customize the index URL for your CUDA version before running setup.sh.
# See https://pytorch.org/get-started for the right --index-url.
export PYTORCH_INSTALL_CMD="pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130"
