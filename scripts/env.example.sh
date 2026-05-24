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
# Which backend to load. Supported: "faster-whisper", "qwen", "anime-whisper".
# Read by BOTH server and client:
#   - server: selects which model implementation to load
#   - client (file mode): selects the SRT generation path
#       * "faster-whisper": request segment timestamps and map Whisper's
#         own segments directly to SRT entries (no word-level pass)
#       * "qwen": request word timestamps, run punctuation attachment +
#         entries_from_words to build SRT entries from aligned words
export TRANSCRIPT_BACKEND="faster-whisper"

# TRANSCRIPT_MODEL_ID identifies the model both as the HuggingFace repo to
# load and as the model name returned by /v1/models (and required in the
# `model` field of /v1/audio/transcriptions).

# --- Faster Whisper backend (TRANSCRIPT_BACKEND=faster-whisper) ---
# CTranslate2-converted Whisper. Available repos:
#   Systran/faster-whisper-large-v3, ...-large-v2, ...-medium,
#   Systran/faster-whisper-small, ...-base, ...-tiny.
export TRANSCRIPT_MODEL_ID="Systran/faster-whisper-large-v3"
# Optional tuning (auto-derived if empty):
export TRANSCRIPT_DEVICE=""         # "cuda", "cpu", or empty (auto)
export TRANSCRIPT_COMPUTE_TYPE=""   # "float16", "int8_float16", "int8", or empty (auto)
export TRANSCRIPT_BEAM_SIZE="5"

# --- Qwen3-ASR backend (TRANSCRIPT_BACKEND=qwen) ---
# Override TRANSCRIPT_MODEL_ID above when switching:
#   export TRANSCRIPT_MODEL_ID="Qwen/Qwen3-ASR-1.7B"
export TRANSCRIPT_MODEL_PATH=""  # Leave empty to auto-download from HuggingFace

# Forced aligner for word/segment timestamps (required for timestamp_granularities)
export TRANSCRIPT_ALIGNER_ID="Qwen/Qwen3-ForcedAligner-0.6B"
export TRANSCRIPT_ALIGNER_PATH=""  # Leave empty to auto-download

# --- Anime Whisper backend (TRANSCRIPT_BACKEND=anime-whisper) ---
# Japanese-only ASR fine-tuned on anime/galgame speech (litagin/anime-whisper).
# Override TRANSCRIPT_MODEL_ID above when switching:
#   export TRANSCRIPT_MODEL_ID="litagin/anime-whisper"
# Word/segment timestamps reuse TRANSCRIPT_ALIGNER_ID (Qwen ForcedAligner above).
# Generation tuning (defaults match the model card's recommended setup):
export TRANSCRIPT_NO_REPEAT_NGRAM_SIZE="5"   # raise toward 10 if repetition appears
export TRANSCRIPT_REPETITION_PENALTY="1.0"
export TRANSCRIPT_CHUNK_LENGTH_S="30.0"
export TRANSCRIPT_BATCH_SIZE="16"            # lower if you hit OOM

# Client connects here to reach the transcription server
# Use 127.0.0.1 instead of localhost on Windows - avoids IPv6 loopback delay
export TRANSCRIPT_BASE_URL="http://127.0.0.1:${TRANSCRIPT_PORT}/v1"
export TRANSCRIPT_API_KEY="not-needed-locally"  # Set to a real key for secured/remote backends

# ============================================================
# LLM Model Backend: Ollama (OpenAI-compatible)
# ============================================================
export LLM_BASE_URL="http://127.0.0.1:11434/v1"
export LLM_MODEL_ID="frob/qwen3.5-instruct:4b"
export LLM_API_KEY="ollama"  # Ollama ignores this but the OpenAI client requires a value

# Optional: model used for transcription when the client is run with --llm-asr.
# Routes audio through the LLM backend (Ollama) instead of the FastAPI server.
export LLM_ASR_MODEL_ID="gemma4:e4b"

# ============================================================
# Client VAD Segmentation
# ============================================================
# Upper bound (seconds) on bundled speech segments and on hard-sliced pieces
# before they are sent to the ASR. Defaults to 30 if unset.
#export MAX_SEGMENT_SECONDS="30"
#export HARD_SLICE_SECONDS="30"

# ============================================================
# Model Lifecycle: Idle Unload
# ============================================================
# After IDLE_UNLOAD_SECONDS without requests, models are unloaded to free VRAM
export IDLE_UNLOAD_SECONDS="120"
export IDLE_CHECK_SECONDS="10"

# ============================================================
# PyTorch Installation
# ============================================================
# Customize the index URL for your CUDA version before running setup.sh.
# See https://pytorch.org/get-started for the right --index-url.
export PYTORCH_INSTALL_CMD="pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130"
