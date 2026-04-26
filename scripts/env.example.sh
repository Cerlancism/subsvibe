#!/usr/bin/env bash
# SubsVibe server environment configuration
# Copy to env.sh and customize for your setup

# ============================================================
# FastAPI Server Binding
# ============================================================
export TRANSCRIPT_HOST="0.0.0.0"
export TRANSCRIPT_PORT="8000"

# ============================================================
# Transcription (ASR) Model Backend: Qwen3-ASR
# ============================================================
# Model identifier (used in API responses)
export TRANSCRIPT_MODEL_NAME="qwen3-asr"

# Model weights location
export TRANSCRIPT_MODEL_ID="Qwen/Qwen3-ASR-1.7B"
export TRANSCRIPT_MODEL_PATH=""  # Leave empty to auto-download from HuggingFace

# Forced aligner for word/segment timestamps (required for timestamp_granularities)
export TRANSCRIPT_ALIGNER_ID="Qwen/Qwen3-ForcedAligner-0.6B"
export TRANSCRIPT_ALIGNER_PATH=""  # Leave empty to auto-download

# Client connects here to reach the transcription server
# Use 127.0.0.1 instead of localhost on Windows — avoids IPv6 loopback delay
export TRANSCRIPT_BASE_URL="http://127.0.0.1:${TRANSCRIPT_PORT}/v1"
export TRANSCRIPT_API_KEY="not-needed-locally"  # Set to a real key for secured/remote backends

# ============================================================
# LLM Model Backend: Ollama (OpenAI-compatible)
# ============================================================
export LLM_BASE_URL="http://127.0.0.1:11434/v1"
export LLM_MODEL_NAME="qwen3.5-instruct:4b"
export LLM_API_KEY="ollama"  # Ollama ignores this but the OpenAI client requires a value

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
export PYTORCH_INSTALL_CMD="pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps"
