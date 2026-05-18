#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

subsvibe() { "$REPO_ROOT/scripts/core/python.sh" "$REPO_ROOT/client/client.py" "$@"; }

subsvibe "$@"

# File mode (generates .srt next to the input):
# subsvibe --input "$@"

# File mode with language hint (ISO-639-1, skips auto-detect):
# subsvibe --language ja --input "$@"

# File mode with prompt hint (bias ASR toward proper nouns / jargon):
# subsvibe --prompt "Speakers: Hatsune Miku, Kagamine Rin. Topic: Vocaloid concert." --input "$@"

# File mode using Gemma-4 e4b with reference subtitle, history context, and custom prompt:
# subsvibe --llm-asr --language ja --prompt "Anime stream. Characters: 鳴海ニコ, 神楽メア." --context-src "reference.srt" --history 3 --input "$@"

# Live loopback mode:
# subsvibe --live --translate

# Live loopback mode with language hint:
# subsvibe --live --translate --language ja

# Live loopback mode with prompt hint:
# subsvibe --live --language ja --prompt "Anime stream. Characters: 鳴海ニコ, 神楽メア."

# Live loopback mode using Gemma-4 e4b (multimodal LLM ASR via Ollama, LLM_BASE_URL):
# subsvibe --live --llm-asr

# File mode using Gemma-4 e4b (writes .srt next to the input):
# subsvibe --llm-asr --input "$@"

# Override the model used with --llm-asr (defaults to LLM_ASR_MODEL_ID):
# subsvibe --live --llm-asr --model gemma4:e4b
