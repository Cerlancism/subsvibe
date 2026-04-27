#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$REPO_ROOT/scripts/core/venv.sh"
source "$REPO_ROOT/scripts/env.sh"

# File mode (generates .srt next to the input):
PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --input "$@"

# File mode with language hint (ISO-639-1, skips auto-detect):
# PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --language ja --input "$@"

# File mode with prompt hint (bias ASR toward proper nouns / jargon):
# PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --prompt "Speakers: Hatsune Miku, Kagamine Rin. Topic: Vocaloid concert." --input "$@"

# Live loopback mode:
# PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --live --translate

# Live loopback mode with language hint:
# PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --live --translate --language ja

# Live loopback mode with prompt hint:
# PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --live --language ja --prompt "Anime stream. Characters: 鳴海ニコ, 神楽メア."

