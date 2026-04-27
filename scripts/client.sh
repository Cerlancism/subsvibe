#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$REPO_ROOT/scripts/core/venv.sh"
source "$REPO_ROOT/scripts/env.sh"

# File mode (generates .srt next to the input):
PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --input "$@"

# File mode with language hint (ISO-639-1, skips auto-detect):
# PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --language ja --input "$@"

# Live loopback mode:
# PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --live --translate

# Live loopback mode with language hint:
# PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --live --translate --language ja

