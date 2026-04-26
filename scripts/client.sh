#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$REPO_ROOT/scripts/core/venv.sh"
source "$REPO_ROOT/scripts/env.sh"

# File mode (with optional --translate):
# PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --timestamps segment --translate --input "$@"
# PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --translate --input "$@"

# Live loopback mode:
PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --live --translate

