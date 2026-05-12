#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"
export PYTHONPATH="$REPO_ROOT"

"$REPO_ROOT/.venv/Scripts/python.exe" "$REPO_ROOT/server/server.py" &
PYTHON_PID=$!

trap 'kill -TERM "$PYTHON_PID" 2>/dev/null; wait "$PYTHON_PID" 2>/dev/null' INT TERM

wait "$PYTHON_PID"
