#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# resolve venv Python binary (Windows Scripts/ or Unix bin/)
if [[ -f "$REPO_ROOT/.venv/Scripts/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/Scripts/python"
elif [[ -f "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    echo "error: virtualenv not found at $REPO_ROOT/.venv — run setup first"
    exit 1
fi

cd "$REPO_ROOT/server"
"$PYTHON" download_models.py "$@"
