#!/usr/bin/env bash
# Sourced by other scripts. Expects REPO_ROOT to be set by the caller.
# Provides: PYTHON, PIP

if [[ -f "$REPO_ROOT/.venv/Scripts/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/Scripts/python"
    PIP="$REPO_ROOT/.venv/Scripts/pip"
elif [[ -f "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
    PIP="$REPO_ROOT/.venv/bin/pip"
else
    echo "error: virtualenv not found — run: bash scripts/setup.sh"
    exit 1
fi
