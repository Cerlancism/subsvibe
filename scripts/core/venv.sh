#!/usr/bin/env bash
# Sourced by other scripts. Expects REPO_ROOT to be set by the caller.
# Provides: PYTHON, PIP
# SKIP_VENV=1 (e.g. Colab, via scripts/setup.colab.sh) uses the system
# interpreter instead of the project venv.

if [[ "${SKIP_VENV:-}" == "1" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
    PIP="$(command -v pip3 || command -v pip)"
elif [[ -f "$REPO_ROOT/.venv/Scripts/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/Scripts/python"
    PIP="$REPO_ROOT/.venv/Scripts/pip"
elif [[ -f "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
    PIP="$REPO_ROOT/.venv/bin/pip"
else
    echo "error: virtualenv not found - run: bash scripts/setup.sh"
    exit 1
fi
