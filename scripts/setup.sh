#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Creating virtual environment..."
python -m venv "$REPO_ROOT/.venv"

source "$REPO_ROOT/scripts/core/venv.sh"
source "$REPO_ROOT/scripts/env.sh"

echo "Installing PyTorch..."
eval "$PYTORCH_INSTALL_CMD"

echo "Installing pip-tools..."
"$PIP" install --quiet pip-tools

echo "Compiling dependencies..."
"$PYTHON" -m piptools compile "$REPO_ROOT/requirements/requirements.in" \
    -o "$REPO_ROOT/requirements/requirements.txt"

echo "Installing dependencies..."
"$PYTHON" -m piptools sync "$REPO_ROOT/requirements/requirements.txt"

echo "Downloading models..."
bash "$REPO_ROOT/scripts/core/download_models.sh" --timestamps

echo "Setup complete. Run: bash scripts/server.sh"
