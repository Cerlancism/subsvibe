#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Creating virtual environment..."
python -m venv "$REPO_ROOT/.venv"

source "$REPO_ROOT/scripts/core/venv.sh"
source "$REPO_ROOT/scripts/env.sh"

echo "Installing PyTorch..."
eval "${PYTORCH_INSTALL_CMD/pip3/$PIP}"

echo "Installing pip-tools..."
"$PIP" install --quiet pip-tools

echo "Compiling dependencies..."
(cd "$REPO_ROOT" && "$PYTHON" -m piptools compile requirements.in -o requirements.txt)

echo "Installing dependencies..."
(cd "$REPO_ROOT" && "$PYTHON" -m piptools sync requirements.txt)

echo "Downloading UniDic (Japanese dictionary, ~1 GB)..."
"$PYTHON" -m unidic download

echo "Downloading models..."
bash "$REPO_ROOT/scripts/core/download_models.sh" --timestamps

echo "Setup complete. Run: bash scripts/server.sh"
