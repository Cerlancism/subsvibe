#!/usr/bin/env bash
# Setup for Google Colab: system interpreter, no virtualenv.
# Colab ships a CUDA-enabled PyTorch, so the torch install is normally a no-op.
# Deliberately avoids pip-sync - syncing against the system interpreter would
# uninstall Colab's preinstalled packages.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$REPO_ROOT/scripts/env.sh" ]]; then
    echo "Generating scripts/env.sh for Colab..."
    cp "$REPO_ROOT/scripts/env.example.sh" "$REPO_ROOT/scripts/env.sh"

    # Default PyPI wheels are CUDA-enabled on Linux; no --index-url needed.
    sed -i 's|^export PYTORCH_INSTALL_CMD=.*|export PYTORCH_INSTALL_CMD="pip3 install torch torchvision"|' \
        "$REPO_ROOT/scripts/env.sh"

    # Colab GPU defaults: larger ASR and LLM models than the local defaults.
    sed -i 's|^export TRANSCRIPT_MODEL_ID=.*|export TRANSCRIPT_MODEL_ID="Systran/faster-whisper-large-v3"|' \
        "$REPO_ROOT/scripts/env.sh"
    sed -i 's|^export LLM_MODEL_ID=.*|export LLM_MODEL_ID="qwen3.5:9b"|' \
        "$REPO_ROOT/scripts/env.sh"

    cat >> "$REPO_ROOT/scripts/env.sh" <<'EOF'

# ============================================================
# Colab: run on the system interpreter, no virtualenv
# ============================================================
export SKIP_VENV="1"
EOF
else
    echo "scripts/env.sh already exists, keeping it."
fi

source "$REPO_ROOT/scripts/env.sh"
source "$REPO_ROOT/scripts/core/venv.sh"

echo "Installing PyTorch..."
eval "${PYTORCH_INSTALL_CMD/pip3/$PIP}"

echo "Installing dependencies..."
(cd "$REPO_ROOT" && "$PIP" install -r requirements.in)

echo "Downloading models..."
bash "$REPO_ROOT/scripts/core/download_models.sh" --timestamps

echo "Setup complete. Run: bash scripts/server.sh"
