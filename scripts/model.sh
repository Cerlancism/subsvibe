#!/usr/bin/env bash
# Manage the ASR model on the SubsVibe server.
# Usage:
#   scripts/model.sh health      — check server health and model state
#   scripts/model.sh load        — load the ASR model
#   scripts/model.sh unload      — unload the ASR model
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$REPO_ROOT/scripts/core/venv.sh"
source "$REPO_ROOT/scripts/env.sh"

ACTION="${1:-}"

case "$ACTION" in
  health)
    PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --health
    ;;
  load)
    PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --load
    ;;
  unload)
    PYTHONPATH="$REPO_ROOT" "$PYTHON" "$REPO_ROOT/client/client.py" --unload
    ;;
  *)
    echo "usage: $(basename "$0") {health|load|unload}"
    exit 1
    ;;
esac
