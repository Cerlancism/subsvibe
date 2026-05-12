#!/usr/bin/env bash
# Run a python command using the project venv with PYTHONPATH=REPO_ROOT.
# Usage:
#   scripts/core/python.sh path/to/script.py [args...]
#   scripts/core/python.sh -m pytest tests/foo.py
#   scripts/core/python.sh -c "import torch; print(torch.cuda.is_available())"
# Does NOT source scripts/env.sh - callers that need TRANSCRIPT_*/LLM_* env
# vars should source it themselves before calling.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/core/venv.sh"

exec env PYTHONPATH="$REPO_ROOT" "$PYTHON" "$@"
