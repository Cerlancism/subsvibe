#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec "$REPO_ROOT/scripts/core/python.sh" "$REPO_ROOT/scripts/dev/check_cuda.py"
