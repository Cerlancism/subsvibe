#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$REPO_ROOT/server"
exec "$REPO_ROOT/scripts/core/python.sh" download_models.py "$@"
