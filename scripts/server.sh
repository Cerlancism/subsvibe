#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

exec "$REPO_ROOT/scripts/core/python.sh" "$REPO_ROOT/server/server.py"
