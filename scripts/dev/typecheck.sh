#!/usr/bin/env bash
# AST parse-check across client/server/utils/tests. Swap to pyright for real
# type checking: `pip install pyright && python -m pyright client server utils tests`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec "$REPO_ROOT/scripts/core/python.sh" "$REPO_ROOT/scripts/dev/typecheck.py"
