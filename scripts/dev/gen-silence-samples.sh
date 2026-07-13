#!/usr/bin/env bash
# Generate the silence test samples under tests/samples/ with ffmpeg.
# The mp3s are gitignored; run this after a fresh clone before
# tests/test_silence_hallucinations.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$REPO_ROOT/tests/samples"

command -v ffmpeg >/dev/null || { echo "error: ffmpeg not found on PATH"; exit 1; }

DURATIONS=(1 2 3 5 10 15 30)

for s in "${DURATIONS[@]}"; do
    out="$OUT_DIR/silence_${s}s.mp3"
    ffmpeg -y -loglevel error -f lavfi -i anullsrc=r=16000:cl=mono -t "$s" -b:a 64k "$out"
    echo "wrote $out"
done
