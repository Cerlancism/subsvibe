"""Download ASR (and forced aligner, when applicable) models before first request."""
import argparse
import os
import model as _model

TRANSCRIPT_BACKEND = os.environ.get("TRANSCRIPT_BACKEND", "qwen")
TRANSCRIPT_ALIGNER_ID = os.environ.get("TRANSCRIPT_ALIGNER_ID", "Qwen/Qwen3-ForcedAligner-0.6B")

parser = argparse.ArgumentParser(description="Download SubsVibe transcription models")
parser.add_argument("--timestamps", action="store_true", help="Also download the forced aligner model (qwen backend only)")
args = parser.parse_args()

print(f"Backend: {TRANSCRIPT_BACKEND}")
print("Downloading ASR model...")
_model.load_model()
print("ASR model ready.")

if args.timestamps:
    if TRANSCRIPT_BACKEND in {"faster-whisper", "faster_whisper"}:
        print("faster-whisper backend has built-in word/segment timestamps; no separate aligner to download.")
    else:
        print(f"Downloading forced aligner: {TRANSCRIPT_ALIGNER_ID}")
        _model.load_aligner()
        print("Forced aligner ready.")
