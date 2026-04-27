"""Download ASR and forced aligner models from HuggingFace before first request."""
import argparse
import os
import model as _model

TRANSCRIPT_MODEL_ID = os.environ.get("TRANSCRIPT_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
TRANSCRIPT_ALIGNER_ID = os.environ.get("TRANSCRIPT_ALIGNER_ID", "Qwen/Qwen3-ForcedAligner-0.6B")

parser = argparse.ArgumentParser(description="Download SubsVibe transcription models")
parser.add_argument("--timestamps", action="store_true", help="Also download the forced aligner model")
args = parser.parse_args()

print(f"Downloading ASR model: {TRANSCRIPT_MODEL_ID}")
_model.load_model()
print("ASR model ready.")

if args.timestamps:
    print(f"Downloading forced aligner: {TRANSCRIPT_ALIGNER_ID}")
    _model.load_aligner()
    print("Forced aligner ready.")
