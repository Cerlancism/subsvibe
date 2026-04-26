"""Download ASR and forced aligner models from HuggingFace before first request."""
import argparse
import model as _model

parser = argparse.ArgumentParser(description="Download SubsVibe transcription models")
parser.add_argument("--timestamps", action="store_true", help="Also download the forced aligner model")
args = parser.parse_args()

print(f"Downloading ASR model: {_model.TRANSCRIPT_MODEL_ID}")
_model.get_model()
print("ASR model ready.")

if args.timestamps:
    print(f"Downloading forced aligner: {_model.TRANSCRIPT_ALIGNER_ID}")
    _model.get_timestamp_model()
    print("Forced aligner ready.")
