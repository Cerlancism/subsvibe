"""Probe what ASR models hallucinate when given pure silence.

Whisper-family models rarely return empty text for silent audio; instead they
emit language-specific filler ("Thank you.", "ご視聴ありがとうございました", ...).
This manual integration test feeds silence samples to the running transcription
server once per (model x sample x language x repeat) and records every distinct
non-empty output into the dataset that server/hallucination_filter.py uses as its
blocklist. Per-sample run counts are kept too, so sweeping samples of different
lengths (tests/samples/silence_*s.mp3) shows which durations trigger
hallucinations most often.

The samples are gitignored, not checked in. Generate them first (needs ffmpeg
on PATH; writes 16 kHz mono 64 kbps mp3s of 1/2/3/5/10/15/30 seconds):

    bash scripts/dev/gen-silence-samples.sh

Usage (server must be running, see scripts/server.sh). Start it with
TRANSCRIPT_SILENCE_FILTER=0 - otherwise the server blanks already-known
hallucinations and the _runs counts under-report:

    python tests/test_silence_hallucinations.py
    python tests/test_silence_hallucinations.py --models Systran/faster-whisper-tiny
    python tests/test_silence_hallucinations.py --samples tests/samples/silence_*.mp3
    python tests/test_silence_hallucinations.py --languages ja --repeats 5

The backend kind is taken from TRANSCRIPT_BACKEND (it must match the server's),
and the model list defaults to that backend's common variants. Models are
switched server-side via the `model` form field, so each run of this script can
sweep every size variant of the configured backend in one go.

Results merge into server/data/silence_hallucinations.json - re-runs and runs
against other backends extend the dataset, never overwrite it. Top-level keys
are backend kinds; `_meta` describes the dataset and `_runs` holds the
per-sample run/hallucination counts behind the blocklists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import openai
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = [ROOT / "tests" / "samples" / "silence_5s.mp3"]
DEFAULT_OUT = ROOT / "server" / "data" / "silence_hallucinations.json"
DEFAULT_LANGUAGES = ["en", "es", "zh", "ja", "ko"]

TRANSCRIPT_HOST = os.environ.get("TRANSCRIPT_HOST", "127.0.0.1")
TRANSCRIPT_PORT = os.environ.get("TRANSCRIPT_PORT", "8000")
TRANSCRIPT_BASE_URL = os.environ.get(
    "TRANSCRIPT_BASE_URL", f"http://{TRANSCRIPT_HOST}:{TRANSCRIPT_PORT}/v1"
)
TRANSCRIPT_API_KEY = os.environ.get("TRANSCRIPT_API_KEY", "not-needed-locally")
TRANSCRIPT_BACKEND = os.environ.get("TRANSCRIPT_BACKEND", "qwen")

# First request after a model switch pays the full load (and possibly download).
REQUEST_TIMEOUT = 900.0

DEFAULT_MODELS: dict[str, list[str]] = {
    "faster-whisper": [
        "Systran/faster-whisper-large-v3",
        "Systran/faster-whisper-medium",
        "Systran/faster-whisper-small",
        "Systran/faster-whisper-base",
        "Systran/faster-whisper-tiny",
    ],
    "qwen": ["Qwen/Qwen3-ASR-1.7B"],
    "anime-whisper": ["litagin/anime-whisper"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend", default=TRANSCRIPT_BACKEND,
        help="backend kind used as the dataset's top-level key; must match the "
             f"server's TRANSCRIPT_BACKEND (default: {TRANSCRIPT_BACKEND!r})",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="model ids to sweep (default: the backend's common variants)",
    )
    parser.add_argument(
        "--languages", nargs="+", default=DEFAULT_LANGUAGES,
        help=f"ISO-639-1 codes to probe (default: {' '.join(DEFAULT_LANGUAGES)})",
    )
    parser.add_argument("--repeats", type=int, default=3, help="runs per combo (default: 3)")
    parser.add_argument(
        "--samples", nargs="+", type=Path, default=DEFAULT_SAMPLES,
        help="silence audio files (default: tests/samples/silence_5s.mp3)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="dataset JSON to merge into")
    return parser.parse_args()


def probe(
    client: OpenAI,
    sample: bytes,
    filename: str,
    model: str,
    language: str,
) -> str:
    result = client.audio.transcriptions.create(
        model=model,
        file=(filename, sample, "audio/mpeg"),
        language=language,
        response_format="json",
        timeout=REQUEST_TIMEOUT,
    )
    return (result if isinstance(result, str) else getattr(result, "text", "") or "").strip()


class Combo:
    """Observations for one (model, sample, language) cell."""

    def __init__(self) -> None:
        self.texts: set[str] = set()
        self.runs = 0
        self.hallucinated = 0

    def record(self, text: str) -> None:
        self.runs += 1
        if text:
            self.texts.add(text)
            self.hallucinated += 1


def merge_dataset(
    out_path: Path,
    backend: str,
    results: dict[str, dict[str, dict[str, Combo]]],
) -> dict:
    data: dict = {}
    if out_path.exists():
        data = json.loads(out_path.read_text(encoding="utf-8"))

    meta = data.setdefault("_meta", {})
    meta["description"] = (
        "Texts that ASR models emit for pure silence, per backend/model/language. "
        "Built by tests/test_silence_hallucinations.py; lists hold distinct "
        "non-empty outputs observed across runs (empty output = no hallucination). "
        "_runs counts runs/hallucinated per silence sample length."
    )
    meta.pop("sample", None)  # superseded by the per-sample _runs section
    samples_seen = set(meta.get("samples", []))
    meta["updated"] = date.today().isoformat()

    backend_map = data.setdefault(backend, {})
    runs_map = data.setdefault("_runs", {}).setdefault(backend, {})
    for model, per_sample in results.items():
        model_texts = backend_map.setdefault(model, {})
        model_runs = runs_map.setdefault(model, {})
        for sample_name, per_lang in per_sample.items():
            samples_seen.add(sample_name)
            for lang, combo in per_lang.items():
                merged = set(model_texts.get(lang, []))
                merged.update(combo.texts)
                model_texts[lang] = sorted(merged)

                cell = model_runs.setdefault(lang, {}).setdefault(
                    sample_name, {"runs": 0, "hallucinated": 0}
                )
                cell["runs"] += combo.runs
                cell["hallucinated"] += combo.hallucinated

    meta["samples"] = sorted(samples_seen)
    return data


def main() -> int:
    args = parse_args()

    samples: list[tuple[str, bytes]] = []
    for path in args.samples:
        if not path.is_file():
            print(f"sample not found: {path}", file=sys.stderr)
            return 1
        samples.append((path.name, path.read_bytes()))

    models = args.models or DEFAULT_MODELS.get(args.backend)
    if not models:
        print(f"no default models for backend {args.backend!r}; pass --models", file=sys.stderr)
        return 1
    if args.backend == "anime-whisper" and set(args.languages) != {"ja"}:
        print("note: anime-whisper ignores the language field (Japanese-only); "
              "non-ja results just duplicate ja")

    client = OpenAI(api_key=TRANSCRIPT_API_KEY, base_url=TRANSCRIPT_BASE_URL)
    print(f"server={TRANSCRIPT_BASE_URL} backend={args.backend}")
    print(f"samples={' '.join(name for name, _ in samples)} repeats={args.repeats}")

    # model -> sample name -> language -> observations
    results: dict[str, dict[str, dict[str, Combo]]] = {}
    failures = 0

    for model in models:
        print(f"\n=== {model} ===")
        per_sample = results.setdefault(model, {})
        bad_model = False
        for sample_name, sample_bytes in samples:
            per_lang = per_sample.setdefault(sample_name, {})
            for lang in args.languages:
                combo = per_lang.setdefault(lang, Combo())
                for rep in range(args.repeats):
                    t0 = time.monotonic()
                    try:
                        text = probe(client, sample_bytes, sample_name, model, lang)
                    except openai.BadRequestError as exc:
                        # Bad model id: the server rejects the switch and keeps
                        # its previous model, so every request for this model
                        # fails the same way - skip the rest of its sweep.
                        print(f"  [{sample_name} {lang}] 400 from server, skipping model: {exc}")
                        failures += 1
                        bad_model = True
                        break
                    except openai.APIConnectionError:
                        print(f"cannot reach {TRANSCRIPT_BASE_URL} - is scripts/server.sh running?",
                              file=sys.stderr)
                        return 1
                    elapsed = time.monotonic() - t0
                    combo.record(text)
                    print(f"  [{sample_name} {lang}] run {rep + 1}/{args.repeats} ({elapsed:.1f}s): {text!r}")
                if bad_model:
                    break
            if bad_model:
                results.pop(model, None)
                break
        if bad_model:
            continue

        print(f"  --- {model} summary ---")
        for sample_name, per_lang in per_sample.items():
            for lang, combo in per_lang.items():
                if combo.texts:
                    detail = f"{combo.hallucinated}/{combo.runs} hallucinated: " + " | ".join(sorted(combo.texts))
                else:
                    detail = f"0/{combo.runs} hallucinated"
                print(f"  {sample_name} {lang}: {detail}")

    if any(
        combo.runs
        for per_sample in results.values()
        for per_lang in per_sample.values()
        for combo in per_lang.values()
    ):
        data = merge_dataset(args.out, args.backend, results)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\ndataset merged into {args.out}")
    else:
        print("\nno runs completed - dataset not modified")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
