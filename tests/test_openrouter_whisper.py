"""Probe what OpenRouter's Whisper models return through the OpenAI audio API.

Manual capability check, not a pass/fail gate: it talks to OpenRouter directly
(no SubsVibe server involved). Two modes:

Window probes (default): per model, one request per probe, all on the same
30 s audio window, recording what each request shape gives back. Run before
wiring the `openrouter` server backend so its assumptions (native
segments/words under verbose_json, language auto-detect, latency against the
60 s upstream timeout) rest on observed responses rather than on the docs.

    json            response_format=json, language given          -> text, usage
    json-auto       response_format=json, language omitted        -> text (auto id)
    verbose         verbose_json, granularities=[segment]         -> segments?
    verbose-words   verbose_json, granularities=[word]            -> words?
    verbose-both    verbose_json, granularities=[segment, word]   -> both?
    verbose-auto    verbose_json, language omitted                -> `language` field?

Chunked file mode (--chunked): Whisper on OpenRouter is not long-form, so the
whole file goes through the real file-mode path - `transcribe_file` in
./client/client.py with its coarse chunker (default 30/5 s profile; the
`openrouter` name has no row yet) and the word-aligner SRT path
(attach_punctuation + entries_from_words + write_srt) - with the ASR client
pointed at OpenRouter. Output: tests/data/<stem>.out.openrouter.<slug>.chunked.srt.

Docs (https://openrouter.ai/docs/guides/overview/multimodal/stt) say: multipart
fields honoured are file/model/language/temperature/response_format/
timestamp_granularities[]; `prompt` is accepted but ignored; text/srt/vtt
formats 400 (verified 2026-10-11: "Only json and verbose_json are supported");
some models 400 on verbose_json; multipart body cap 25 MB; upstream timeout
60 s. `provider.order`/`only` are NOT applied to this endpoint, so whisper
requests route between DeepInfra and Groq (and Together for large-v3) on
their own - the `X-Generation-Id` header is kept per probe so a run can be
looked up on the dashboard if the provider matters.

Costs real money (fractions of a cent per 30 s window at 2026-10 pricing; a
--chunked run of a 15 min file is ~30 requests per model).

Usage (python.sh sources scripts/env.sh, which exports OPENROUTER_API_KEY):

    scripts/core/python.sh tests/test_openrouter_whisper.py
    scripts/core/python.sh tests/test_openrouter_whisper.py --input tests/data/foo.mp3 --offset 120 --seconds 30
    scripts/core/python.sh tests/test_openrouter_whisper.py --models openai/whisper-large-v3-turbo --probes json verbose
    scripts/core/python.sh tests/test_openrouter_whisper.py --flac       # also send the window as FLAC
    scripts/core/python.sh tests/test_openrouter_whisper.py --language ""  # never send language
    scripts/core/python.sh tests/test_openrouter_whisper.py --chunked    # whole file -> .srt per model
    scripts/core/python.sh tests/test_openrouter_whisper.py --chunked --models microsoft/mai-transcribe-2   # any model through the chunker

Window results print as a table and are written in full (raw response bodies
included) to --out, default tests/data/openrouter_probe.json (gitignored).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import openai
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

from openrouter_common import (  # noqa: E402
    DEFAULT_INPUT,
    OPENROUTER_BASE_URL,
    SAMPLE_RATE,
    decode_window,
    encode_flac,
    encode_wav,
    make_client,
    out_base,
    print_table,
    require_key,
    run_chunked_file_mode,
)
from utils.logging_config import setup_logging  # noqa: E402

DEFAULT_MODELS = [
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
]
DEFAULT_OUT = DEFAULT_INPUT.parent / "openrouter_probe.json"
# A 30 s window never needs the long-form timeout.
WINDOW_TIMEOUT_SECONDS = 75.0

# name -> (response_format, timestamp_granularities, send_language)
PROBES: dict[str, tuple[str, list[str] | None, bool]] = {
    "json": ("json", None, True),
    "json-auto": ("json", None, False),
    "verbose": ("verbose_json", ["segment"], True),
    "verbose-words": ("verbose_json", ["word"], True),
    "verbose-both": ("verbose_json", ["segment", "word"], True),
    "verbose-auto": ("verbose_json", ["segment"], False),
}


# --- window probes -------------------------------------------------------


def run_probe(
    client: OpenAI,
    *,
    model: str,
    probe: str,
    audio: bytes,
    filename: str,
    language: str | None,
) -> dict:
    response_format, granularities, send_language = PROBES[probe]
    kwargs: dict = {
        "model": model,
        "file": (filename, audio),
        "response_format": response_format,
        "temperature": 0.0,
    }
    if send_language and language:
        kwargs["language"] = language
    if granularities:
        kwargs["timestamp_granularities"] = granularities

    record: dict = {
        "model": model,
        "probe": probe,
        "request": {k: v for k, v in kwargs.items() if k != "file"},
        "bytes": len(audio),
        "filename": filename,
    }
    t0 = time.perf_counter()
    try:
        raw = client.audio.transcriptions.with_raw_response.create(**kwargs)
        record["status"] = raw.status_code
        record["headers"] = {
            k: v
            for k, v in raw.headers.items()
            if k.lower() in ("x-generation-id", "content-type", "x-request-id")
        }
        # Raw body, not raw.parse(): the SDK model would drop provider-specific
        # fields (usage, confidence, speaker) that this probe exists to see.
        body = json.loads(raw.text)
        record["body"] = body
        record.update(summarize(body))
    except openai.APIStatusError as e:
        record["status"] = e.status_code
        record["error"] = _error_payload(e)
    except openai.APIError as e:  # timeouts, connection errors
        record["status"] = None
        record["error"] = f"{type(e).__name__}: {e}"
    record["latency_s"] = round(time.perf_counter() - t0, 3)
    return record


def _error_payload(e: openai.APIStatusError):
    try:
        return e.response.json()
    except Exception:
        return e.response.text if e.response is not None else str(e)


def summarize(body: dict) -> dict:
    segments = body.get("segments")
    words = body.get("words")
    out = {
        "keys": sorted(body.keys()),
        "text": body.get("text", ""),
        "language": body.get("language"),
        "duration": body.get("duration"),
        "n_segments": len(segments) if isinstance(segments, list) else None,
        "n_words": len(words) if isinstance(words, list) else None,
        "usage": body.get("usage"),
    }
    if segments:
        out["segment_keys"] = sorted(segments[0].keys())
        out["first_segment"] = segments[0]
        out["last_segment"] = segments[-1]
    if words:
        out["word_keys"] = sorted(words[0].keys())
        out["first_words"] = words[:5]
    return out


def print_records(records: list[dict]) -> None:
    cols = ("model", "probe", "fmt", "status", "lat_s", "lang", "dur", "segs", "words", "text")
    rows = []
    for r in records:
        err = r.get("error")
        text = r.get("text") or (json.dumps(err, ensure_ascii=False)[:60] if err else "")
        rows.append(
            (
                r["model"].split("/")[-1],
                r["probe"],
                Path(r["filename"]).suffix[1:],
                str(r.get("status")),
                f"{r['latency_s']:.2f}",
                str(r.get("language") or "-"),
                str(r.get("duration") if r.get("duration") is not None else "-"),
                str(r.get("n_segments") if r.get("n_segments") is not None else "-"),
                str(r.get("n_words") if r.get("n_words") is not None else "-"),
                text[:48].replace("\n", " "),
            )
        )
    print_table(cols, rows)


def run_windows(args, language: str | None) -> None:
    pcm = decode_window(args.input, args.offset, args.seconds)
    payloads = [("window.wav", encode_wav(pcm))]
    if args.flac:
        payloads.append(("window.flac", encode_flac(pcm)))
    print(f"window: {args.input.name} @ {args.offset}s +{len(pcm) / SAMPLE_RATE:.1f}s")
    for name, data in payloads:
        print(f"  {name}: {len(data) / 1e6:.2f} MB")
    if args.dry_run:
        return

    client = make_client(WINDOW_TIMEOUT_SECONDS)
    records: list[dict] = []
    for model in args.models:
        for filename, audio in payloads:
            for probe in args.probes:
                print(f"-> {model} {probe} ({filename}) ...", end="", flush=True)
                rec = run_probe(
                    client, model=model, probe=probe, audio=audio, filename=filename, language=language
                )
                records.append(rec)
                print(f" {rec.get('status')} in {rec['latency_s']:.2f}s")

    print()
    print_records(records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "_meta": {
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "base_url": OPENROUTER_BASE_URL,
            "input": str(args.input),
            "offset": args.offset,
            "seconds": args.seconds,
            "language": language,
        },
        "records": records,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


# --- chunked file mode ----------------------------------------------------


def run_chunked(args, language: str | None) -> None:
    if args.dry_run:
        print(f"chunked: would run transcribe_file on {args.input.name} for {', '.join(args.models)}")
        return
    setup_logging(logging.INFO)
    client = make_client()
    base = out_base(args.input)
    results = []
    for model in args.models:
        slug = model.split("/")[-1]
        srt_path = base.with_name(f"{base.name}.out.openrouter.{slug}.chunked.srt")
        results.append(run_chunked_file_mode(
            model=model, client=client, path=args.input, language=language, srt_path=srt_path,
        ))
    print()
    print_table(
        ("model", "ok", "lat_s", "requests", "billed_s", "cost", "srt"),
        [(r["model"].split("/")[-1], "ok" if r["ok"] else "FAIL", f"{r['latency_s']:.1f}",
          str(r["requests"]), f"{r['seconds']:.0f}", f"{r['cost']:.4f}", Path(r["srt"]).name)
         for r in results],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--offset", type=float, default=60.0, help="window start in seconds")
    ap.add_argument("--seconds", type=float, default=30.0, help="window length in seconds")
    ap.add_argument("--language", default="ja", help="ISO-639-1 for the language-given probes; '' never sends it")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--probes", nargs="+", default=list(PROBES), choices=list(PROBES))
    ap.add_argument("--flac", action="store_true", help="send each probe as FLAC too, not only WAV")
    ap.add_argument("--chunked", action="store_true", help="instead of window probes: whole file through transcribe_file -> .srt")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true", help="decode + encode, print sizes, send nothing")
    args = ap.parse_args()

    if not args.dry_run:
        require_key()
    if not args.input.exists():
        raise SystemExit(f"error: input not found: {args.input}")
    language = args.language or None

    if args.chunked:
        run_chunked(args, language)
    else:
        run_windows(args, language)


if __name__ == "__main__":
    main()
