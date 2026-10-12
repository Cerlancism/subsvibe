"""Long-form probe: mai-transcribe-2 one-shot / split through OpenRouter, SRT out.

Manual capability check, not a pass/fail gate. It talks to OpenRouter directly
(no SubsVibe server) and answers, for microsoft/mai-transcribe-2, what a
minutes-long request gives back and what our existing word-timestamp path
makes of it:

  - does a one-shot request of the whole file (~15 min) get through the
    25 MB multipart cap and the 60 s upstream timeout, and how long it takes;
  - whether splitting the file into two halves at a quiet seam near the
    midpoint changes the transcript (compare the two .srt outputs);
  - whether Azure diarization (`provider.options.azure.diarization.enabled`)
    is honoured, whether the `speaker` field lands on words, on segments, or
    on neither, and whether the multipart path forwards `provider` at all
    (docs only show it in the JSON body) - see --diarize-transport.

Whisper is not long-form; its probes and chunked file-mode run live in
test_openrouter_whisper.py.

Audio is sent as FLAC (16-bit mono 16 kHz): 15 min of WAV is ~29 MB, over the
cap. Note FLAC gains little on noisy material (the sample compresses to
~21 MB, whose base64 is ~28.5 MB), so the one-shot fits multipart but not
the JSON body on Azure, where base64 over the cap is rejected.

Output per run, all under tests/data (gitignored):

    <stem>.out.openrouter.<model slug>.<mode>[.diar].srt    entries from words
    <stem>.out.openrouter.<model slug>.<mode>[.diar].json   raw response(s)
    <stem>.out.openrouter.longform.json                      run summary

SRT entries are built exactly as file mode does on a word-timestamp backend:
attach_punctuation (./utils/text.py) + entries_from_words + write_srt
(./client/subtitle.py). With diarization, words are grouped into speaker runs
first and each entry is prefixed with its speaker label, so a speaker change
always starts a new entry.

Usage (python.sh sources scripts/env.sh, which exports OPENROUTER_API_KEY):

    scripts/core/python.sh tests/test_openrouter_longform.py --dry-run     # decode, seam, sizes; no network
    scripts/core/python.sh tests/test_openrouter_longform.py               # oneshot + split, plain + diarized
    scripts/core/python.sh tests/test_openrouter_longform.py --modes split --no-diarize
    scripts/core/python.sh tests/test_openrouter_longform.py --diarize-transport multipart   # does multipart carry provider?
    scripts/core/python.sh tests/test_openrouter_longform.py --language ""  # auto language id
    scripts/core/python.sh tests/test_openrouter_longform.py --rebuild      # re-derive .srt from saved .json, no network
    scripts/core/python.sh tests/test_openrouter_longform.py --diarize-only --max-speakers 11 --modes oneshot
    scripts/core/python.sh tests/test_openrouter_longform.py --modes oneshot --no-diarize --style clean
    scripts/core/python.sh tests/test_openrouter_longform.py --input tests/data/sample_100mins_ja.mp4 --modes oneshot --no-diarize --codec ogg --transport json --language ""

mai returns every character as its own word and punctuation marks as words of
their own, where our word path expects aligner words with punctuation in
`trailing` (what entries_from_words flushes on). words_from_body runs
normalize_words from openrouter_common first, which folds each mark into the
previous word's `trailing`; attach_punctuation then returns such words
untouched instead of re-deriving the marks from the response text.

Costs real money: a full run sends the file four times (one-shot, split, and
both again diarized).
"""
from __future__ import annotations

import argparse
import base64
import functools
import json
import subprocess
import tempfile
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import httpx
import openai
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

from openrouter_common import (  # noqa: E402
    DEFAULT_INPUT,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_TIMEOUT_SECONDS,
    SAMPLE_RATE,
    decode_file,
    encode_flac,
    encode_wav,
    make_client,
    normalize_words,
    out_base,
    print_table,
    quiet_seam,
    require_key,
)
from subtitle import entries_from_words, write_srt  # noqa: E402
from utils.text import attach_punctuation  # noqa: E402

MAI_MODEL = "microsoft/mai-transcribe-2"
DEFAULT_MODELS = [MAI_MODEL]
MODES = ("oneshot", "split")

# Azure diarization switch, JSON-body shape per the OpenRouter STT guide.
# `maxSpeakers` is Azure's own hint (fast transcription API, 2-35); OpenRouter
# forwards provider.options.<tag>.* with the provider's field names.
def diarize_provider(max_speakers: int | None) -> dict:
    diarization: dict = {"enabled": True}
    if max_speakers is not None:
        diarization["maxSpeakers"] = max_speakers
    return {"options": {"azure": {"diarization": diarization}}}


def azure_provider(*, diarize: bool, max_speakers: int | None, style: str | None) -> dict | None:
    """provider.options.azure.* for the JSON body: diarization and/or
    enhancedMode.modelOptions.transcribeStyle ("verbatim" default, "clean"
    drops fillers). None when nothing is requested so the plain multipart
    path stays plain."""
    azure: dict = {}
    if diarize:
        azure.update(diarize_provider(max_speakers)["options"]["azure"])
    if style:
        azure["enhancedMode"] = {"modelOptions": {"transcribeStyle": style}}
    return {"options": {"azure": azure}} if azure else None

Piece = tuple[float, float, bytes]  # (start_s, end_s, flac)


# --- requests ------------------------------------------------------------


def _ffmpeg_encode(pcm: np.ndarray, ext: str, *codec_args: str) -> bytes:
    """Lossy encode via ffmpeg (mono 16 kHz in). Exists to get a 100 min file
    under OpenRouter's 50 MB body cap; FLAC of that is 138 MB."""
    pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"audio.{ext}"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
             *codec_args, str(out)],
            input=pcm_int16.tobytes(), check=True,
        )
        return out.read_bytes()


def encode_ogg(pcm: np.ndarray, bitrate: str = "32k") -> bytes:
    return _ffmpeg_encode(pcm, "ogg", "-c:a", "libopus", "-b:a", bitrate, "-application", "voip")


def encode_mp3(pcm: np.ndarray, bitrate: str = "48k") -> bytes:
    return _ffmpeg_encode(pcm, "mp3", "-c:a", "libmp3lame", "-b:a", bitrate)


def encode_m4a(pcm: np.ndarray, bitrate: str = "64k") -> bytes:
    return _ffmpeg_encode(pcm, "m4a", "-c:a", "libfdk_aac", "-profile:a", "aac_he", "-b:a", bitrate, "-movflags", "+faststart")


# name -> (multipart filename, mime, encoder). Azure via OpenRouter (2026-10-12,
# 15 min probe): wav, flac, ogg/opus, mp3 accepted; every AAC (fdk HE-AAC,
# fdk LC, native, m4a and ADTS) and webm/opus rejected with provider 400.
# Lossy encoders take a bitrate (--bitrate); the defaults are what fit 100 min
# under the 50 MB JSON cap. Opus 32k vs FLAC moved ~4% of characters.
CODECS = {
    "flac": ("audio.flac", "audio/flac", encode_flac),
    "wav": ("audio.wav", "audio/wav", encode_wav),
    "ogg": ("audio.ogg", "audio/ogg", encode_ogg),
    "mp3": ("audio.mp3", "audio/mpeg", encode_mp3),
    "m4a": ("audio.m4a", "audio/mp4", encode_m4a),
}
LOSSY = ("ogg", "mp3", "m4a")


def post_multipart(client: OpenAI, *, model: str, flac: bytes, codec: str, language: str | None, provider: dict | None) -> dict:
    filename, mime, _ = CODECS[codec]
    kwargs: dict = {
        "model": model,
        "file": (filename, flac, mime),
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment", "word"],
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language
    if provider:
        # The SDK flattens nested dicts into bracketed form fields
        # (provider[options][azure][diarization][enabled]); whether OpenRouter
        # reads those is exactly what --diarize-transport multipart tests.
        kwargs["extra_body"] = {"provider": provider}
    raw = client.audio.transcriptions.with_raw_response.create(**kwargs)
    return {
        "status": raw.status_code,
        "generation_id": raw.headers.get("x-generation-id"),
        "body": json.loads(raw.text),
    }


def post_json(http: httpx.Client, *, model: str, flac: bytes, codec: str, language: str | None, provider: dict | None) -> dict:
    body: dict = {
        "model": model,
        "input_audio": {"data": base64.b64encode(flac).decode("ascii"), "format": "mp4" if codec == "m4a" else codec},
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment", "word"],
        "temperature": 0.0,
    }
    if language:
        body["language"] = language
    if provider:
        body["provider"] = provider
    resp = http.post("/audio/transcriptions", json=body)
    out = {"status": resp.status_code, "generation_id": resp.headers.get("x-generation-id")}
    try:
        out["body"] = resp.json()
    except ValueError:
        out["body"] = {"error": resp.text}
    if resp.status_code >= 400:
        raise openai.APIStatusError(message=resp.text, response=resp, body=out["body"])
    return out


def request(
    *,
    transport: str,
    client: OpenAI,
    http: httpx.Client,
    model: str,
    flac: bytes,
    codec: str,
    language: str | None,
    provider: dict | None,
) -> dict:
    t0 = time.perf_counter()
    record: dict = {"transport": transport, "codec": codec, "bytes": len(flac), "provider": provider}
    try:
        if transport == "json":
            record.update(post_json(http, model=model, flac=flac, codec=codec, language=language, provider=provider))
        else:
            record.update(post_multipart(client, model=model, flac=flac, codec=codec, language=language, provider=provider))
    except openai.APIStatusError as e:
        record["status"] = e.status_code
        try:
            record["error"] = e.response.json()
        except Exception:
            record["error"] = e.response.text if e.response is not None else str(e)
    except (openai.APIError, httpx.HTTPError) as e:
        record["status"] = None
        record["error"] = f"{type(e).__name__}: {e}"
    record["latency_s"] = round(time.perf_counter() - t0, 3)
    return record


# --- words -> entries ----------------------------------------------------


def words_from_body(body: dict, offset: float) -> tuple[list[dict], str, dict]:
    """Words shifted by `offset`, the full text, and a summary of what the
    response carried (speaker on words / segments, counts). normalize_words
    folds punctuation-only words into the previous word's `trailing` and
    end, so the words already carry the marks entries_from_words flushes on
    and attach_punctuation leaves them as they are."""
    text = (body.get("text") or "").strip()
    raw_words = body.get("words") or []
    segments = body.get("segments") or []
    seg_speakers = [s for s in segments if s.get("speaker") not in (None, "")]
    words: list[dict] = []
    normalized = normalize_words([dict(w) for w in raw_words])
    n_punct = len(raw_words) - len(normalized)
    for w in normalized:
        token = str(w.get("word") or w.get("text") or "").strip()
        if not token:
            continue
        w_start = float(w.get("start", 0.0))
        w_end = float(w.get("end", 0.0))
        word = {
            "text": str(token),
            "trailing": str(w.get("trailing") or ""),
            "start": offset + w_start,
            "end": offset + w_end,
        }
        speaker = w.get("speaker")
        if speaker in (None, "") and seg_speakers:
            mid = (w_start + w_end) / 2
            for s in seg_speakers:
                if float(s.get("start", 0)) <= mid <= float(s.get("end", 0)):
                    speaker = s["speaker"]
                    break
        if speaker not in (None, ""):
            word["speaker"] = str(speaker)
        words.append(word)
    summary = {
        "n_segments": len(segments),
        "n_words": len(raw_words),
        "n_punctuation_words": n_punct,
        "speaker_on_words": any(w.get("speaker") not in (None, "") for w in raw_words),
        "speaker_on_segments": bool(seg_speakers),
        "speakers": sorted({w["speaker"] for w in words if "speaker" in w}),
        "language": body.get("language"),
        "duration": body.get("duration"),
        "usage": body.get("usage"),
        "keys": sorted(body.keys()),
    }
    if segments:
        summary["segment_keys"] = sorted(segments[0].keys())
    if raw_words:
        summary["word_keys"] = sorted(raw_words[0].keys())
    return words, text, summary


def entries_from_response(words: list[dict], text: str, *, fallback: tuple[float, float]) -> list[dict]:
    """File mode's word path (attach_punctuation + entries_from_words), with
    speaker runs kept apart and labelled when the words carry speakers."""
    if not words:
        return [{"start": fallback[0], "end": fallback[1], "text": text}] if text else []
    enriched = attach_punctuation(words, text)
    if not any("speaker" in w for w in enriched):
        return entries_from_words(enriched)
    entries: list[dict] = []
    run: list[dict] = []
    speaker: str | None = None
    for w in enriched:
        s = w.get("speaker")
        if run and s != speaker:
            entries.extend(_label(entries_from_words(run), speaker))
            run = []
        speaker = s
        run.append(w)
    if run:
        entries.extend(_label(entries_from_words(run), speaker))
    return entries


def _label(entries: list[dict], speaker: str | None) -> list[dict]:
    if speaker in (None, ""):
        return entries
    return [{**e, "text": f"[{speaker}] {e['text']}"} for e in entries]


# --- runs ----------------------------------------------------------------


def run_one(
    *,
    model: str,
    mode: str,
    diarize: bool,
    max_speakers: int | None,
    style: str | None,
    pieces: list[Piece],
    transport: str,
    codec: str,
    bitrate: str | None,
    client: OpenAI,
    http: httpx.Client,
    language: str | None,
    base: Path,
) -> dict:
    provider = azure_provider(diarize=diarize, max_speakers=max_speakers, style=style)
    tag = mode
    if codec in LOSSY:
        tag += f".{codec}{bitrate or ''}"
    if diarize:
        tag += f".diar{max_speakers}" if max_speakers is not None else ".diar"
    if style:
        tag += f".{style}"
    slug = model.split("/")[-1]
    srt_path = base.with_name(f"{base.name}.out.openrouter.{slug}.{tag}.srt")
    json_path = srt_path.with_suffix(".json")

    all_entries: list[dict] = []
    requests: list[dict] = []
    ok = True
    for start, end, flac in pieces:
        print(f"-> {model} {tag} [{start:.1f}-{end:.1f}s, {len(flac) / 1e6:.2f} MB {codec}, {transport}] ...", end="", flush=True)
        rec = request(transport=transport, client=client, http=http, model=model, flac=flac, codec=codec,
                      language=language, provider=provider)
        rec["piece"] = {"start": start, "end": end}
        print(f" {rec.get('status')} in {rec['latency_s']:.1f}s")
        if "body" in rec and rec.get("status") == 200 and "error" not in rec["body"]:
            words, text, summary = words_from_body(rec["body"], start)
            rec["summary"] = summary
            all_entries.extend(entries_from_response(words, text, fallback=(start, end)))
        else:
            ok = False
            print(f"   error: {json.dumps(rec.get('error') or rec.get('body'), ensure_ascii=False)[:400]}")
        requests.append(rec)

    result = {
        "model": model,
        "mode": mode,
        "diarize": diarize,
        "style": style,
        "transport": transport,
        "codec": codec,
        "ok": ok,
        "n_entries": len(all_entries),
        "latency_s": round(sum(r["latency_s"] for r in requests), 3),
        "cost": sum(((r.get("summary") or {}).get("usage") or {}).get("cost", 0) or 0 for r in requests),
        "speakers": sorted({s for r in requests for s in (r.get("summary") or {}).get("speakers", [])}),
        "requests": requests,
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if all_entries:
        write_srt(all_entries, srt_path)
        print(f"   wrote {srt_path.name} ({len(all_entries)} entries before post-processing)")
    return result


def rebuild(args) -> None:
    """Re-derive every .srt from the saved raw .json files, no requests."""
    base = out_base(args.input)
    paths = sorted(base.parent.glob(f"{base.name}.out.openrouter.*.json"))
    paths = [p for p in paths if not p.name.endswith(".longform.json")]
    if not paths:
        raise SystemExit(f"error: no saved runs next to {args.input}")
    for json_path in paths:
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        entries: list[dict] = []
        for rec in saved.get("requests", []):
            if rec.get("status") != 200 or "body" not in rec:
                continue
            start, end = rec["piece"]["start"], rec["piece"]["end"]
            words, text, summary = words_from_body(rec["body"], start)
            rec["summary"] = summary
            entries.extend(entries_from_response(words, text, fallback=(start, end)))
        json_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        srt_path = json_path.with_suffix(".srt")
        if entries:
            write_srt(entries, srt_path)
        print(f"rebuilt {srt_path.name} ({len(entries)} entries before post-processing)")


def prepare_pieces(args) -> tuple[dict[str, list[Piece]], float]:
    """Decode once, pick the seam, encode the one-shot and the halves (--codec)."""
    pcm = decode_file(args.input)
    total = len(pcm) / SAMPLE_RATE
    seam = quiet_seam(pcm, args.split_at if args.split_at is not None else total / 2)
    seam_i = int(seam * SAMPLE_RATE)
    print(f"input: {args.input.name} {total:.1f}s; seam at {seam:.2f}s")

    t0 = time.perf_counter()
    encode = CODECS[args.codec][2]
    if args.bitrate:
        if args.codec not in LOSSY:
            raise SystemExit(f"error: --bitrate only applies to {', '.join(LOSSY)}")
        encode = functools.partial(encode, bitrate=args.bitrate)
    pieces_by_mode: dict[str, list[Piece]] = {}
    if "oneshot" in args.modes:
        pieces_by_mode["oneshot"] = [(0.0, total, encode(pcm))]
    if "split" in args.modes:
        pieces_by_mode["split"] = [
            (0.0, seam, encode(pcm[:seam_i])),
            (seam, total, encode(pcm[seam_i:])),
        ]
    print(f"{args.codec} encode: {time.perf_counter() - t0:.1f}s")
    for mode, pieces in pieces_by_mode.items():
        for start, end, flac in pieces:
            b64 = len(flac) * 4 / 3
            print(f"  {mode}: {start:.1f}-{end:.1f}s  {args.codec} {len(flac) / 1e6:.2f} MB  (base64 {b64 / 1e6:.2f} MB)")
    return pieces_by_mode, seam


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--language", default="ja", help="ISO-639-1; '' lets the model detect")
    ap.add_argument("--split-at", type=float, default=None, help="seam target in seconds (default: midpoint)")
    ap.add_argument("--transport", choices=("multipart", "json"), default="multipart", help="plain runs")
    ap.add_argument("--bitrate", default=None, help="lossy codec bitrate, e.g. 64k (defaults: ogg 32k, mp3 48k, m4a 64k); output tag gets .<codec><bitrate>")
    ap.add_argument("--codec", choices=tuple(CODECS), default="flac", help="audio sent: flac/wav lossless; ogg (opus 32k) / mp3 (48k) lossy and accepted; m4a (HE-AAC) kept as the rejected case")
    ap.add_argument("--no-diarize", action="store_true", help="skip the diarized runs")
    ap.add_argument("--diarize-only", action="store_true", help="skip the plain runs")
    ap.add_argument("--max-speakers", type=int, default=None,
                    help="Azure diarization.maxSpeakers hint (2-35); output tag becomes .diar<N>")
    ap.add_argument("--style", choices=("verbatim", "clean"), default=None,
                    help="Azure transcribeStyle; sent in the JSON body, output tag gets .<style>")
    ap.add_argument("--diarize-transport", choices=("multipart", "json"), default="json",
                    help="how the provider.options.azure.diarization field is sent (docs show JSON body only)")
    ap.add_argument("--dry-run", action="store_true", help="decode, pick the seam, print sizes, send nothing")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the .srt files from saved .json runs; no requests")
    args = ap.parse_args()

    if args.rebuild:
        rebuild(args)
        return
    if not args.dry_run:
        require_key()
    if not args.input.exists():
        raise SystemExit(f"error: input not found: {args.input}")

    pieces_by_mode, seam = prepare_pieces(args)
    if args.dry_run:
        return

    client = make_client()
    http = httpx.Client(base_url=OPENROUTER_BASE_URL, timeout=OPENROUTER_TIMEOUT_SECONDS, trust_env=True,
                        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"})
    language = args.language or None
    base = out_base(args.input)

    results: list[dict] = []
    for model in args.models:
        for mode in args.modes:
            pieces = pieces_by_mode[mode]
            # provider options only travel in the JSON body
            plain_transport = "json" if args.style else args.transport
            variants: list[tuple[bool, str]] = [] if args.diarize_only else [(False, plain_transport)]
            if not args.no_diarize:
                variants.append((True, args.diarize_transport))
            for diarize, transport in variants:
                results.append(run_one(
                    model=model, mode=mode, diarize=diarize, max_speakers=args.max_speakers,
                    style=args.style, pieces=pieces, transport=transport, codec=args.codec, bitrate=args.bitrate,
                    client=client, http=http, language=language, base=base,
                ))

    print()
    print_table(
        ("model", "mode", "diar", "transport", "ok", "lat_s", "cost", "entries", "speakers"),
        [(
            r["model"].split("/")[-1], r["mode"], "y" if r["diarize"] else "-", r["transport"],
            "ok" if r["ok"] else "FAIL", f"{r['latency_s']:.1f}", f"{r['cost']:.4f}", str(r["n_entries"]),
            ",".join(r["speakers"]) or "-",
        ) for r in results],
    )

    summary_path = base.with_name(f"{base.name}.out.openrouter.longform.json")
    summary_path.write_text(json.dumps({
        "_meta": {
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input": str(args.input), "seam": seam, "language": language,
        },
        "runs": [{k: v for k, v in r.items() if k != "requests"} for r in results],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
