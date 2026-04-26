from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI

from utils.logging_config import setup_logging

setup_logging()
log = logging.getLogger("subsvibe.client")

TRANSCRIPT_HOST = os.environ.get("TRANSCRIPT_HOST", "127.0.0.1")
TRANSCRIPT_PORT = os.environ.get("TRANSCRIPT_PORT", "8000")
TRANSCRIPT_MODEL_NAME = os.environ.get("TRANSCRIPT_MODEL_NAME", "qwen3-asr")
TRANSCRIPT_BASE_URL = os.environ.get("TRANSCRIPT_BASE_URL", f"http://{TRANSCRIPT_HOST}:{TRANSCRIPT_PORT}")
TRANSCRIPT_API_KEY = os.environ.get("TRANSCRIPT_API_KEY", "not-needed-locally")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "qwen3.5-instruct:4b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

_client = OpenAI(api_key=TRANSCRIPT_API_KEY, base_url=TRANSCRIPT_BASE_URL)
_llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


_TRANSLATE_SYSTEM = (
    "You are a subtitle translator. Translate the given text to English. "
    "Output only the translation, no explanations."
)


def _translate(text: str) -> str:
    t0 = time.monotonic()
    resp = _llm_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": _TRANSLATE_SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=0.3,
    )
    elapsed = time.monotonic() - t0
    result = resp.choices[0].message.content or ""
    log.info("translated in %.2fs", elapsed)
    return result.strip()


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _print_timestamps(done: dict, granularity: str, *, translate: bool = False) -> None:
    if granularity == "word":
        for w in done.get("words") or []:
            start = _fmt_ts(_get(w, "start", 0))
            end = _fmt_ts(_get(w, "end", 0))
            print(f"  [{start} -> {end}]  {_get(w, 'word', '')}")
    elif granularity == "segment":
        for seg in done.get("segments") or []:
            start = _fmt_ts(_get(seg, "start", 0))
            end = _fmt_ts(_get(seg, "end", 0))
            text = _get(seg, "text", "")
            print(f"  [{start} -> {end}]  {text}")
            if translate and text:
                print(f"               -> {_translate(text)}")


def transcribe_file(
    path: Path,
    *,
    model: str,
    language: str | None,
    response_format: str,
    timestamps: str,
    translate: bool,
) -> None:
    with path.open("rb") as f:
        size = path.stat().st_size
        log.info("sending %s (%dB) -> %s", path.name, size, TRANSCRIPT_BASE_URL)
        kwargs: dict = dict(
            model=model,
            file=(path.name, f),
            response_format=response_format,  # type: ignore[arg-type]
        )
        if language:
            kwargs["language"] = language
        t0 = time.monotonic()
        result = _client.audio.transcriptions.create(**kwargs)
        elapsed = time.monotonic() - t0
    text = result if isinstance(result, str) else result.text  # type: ignore[union-attr]
    log.info("received in %.2fs — %r", elapsed, text[:80])
    print(text)
    if timestamps != "none":
        words = getattr(result, "words", []) or []
        segments = getattr(result, "segments", []) or []
        if timestamps == "word" and words:
            _print_timestamps({"words": words}, "word", translate=translate)
        elif timestamps == "segment" and segments:
            _print_timestamps({"segments": segments}, "segment", translate=translate)


def transcribe_file_stream(
    path: Path,
    *,
    model: str,
    language: str | None,
    timestamps: str,
    translate: bool,
) -> str:
    with path.open("rb") as f:
        size = path.stat().st_size
        log.info("sending %s (%dB) -> %s", path.name, size, TRANSCRIPT_BASE_URL)
        kwargs: dict = dict(
            model=model,
            file=(path.name, f),
            stream=True,
        )
        if language:
            kwargs["language"] = language
        if timestamps != "none":
            kwargs["timestamp_granularities"] = [timestamps]

        t0 = time.monotonic()
        log.info("streaming begin")
        full_text = ""
        done_words: list = []
        done_segments: list = []
        stream = _client.audio.transcriptions.create(**kwargs)
        for event in stream:  # type: ignore[union-attr]
            etype = getattr(event, "type", None)
            if etype == "transcript.text.delta":
                print(event.delta, end="", flush=True)
            elif etype == "transcript.text.done":
                full_text = getattr(event, "text", full_text)
                done_words = getattr(event, "words", []) or []
                done_segments = getattr(event, "segments", []) or []
        print()
        elapsed = time.monotonic() - t0
    log.info("streaming end — %.2fs", elapsed)

    if timestamps == "word" and done_words:
        _print_timestamps({"words": done_words}, "word", translate=translate)
    elif timestamps == "segment" and done_segments:
        _print_timestamps({"segments": done_segments}, "segment", translate=translate)

    return full_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SubsVibe client - real time transcription and translation.",
    )
    parser.add_argument("-i", "--input", type=Path, default=None, help="Audio file to transcribe (mp3, wav, …)")

    parser.add_argument("--model", default=TRANSCRIPT_MODEL_NAME, help="Model name")
    parser.add_argument("--language", default=None, help="ISO-639-1 language code (default: auto-detect)")
    parser.add_argument(
        "--format",
        dest="response_format",
        default="json",
        choices=["json", "text", "verbose_json"],
        help="Response format (default: json, ignored when streaming)",
    )
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming (return full result at once)")
    parser.add_argument(
        "--timestamps",
        default="none",
        choices=["none", "segment", "word"],
        help="Include timestamps in output (default: none)",
    )
    parser.add_argument("--translate", action="store_true", help="Translate each segment to English via LLM")

    args = parser.parse_args()

    if args.input is not None:
        if not args.input.exists():
            parser.error(f"File not found: {args.input}")
        try:
            if args.no_stream:
                fmt = args.response_format
                if args.timestamps != "none" and fmt == "json":
                    fmt = "verbose_json"
                transcribe_file(
                    args.input,
                    model=args.model,
                    language=args.language,
                    response_format=fmt,
                    timestamps=args.timestamps,
                    translate=args.translate,
                )
            else:
                transcribe_file_stream(
                    args.input,
                    model=args.model,
                    language=args.language,
                    timestamps=args.timestamps,
                    translate=args.translate,
                )
        except APIConnectionError:
            sys.exit(f"error: could not connect to transcription server at {TRANSCRIPT_BASE_URL}")
        except APIStatusError as exc:
            sys.exit(f"error: server returned {exc.status_code}: {exc.message}")


if __name__ == "__main__":
    main()
