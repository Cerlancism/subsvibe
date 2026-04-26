from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import av
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


def _get_audio_duration(path: Path) -> float:
    try:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            return float(stream.duration * stream.time_base)
    except Exception as e:
        log.warning("could not get audio duration: %s", e)
        return 0.0


_TRANSLATE_SYSTEM = (
    "You are a subtitle translator. Translate the given text to English. "
    "Output only the translation, no explanations."
)


def _translate(text: str) -> str:
    resp = _llm_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": _TRANSLATE_SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


def _print_timestamps(words: list, segments: list, granularity: str, *, translate: bool = False) -> None:
    if granularity == "word":
        for w in words:
            start = _fmt_ts(w.get("start", 0))
            end = _fmt_ts(w.get("end", 0))
            print(f"  [{start} -> {end}]  {w.get('word', w.get('text', ''))}")
    elif granularity == "segment":
        t0 = time.monotonic()
        for seg in segments:
            start = _fmt_ts(seg.get("start", 0))
            end = _fmt_ts(seg.get("end", 0))
            text = seg.get("text", "")
            prefix = f"  [{start} -> {end}]  "
            print(f"{prefix}{text}")
            if translate and text:
                print(f"{' ' * (len(prefix) - 3)}-> {_translate(text)}")
        if translate:
            log.info("translation done in %.2fs", time.monotonic() - t0)


def transcribe_file(
    path: Path,
    *,
    model: str,
    language: str | None,
    timestamps: str,
    translate: bool,
) -> None:
    audio_duration = _get_audio_duration(path)
    want_timestamps = timestamps != "none"
    fmt = "verbose_json" if want_timestamps else "json"

    with path.open("rb") as f:
        size = path.stat().st_size
        log.info("sending %s (%dB) -> %s", path.name, size, TRANSCRIPT_BASE_URL)
        kwargs: dict = dict(
            model=model,
            file=(path.name, f),
            response_format=fmt,
        )
        if language:
            kwargs["language"] = language
        t0 = time.monotonic()
        result = _client.audio.transcriptions.create(**kwargs)
        elapsed = time.monotonic() - t0

    text = result if isinstance(result, str) else result.text
    log.info("received in %.2fs — %r", elapsed, text[:80])

    if want_timestamps:
        words = list(getattr(result, "words", None) or [])
        segments = list(getattr(result, "segments", None) or [])
        _print_timestamps(
            [w if isinstance(w, dict) else w.__dict__ for w in words],
            [s if isinstance(s, dict) else s.__dict__ for s in segments],
            timestamps,
            translate=translate,
        )
    else:
        print(text)
        if translate:
            t_start = time.monotonic()
            translation = _translate(text)
            t_translate = time.monotonic() - t_start
            print(f"  -> {translation}")
            total_time = elapsed + t_translate
            buffer = audio_duration - total_time
            real_time_pct = 100.0 * audio_duration / total_time if total_time > 0 else 0
            log.info(
                "audio=%.2fs, transcript=%.2fs, translate=%.2fs, total=%.2fs, buffer=%.2fs (%.1f%% real-time)",
                audio_duration, elapsed, t_translate, total_time, buffer, real_time_pct,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SubsVibe client - real time transcription and translation.",
    )
    parser.add_argument("-i", "--input", type=Path, default=None, help="Audio file to transcribe (mp3, wav, …)")
    parser.add_argument("--model", default=TRANSCRIPT_MODEL_NAME, help="Model name")
    parser.add_argument("--language", default=None, help="ISO-639-1 language code (default: auto-detect)")
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
            transcribe_file(
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
