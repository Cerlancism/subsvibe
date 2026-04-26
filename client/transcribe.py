from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("subsvibe.client")


def _on_request(request: httpx.Request) -> None:
    log.info("httpx: dispatching %s %s", request.method, request.url)


def _on_response(response: httpx.Response) -> None:
    log.info("httpx: response %d in-flight", response.status_code)


_http_client = httpx.Client(event_hooks={"request": [_on_request], "response": [_on_response]})

TRANSCRIPT_HOST = os.environ.get("TRANSCRIPT_HOST", "localhost")
TRANSCRIPT_PORT = os.environ.get("TRANSCRIPT_PORT", "8000")
TRANSCRIPT_MODEL_NAME = os.environ.get("TRANSCRIPT_MODEL_NAME", "qwen3-asr")
TRANSCRIPT_BASE_URL = os.environ.get("TRANSCRIPT_BASE_URL", f"http://{TRANSCRIPT_HOST}:{TRANSCRIPT_PORT}")
TRANSCRIPT_API_KEY = os.environ.get("TRANSCRIPT_API_KEY", "not-needed-locally")

_client = OpenAI(api_key=TRANSCRIPT_API_KEY, base_url=TRANSCRIPT_BASE_URL, http_client=_http_client)


def transcribe_file(
    path: Path,
    *,
    model: str,
    language: str | None,
    response_format: str,
) -> str:
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
    return text


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
        help="Response format (default: json)",
    )

    args = parser.parse_args()

    if args.input is not None:
        if not args.input.exists():
            parser.error(f"File not found: {args.input}")
        try:
            text = transcribe_file(
                args.input,
                model=args.model,
                language=args.language,
                response_format=args.response_format,
            )
        except APIConnectionError:
            sys.exit(f"error: could not connect to transcription server at {TRANSCRIPT_BASE_URL}")
        except APIStatusError as exc:
            sys.exit(f"error: server returned {exc.status_code}: {exc.message}")
        print(text)


if __name__ == "__main__":
    main()
