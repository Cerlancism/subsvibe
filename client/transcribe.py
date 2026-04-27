from __future__ import annotations

import logging
import os

from openai import OpenAI

log = logging.getLogger("subsvibe.transcribe")

TRANSCRIPT_HOST = os.environ.get("TRANSCRIPT_HOST", "127.0.0.1")
TRANSCRIPT_PORT = os.environ.get("TRANSCRIPT_PORT", "8000")
TRANSCRIPT_MODEL_NAME = os.environ.get("TRANSCRIPT_MODEL_NAME", "qwen3-asr")
TRANSCRIPT_BASE_URL = os.environ.get("TRANSCRIPT_BASE_URL", f"http://{TRANSCRIPT_HOST}:{TRANSCRIPT_PORT}")
TRANSCRIPT_API_KEY = os.environ.get("TRANSCRIPT_API_KEY", "not-needed-locally")

client = OpenAI(api_key=TRANSCRIPT_API_KEY, base_url=TRANSCRIPT_BASE_URL)
