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

# Qwen3-ASR accepts only canonical English language names. Map ISO-639-1
# codes (and a few common aliases) to those names so the user can pass either.
LANGUAGE_NONE_VALUES = {"", "auto", "detect", "none"}
_LANGUAGE_ALIASES = {
    "zh": "Chinese", "zh-cn": "Chinese", "zh-tw": "Chinese", "cmn": "Chinese", "mandarin": "Chinese",
    "en": "English",
    "yue": "Cantonese", "zh-yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino", "tl": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}
SUPPORTED_LANGUAGES = {
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish",
    "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai",
    "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish",
    "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek",
    "Romanian", "Hungarian", "Macedonian",
}


def normalize_language(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    lowered = text.lower()
    if lowered in LANGUAGE_NONE_VALUES:
        return None
    if lowered in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[lowered]
    canonical = text[:1].upper() + text[1:].lower()
    if canonical in SUPPORTED_LANGUAGES:
        return canonical
    raise ValueError(
        f"unsupported language {value!r}; pass an ISO-639-1 code "
        f"(e.g. ja, zh, en) or a canonical name like {sorted(SUPPORTED_LANGUAGES)}"
    )
