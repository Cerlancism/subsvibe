"""Blank known silence hallucinations from transcription output.

ASR models emit filler text for pure silence ("Thank you.",
"ご視聴ありがとうございました", ...). server/data/silence_hallucinations.json -
built by tests/test_silence_hallucinations.py - records every such text per
backend/model/language. This module loads the entries for the configured
backend and reports a match when the whole output, for the active model and
the utterance's language, equals a recorded text once punctuation, symbols,
whitespace and case are stripped. Partial matches (a hallucination embedded
in real speech) are never touched.

On by default; set TRANSCRIPT_SILENCE_FILTER=0 to disable.
"""
from __future__ import annotations

import json
import logging
import os
import unicodedata
from functools import lru_cache
from pathlib import Path

from utils.language import to_iso_code

log = logging.getLogger("subsvibe.server")

DATA_PATH = Path(__file__).resolve().parent / "data" / "silence_hallucinations.json"

ENABLED = os.environ.get("TRANSCRIPT_SILENCE_FILTER", "1").strip().lower() not in {
    "0", "false", "off", "no",
}


def _normalize(text: str) -> str:
    """Strip whitespace, punctuation and symbols and casefold, so the
    comparison sees only letters/digits ('Thank you.' -> 'thankyou')."""
    return "".join(
        ch for ch in text.casefold()
        if not ch.isspace() and unicodedata.category(ch)[0] not in "PS"
    )


@lru_cache(maxsize=1)
def _blocklists() -> dict[str, dict[str, frozenset[str]]]:
    """model id -> ISO language -> normalized hallucination texts, holding
    only the configured backend's section of the dataset."""
    backend = os.environ.get("TRANSCRIPT_BACKEND", "qwen")
    try:
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("silence filter inactive - cannot read %s: %s", DATA_PATH, exc)
        return {}
    blocklists = {
        model: {
            lang: frozenset(_normalize(t) for t in texts)
            for lang, texts in per_lang.items()
        }
        for model, per_lang in data.get(backend, {}).items()
    }
    log.info(
        "silence filter: blocklists for backend %r cover %d model(s)",
        backend, len(blocklists),
    )
    return blocklists


def is_silence_hallucination(text: str, model: str, language: str | None) -> bool:
    """True when the whole text matches a silence hallucination recorded for
    this model and language. When the language is unknown (no request value
    and no detection), all of the model's languages are checked instead."""
    if not ENABLED or not text:
        return False
    per_lang = _blocklists().get(model)
    if not per_lang:
        return False
    iso = to_iso_code(language)
    if iso is not None:
        entries = per_lang.get(iso, frozenset())
    else:
        entries = frozenset().union(*per_lang.values())
    return _normalize(text) in entries
