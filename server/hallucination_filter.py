"""Blank known non-speech (silence/noise) hallucinations from transcription output.

ASR models emit filler text both for pure silence ("Thank you.",
"ご視聴ありがとうございました", ...) and for non-speech background noise/music
(BGM stings, channel-promo overlays). Two datasets record every such text per
backend/model/language:

- server/data/silence_hallucinations.json - built by
  tests/test_silence_hallucinations.py (pure silence).
- server/data/noise_hallucinations.json - the noise/music variant; same
  structure and same matching, separately toggleable.

This module loads the entries for the configured backend from both files
(union) and reports a match when the whole output, for the active model and
the utterance's language, equals a recorded text once punctuation, symbols,
whitespace and case are stripped. Partial matches (a hallucination embedded
in real speech) are never touched.

Each source is on by default; set TRANSCRIPT_SILENCE_FILTER=0 and/or
TRANSCRIPT_NOISE_FILTER=0 to disable them individually.
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

DATA_DIR = Path(__file__).resolve().parent / "data"
SILENCE_DATA_PATH = DATA_DIR / "silence_hallucinations.json"
NOISE_DATA_PATH = DATA_DIR / "noise_hallucinations.json"


def _enabled(var: str) -> bool:
    return os.environ.get(var, "1").strip().lower() not in {"0", "false", "off", "no"}


SILENCE_ENABLED = _enabled("TRANSCRIPT_SILENCE_FILTER")
NOISE_ENABLED = _enabled("TRANSCRIPT_NOISE_FILTER")


def _normalize(text: str) -> str:
    """Strip whitespace, punctuation and symbols and casefold, so the
    comparison sees only letters/digits ('Thank you.' -> 'thankyou')."""
    return "".join(
        ch for ch in text.casefold()
        if not ch.isspace() and unicodedata.category(ch)[0] not in "PS"
    )


def _load_source(path: Path, backend: str) -> dict[str, dict[str, set[str]]]:
    """model id -> ISO language -> normalized texts, for one dataset file and
    the configured backend. Missing/unreadable file yields no entries."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("hallucination filter: cannot read %s: %s", path.name, exc)
        return {}
    return {
        model: {
            lang: {_normalize(t) for t in texts}
            for lang, texts in per_lang.items()
        }
        for model, per_lang in data.get(backend, {}).items()
    }


@lru_cache(maxsize=1)
def _blocklists() -> dict[str, dict[str, frozenset[str]]]:
    """model id -> ISO language -> normalized hallucination texts, merging the
    enabled dataset sources for the configured backend."""
    backend = os.environ.get("TRANSCRIPT_BACKEND", "qwen")
    sources = []
    if SILENCE_ENABLED:
        sources.append(_load_source(SILENCE_DATA_PATH, backend))
    if NOISE_ENABLED:
        sources.append(_load_source(NOISE_DATA_PATH, backend))

    merged: dict[str, dict[str, set[str]]] = {}
    for source in sources:
        for model, per_lang in source.items():
            dest = merged.setdefault(model, {})
            for lang, entries in per_lang.items():
                dest.setdefault(lang, set()).update(entries)

    blocklists = {
        model: {lang: frozenset(entries) for lang, entries in per_lang.items()}
        for model, per_lang in merged.items()
    }
    log.info(
        "hallucination filter: backend %r, silence=%s noise=%s, %d model(s)",
        backend, SILENCE_ENABLED, NOISE_ENABLED, len(blocklists),
    )
    return blocklists


def is_hallucination(text: str, model: str, language: str | None) -> bool:
    """True when the whole text matches a known silence or noise hallucination
    recorded for this model and language. When the language is unknown (no
    request value and no detection), all of the model's languages are checked
    instead."""
    if not text or not (SILENCE_ENABLED or NOISE_ENABLED):
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
