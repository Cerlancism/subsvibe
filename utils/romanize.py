"""Per-language romanization for the live subtitle display.

`make_romanizer(language)` returns a `str -> str` callable that transliterates a
source-language utterance into Latin script for the renderer's romaji line:

- Japanese (`ja`) -> pykakasi (kana/kanji -> Hepburn romaji)
- Chinese (`zh`)  -> pypinyin (tone-marked pinyin)
- Korean (`ko`)   -> korean-romanizer (Revised Romanization)
- anything else / unknown / auto-detect (None) -> anyascii (generic best-effort)

Romanization is a display aid only: it never feeds the LLM translator or the ASR
prompt. Every backend is best-effort and total — a per-string failure logs at
debug and returns "" so the live loop never crashes on odd input. Pure-ASCII
input is skipped (returns "") so English-looking text gets no spurious romaji
line.
"""
from __future__ import annotations

import logging
from typing import Callable

from utils.language import to_iso_code

log = logging.getLogger("subsvibe.romanize")


def _needs_romanization(text: str) -> bool:
    """True if `text` has any non-ASCII char worth transliterating."""
    return any(ord(ch) > 127 for ch in text)


def _collapse(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip."""
    return " ".join(text.split())


def _make_kakasi() -> Callable[[str], str]:
    import pykakasi

    converter = pykakasi.kakasi()

    def romanize(text: str) -> str:
        if not text or not _needs_romanization(text):
            return ""
        try:
            parts = converter.convert(text)
            return _collapse(" ".join(p["hepburn"] for p in parts))
        except Exception:  # best-effort: never break the live loop
            log.debug("kakasi romanize failed for %r", text[:40], exc_info=True)
            return ""

    return romanize


def _make_pinyin() -> Callable[[str], str]:
    from pypinyin import Style, pinyin

    def romanize(text: str) -> str:
        if not text or not _needs_romanization(text):
            return ""
        try:
            # pinyin() returns a list of single-item lists; non-Han runs pass
            # through verbatim as their own item.
            return _collapse(" ".join(item[0] for item in pinyin(text, style=Style.TONE)))
        except Exception:
            log.debug("pinyin romanize failed for %r", text[:40], exc_info=True)
            return ""

    return romanize


def _make_korean() -> Callable[[str], str]:
    from korean_romanizer.romanizer import Romanizer

    def romanize(text: str) -> str:
        if not text or not _needs_romanization(text):
            return ""
        try:
            return _collapse(Romanizer(text).romanize())
        except Exception:
            log.debug("korean romanize failed for %r", text[:40], exc_info=True)
            return ""

    return romanize


def _make_anyascii() -> Callable[[str], str]:
    from anyascii import anyascii

    def romanize(text: str) -> str:
        if not text or not _needs_romanization(text):
            return ""
        try:
            return _collapse(anyascii(text))
        except Exception:
            log.debug("anyascii romanize failed for %r", text[:40], exc_info=True)
            return ""

    return romanize


def make_romanizer(language: str | None) -> Callable[[str], str]:
    """Return a romanizer callable for `language` (ISO code or canonical name).

    `ja` -> pykakasi, `zh` -> pypinyin, `ko` -> korean-romanizer, everything
    else (including None / auto-detect) -> anyascii. The callable maps "" for
    empty or pure-ASCII input, and never raises."""
    iso = to_iso_code(language)
    if iso == "ja":
        return _make_kakasi()
    if iso == "zh":
        return _make_pinyin()
    if iso == "ko":
        return _make_korean()
    return _make_anyascii()
