"""Per-language romanization for the live subtitle display.

`make_romanizer(language)` returns a `str -> str` callable that transliterates a
source-language utterance into Latin script for the renderer's romaji line:

- Chinese (`zh`)  -> pypinyin (tone-marked pinyin)
- Japanese (`ja`) -> cutlet (fugashi/MeCab morphological analysis -> Hepburn romaji)
- Korean (`ko`)   -> korean-romanizer (Revised Romanization)
- anything else / unknown / auto-detect (None) -> anyascii (generic best-effort)

Romanization is a display aid only: it never feeds the LLM translator or the ASR
prompt. Every backend is best-effort and total — a per-string failure logs at
debug and returns "" so the live loop never crashes on odd input. Pure-ASCII
input is skipped (returns "") so English-looking text gets no spurious romaji
line.

The returned callable is thread-safe: calls are serialized under a
per-romanizer lock (see `_serialized`). The live pipeline shares one instance
across the renderer and worker threads, and the underlying engines (MeCab
taggers in particular) are not guaranteed safe under concurrent calls.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

from utils.language import to_iso_code

log = logging.getLogger("subsvibe.romanize")


def _needs_romanization(text: str) -> bool:
    """True if `text` has any non-ASCII char worth transliterating."""
    return any(ord(ch) > 127 for ch in text)


def _collapse(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip."""
    return " ".join(text.split())


def _serialized(fn: Callable[[str], str]) -> Callable[[str], str]:
    """Serialize calls to `fn` under a per-romanizer lock.

    One romanizer instance is shared across threads in the live pipeline
    (renderer refreshes on the emitting workers, the commit-path draft, the
    async corrector worker's revise re-render). The engines underneath —
    MeCab/fugashi taggers in particular — are not guaranteed thread-safe, so
    thread-safety is made part of make_romanizer's contract here. Calls are
    ms-scale, so the lock is effectively uncontended."""
    lock = threading.Lock()

    def wrapper(text: str) -> str:
        with lock:
            return fn(text)

    return wrapper


def _make_cutlet() -> Callable[[str], str]:
    import cutlet

    # Morphological analysis (fugashi/MeCab + unidic-lite) segments words, so the
    # grammatical particles は/へ/を romanize as wa/e/o rather than their literal
    # kana readings ha/he/wo. cutlet's particle flags are inverted from intuition:
    # use_he/use_wo being True *keeps* the literal he/wo, so for natural Hepburn
    # both must be False (use_wa already defaults True). use_foreign_spelling=False
    # keeps katakana loanwords phonetic (コーヒー -> koohii) instead of guessing
    # English spellings.
    katsu = cutlet.Cutlet()
    katsu.use_he = False
    katsu.use_wo = False
    katsu.use_foreign_spelling = False

    def romanize(text: str) -> str:
        if not text or not _needs_romanization(text):
            return ""
        try:
            return _collapse(katsu.romaji(text))
        except Exception:  # best-effort: never break the live loop
            log.debug("cutlet romanize failed for %r", text[:40], exc_info=True)
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

    `ja` -> cutlet, `zh` -> pypinyin, `ko` -> korean-romanizer, everything
    else (including None / auto-detect) -> anyascii. The callable maps "" for
    empty or pure-ASCII input, never raises, and is thread-safe (calls are
    serialized — see `_serialized`)."""
    iso = to_iso_code(language)
    if iso == "ja":
        return _serialized(_make_cutlet())
    if iso == "zh":
        return _serialized(_make_pinyin())
    if iso == "ko":
        return _serialized(_make_korean())
    return _serialized(_make_anyascii())
