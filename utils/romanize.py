"""Per-language romanization for the live subtitle display.

`make_romanizer(language)` returns a `str -> str` callable that transliterates a
source-language utterance into Latin script for the renderer's romaji line:

- Chinese (`zh`)  -> pypinyin (tone-marked pinyin)
- Japanese (`ja`) -> cutlet (fugashi/MeCab + full UniDic -> Hepburn romaji)
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

from utils.data.ja_romaji_exceptions import JA_ROMAJI_EXCEPTIONS
from utils.language import to_iso_code

log = logging.getLogger("subsvibe.romanize")


def _needs_romanization(text: str) -> bool:
    """True if `text` has any non-ASCII char worth transliterating."""
    return any(ord(ch) > 127 for ch in text)


def _collapse(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip."""
    return " ".join(text.split())


def _make_cutlet() -> Callable[[str], str]:
    import cutlet

    # Morphological analysis (fugashi/MeCab + full UniDic) segments words, so the
    # grammatical particles は/へ/を romanize as wa/e/o rather than their literal
    # kana readings ha/he/wo. Full UniDic (downloaded by setup.sh) is used over the
    # bundled unidic-lite for far better name / rare-word coverage and a more
    # reliable reading field; cutlet auto-prefers it when present, falling back to
    # unidic-lite otherwise. use_foreign_spelling=False keeps katakana loanwords
    # phonetic (コーヒー -> koohii) instead of guessing English spellings.
    # Hepburn particle rules: は->wa, へ->e, を->o. cutlet's flags are inverted
    # from intuition — use_he/use_wo being True *keeps* the literal he/wo, so for
    # natural Hepburn both must be False (use_wa already defaults True for hepburn).
    katsu = cutlet.Cutlet()
    katsu.use_he = False
    katsu.use_wo = False
    katsu.use_foreign_spelling = False

    # Override the handful of lexicalized greetings cutlet mis-sounds (e.g. こんにちは
    # -> "konnichiha"). See utils/data/ja_romaji_exceptions.py for the rationale and
    # how to re-verify the set after a cutlet/UniDic upgrade.
    katsu.exceptions.update(JA_ROMAJI_EXCEPTIONS)

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
    empty or pure-ASCII input, and never raises."""
    iso = to_iso_code(language)
    if iso == "ja":
        return _make_cutlet()
    if iso == "zh":
        return _make_pinyin()
    if iso == "ko":
        return _make_korean()
    return _make_anyascii()
