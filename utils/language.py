"""Shared language-code helpers.

The transcription pipeline mixes two language identifier conventions:
- Qwen3-ASR expects canonical English names ("Japanese", "Chinese").
- Faster Whisper expects ISO-639-1 codes ("ja", "zh").

Either form is accepted on the wire; backends translate as needed.

Known issues / asymmetries to revisit:
- ISO_TO_NAME covers only the ~30 languages Qwen3-ASR supports. Whisper
  supports ~99, so passing e.g. `sw` (Swahili) makes `to_canonical_name`
  raise client-side even though faster-whisper would handle it.
- `to_iso_code` falls back to a lowercased pass-through for unknown input
  (faster-whisper rejects at the backend boundary), while
  `to_canonical_name` raises eagerly. The two helpers should fail in
  symmetric ways - either both pass-through or both raise.
- Validation lives at the CLI edge but the wire format claim ("either form
  is accepted") is gated to qwen's subset. If we widen the table to
  Whisper's full set, the backend boundary becomes the right place to
  reject "supported by client, unsupported by this backend".
"""
from __future__ import annotations

LANGUAGE_NONE_VALUES = frozenset({"", "auto", "detect", "none"})

# Primary mapping: ISO-639-1 code -> canonical English name.
ISO_TO_NAME: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
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
    "tl": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}

_NAME_TO_ISO: dict[str, str] = {name.lower(): code for code, name in ISO_TO_NAME.items()}

# Extra aliases users may type; resolved to a primary ISO code above.
_ALIAS_TO_ISO: dict[str, str] = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "cmn": "zh",
    "mandarin": "zh",
    "zh-yue": "yue",
    "fil": "tl",
}


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if text.lower() in LANGUAGE_NONE_VALUES:
        return None
    return text


def _resolve_iso(lowered: str) -> str | None:
    if lowered in ISO_TO_NAME:
        return lowered
    if lowered in _NAME_TO_ISO:
        return _NAME_TO_ISO[lowered]
    if lowered in _ALIAS_TO_ISO:
        return _ALIAS_TO_ISO[lowered]
    return None


def to_canonical_name(value: str | None) -> str | None:
    """Return canonical English name (Qwen-style), or None for empty/auto.
    Raises ValueError on unknown identifiers."""
    text = _clean(value)
    if text is None:
        return None
    iso = _resolve_iso(text.lower())
    if iso is None:
        raise ValueError(
            f"unsupported language {value!r}; pass an ISO-639-1 code "
            f"(e.g. ja, zh, en) or canonical name (e.g. Japanese)"
        )
    return ISO_TO_NAME[iso]


def to_iso_code(value: str | None) -> str | None:
    """Return ISO-639-1 code (Whisper-style), or None for empty/auto.
    Unknown values pass through lowercased."""
    text = _clean(value)
    if text is None:
        return None
    lowered = text.lower()
    return _resolve_iso(lowered) or lowered


# Scripts that conventionally do not separate words with spaces. Used by
# callers that need to join tokenised pieces back into a display string
# (e.g. the live pipeline's tail-prov preview) without inserting spurious
# whitespace for CJK / SE-Asian languages.
SPACELESS_ISO: frozenset[str] = frozenset({"ja", "zh", "yue", "th", "lo", "my", "km"})


def is_spaceless(value: str | None) -> bool:
    """True if the language conventionally writes without word-separating
    spaces. False for None (auto-detect) and unknown values."""
    iso = to_iso_code(value)
    return iso in SPACELESS_ISO if iso else False


# Languages written in CJK scripts (Han / kana / hangul), for which a Latin-
# script romanization line is worth showing by default. `yue` (Cantonese)
# shares Han script with Mandarin.
CJK_ISO: frozenset[str] = frozenset({"zh", "ja", "ko", "yue"})


def is_cjk(value: str | None) -> bool:
    """True if the language is written in a CJK script (Chinese / Japanese /
    Korean). False for None (auto-detect) and unknown values."""
    iso = to_iso_code(value)
    return iso in CJK_ISO if iso else False
