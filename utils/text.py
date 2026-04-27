from __future__ import annotations

SENTENCE_END_MARKERS = frozenset((".", "!", "?", "。", "！", "？"))
SOFT_BREAK_MARKERS = frozenset((",", "、", "，", ";", "；", ":", "："))
CLOSING_PUNCTUATION = frozenset(".,!?;:)]}、。，！？；：」』）》〉】")
OPENING_PUNCTUATION = frozenset("([{'\"「『《〈【")

SUBTITLE_MAX_LINE_CJK = 16
SUBTITLE_MAX_LINE_LATIN = 42


def contains_cjk(value: str) -> bool:
    for ch in value:
        code = ord(ch)
        if (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0x3040 <= code <= 0x30FF
            or 0xF900 <= code <= 0xFAFF
        ):
            return True
    return False


def max_line_chars(text: str) -> int:
    return SUBTITLE_MAX_LINE_CJK if contains_cjk(text) else SUBTITLE_MAX_LINE_LATIN


def is_overlong(text: str) -> bool:
    return len(text) >= max_line_chars(text)
