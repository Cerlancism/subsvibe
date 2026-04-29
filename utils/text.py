from __future__ import annotations

FULLSTOP_MARKERS = frozenset((".", "。", "．", "｡"))
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


def strip_trailing_fullstop(text: str) -> str:
    return text[:-1].rstrip() if text and text[-1] in FULLSTOP_MARKERS else text


def attach_punctuation(words: list[dict], full_text: str) -> list[dict]:
    """Walk full_text and append surrounding punctuation/whitespace to each
    aligner word, since the forced aligner strips all non-letter chars.

    Each input word must have a `text` (or `word`) field; output words get
    `text` and `trailing` fields populated, with all other fields preserved."""
    def _word_text(w: dict) -> str:
        return str(w.get("text") or w.get("word") or "")

    enriched: list[dict] = [{**w, "text": "", "trailing": ""} for w in words]
    if not enriched or not full_text:
        return enriched

    cursor = 0
    n = len(full_text)
    for idx, word in enumerate(enriched):
        token = _word_text(words[idx])
        if not token:
            continue
        match = full_text.find(token, cursor)
        if match < 0:
            word["text"] = token
            continue
        leading = full_text[cursor:match]
        if leading and idx > 0:
            enriched[idx - 1]["trailing"] += leading
        word["text"] = token
        cursor = match + len(token)

    if cursor < n:
        enriched[-1]["trailing"] += full_text[cursor:]

    return enriched
