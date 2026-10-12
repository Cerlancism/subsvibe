from __future__ import annotations

import logging
import os

log = logging.getLogger("subsvibe.text")

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


HALLUCINATION_REPEAT_THRESHOLD = 20
HALLUCINATION_PATTERN_MAX_LEN = 20


def strip_hallucinations(text: str, threshold: int = HALLUCINATION_REPEAT_THRESHOLD) -> str:
    """Collapse runaway character and short-pattern repetitions that ASR
    models sometimes emit (whisper-family especially). A run of any char
    or short pattern longer than `threshold` collapses to one copy."""
    if not text:
        return text

    def fix_char_repeats(s: str, thresh: int) -> str:
        out: list[str] = []
        i, n = 0, len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                out.append(s[i])
            else:
                out.append(s[i:i + count])
            i += count
        return "".join(out)

    def fix_pattern_repeats(s: str, thresh: int, max_len: int = HALLUCINATION_PATTERN_MAX_LEN) -> str:
        n = len(s)
        if n < thresh * 2:
            return s
        i = 0
        out: list[str] = []
        while i <= n - thresh * 2:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i:i + k]
                if not all(s[i + r * k:i + r * k + k] == pattern for r in range(1, thresh)):
                    continue
                end = i + thresh * k
                while end + k <= n and s[end:end + k] == pattern:
                    end += k
                out.append(pattern)
                out.append(fix_pattern_repeats(s[end:], thresh, max_len))
                i = n
                found = True
                break
            if not found:
                out.append(s[i])
                i += 1
        if i < n:
            out.append(s[i:])
        return "".join(out)

    return fix_pattern_repeats(fix_char_repeats(text, threshold), threshold)


# How far past the cursor attach_punctuation looks for the next word. Between
# two consecutive aligner words the full text holds only punctuation and
# whitespace, so a hit further out than this is a false one (a one-character
# CJK token recurs every few characters) rather than the word itself.
ATTACH_SEARCH_SLACK = 24


def _has_word_chars(s: str) -> bool:
    return any(c.isalnum() for c in s)


def attach_punctuation(words: list[dict], full_text: str) -> list[dict]:
    """Walk full_text and append surrounding punctuation/whitespace to each
    aligner word, since the forced aligner strips all non-letter chars.

    Each input word must have a `text` (or `word`) field; output words get
    `text` and `trailing` fields populated, with all other fields preserved.
    Words that already carry a non-empty `trailing` (a backend that emits
    marks as tokens and folded them in itself) are returned as they are,
    with `text` filled in; full_text is not consulted.

    A token that cannot be located in full_text is treated as a spelling
    difference, not as missing: the search for later words stays local
    (ATTACH_SEARCH_SLACK plus the unmatched tokens' length, and a hit that
    skips word characters must be confirmed by the following token), and the
    span of full_text that stands in for the unmatched run goes into the
    run's last word as `trailing` with those words' own `text` left empty.
    The output text is therefore always full_text in order, never a token
    plus its full_text spelling twice, and a miss cannot cascade to the end
    of the text. When the skipped span holds no word characters the tokens
    were genuinely absent from full_text and keep their own text. If no
    token matches at all, the words keep their text and full_text is ignored
    (one warning), so timing survives at the cost of punctuation."""
    def _word_text(w: dict) -> str:
        return str(w.get("text") or w.get("word") or "")

    def _trailing(w: dict) -> str:
        return str(w.get("trailing") or "")

    if any(_trailing(w) for w in words):
        return [{**w, "text": _word_text(w), "trailing": _trailing(w)} for w in words]

    enriched: list[dict] = [{**w, "text": "", "trailing": ""} for w in words]
    if not enriched or not full_text:
        return enriched

    tokens = [_word_text(w) for w in words]
    cursor = 0
    n = len(full_text)
    pending: list[int] = []  # words not located since the last match
    matched = 0
    collapsed = 0

    def _find(token: str, start: int, budget: int) -> int:
        return full_text.find(token, start, min(n, start + budget))

    def _settle_pending(span: str) -> None:
        """`span` is the full_text between the last match and the next one;
        it stands in for the pending tokens when it holds word characters."""
        nonlocal collapsed
        if not pending:
            return
        if _has_word_chars(span):
            collapsed += len(pending)
        else:
            for i in pending:
                enriched[i]["text"] = tokens[i]
        enriched[pending[-1]]["trailing"] += span
        pending.clear()

    for idx, word in enumerate(enriched):
        token = tokens[idx]
        if not token:
            continue
        budget = len(token) + ATTACH_SEARCH_SLACK + sum(len(tokens[i]) for i in pending)
        match = _find(token, cursor, budget)
        if match >= 0 and _has_word_chars(full_text[cursor:match]):
            nxt = next((t for t in tokens[idx + 1:] if t), None)
            if nxt is not None and _find(nxt, match + len(token), len(nxt) + ATTACH_SEARCH_SLACK) < 0:
                match = -1
        if match < 0:
            pending.append(idx)
            continue
        leading = full_text[cursor:match]
        if pending:
            _settle_pending(leading)
        elif leading and idx > 0:
            enriched[idx - 1]["trailing"] += leading
        word["text"] = token
        cursor = match + len(token)
        matched += 1

    if matched == 0:
        log.warning(
            "attach_punctuation: none of %d words found in text (%d chars); keeping word text without punctuation",
            len(tokens), n,
        )
        for i, w in enumerate(enriched):
            w["text"] = tokens[i]
            w["trailing"] = ""
        return enriched

    tail = full_text[cursor:]
    if pending:
        _settle_pending(tail)
    elif tail:
        enriched[-1]["trailing"] += tail
    if collapsed:
        log.debug("attach_punctuation: %d of %d words not found verbatim; spelled from text", collapsed, len(tokens))
    return enriched


# Line ending used for files SubsVibe writes (SRT subtitles, log files).
# Python text mode translates "\n" to os.linesep, which makes output CRLF on
# Windows and LF elsewhere; SUBSVIBE_NEWLINE pins it instead. Accepted: "lf",
# "crlf", or "auto". Unset (or unrecognised) means auto.
_NEWLINE_MODES = {"lf": "\n", "crlf": "\r\n", "auto": None}


def output_newline() -> str | None:
    """Return the `newline=` argument for open()/FileHandler, per
    SUBSVIBE_NEWLINE. None means "translate to the platform default"."""
    mode = os.environ.get("SUBSVIBE_NEWLINE", "auto").strip().lower()
    return _NEWLINE_MODES.get(mode, None)
