from __future__ import annotations

import logging
from pathlib import Path

from utils.text import (
    CLOSING_PUNCTUATION,
    OPENING_PUNCTUATION,
    SENTENCE_END_MARKERS,
    SOFT_BREAK_MARKERS,
    contains_cjk,
    is_overlong,
    max_line_chars,
)

log = logging.getLogger("subsvibe.subtitle")

SRT_MIN_DURATION_SECONDS = 0.5
SRT_READING_BUFFER_SECONDS = 1.0
SRT_NEXT_GAP_SECONDS = 0.08

SRT_MAX_LINES = 2
SRT_WRAP_RATIO = 2.0

SRT_CPS_CJK = 9.0
SRT_CPS_LATIN = 17.0

WORD_GAP_FLUSH_SECONDS = 1.0


def _join_word_tokens(tokens: list[str]) -> str:
    """Join word/trailing tokens with the same script-aware spacing the server
    uses, so adjacent CJK chars and punctuation don't get extra spaces."""
    text = ""
    for token in tokens:
        piece = token.strip()
        if not piece:
            continue
        if not text:
            text = piece
            continue
        prev, nxt = text[-1], piece[0]
        if (
            nxt in CLOSING_PUNCTUATION
            or prev in OPENING_PUNCTUATION
            or (contains_cjk(prev) and contains_cjk(nxt))
        ):
            text += piece
        else:
            text += f" {piece}"
    return text.strip()


def _accumulated_text(words: list[dict]) -> str:
    parts: list[str] = []
    for w in words:
        parts.append(str(w.get("text", "") or ""))
        trailing = str(w.get("trailing", "") or "")
        if trailing:
            parts.append(trailing)
    return _join_word_tokens(parts).strip()


def _endswith_any(s: str, markers: frozenset[str]) -> bool:
    return bool(s) and s[-1] in markers


def entries_from_words(words: list[dict]) -> list[dict]:
    """Group aligner words into subtitle entries on word boundaries.

    Flushes on: gap >= WORD_GAP_FLUSH_SECONDS, accumulated text reaching the
    2-line budget, sentence-end punctuation, or soft-break punctuation when the
    accumulator already fills one line. Timestamps come from the words' actual
    start/end so splits land on real boundaries (no character-proportional
    interpolation)."""
    entries: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        if not current:
            return
        text = _accumulated_text(current)
        if text:
            entries.append({
                "start": round(float(current[0]["start"]), 3),
                "end": round(float(current[-1]["end"]), 3),
                "text": text,
            })
        current.clear()

    for word in words:
        if current:
            gap = float(word["start"]) - float(current[-1]["end"])
            if gap >= WORD_GAP_FLUSH_SECONDS:
                flush()
            else:
                # would adding this word push us over the 2-line budget?
                tentative = _accumulated_text(current + [word])
                if len(tentative) > max_line_chars(tentative) * SRT_MAX_LINES:
                    flush()
        current.append(word)
        trailing = str(word.get("trailing", "") or "").rstrip()
        if _endswith_any(trailing, SENTENCE_END_MARKERS):
            flush()
        elif _endswith_any(trailing, SOFT_BREAK_MARKERS) and is_overlong(_accumulated_text(current)):
            flush()

    flush()
    return entries


def _srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _target_cps(text: str) -> float:
    return SRT_CPS_CJK if contains_cjk(text) else SRT_CPS_LATIN


def _find_split_point(text: str, budget: int) -> int:
    assert budget >= 1
    window = text[:budget]
    for markers in (SENTENCE_END_MARKERS, SOFT_BREAK_MARKERS):
        for i in range(len(window) - 1, -1, -1):
            if window[i] in markers:
                return i + 1
    space = window.rfind(" ")
    if space > 0:
        return space + 1
    return budget


def _split_text_at_boundaries(text: str, budget: int) -> list[str]:
    pieces: list[str] = []
    remaining = text
    while len(remaining) > budget:
        cut = _find_split_point(remaining, budget)
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        pieces.append(remaining)
    return [p for p in pieces if p]


def _split_overlong(entries: list[dict]) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        text = e["text"].strip()
        budget = max_line_chars(text) * SRT_MAX_LINES
        if len(text) <= budget:
            out.append(e)
            continue
        pieces = _split_text_at_boundaries(text, budget)
        if len(pieces) <= 1:
            out.append(e)
            continue
        total_chars = sum(len(p) for p in pieces)
        duration = e["end"] - e["start"]
        cursor = e["start"]
        for piece in pieces:
            share = duration * (len(piece) / total_chars)
            out.append({"start": cursor, "end": cursor + share, "text": piece})
            cursor += share
        out[-1]["end"] = e["end"]
    return out


def _wrap_two_lines(text: str) -> str:
    line_max = max_line_chars(text)
    if len(text) <= int(line_max * SRT_WRAP_RATIO):
        return text
    midpoint = len(text) // 2
    best = -1
    for offset in range(midpoint):
        directions = (0,) if offset == 0 else (-1, 1)
        for direction in directions:
            i = midpoint + direction * offset
            if 0 < i < len(text):
                if text[i - 1] in SENTENCE_END_MARKERS or text[i - 1] in SOFT_BREAK_MARKERS:
                    best = i
                    break
                if text[i] == " ":
                    best = i
                    break
        if best != -1:
            break
    if best == -1:
        best = midpoint
    line1 = text[:best].rstrip()
    line2 = text[best:].lstrip()
    return f"{line1}\n{line2}"


def _can_merge(a: dict, b: dict) -> bool:
    merged = f"{a['text'].strip()} {b['text'].strip()}".strip()
    line_max = max_line_chars(merged)
    if len(merged) > line_max * SRT_MAX_LINES:
        return False
    duration = b["end"] - a["start"]
    if duration <= 0:
        return False
    return (len(merged) / duration) <= _target_cps(merged)


def _normalize_durations(entries: list[dict]) -> list[dict]:
    """Extend each entry by up to SRT_READING_BUFFER_SECONDS, capped before the
    next entry. If an entry still can't meet SRT_MIN_DURATION_SECONDS, merge it
    forward when reading-speed budgets allow (best-effort: a merge that would
    breach line or CPS limits is skipped, leaving the entry under-duration)."""
    out: list[dict] = [dict(e) for e in entries]
    i = 0
    while i < len(out):
        e = out[i]
        target_end = e["end"] + SRT_READING_BUFFER_SECONDS
        if i + 1 < len(out):
            target_end = min(target_end, out[i + 1]["start"] - SRT_NEXT_GAP_SECONDS)
        new_end = max(e["end"], target_end)

        if new_end - e["start"] >= SRT_MIN_DURATION_SECONDS or i + 1 >= len(out):
            e["end"] = new_end
            i += 1
            continue

        nxt = out[i + 1]
        if not _can_merge(e, nxt):
            e["end"] = new_end
            i += 1
            continue

        merged_text = f"{e['text'].strip()} {nxt['text'].strip()}".strip()
        out[i + 1] = {"start": e["start"], "end": nxt["end"], "text": merged_text}
        del out[i]
    return out


def write_srt(entries: list[dict], out_path: Path) -> None:
    entries = _split_overlong(entries)
    entries = _normalize_durations(entries)
    with out_path.open("w", encoding="utf-8") as f:
        for i, e in enumerate(entries, 1):
            f.write(f"{i}\n")
            f.write(f"{_srt_timestamp(e['start'])} --> {_srt_timestamp(e['end'])}\n")
            f.write(f"{_wrap_two_lines(e['text'].strip())}\n\n")
    log.info("wrote %d subtitle(s) to %s", len(entries), out_path)
