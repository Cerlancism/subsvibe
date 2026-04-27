from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("subsvibe.subtitle")

SRT_MIN_DURATION_SECONDS = 0.5
SRT_READING_BUFFER_SECONDS = 1.0
SRT_NEXT_GAP_SECONDS = 0.08

# Line/sub length budgets per script
SRT_MAX_LINE_CJK = 16
SRT_MAX_LINE_LATIN = 42
SRT_MAX_LINES = 2

# Reading-speed targets (characters per second)
SRT_CPS_CJK = 9.0
SRT_CPS_LATIN = 17.0

SOFT_BREAK_MARKERS = (",", "、", "，", ";", "；", ":", "：")
SENTENCE_END_MARKERS = (".", "!", "?", "。", "！", "？")


def _srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _is_cjk(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0x3040 <= code <= 0x30FF
            or 0xF900 <= code <= 0xFAFF
        ):
            return True
    return False


def _max_line_chars(text: str) -> int:
    return SRT_MAX_LINE_CJK if _is_cjk(text) else SRT_MAX_LINE_LATIN


def _target_cps(text: str) -> float:
    return SRT_CPS_CJK if _is_cjk(text) else SRT_CPS_LATIN


def _find_split_point(text: str, budget: int) -> int:
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
    """Greedy split: prefer sentence-end, then soft-break, then space, falling
    back to a hard cut at the budget if no boundary is available."""
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
    """Split any entry whose text exceeds MAX_LINES * max_line_chars into
    multiple entries, breaking at clause boundaries (sentence-end > comma >
    space). Timestamps are interpolated proportionally by character count."""
    out: list[dict] = []
    for e in entries:
        text = e["text"].strip()
        budget = _max_line_chars(text) * SRT_MAX_LINES
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
    """If text exceeds one line but fits in two, insert a newline at the best
    boundary near the midpoint."""
    line_max = _max_line_chars(text)
    if len(text) <= line_max:
        return text
    midpoint = len(text) // 2
    best = -1
    for offset in range(midpoint):
        for direction in (-1, 1):
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
    """Return True if merging a+b would still respect length and reading-speed budgets."""
    merged = f"{a['text'].strip()} {b['text'].strip()}".strip()
    line_max = _max_line_chars(merged)
    if len(merged) > line_max * SRT_MAX_LINES:
        return False
    duration = b["end"] - a["start"]
    if duration <= 0:
        return False
    return (len(merged) / duration) <= _target_cps(merged)


def _normalize_durations(entries: list[dict]) -> list[dict]:
    """Extend each entry by up to SRT_READING_BUFFER_SECONDS, capped before the
    next entry. If an entry still can't meet SRT_MIN_DURATION_SECONDS, merge it
    forward when reading-speed budgets allow."""
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
