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
    strip_trailing_fullstop,
)

log = logging.getLogger("subsvibe.subtitle")

SRT_MIN_DURATION_SECONDS = 0.5
SRT_READING_BUFFER_SECONDS = 1.0
SRT_NEXT_GAP_SECONDS = 0.08
SRT_TAIL_EXTEND_SECONDS = 1.0
SRT_TAIL_EXTEND_GAP_SECONDS = 0.005

SRT_MAX_LINES = 2
SRT_WRAP_RATIO = 2.0

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
        text = _accumulated_text(current).rstrip()
        while text and text[-1] in SOFT_BREAK_MARKERS:
            text = text[:-1].rstrip()
        text = strip_trailing_fullstop(text)
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
        duration = e["end"] - e["start"]
        if duration < SRT_MIN_DURATION_SECONDS * len(pieces):
            log.warning(
                "kept overlong entry [%.3f-%.3f] (%d chars, %d pieces) intact - "
                "splitting would produce sub-minimum pieces",
                e["start"], e["end"], len(text), len(pieces),
            )
            out.append(e)
            continue
        total_chars = sum(len(p) for p in pieces)
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


def _merge_degenerate(entries: list[dict]) -> list[dict]:
    """Repair entries where start == end (aligner returned zero-span words for
    a whole run). For each maximal run of degenerate entries, collapse the run
    into a single entry spanning from the previous valid entry's end (or the
    run's pinned timestamp) to the next valid entry's start, concatenating
    their text. The run also absorbs subsequent entries whose start equals
    the run's pinned timestamp, even if their end > start - those come from
    the same bad aligner result and have nonsensical end values."""
    out: list[dict] = [dict(e) for e in entries]
    result: list[dict] = []
    n = len(out)
    i = 0
    while i < n:
        if out[i]["end"] > out[i]["start"]:
            result.append(out[i])
            i += 1
            continue
        pinned = out[i]["start"]
        j = i
        while j < n and (out[j]["end"] <= out[j]["start"] or out[j]["start"] == pinned):
            j += 1
        prev_end = result[-1]["end"] if result else pinned
        next_start = out[j]["start"] if j < n else pinned + SRT_MIN_DURATION_SECONDS * (j - i)
        span_start = max(prev_end, pinned)
        span_end = max(next_start, span_start)
        count = j - i
        merged_text = " ".join(out[k]["text"].strip() for k in range(i, j) if out[k]["text"].strip()).strip()
        log.debug(
            "merged %d degenerate entr%s pinned at %.3fs into one across [%.3f–%.3f]",
            count, "y" if count == 1 else "ies", pinned, span_start, span_end,
        )
        result.append({
            "start": round(span_start, 3),
            "end": round(span_end, 3),
            "text": merged_text,
        })
        i = j
    return result


def _fix_overlaps(entries: list[dict]) -> list[dict]:
    """Ensure each entry starts strictly after the previous one ends. When the
    next entry's start <= previous end, push it to prev_end + 1ms (and extend
    its end if needed to keep start < end)."""
    out: list[dict] = [dict(e) for e in entries]
    for i in range(1, len(out)):
        prev_end = out[i - 1]["end"]
        if out[i]["start"] <= prev_end:
            new_start = round(prev_end + 0.001, 3)
            log.debug(
                "overlap: entry %d start=%.3f <= prev end=%.3f, bumped to %.3f",
                i, out[i]["start"], prev_end, new_start,
            )
            out[i]["start"] = new_start
            if out[i]["end"] < new_start:
                out[i]["end"] = new_start
    return out


def _normalize_durations(entries: list[dict]) -> list[dict]:
    """Ensure every entry meets SRT_MIN_DURATION_SECONDS. Entries already long
    enough are left untouched (real word-end timestamp preserved). Short entries
    are extended forward, capped at SRT_NEXT_GAP_SECONDS before the next entry;
    if that's still not enough, merged forward into the next entry. The merge
    can produce overlong text for packed runs of short entries - callers should
    re-run _split_overlong afterward."""
    out: list[dict] = [dict(e) for e in entries]
    i = 0
    while i < len(out):
        e = out[i]
        if e["end"] - e["start"] >= SRT_MIN_DURATION_SECONDS:
            i += 1
            continue

        target_end = e["start"] + SRT_MIN_DURATION_SECONDS
        if i + 1 < len(out):
            target_end = min(target_end, out[i + 1]["start"] - SRT_NEXT_GAP_SECONDS)
        new_end = max(e["end"], target_end)

        if new_end - e["start"] >= SRT_MIN_DURATION_SECONDS or i + 1 >= len(out):
            e["end"] = new_end
            i += 1
            continue

        nxt = out[i + 1]
        merged_text = f"{e['text'].strip()} {nxt['text'].strip()}".strip()
        out[i + 1] = {"start": e["start"], "end": nxt["end"], "text": merged_text}
        del out[i]

    for i, e in enumerate(out):
        cap = e["end"] + SRT_TAIL_EXTEND_SECONDS
        if i + 1 < len(out):
            cap = min(cap, out[i + 1]["start"] - SRT_TAIL_EXTEND_GAP_SECONDS)
        if cap > e["end"]:
            e["end"] = round(cap, 3)
    return out


def write_srt(entries: list[dict], out_path: Path, *, normalize_durations: bool = True) -> None:
    """`normalize_durations=False` skips the _normalize_durations pass for
    entries whose timings come straight from the ASR model's own segments
    (faster-whisper): those timestamps are trusted as-is, including sub-minimum
    durations and tail gaps. Forced-aligner-derived entries (word paths) keep
    the pass — aligner word timings routinely need the min-duration repair."""
    log.info("post-processing %d subtitle entry(ies)", len(entries))
    entries = [e for e in entries if e["text"].strip()]
    log.info("after dropping empty entries: %d entry(ies)", len(entries))
    entries = _merge_degenerate(entries)
    log.info("after _merge_degenerate: %d entry(ies)", len(entries))
    entries = _fix_overlaps(entries)
    log.info("after _fix_overlaps: %d entry(ies)", len(entries))
    if normalize_durations:
        entries = _normalize_durations(entries)
        log.info("after _normalize_durations: %d entry(ies)", len(entries))
    else:
        log.info("skipping _normalize_durations (segment-timed backend)")
    entries = _split_overlong(entries)
    log.info("after _split_overlong: %d entry(ies)", len(entries))
    if entries:
        lengths = sorted(len(e["text"]) for e in entries)
        n = len(lengths)
        median = lengths[n // 2] if n % 2 else (lengths[n // 2 - 1] + lengths[n // 2]) / 2
        log.info(
            "entry text length - avg=%.1f median=%.1f min=%d max=%d",
            sum(lengths) / n, median, lengths[0], lengths[-1],
        )
    with out_path.open("w", encoding="utf-8") as f:
        for i, e in enumerate(entries, 1):
            f.write(f"{i}\n")
            f.write(f"{_srt_timestamp(e['start'])} --> {_srt_timestamp(e['end'])}\n")
            f.write(f"{_wrap_two_lines(e['text'].strip())}\n\n")
    log.info("wrote %d subtitle(s) to %s", len(entries), out_path)
