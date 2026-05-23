from __future__ import annotations

import re
from pathlib import Path

_SRT_TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})")


def _parse_srt_timestamp(s: str) -> float:
    m = _SRT_TIME_RE.match(s.strip())
    if not m:
        raise ValueError(f"invalid SRT timestamp: {s!r}")
    h, mi, sec, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(sec) + int(ms.ljust(3, "0")) / 1000.0


def read_srt(path: Path) -> list[dict]:
    """Parse an SRT file into [{start, end, text}] entries sorted by start (seconds)."""
    raw = path.read_text(encoding="utf-8-sig")
    entries: list[dict] = []
    for block in re.split(r"\r?\n\r?\n+", raw.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if len(lines) < 2:
            continue
        # First line is usually an index; the timing line contains "-->".
        timing_idx = 0 if "-->" in lines[0] else 1
        if timing_idx >= len(lines) or "-->" not in lines[timing_idx]:
            continue
        start_s, _, end_s = lines[timing_idx].partition("-->")
        try:
            start = _parse_srt_timestamp(start_s)
            end = _parse_srt_timestamp(end_s)
        except ValueError:
            continue
        text = " ".join(ln.strip() for ln in lines[timing_idx + 1 :]).strip()
        if text:
            entries.append({"start": start, "end": end, "text": text})
    entries.sort(key=lambda e: e["start"])
    return entries


def overlapping_text(entries: list[dict], start: float, end: float) -> dict | None:
    """Find entries whose time range overlaps [start, end] and return
    {start, end, text} where start/end are the actual reference timings
    spanning the matched entries. Returns None when nothing overlaps."""
    matched = [e for e in entries if e["end"] > start and e["start"] < end]
    if not matched:
        return None
    text = "\n".join(e["text"] for e in matched).strip()
    if not text:
        return None
    return {
        "start": matched[0]["start"],
        "end": matched[-1]["end"],
        "text": text,
    }
