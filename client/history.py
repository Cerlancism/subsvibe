"""Shared helpers for the --history / --history-seconds ASR prompt window.

Used by both file-mode (client.py) and live-mode (pipeline.py) so the two
paths apply identical count/time-window semantics and identical prompt
composition rules."""
from __future__ import annotations


def select_history(
    buf: list[tuple[float, str]],
    *,
    count: int,
    seconds: float,
    now: float,
) -> list[str]:
    """Return the texts from `buf` that should be included in the next
    segment's prompt. `buf` is a list of (end_time, text) pairs in append
    order. `now` is the new segment's start time — entries whose end is
    older than now - seconds are filtered out when seconds > 0. count caps
    the most-recent N when > 0."""
    if not buf or (count <= 0 and seconds <= 0):
        return []
    window = buf
    if seconds > 0:
        cutoff = now - seconds
        window = [(t, txt) for t, txt in window if t >= cutoff]
    if count > 0:
        window = window[-count:]
    return [txt for _, txt in window]


def compose_prompt(*parts: str | None) -> str | None:
    """Flatten ordered prompt parts into a newline-joined string. Whisper's
    initial_prompt is tokenised as prior speech, so we pass raw text without
    headers."""
    kept = [p for p in parts if p]
    return "\n".join(kept) if kept else None
