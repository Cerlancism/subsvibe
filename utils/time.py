from __future__ import annotations


def format_timestamp(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"
    if m:
        return f"{int(m):02d}:{s:05.2f}"
    return f"{s:.2f}"


def format_hms(seconds: float) -> str:
    """Format a duration as HH:MM:SS (whole seconds)."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
