"""Live terminal renderer backed by rich.live.

Committed subtitle lines scroll above an atomically-refreshed "live region"
that holds the in-progress (provisional) utterance. Logs are routed through
the same rich Console (via RichHandler) so they also scroll above the live
region instead of colliding with the in-place writes.

Threading: rich.live.Live is safe to update from any thread once started.
"""
from __future__ import annotations

import logging
import sys

from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.text import Text


class LiveRenderer:
    """Two-line provisional region + scrolling committed history."""

    def __init__(self) -> None:
        # Send everything to stderr so a user redirecting stdout (e.g. piping
        # to a file) still sees the UI. Subtitles are part of the UI here, not
        # data output.
        self._console = Console(file=sys.stderr, soft_wrap=True)
        self._prov_transcript: str = ""
        self._prov_translation: str | None = None
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=12,
            transient=False,
        )

    def __enter__(self) -> "LiveRenderer":
        self._live.start()
        self._install_log_handler()
        return self

    def __exit__(self, *exc) -> None:
        self._restore_log_handler()
        self._live.stop()

    def _render(self) -> Group:
        items: list = []
        if self._prov_transcript:
            items.append(Text(self._prov_transcript, style="bright_white"))
            if self._prov_translation:
                items.append(Text(f"  → {self._prov_translation}", style="cyan"))
        return Group(*items)

    def commit(self, transcript: str, translation: str | None) -> None:
        """Print a finalised utterance above the live region."""
        if translation:
            self._console.print(Text(transcript, style="bold white"))
            self._console.print(Text(f"  → {translation}", style="bold cyan"))
        else:
            self._console.print(Text(transcript, style="bold white"))
        # New utterance starts: clear the provisional region.
        self._prov_transcript = ""
        self._prov_translation = None
        self._live.update(self._render())

    def provisional(self, transcript: str, translation: str | None = None) -> None:
        """Refresh the in-place provisional region."""
        self._prov_transcript = transcript
        self._prov_translation = translation
        self._live.update(self._render())

    def _install_log_handler(self) -> None:
        root = logging.getLogger()
        self._saved_handlers = root.handlers[:]
        self._saved_level = root.level
        root.handlers = []
        self._rich_handler = RichHandler(
            console=self._console,
            show_path=False,
            show_time=True,
            rich_tracebacks=True,
            markup=False,
            log_time_format="%H:%M:%S",
        )
        # Include the logger name (e.g. "subsvibe.pipeline") in each line.
        self._rich_handler.setFormatter(logging.Formatter("%(name)-18s %(message)s"))
        root.addHandler(self._rich_handler)

    def _restore_log_handler(self) -> None:
        root = logging.getLogger()
        root.removeHandler(self._rich_handler)
        for h in self._saved_handlers:
            root.addHandler(h)
        root.setLevel(self._saved_level)
