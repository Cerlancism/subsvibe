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
from datetime import datetime

from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.text import Text

# Subtitle UI palette. Provisional (in-progress) text is dim so the eye treats
# the brighter committed lines as the anchor.
STYLE_TIMESTAMP = "grey42"
STYLE_PROV_TRANSCRIPT = "grey80"
STYLE_PROV_TRANSLATION = "cyan"
STYLE_COMMIT_TRANSCRIPT = "bold white"
STYLE_COMMIT_TRANSLATION = "bold cyan"


class LiveRenderer:
    """Two-line provisional region + scrolling committed history."""

    def __init__(self) -> None:
        # Send everything to stderr so a user redirecting stdout (e.g. piping
        # to a file) still sees the UI. Subtitles are part of the UI here, not
        # data output.
        self._console = Console(file=sys.stderr, soft_wrap=True)
        self._prov_transcript: str = ""
        self._prov_translation: str | None = None
        # Frozen at the first provisional update of an utterance, reused on
        # commit so the timestamp visually anchors the same line through
        # provisional → final. Lag refreshes on every update — it tracks how
        # stale the on-screen line is right now.
        self._prov_ts: str | None = None
        self._prov_lag: float | None = None
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
            header = self._header(self._prov_ts, self._prov_lag)
            if header is not None:
                items.append(header)
            items.append(Text(self._prov_transcript, style=STYLE_PROV_TRANSCRIPT))
            if self._prov_translation:
                items.append(Text(self._prov_translation, style=STYLE_PROV_TRANSLATION))
        return Group(*items)

    @staticmethod
    def _header(ts: str | None, lag: float | None) -> Text | None:
        if ts is None and lag is None:
            return None
        parts: list = []
        if ts is not None:
            parts.append((ts, STYLE_TIMESTAMP))
        if lag is not None:
            if parts:
                parts.append(("  ", STYLE_TIMESTAMP))
            parts.append((f"lag={lag:.2f}s", STYLE_TIMESTAMP))
        return Text.assemble(*parts)

    def commit(self, transcript: str, translation: str | None, *, lag: float | None = None) -> None:
        """Print a finalised utterance above the live region."""
        # Reuse the frozen provisional timestamp if we have one, so the same
        # line that was previewed lands with the same time. Falls back to
        # "now" if commit fires without any prior provisional (e.g. a final
        # event with no preview frames).
        ts = self._prov_ts or datetime.now().strftime("%H:%M:%S.%f")[:-3]
        # Clear the live region first so the background refresh thread can't
        # redraw the stale provisional between the two prints below — that
        # briefly duplicates the just-committed line on screen.
        self._prov_transcript = ""
        self._prov_translation = None
        self._prov_ts = None
        self._prov_lag = None
        self._live.update(self._render())
        header = self._header(ts, lag) or Text(ts, style=STYLE_TIMESTAMP)
        if translation:
            # Print as one Group so a log record can't slip between the
            # header / transcript / translation triple.
            self._console.print(Group(
                header,
                Text(transcript, style=STYLE_COMMIT_TRANSCRIPT),
                Text(translation, style=STYLE_COMMIT_TRANSLATION),
            ))
        else:
            self._console.print(Group(
                header,
                Text(transcript, style=STYLE_COMMIT_TRANSCRIPT),
            ))

    def provisional(self, transcript: str, translation: str | None = None, *, lag: float | None = None) -> None:
        """Refresh the in-place provisional region."""
        if self._prov_ts is None:
            self._prov_ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._prov_transcript = transcript
        self._prov_translation = translation
        self._prov_lag = lag
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
            # Callable form lets us trim microseconds (%f, 6 digits) to ms (3).
            log_time_format=lambda dt: dt.strftime("%H:%M:%S.%f")[:-3],
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
