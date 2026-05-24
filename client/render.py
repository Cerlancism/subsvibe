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
import threading
from datetime import datetime
from difflib import SequenceMatcher

from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.text import Text

# Provisional text is dim so committed lines read as the anchor. When a final
# is pending translation alongside a fresher provisional, the latter renders
# in the NEXT_* palette so they read as "anchored | tentative".
STYLE_TIMESTAMP = "grey42"
STYLE_PROV_TRANSCRIPT = "grey80"
STYLE_PROV_TRANSLATION = "cyan"
STYLE_COMMIT_TRANSCRIPT = "bold white"
STYLE_COMMIT_TRANSLATION = "bold cyan"
STYLE_NEXT_TRANSCRIPT = "grey50"
STYLE_NEXT_TRANSLATION = "color(31)"
STYLE_SEPARATOR = "grey42"
SEPARATOR = "  "

# Seconds to keep a committed line visible in the live region before scrolling
# it into history. Cancelled if the next provisional arrives first.
COMMIT_HOLD_SECONDS = 3.0

# Similarity threshold for carrying a prior provisional/pending translation
# into a freshly arrived pending-final. ASR refinements that nudge the
# transcript (punctuation, casing, one or two corrected words) stay above
# this; a wholesale rewrite falls below and the old translation is blanked
# so the viewer never sees a translation paired with mismatched text.
CARRY_TRANSLATION_SIMILARITY = 0.6


def _transcripts_similar(prev: str, new: str) -> bool:
    """True when `new` is close enough to `prev` that an existing translation
    of `prev` is still a useful preview of `new`. Empty inputs return False."""
    if not prev or not new:
        return False
    if prev == new:
        return True
    return SequenceMatcher(None, prev, new).ratio() >= CARRY_TRANSLATION_SIMILARITY


class LiveRenderer:
    """Two-line provisional region + scrolling committed history."""

    def __init__(self) -> None:
        # stderr so a user piping stdout to a file still sees the UI.
        self._console = Console(file=sys.stderr, soft_wrap=True)
        self._prov_transcript: str = ""
        self._prov_translation: str | None = None
        self._prov_key: object | None = None
        self._prov_ts: str | None = None
        self._prov_lag: float | None = None
        self._prov_entries: int | None = None
        self._prov_tag: str | None = None
        # Pending-final: a final whose translation is still in flight. Kept
        # on screen so the next utterance's provisional renders alongside
        # rather than overwriting it.
        self._pending_final_transcript: str = ""
        self._pending_final_translation: str | None = None
        self._pending_final_key: object | None = None
        self._pending_final_ts: str | None = None
        self._pending_final_lag: float | None = None
        self._pending_final_entries: int | None = None
        self._pending_final_tag: str | None = None
        # Held-commit: a committed utterance shown in the live region in
        # committed colors. Flushed to history when the next provisional
        # arrives or after COMMIT_HOLD_SECONDS, whichever comes first.
        self._held_transcript: str = ""
        self._held_translation: str | None = None
        self._held_ts: str | None = None
        self._held_lag: float | None = None
        self._held_entries: int | None = None
        self._held_tag: str | None = None
        self._hold_timer: threading.Timer | None = None
        self._hold_lock = threading.Lock()
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
        with self._hold_lock:
            if self._held_transcript:
                self._flush_held_locked()
        self._restore_log_handler()
        self._live.stop()

    def _render(self) -> Group:
        has_held = bool(self._held_transcript)
        has_pending = bool(self._pending_final_transcript)
        has_prov = bool(self._prov_transcript)
        # Suppress the next utterance's provisional while the previous final
        # is still waiting for its translation. Showing both together causes
        # the line to flicker between layouts as the prov updates. Once the
        # pending translation arrives (or the final commits / times out) the
        # prov becomes visible again on its next update.
        if has_pending and self._pending_final_translation is None:
            has_prov = False
        # Always reserve a 3-line block (header / transcript / translation)
        # so the live region's height stays stable across silence gaps and
        # the moment a translation lands. Empty Text() placeholders fill
        # any slot whose content hasn't arrived yet.
        if not (has_held or has_pending or has_prov):
            return Group(Text(""), Text(""), Text(""))

        items: list = []
        if has_held:
            held_header = self._header(self._held_ts, self._held_lag, self._held_entries, self._held_tag)
            items.append(held_header if held_header is not None else Text(""))
            items.append(Text(self._held_transcript, style=STYLE_COMMIT_TRANSCRIPT))
            items.append(
                Text(self._held_translation, style=STYLE_COMMIT_TRANSLATION)
                if self._held_translation else Text("")
            )

        if has_pending or has_prov:
            # Build a per-slot header for each side and join with the same
            # SEPARATOR the transcript/translation lines use. Anchoring a
            # single shared header on the older slot while reading lag from
            # the newer one was confusing — the header's ts and lag now
            # always reflect the same content.
            pending_header = (
                self._header(
                    self._pending_final_ts, self._pending_final_lag,
                    self._pending_final_entries, self._pending_final_tag,
                )
                if has_pending else None
            )
            prov_header = (
                self._header(
                    self._prov_ts, self._prov_lag,
                    self._prov_entries, self._prov_tag,
                )
                if has_prov else None
            )
            header = self._compose_headers(pending_header, prov_header)
            items.append(header if header is not None else Text(""))
            transcript_line = self._compose(
                self._pending_final_transcript, STYLE_PROV_TRANSCRIPT,
                self._prov_transcript if has_prov else None, STYLE_NEXT_TRANSCRIPT,
            )
            items.append(transcript_line if transcript_line is not None else Text(""))
            translation_line = self._compose(
                self._pending_final_translation, STYLE_PROV_TRANSLATION,
                self._prov_translation if has_prov else None, STYLE_NEXT_TRANSLATION,
            )
            items.append(translation_line if translation_line is not None else Text(""))
        return Group(*items)

    @staticmethod
    def _compose(left: str | None, left_style: str, right: str | None, right_style: str) -> Text | None:
        """Render `left  right` when both are present; if only one side has
        content it gets the primary `left_style` (nothing to defer to)."""
        left = left or ""
        right = right or ""
        if left and right:
            return Text.assemble((left, left_style), (SEPARATOR, STYLE_SEPARATOR), (right, right_style))
        if left:
            return Text(left, style=left_style)
        if right:
            return Text(right, style=left_style)
        return None

    @staticmethod
    def _compose_headers(left: Text | None, right: Text | None) -> Text | None:
        """Join two pre-styled mini-headers with the shared separator.
        Mirrors `_compose` for the content rows so the columns visually
        align (within soft-wrap limits)."""
        if left and right:
            return Text.assemble(left, (SEPARATOR, STYLE_SEPARATOR), right)
        return left or right

    @staticmethod
    def _header(
        ts: str | None,
        lag: float | None,
        entries: int | None = None,
        tag: str | None = None,
    ) -> Text | None:
        if ts is None and lag is None and entries is None and tag is None:
            return None
        parts: list = []
        if ts is not None:
            parts.append((ts, STYLE_TIMESTAMP))
        if lag is not None:
            if parts:
                parts.append(("  ", STYLE_TIMESTAMP))
            parts.append((f"lag={lag:.2f}s", STYLE_TIMESTAMP))
        if entries is not None:
            if parts:
                parts.append(("  ", STYLE_TIMESTAMP))
            parts.append((f"n={entries}", STYLE_TIMESTAMP))
        if tag is not None:
            if parts:
                parts.append(("  ", STYLE_TIMESTAMP))
            parts.append((tag, STYLE_TIMESTAMP))
        return Text.assemble(*parts)

    def pending_final(
        self,
        transcript: str,
        *,
        key: object,
        lag: float | None = None,
        entries: int | None = None,
        tag: str | None = None,
        ts: str | None = None,
    ) -> None:
        """Park a final's transcript in the live region while its translation
        is still in flight. The next utterance's provisional (if any) will
        render alongside rather than replace it.

        If a previous pending-final is still on screen (translate is behind),
        it gets overwritten here — but the pipeline never drops finals from
        the translate queue, so its commit() will still arrive and write it
        to scrollback then. The live region only ever shows the freshest one.

        `ts` lets the caller anchor the header to a fixed reference (e.g. the
        audio position of this slice) so it stays stable across re-emits and
        slot overwrites. Falls back to first-call wall time if omitted."""
        with self._hold_lock:
            # Carry over an existing prov translation when prov matches the
            # finalising utterance, OR a pending translation if pending was
            # already this utterance (e.g. promoted from a prior prov_transcript
            # call when a newer utterance overtook the prov slot).
            prov_same = self._prov_key == key
            pending_same = self._pending_final_key == key
            carried_translation: str | None = None
            # Only carry the prior translation if its source text is close
            # enough to the new transcript that it still reads as a useful
            # preview. A wholesale rewrite blanks the slot until the real
            # translation lands via commit().
            if prov_same and _transcripts_similar(self._prov_transcript, transcript):
                carried_translation = self._prov_translation
            elif pending_same and _transcripts_similar(self._pending_final_transcript, transcript):
                carried_translation = self._pending_final_translation

            self._pending_final_transcript = transcript
            self._pending_final_translation = carried_translation
            self._pending_final_key = key
            self._pending_final_ts = ts or datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self._pending_final_lag = lag
            self._pending_final_entries = entries
            self._pending_final_tag = tag
            if prov_same:
                self._prov_transcript = ""
                self._prov_translation = None
                self._prov_key = None
                self._prov_ts = None
                self._prov_lag = None
                self._prov_entries = None
                self._prov_tag = None
            if self._held_transcript:
                self._flush_held_locked()
            else:
                self._live.update(self._render())

    def commit(
        self,
        transcript: str,
        translation: str | None,
        *,
        key: object | None = None,
        lag: float | None = None,
        entries: int | None = None,
        tag: str | None = None,
        ts: str | None = None,
    ) -> None:
        """Place a finalised utterance in the live region in committed colors.
        It stays there until the next provisional arrives or until
        COMMIT_HOLD_SECONDS elapses, whichever comes first.

        `key` identifies the utterance being committed. The pending/prov
        slots are only cleared when their keys match — late translations
        for an older utterance no longer evict a fresher pending/prov.

        `ts` is preferred when given (caller-supplied audio-time anchor).
        Falls back to whatever was stored in the matching slot, then to now()."""
        with self._hold_lock:
            commits_pending = key is None or self._pending_final_key == key
            commits_prov = key is None or self._prov_key == key
            resolved_ts = (
                ts
                or (self._pending_final_ts if commits_pending else None)
                or (self._prov_ts if commits_prov else None)
                or datetime.now().strftime("%H:%M:%S.%f")[:-3]
            )
            if commits_pending:
                self._pending_final_transcript = ""
                self._pending_final_translation = None
                self._pending_final_key = None
                self._pending_final_ts = None
                self._pending_final_lag = None
                self._pending_final_entries = None
                self._pending_final_tag = None
            # Clearing the prov slot avoids the "bumping" effect where
            # leftover provisional state for the same utterance sits next
            # to the held line and resizes as updates land.
            if commits_prov:
                self._prov_transcript = ""
                self._prov_translation = None
                self._prov_key = None
                self._prov_ts = None
                self._prov_lag = None
                self._prov_entries = None
                self._prov_tag = None
            # If a previous held is on screen, flush it first. The flush
            # scrolls it to scrollback. We then set the NEW held and update.
            if self._held_transcript:
                self._flush_held_locked()
            self._held_transcript = transcript
            self._held_translation = translation if translation else None
            self._held_ts = resolved_ts
            self._held_lag = lag
            self._held_entries = entries
            self._held_tag = tag
            self._live.update(self._render())
            self._restart_hold_timer()

    def _restart_hold_timer(self) -> None:
        if self._hold_timer is not None:
            self._hold_timer.cancel()
        timer = threading.Timer(COMMIT_HOLD_SECONDS, self._on_hold_expired)
        timer.daemon = True
        self._hold_timer = timer
        timer.start()

    def _on_hold_expired(self) -> None:
        with self._hold_lock:
            if self._held_transcript:
                self._flush_held_locked()

    def _flush_held_locked(self) -> None:
        """Scroll the held line into history. Caller must hold _hold_lock.

        Order matters and is subtle: rich.Live.update() only sets the
        pending renderable; it doesn't push it to LiveRender until
        refresh() runs. process_renderables (which fires during
        console.print) emits LiveRender.renderable AFTER the printed
        content, using whatever was most recently *refreshed* into it.
        If we just call update() and then print, the stale held line
        re-renders under the newly-scrolled copy.

        Fix: clear held state, then refresh() so the cleared renderable
        is pushed into LiveRender, then print. Callers should set up any
        new state (prov, pending-final, replacement held) BEFORE calling
        this method so the refresh inside picks up the right target."""
        if not self._held_transcript:
            return
        if self._hold_timer is not None:
            self._hold_timer.cancel()
            self._hold_timer = None
        ts = self._held_ts or datetime.now().strftime("%H:%M:%S.%f")[:-3]
        header = self._header(ts, self._held_lag, self._held_entries, self._held_tag) or Text(ts, style=STYLE_TIMESTAMP)
        # Stable 3-line block: header / transcript / translation (or blank).
        lines = [
            header,
            Text(self._held_transcript, style=STYLE_COMMIT_TRANSCRIPT),
            Text(self._held_translation, style=STYLE_COMMIT_TRANSLATION)
            if self._held_translation else Text(""),
        ]
        self._held_transcript = ""
        self._held_translation = None
        self._held_ts = None
        self._held_lag = None
        self._held_entries = None
        self._held_tag = None
        # update() sets pending renderable; refresh() pushes it into LiveRender
        # so the next process_renderables call (during print) sees it.
        self._live.update(self._render())
        self._live.refresh()
        self._console.print(Group(*lines))

    def provisional_transcript(
        self,
        transcript: str,
        *,
        key: object,
        lag: float | None = None,
        entries: int | None = None,
        tag: str | None = None,
        ts: str | None = None,
    ) -> None:
        """Refresh the in-place provisional transcript. `key` identifies the
        utterance so late translations for a previous one can be ignored.

        If the previous prov was for a different utterance and no pending
        slot is occupied, promote it to pending so the older utterance
        stays visible while we wait for its final/translation. Without
        this, the user sees U1's prov vanish, then briefly reappear when
        U1's pending_final lands.

        `ts` lets the caller anchor the header (audio-time). Falls back to
        first-call wall time if omitted."""
        with self._hold_lock:
            new_utt = self._prov_key != key
            if new_utt and self._prov_transcript and not self._pending_final_transcript:
                # Promote the previous prov into the pending slot so the
                # older utterance stays on screen until its commit arrives.
                self._pending_final_transcript = self._prov_transcript
                self._pending_final_translation = self._prov_translation
                self._pending_final_key = self._prov_key
                self._pending_final_ts = self._prov_ts
                self._pending_final_lag = self._prov_lag
                self._pending_final_entries = self._prov_entries
                self._pending_final_tag = self._prov_tag
            if ts is not None:
                self._prov_ts = ts
            elif self._prov_ts is None or new_utt:
                self._prov_ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            if new_utt:
                self._prov_translation = None
            self._prov_key = key
            self._prov_transcript = transcript
            self._prov_lag = lag
            self._prov_entries = entries
            self._prov_tag = tag
            if self._held_transcript:
                self._flush_held_locked()
            else:
                self._live.update(self._render())

    def discard_provisional(self, key: object) -> None:
        """Clear the prov slot if its key matches; no-op otherwise.

        Used by the pipeline to retire a tail prov when its utterance closes
        or a re-transcription leaves no new tail to display. Without this,
        a tail's namespaced key means commits can no longer accidentally
        evict it, so explicit cleanup is required at utterance boundaries."""
        with self._hold_lock:
            if self._prov_key != key:
                return
            self._prov_transcript = ""
            self._prov_translation = None
            self._prov_key = None
            self._prov_ts = None
            self._prov_lag = None
            self._prov_entries = None
            self._prov_tag = None
            self._live.update(self._render())

    def provisional_translation(
        self,
        translation: str,
        *,
        key: object,
        lag: float | None = None,
    ) -> None:
        """Refresh just the translation line. Routes to the slot whose key
        matches: prov if it's still the current utterance, or pending if
        this utterance was promoted there by a newer prov arriving. Dropped
        if neither slot matches (utterance has scrolled away)."""
        with self._hold_lock:
            if self._prov_key == key:
                self._prov_translation = translation
                if lag is not None:
                    self._prov_lag = lag
            elif self._pending_final_key == key:
                self._pending_final_translation = translation
                if lag is not None:
                    self._pending_final_lag = lag
            else:
                return
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
            log_time_format=lambda dt: dt.strftime("%H:%M:%S.%f")[:-3] + " ",
        )
        self._rich_handler.setFormatter(logging.Formatter("%(name)-18s %(message)s"))
        root.addHandler(self._rich_handler)

    def _restore_log_handler(self) -> None:
        root = logging.getLogger()
        root.removeHandler(self._rich_handler)
        for h in self._saved_handlers:
            root.addHandler(h)
        root.setLevel(self._saved_level)
