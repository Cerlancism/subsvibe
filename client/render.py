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
from typing import Callable

from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.text import Text
from rich.theme import Theme

log = logging.getLogger("subsvibe.render")

# Provisional text is dim so committed lines read as the anchor. Styles are
# semantic theme keys resolved against the palette picked at construction
# (no auto-detection — terminals don't reliably report their background).
STYLE_TIMESTAMP = "subsvibe.timestamp"
STYLE_PROV_TRANSCRIPT = "subsvibe.prov.transcript"
STYLE_PROV_ROMAJI = "subsvibe.prov.romaji"
STYLE_PROV_TRANSLATION = "subsvibe.prov.translation"
STYLE_COMMIT_TRANSCRIPT = "subsvibe.commit.transcript"
STYLE_COMMIT_ROMAJI = "subsvibe.commit.romaji"
STYLE_COMMIT_TRANSLATION = "subsvibe.commit.translation"

# "dark" is the original palette; "light" swaps in darker inks so prov/commit
# contrast survives a white background (light greys, yellow and cyan wash out).
THEMES: dict[str, dict[str, str]] = {
    "dark": {
        STYLE_TIMESTAMP: "grey42",
        # The pre-theme code said "grey80", which is not a valid rich color
        # (the xterm greys jump 78->82) and silently rendered UNSTYLED;
        # grey82 realises the original dim-prov intent.
        STYLE_PROV_TRANSCRIPT: "grey82",
        STYLE_PROV_ROMAJI: "yellow",
        STYLE_PROV_TRANSLATION: "cyan",
        STYLE_COMMIT_TRANSCRIPT: "bold white",
        STYLE_COMMIT_ROMAJI: "bright_yellow",
        STYLE_COMMIT_TRANSLATION: "bold cyan",
    },
    "light": {
        STYLE_TIMESTAMP: "grey42",
        STYLE_PROV_TRANSCRIPT: "grey46",
        STYLE_PROV_ROMAJI: "dark_goldenrod",
        STYLE_PROV_TRANSLATION: "dark_cyan",
        STYLE_COMMIT_TRANSCRIPT: "bold black",
        STYLE_COMMIT_ROMAJI: "bold dark_goldenrod",
        STYLE_COMMIT_TRANSLATION: "bold dark_cyan",
    },
}

class LiveRenderer:
    """Single-utterance provisional region + held-commit + scrolling history.

    Finals are queue-first in the pipeline (transcript+translation arrive
    together via commit()), so the renderer only ever tracks ONE in-progress
    utterance in the live region at a time. The held slot briefly shows the
    just-committed line in committed colors before scrolling it to history.

    The held line's translation stays REVISABLE while it's on screen: the
    pipeline can re-translate it together with the following utterance
    (cross-VAD pair translation) and land the refined text via
    revise_held_translation BEFORE the line scrolls to scrollback, so the
    refined translation is what gets frozen into history."""

    def __init__(
        self,
        romanizer: Callable[[str], str] | None = None,
        theme: str = "dark",
    ) -> None:
        # stderr so a user piping stdout to a file still sees the UI.
        self._console = Console(
            file=sys.stderr, soft_wrap=True, theme=Theme(THEMES[theme]),
        )
        # Romanizer maps a source-language transcript to a Latin-script
        # pronunciation line (romaji/pinyin/translit). None disables the romaji
        # line entirely (blocks stay 3 lines). When set, the romaji line is
        # always reserved (blank for pure-ASCII utterances) so the live region
        # height is stable within a romanizing session.
        self._romanizer = romanizer
        self._prov_transcript: str = ""
        self._prov_translation: str | None = None
        self._prov_key: object | None = None
        self._prov_ts: str | None = None
        self._prov_lag: float | None = None
        self._prov_entries: int | None = None
        self._prov_tag: str | None = None
        self._prov_duration: float | None = None
        self._prov_gain_db: float | None = None
        # Held-commit: a committed utterance shown in the live region in
        # committed colors. Flushed to history when the next commit or
        # provisional arrives — no time-based flush, so the held line stays
        # visible during silences and remains available as the carry source
        # for the next prov's translation placeholder.
        self._held_transcript: str = ""
        self._held_translation: str | None = None
        # Precomputed romaji for the held line, supplied by the caller (the
        # pipeline's LLM corrector for committed JA lines). When set, it is used
        # verbatim for the held line AND its scrollback copy, overriding the
        # on-the-fly `_romaji()` cutlet draft. None falls back to on-the-fly
        # derivation — the path provisionals and non-JA languages always take.
        self._held_romaji: str | None = None
        # True while the held line's translation is still being refined (set by
        # revise_held_translation). Renders the translation in the provisional
        # color so the viewer can see it's not yet final; reset to committed
        # color once a fresh final takes the slot. On flush the scrollback copy
        # is always committed-colored regardless (scrollback is immutable).
        self._held_translation_revisable: bool = False
        # Key of the held utterance. Lets revise_held_translation() rewrite the
        # held line's translation in place (the held line is committed but its
        # translation stays revisable until a newer line takes the held slot).
        self._held_key: object | None = None
        self._held_ts: str | None = None
        self._held_lag: float | None = None
        self._held_entries: int | None = None
        self._held_tag: str | None = None
        self._held_duration: float | None = None
        self._held_gain_db: float | None = None
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

    def _romaji(self, transcript: str) -> str:
        """Romanize `transcript` for the current slot, or "" if disabled / not
        applicable. Derived on the fly so it always tracks whatever transcript
        is in the slot (prov refresh, squash) with no extra state to sync.

        This is the cutlet/rule-based path used for provisionals and for any
        committed line without a precomputed override (see `_held_romaji_line`)."""
        if not self._romanizer or not transcript:
            return ""
        return self._romanizer(transcript)

    def _held_romaji_line(self) -> str:
        """Romaji for the held line: the caller's precomputed value if supplied
        (the LLM corrector for committed JA lines), else the on-the-fly cutlet
        draft. Only meaningful when a romanizer is active."""
        if self._held_romaji is not None:
            return self._held_romaji
        return self._romaji(self._held_transcript)

    def _render(self) -> Group:
        has_held = bool(self._held_transcript)
        has_prov = bool(self._prov_transcript)
        # With a romanizer present, each block reserves a 4th line (header /
        # transcript / romaji / translation); without one, the old 3-line block.
        # Reserve the empty-state block at the same height so the live region
        # doesn't jump across silence gaps.
        block_lines = 4 if self._romanizer else 3
        if not (has_held or has_prov):
            return Group(*(Text("") for _ in range(block_lines)))

        items: list = []
        if has_held:
            held_header = self._header(
                self._held_ts, self._held_lag, self._held_entries, self._held_tag,
                duration=self._held_duration,
                gain_db=self._held_gain_db,
            )
            items.append(held_header if held_header is not None else Text(""))
            items.append(Text(self._held_transcript, style=STYLE_COMMIT_TRANSCRIPT))
            if self._romanizer:
                romaji = self._held_romaji_line()
                items.append(Text(romaji, style=STYLE_COMMIT_ROMAJI) if romaji else Text(""))
            # While the held translation is still revisable (being refined
            # against a following utterance), show it in the provisional color so
            # the viewer reads it as not-yet-final; otherwise committed color.
            held_translation_style = (
                STYLE_PROV_TRANSLATION if self._held_translation_revisable
                else STYLE_COMMIT_TRANSLATION
            )
            items.append(
                Text(self._held_translation, style=held_translation_style)
                if self._held_translation else Text("")
            )

        if has_prov:
            prov_header = self._header(
                self._prov_ts, self._prov_lag,
                self._prov_entries, self._prov_tag,
                duration=self._prov_duration,
                length=len(self._prov_transcript),
                gain_db=self._prov_gain_db,
            )
            items.append(prov_header if prov_header is not None else Text(""))
            items.append(Text(self._prov_transcript, style=STYLE_PROV_TRANSCRIPT))
            if self._romanizer:
                romaji = self._romaji(self._prov_transcript)
                items.append(Text(romaji, style=STYLE_PROV_ROMAJI) if romaji else Text(""))
            items.append(
                Text(self._prov_translation, style=STYLE_PROV_TRANSLATION)
                if self._prov_translation else Text("")
            )
        return Group(*items)

    @staticmethod
    def _header(
        ts: str | None,
        lag: float | None,
        entries: int | None = None,
        tag: str | None = None,
        duration: float | None = None,
        length: int | None = None,
        gain_db: float | None = None,
    ) -> Text | None:
        if (
            ts is None and lag is None and entries is None
            and tag is None and duration is None and length is None
            and gain_db is None
        ):
            return None
        parts: list = []
        if ts is not None:
            parts.append((ts, STYLE_TIMESTAMP))
        if duration is not None:
            if parts:
                parts.append(("  ", STYLE_TIMESTAMP))
            parts.append((f"dur={duration:.2f}s", STYLE_TIMESTAMP))
        if gain_db is not None:
            if parts:
                parts.append(("  ", STYLE_TIMESTAMP))
            parts.append((f"{gain_db:+.1f}dB", STYLE_TIMESTAMP))
        if lag is not None:
            if parts:
                parts.append(("  ", STYLE_TIMESTAMP))
            parts.append((f"lag={lag:.2f}s", STYLE_TIMESTAMP))
        if entries is not None:
            if parts:
                parts.append(("  ", STYLE_TIMESTAMP))
            parts.append((f"n={entries}", STYLE_TIMESTAMP))
        if length is not None:
            if parts:
                parts.append(("  ", STYLE_TIMESTAMP))
            parts.append((f"prov={length}", STYLE_TIMESTAMP))
        if tag is not None:
            if parts:
                parts.append(("  ", STYLE_TIMESTAMP))
            parts.append((tag, STYLE_TIMESTAMP))
        return Text.assemble(*parts)

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
        duration: float | None = None,
        gain_db: float | None = None,
        romaji: str | None = None,
    ) -> None:
        """Place a finalised utterance in the live region in committed colors.
        It stays visible until the next commit overwrites it or the next
        provisional starts — whichever comes first. There's no time-based
        flush: during long silences the last committed line stays on screen
        as the "last thing said," and as the carry source for the next
        provisional's translation placeholder.

        Cross-VAD pair translation: the pipeline may call
        revise_held_translation BEFORE this commit (refining the prior held
        line against the utterance now being committed). Because the flush
        below uses the held line's CURRENT translation, the refined text is
        what scrolls to scrollback.

        `key` identifies the utterance being committed. The prov slot is
        cleared if its key matches — late commits for an older utterance no
        longer evict a fresher prov.

        `ts` is preferred when given (caller-supplied audio-time anchor).
        Falls back to whatever was stored in the matching prov slot, then to
        now().

        `romaji`, when given, is used verbatim for this line's romaji (and its
        scrollback copy), overriding the on-the-fly cutlet draft — the pipeline
        passes the LLM-corrected romaji for committed JA lines here. None keeps
        on-the-fly derivation (provisionals and non-JA languages)."""
        with self._hold_lock:
            commits_prov = key is None or self._prov_key == key
            log.debug(
                "commit key=%r commits_prov=%s prov_key_was=%r transcript=%r",
                key, commits_prov, self._prov_key, transcript[:30],
            )
            resolved_ts = (
                ts
                or (self._prov_ts if commits_prov else None)
                or datetime.now().strftime("%H:%M:%S.%f")[:-3]
            )
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
                self._prov_duration = None
                self._prov_gain_db = None
            # If a previous held is on screen, flush it first (carrying any
            # cross-VAD refinement landed via revise_held_translation). The
            # flush scrolls it to scrollback. We then set the NEW held and update.
            if self._held_transcript:
                self._flush_held_locked()
            self._held_transcript = transcript
            self._held_translation = translation if translation else None
            self._held_romaji = romaji
            # A fresh final owns the slot now: committed-colored until a
            # following utterance pairs with it and revise_held_translation
            # marks it revisable.
            self._held_translation_revisable = False
            self._held_key = key
            self._held_ts = resolved_ts
            self._held_lag = lag
            self._held_entries = entries
            self._held_tag = tag
            self._held_duration = duration
            self._held_gain_db = gain_db
            self._live.update(self._render())

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
        new state (prov, replacement held) BEFORE calling this method so
        the refresh inside picks up the right target.

        The scrollback copy is ALWAYS committed-colored (scrollback is
        immutable) even if the line was revisable on screen."""
        if not self._held_transcript:
            return
        ts = self._held_ts or datetime.now().strftime("%H:%M:%S.%f")[:-3]
        header = self._header(
            ts, self._held_lag, self._held_entries, self._held_tag,
            duration=self._held_duration,
            gain_db=self._held_gain_db,
        ) or Text(ts, style=STYLE_TIMESTAMP)
        # Stable block: header / transcript / (romaji) / translation (or blank).
        # The romaji line is included only when a romanizer is active, matching
        # the live region's block height.
        lines = [
            header,
            Text(self._held_transcript, style=STYLE_COMMIT_TRANSCRIPT),
        ]
        if self._romanizer:
            romaji = self._held_romaji_line()
            lines.append(Text(romaji, style=STYLE_COMMIT_ROMAJI) if romaji else Text(""))
        lines.append(
            Text(self._held_translation, style=STYLE_COMMIT_TRANSLATION)
            if self._held_translation else Text("")
        )
        self._held_transcript = ""
        self._held_translation = None
        self._held_romaji = None
        self._held_translation_revisable = False
        self._held_key = None
        self._held_ts = None
        self._held_lag = None
        self._held_entries = None
        self._held_tag = None
        self._held_duration = None
        self._held_gain_db = None
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
        duration: float | None = None,
        gain_db: float | None = None,
        inherit_from: object | tuple[object, ...] | None = None,
        keep_held: bool = False,
    ) -> None:
        """Refresh the in-place provisional transcript. `key` identifies the
        utterance so late translations for a previous one can be ignored.

        `inherit_from` lists prior keys whose translation the caller wants
        carried into the new slot — used when the prov key changes for the
        SAME open utterance (slicing path emits a `tail`-namespaced key after
        the cheap-path used the bare utterance key). Carry is unconditional:
        we trust the slicer's boundaries, so the prior translation is a
        useful placeholder until the new LLM call returns.

        `keep_held` pins the held line on screen: it is a committed line whose
        translation is still being revised (array-form translation), so it must
        stay as the upper of the two live lines (held above, tail prov below)
        instead of being flushed to scrollback when the prov gains a translation.

        `ts` lets the caller anchor the header (audio-time). Falls back to
        first-call wall time if omitted."""
        candidates: tuple[object, ...]
        if inherit_from is None:
            candidates = ()
        elif isinstance(inherit_from, tuple):
            candidates = inherit_from
        else:
            candidates = (inherit_from,)
        with self._hold_lock:
            new_utt = self._prov_key != key
            log.debug(
                "provisional_transcript key=%r new_utt=%s prov_key_was=%r text=%r prov_trans_was=%r",
                key, new_utt, self._prov_key, transcript[:30], (self._prov_translation or "")[:30],
            )
            # Same key: keep current translation. New key with an inherit-from
            # hint matching the present prov slot: carry its translation
            # (used for the slicing-tail key rotation within the SAME open
            # utterance — same content, just different key namespace). New
            # key with no candidate match: blank. We do NOT fall back to the
            # held slot's translation across utterance boundaries — pairing
            # a prior utterance's translation under a new utterance's
            # transcript is misleading. The viewer accepts a one-LLM-round-
            # trip gap as the cost of truthful pairing.
            if new_utt:
                carried: str | None = None
                for cand in candidates:
                    if cand == self._prov_key:
                        carried = self._prov_translation
                        break
                self._prov_translation = carried
            if ts is not None:
                self._prov_ts = ts
            elif self._prov_ts is None or new_utt:
                self._prov_ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self._prov_key = key
            self._prov_transcript = transcript
            self._prov_lag = lag
            self._prov_entries = entries
            self._prov_tag = tag
            self._prov_duration = duration
            self._prov_gain_db = gain_db
            # Defer the held flush when the new prov has no translation yet.
            # Otherwise the live region would briefly show transcript-only
            # under a blank translation line while waiting for the LLM. By
            # keeping held on screen until either provisional_translation()
            # fills the prov's own line or the next commit() lands, the
            # viewer sees the prior bold line linger instead of a blank
            # flicker. Tradeoff: region grows 3->6 lines during the overlap.
            if self._held_transcript and self._prov_translation is not None and not keep_held:
                self._flush_held_locked()
            else:
                self._live.update(self._render())

    def discard_provisional(self, key: object) -> None:
        """Clear the prov slot if its key matches; no-op otherwise.

        Used by the pipeline to retire a tail prov when its utterance closes
        or a re-transcription leaves no new tail to display."""
        with self._hold_lock:
            if self._prov_key != key:
                log.debug("discard_provisional NO-OP key=%r prov_key=%r", key, self._prov_key)
                return
            log.debug("discard_provisional FIRED key=%r prov_text=%r", key, self._prov_transcript[:30])
            self._prov_transcript = ""
            self._prov_translation = None
            self._prov_key = None
            self._prov_ts = None
            self._prov_lag = None
            self._prov_entries = None
            self._prov_tag = None
            self._prov_duration = None
            self._prov_gain_db = None
            self._live.update(self._render())

    def revise_held_translation(self, translation: str, *, key: object) -> bool:
        """Rewrite the held line's translation in place WITHOUT flushing it.

        The held line is a committed transcript still on screen; its translation
        stays revisable until a newer line takes the held slot. The array-form
        translator (translate_pair) re-translates this line with the following
        in-progress line as context and lands the refined translation here.

        Returns True if it applied, False if it no-op'd because the held slot has
        already moved on (different utterance) or flushed to scrollback. The
        caller uses the False result to drop its now-stale pairing target so it
        stops feeding an off-screen line into translate_pair."""
        with self._hold_lock:
            if self._held_key != key or not self._held_transcript:
                log.debug("revise_held_translation NO-OP key=%r held_key=%r", key, self._held_key)
                return False
            log.debug("revise_held_translation SET key=%r translation=%r", key, translation[:30])
            self._held_translation = translation if translation else None
            self._held_translation_revisable = True
            self._live.update(self._render())
            return True

    def replace_held(
        self,
        transcript: str,
        translation: str | None,
        *,
        key: object,
        duration: float | None = None,
        lag: float | None = None,
        entries: int | None = None,
        tag: str | None = None,
        gain_db: float | None = None,
        romaji: str | None = None,
    ) -> bool:
        """Replace the held line's TRANSCRIPT and translation in place WITHOUT
        flushing it (the line keeps its slot and its start-time `ts`).

        Used when a following utterance is merged into the held line (squash):
        the model rendered them as one continuous utterance, so the held line
        becomes `A + B` with a single combined translation. The translation is
        committed-colored (it's a finished merge, not a pending refinement).
        `duration` (and any other supplied metadata) updates the header so it
        spans the merged range; `ts` is left as-is so the run keeps its original
        start time. Only the explicitly-passed metadata is overwritten.

        `romaji` overrides this merged line's romaji verbatim (the pipeline's
        LLM corrector for the merged JA text); None falls back to the on-the-fly
        cutlet draft. It is always reassigned here — the merge changed the
        transcript, so any prior precomputed romaji is stale.

        Returns True if it applied, False if it no-op'd because the held slot has
        moved on (different key) or is empty — the caller then commits the
        following utterance as its own line instead."""
        with self._hold_lock:
            if self._held_key != key or not self._held_transcript:
                log.debug("replace_held NO-OP key=%r held_key=%r", key, self._held_key)
                return False
            log.debug("replace_held SET key=%r transcript=%r", key, transcript[:30])
            self._held_transcript = transcript
            self._held_translation = translation if translation else None
            self._held_romaji = romaji
            self._held_translation_revisable = False
            if duration is not None:
                self._held_duration = duration
            if lag is not None:
                self._held_lag = lag
            if entries is not None:
                self._held_entries = entries
            if tag is not None:
                self._held_tag = tag
            if gain_db is not None:
                self._held_gain_db = gain_db
            self._live.update(self._render())
            return True

    def provisional_translation(
        self,
        translation: str,
        *,
        key: object,
        lag: float | None = None,
        keep_held: bool = False,
    ) -> None:
        """Refresh just the translation line. Dropped if the prov slot has
        moved on to a newer utterance.

        `keep_held` pins the held line on screen (array-form translation: the
        held line is a committed line still being revised alongside this tail).
        See provisional_transcript for the layout rationale."""
        with self._hold_lock:
            if self._prov_key != key:
                log.debug("provisional_translation MISMATCH key=%r prov_key=%r", key, self._prov_key)
                return
            log.debug("provisional_translation SET key=%r translation=%r", key, translation[:30])
            self._prov_translation = translation
            if lag is not None:
                self._prov_lag = lag
            # New prov now has its own translation — release the held line
            # that was kept on screen to mask the translation-pending gap
            # (see provisional_transcript). Flush scrolls held to scrollback,
            # leaving the live region at 3 lines with the fully-paired prov.
            # Unless keep_held: the held line is itself a revisable committed
            # line paired with this tail, so it must stay above the tail prov.
            if self._held_transcript and not keep_held:
                self._flush_held_locked()
            else:
                self._live.update(self._render())

    def _install_log_handler(self) -> None:
        root = logging.getLogger()
        self._saved_handlers = root.handlers[:]
        self._saved_level = root.level
        # Inherit the console level from whichever StreamHandler was installed
        # (the first non-file handler); fall back to the root level.
        console_level = next(
            (h.level for h in self._saved_handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)),
            root.level,
        )
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
        self._rich_handler.setLevel(console_level)
        root.addHandler(self._rich_handler)
        for h in self._saved_handlers:
            if isinstance(h, logging.FileHandler):
                root.addHandler(h)

    def _restore_log_handler(self) -> None:
        root = logging.getLogger()
        root.removeHandler(self._rich_handler)
        for h in self._saved_handlers:
            root.addHandler(h)
        root.setLevel(self._saved_level)
