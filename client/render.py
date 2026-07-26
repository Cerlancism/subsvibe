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
from dataclasses import dataclass
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

def _row(content: str, style: str) -> Text:
    """A subtitle row that wraps at console width in BOTH the live region
    and scrollback.

    The Console runs with soft_wrap=True, which makes every print() render
    with no_wrap=True / overflow="ignore" — including Live's refresh, where
    LiveRender then CROPS over-long lines to the console width instead of
    wrapping them (scrollback looked fine only because the terminal itself
    wrapped the uncropped output). Explicit per-Text no_wrap/overflow take
    precedence over those print-level options, so long rows fold onto
    additional lines identically in place and in history."""
    return Text(content, style=style, no_wrap=False, overflow="fold")


# Upper bound on committed blocks waiting in the live region. Blocks whose
# async romaji correction is still pending may NOT scroll to scrollback yet
# (scrollback is immutable — flushing early would freeze the draft), so rapid
# consecutive commits can stack blocks while corrections are in flight. Past
# this cap the oldest block force-flushes with its draft romaji, bounding
# region growth when the corrector is slow or down.
HELD_MAX_BLOCKS = 4


@dataclass
class _HeldBlock:
    """One committed utterance still in the live region (not yet scrolled).

    Blocks queue FIFO; scrollback order always matches commit order because
    only the OLDEST block may flush. A block flushes once BOTH hold:

    - `wants_flush`: a flush trigger has released it (a newer commit landed,
      or the next prov gained its translation — the same triggers that used
      to flush the single held line);
    - not `romaji_pending`: its async romanization correction has settled
      (landed, failed, or was never queued). Corrector-less languages are
      born settled, so their flush timing is identical to the pre-block
      behavior.
    """
    transcript: str
    translation: str | None
    key: object
    # Precomputed romaji, supplied by the pipeline: the rule-based draft at
    # commit time when the language has an async corrector (the correction
    # later lands via settle_held_romaji). Used verbatim for the live region
    # AND the scrollback copy, overriding on-the-fly `_romaji()` derivation.
    # None falls back to on-the-fly derivation — the path corrector-less
    # languages always take.
    romaji: str | None = None
    # True while the async correction for this block is still in flight.
    # Gates the flush (see above) and renders the romaji in the provisional
    # color so the viewer can see it's not yet final; settle_held_romaji
    # clears it and the line flips to the committed color.
    romaji_pending: bool = False
    # True while the translation is being refined against a following
    # utterance (revise_held_translation). Provisional color while set; the
    # scrollback copy is always committed-colored (scrollback is immutable).
    translation_revisable: bool = False
    wants_flush: bool = False
    ts: str | None = None
    lag: float | None = None
    entries: int | None = None
    tag: str | None = None
    duration: float | None = None
    gain_db: float | None = None


class LiveRenderer:
    """Single-utterance provisional region + held-commit blocks + scrolling
    history.

    Finals are queue-first in the pipeline (transcript+translation arrive
    together via commit()), so the renderer only ever tracks ONE in-progress
    utterance in the live region at a time. Just-committed lines show in
    committed colors as held blocks (FIFO, usually just one — see _HeldBlock)
    before scrolling to history.

    Held lines stay REVISABLE while on screen: the pipeline can re-translate
    one together with the following utterance (cross-VAD pair translation) and
    land the refined text via revise_held_translation, and the async
    romanization corrector lands its fix via settle_held_romaji — both BEFORE
    the line scrolls to scrollback, so the refined text is what gets frozen
    into history. A block whose correction is still pending will not scroll
    (up to HELD_MAX_BLOCKS), which is what lets corrections land even when
    several lines commit in quick succession."""

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
        # Held-commit region: committed utterances shown in the live region in
        # committed colors, FIFO (oldest first, newest at the bottom, prov
        # below). A block scrolls to history once released by the next commit /
        # prov trigger AND its async romaji correction has settled — see
        # _HeldBlock. No time-based flush, so the newest committed line stays
        # visible during silences. Usually length 0-1; grows past 1 only while
        # corrections are in flight for rapid consecutive commits (bounded by
        # HELD_MAX_BLOCKS).
        self._held: list[_HeldBlock] = []
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
            # Flush everything, pending corrections included — the session is
            # over, so blocks scroll with whatever romaji they have now.
            while self._held:
                self._flush_first_locked()
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

    def _held_romaji_line(self, block: _HeldBlock) -> str:
        """Romaji for a held block: the pipeline's precomputed value if
        supplied (draft, later async-corrected in place), else the on-the-fly
        rule-based derivation. Only meaningful when a romanizer is active."""
        if block.romaji is not None:
            return block.romaji
        return self._romaji(block.transcript)

    def _render(self) -> Group:
        has_held = bool(self._held)
        has_prov = bool(self._prov_transcript)
        # With a romanizer present, each block reserves a 4th line (header /
        # transcript / romaji / translation); without one, the old 3-line block.
        # Reserve the empty-state block at the same height so the live region
        # doesn't jump across silence gaps.
        block_lines = 4 if self._romanizer else 3
        if not (has_held or has_prov):
            return Group(*(Text("") for _ in range(block_lines)))

        items: list = []
        for block in self._held:
            held_header = self._header(
                block.ts, block.lag, block.entries, block.tag,
                duration=block.duration,
                gain_db=block.gain_db,
            )
            items.append(held_header if held_header is not None else Text(""))
            items.append(_row(block.transcript, STYLE_COMMIT_TRANSCRIPT))
            if self._romanizer:
                romaji = self._held_romaji_line(block)
                # While the async correction is still in flight, keep the
                # romaji in the provisional color — it's a draft, not final.
                # It flips to the committed color when settle_held_romaji
                # lands (or fails, settling on the draft).
                romaji_style = (
                    STYLE_PROV_ROMAJI if block.romaji_pending
                    else STYLE_COMMIT_ROMAJI
                )
                items.append(_row(romaji, romaji_style) if romaji else Text(""))
            # While the held translation is still revisable (being refined
            # against a following utterance), show it in the provisional color so
            # the viewer reads it as not-yet-final; otherwise committed color.
            held_translation_style = (
                STYLE_PROV_TRANSLATION if block.translation_revisable
                else STYLE_COMMIT_TRANSLATION
            )
            items.append(
                _row(block.translation, held_translation_style)
                if block.translation else Text("")
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
            items.append(_row(self._prov_transcript, STYLE_PROV_TRANSCRIPT))
            if self._romanizer:
                romaji = self._romaji(self._prov_transcript)
                items.append(_row(romaji, STYLE_PROV_ROMAJI) if romaji else Text(""))
            items.append(
                _row(self._prov_translation, STYLE_PROV_TRANSLATION)
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
        # Same wrap override as _row — see its docstring.
        return Text.assemble(*parts, no_wrap=False, overflow="fold")

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
        romaji_pending: bool = False,
    ) -> None:
        """Place a finalised utterance in the live region in committed colors.
        It stays visible until released by the next commit or the next
        provisional's translation — and, when `romaji_pending`, until its
        async romaji correction settles (see _HeldBlock). There's no
        time-based flush: during long silences the last committed line stays
        on screen as the "last thing said," and as the carry source for the
        next provisional's translation placeholder.

        Cross-VAD pair translation: the pipeline may call
        revise_held_translation BEFORE this commit (refining the prior held
        line against the utterance now being committed). Because the flush
        uses each block's CURRENT translation, the refined text is what
        scrolls to scrollback.

        `key` identifies the utterance being committed. The prov slot is
        cleared if its key matches — late commits for an older utterance no
        longer evict a fresher prov.

        `ts` is preferred when given (caller-supplied audio-time anchor).
        Falls back to whatever was stored in the matching prov slot, then to
        now().

        `romaji`, when given, is used verbatim for this line's romaji (and its
        scrollback copy), overriding on-the-fly derivation — the pipeline pins
        the rule-based draft here when the language has an async romanization
        corrector, with `romaji_pending=True` so the block waits for (and
        shows dim romaji until) the settle_held_romaji that follows. None
        keeps on-the-fly derivation (provisionals and corrector-less
        languages)."""
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
            # A new commit releases every prior block (the trigger that used
            # to flush the single held line); each actually scrolls once its
            # own correction settles, oldest-first.
            for block in self._held:
                block.wants_flush = True
            self._held.append(_HeldBlock(
                transcript=transcript,
                translation=translation if translation else None,
                key=key,
                romaji=romaji,
                romaji_pending=romaji_pending,
                ts=resolved_ts,
                lag=lag,
                entries=entries,
                tag=tag,
                duration=duration,
                gain_db=gain_db,
            ))
            self._flush_ready_locked()
            self._live.update(self._render())

    def _flush_ready_locked(self) -> None:
        """Scroll every leading flush-ready block into history. Caller must
        hold _hold_lock.

        FIFO only: a block behind a still-pending one waits even if its own
        correction has settled, so scrollback order always matches commit
        order. Runs from BOTH directions of the release/settle race — the
        flush triggers (commit, prov-gains-translation) and settle_held_romaji
        — so no ordering between them is assumed.

        Cap: past HELD_MAX_BLOCKS the oldest block force-flushes with its
        draft romaji regardless of pending state, bounding region growth when
        the corrector is slow or down."""
        while self._held:
            block = self._held[0]
            if len(self._held) > HELD_MAX_BLOCKS:
                if block.romaji_pending:
                    log.debug(
                        "held cap %d exceeded - force-flushing %r with draft romaji",
                        HELD_MAX_BLOCKS, block.transcript[:30],
                    )
            elif not block.wants_flush or block.romaji_pending:
                break
            self._flush_first_locked()

    def _flush_first_locked(self) -> None:
        """Scroll the OLDEST held block into history. Caller must hold
        _hold_lock.

        Order matters and is subtle: rich.Live.update() only sets the
        pending renderable; it doesn't push it to LiveRender until
        refresh() runs. process_renderables (which fires during
        console.print) emits LiveRender.renderable AFTER the printed
        content, using whatever was most recently *refreshed* into it.
        If we just call update() and then print, the stale held block
        re-renders under the newly-scrolled copy.

        Fix: pop the block, then refresh() so the shrunken renderable
        is pushed into LiveRender, then print. Callers should set up any
        new state (prov, replacement blocks) BEFORE calling this method so
        the refresh inside picks up the right target.

        The scrollback copy is ALWAYS committed-colored (scrollback is
        immutable) even if the line was revisable / pending on screen."""
        if not self._held:
            return
        block = self._held.pop(0)
        ts = block.ts or datetime.now().strftime("%H:%M:%S.%f")[:-3]
        header = self._header(
            ts, block.lag, block.entries, block.tag,
            duration=block.duration,
            gain_db=block.gain_db,
        ) or _row(ts, STYLE_TIMESTAMP)
        # Stable block: header / transcript / (romaji) / translation (or blank).
        # The romaji line is included only when a romanizer is active, matching
        # the live region's block height.
        lines = [
            header,
            _row(block.transcript, STYLE_COMMIT_TRANSCRIPT),
        ]
        if self._romanizer:
            romaji = self._held_romaji_line(block)
            lines.append(_row(romaji, STYLE_COMMIT_ROMAJI) if romaji else Text(""))
        lines.append(
            _row(block.translation, STYLE_COMMIT_TRANSLATION)
            if block.translation else Text("")
        )
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
            # Defer the held release when the new prov has no translation yet.
            # Otherwise the live region would briefly show transcript-only
            # under a blank translation line while waiting for the LLM. By
            # keeping held on screen until either provisional_translation()
            # fills the prov's own line or the next commit() lands, the
            # viewer sees the prior bold line linger instead of a blank
            # flicker. Tradeoff: region grows 3->6 lines during the overlap.
            # (Released blocks still wait for their own romaji settle — the
            # flush loop enforces both gates.)
            if self._prov_translation is not None and not keep_held:
                for block in self._held:
                    block.wants_flush = True
            self._flush_ready_locked()
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

    def settle_held_romaji(self, romaji: str | None, *, key: object, transcript: str | None = None) -> bool:
        """Land the async romaji correction for a held block: update its romaji
        (when `romaji` is given; None settles on the current draft — the
        corrector failed or returned the draft verbatim) and clear its
        `romaji_pending` gate, letting the block scroll to scrollback once its
        flush trigger has fired — in whichever order release and settle arrive.

        The pipeline's async romanization corrector (a per-language hook;
        currently only Japanese has one) runs off the commit path, so a
        committed line renders immediately with its rule-based draft in the
        provisional romaji color; the settle flips it to the committed color.
        `transcript`, when given, must still match the block's transcript — a
        squash may have replaced the held text after this correction was
        queued, making the fix stale; the merged text's own queued correction
        (FIFO behind this one) settles the block instead.

        Returns True if it settled the block, False if it no-op'd (block
        already force-flushed / scrolled, or transcript changed). On a
        force-flush no-op the scrollback copy keeps the rule-based draft —
        best-effort, never worse than the rule-based romanizer alone."""
        with self._hold_lock:
            block = next((b for b in self._held if b.key == key), None)
            if block is None:
                log.debug("settle_held_romaji NO-OP key=%r (block gone)", key)
                return False
            if transcript is not None and transcript != block.transcript:
                log.debug("settle_held_romaji STALE key=%r (held transcript changed)", key)
                return False
            log.debug("settle_held_romaji key=%r romaji=%r", key, (romaji or "")[:30])
            if romaji is not None:
                block.romaji = romaji
            block.romaji_pending = False
            self._flush_ready_locked()
            self._live.update(self._render())
            return True

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
            block = next((b for b in self._held if b.key == key), None)
            if block is None:
                log.debug("revise_held_translation NO-OP key=%r (block gone)", key)
                return False
            log.debug("revise_held_translation SET key=%r translation=%r", key, translation[:30])
            block.translation = translation if translation else None
            block.translation_revisable = True
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
        romaji_pending: bool = False,
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
        rule-based draft for the merged text; its async correction follows via
        settle_held_romaji, signalled by `romaji_pending=True`); None falls
        back to on-the-fly derivation. Both are always reassigned here — the
        merge changed the transcript, so any prior precomputed romaji (and any
        in-flight correction for the pre-merge text, rejected later by the
        settle's transcript guard) is stale.

        Returns True if it applied, False if it no-op'd because the block has
        already scrolled (or was force-flushed) — the caller then commits the
        following utterance as its own line instead."""
        with self._hold_lock:
            block = next((b for b in self._held if b.key == key), None)
            if block is None:
                log.debug("replace_held NO-OP key=%r (block gone)", key)
                return False
            log.debug("replace_held SET key=%r transcript=%r", key, transcript[:30])
            block.transcript = transcript
            block.translation = translation if translation else None
            block.romaji = romaji
            block.romaji_pending = romaji_pending
            block.translation_revisable = False
            if duration is not None:
                block.duration = duration
            if lag is not None:
                block.lag = lag
            if entries is not None:
                block.entries = entries
            if tag is not None:
                block.tag = tag
            if gain_db is not None:
                block.gain_db = gain_db
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
            # New prov now has its own translation — release the held blocks
            # that were kept on screen to mask the translation-pending gap
            # (see provisional_transcript). Each scrolls once its own romaji
            # correction settles, leaving the live region with the fully-
            # paired prov. Unless keep_held: the newest held line is itself a
            # revisable committed line paired with this tail, so it must stay
            # above the tail prov.
            if not keep_held:
                for block in self._held:
                    block.wants_flush = True
            self._flush_ready_locked()
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
