"""Live pipeline: capture -> VAD -> ASR -> (translate) -> renderer.

Commit-on-silence model. Each speech segment is transcribed (and translated)
once when VAD confirms its end. Provisional events refresh an in-place
preview line while a segment is still open.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError

from capture import (
    LIVE_ASR_PEAK_NORMALIZE_MAX_DB,
    LIVE_LAG_TOLERANCE_SECONDS,
    LIVE_MAX_SEGMENT_SECONDS,
    LIVE_PROVISIONAL_BACKOFF_SECONDS,
    LIVE_SAMPLE_RATE,
    LIVE_VAD_CHUNK_FRAMES,
    encode_wav,
    get_loopback_mic,
    peak_normalize,
)

# Above this open-utterance duration we ask the transcriber for multiple
# entries so completed pieces can be COMMITTED to scrollback early. The live
# preview keeps refreshing either way: the VAD emits a provisional no more often
# than LIVE_PROVISIONAL_MIN_INTERVAL_SECONDS and only when ASR is idle (see
# live_vad.feed), regardless of segment length, and the trailing piece is always
# rendered as a provisional tail. So the
# difference is NOT preview-vs-nothing — it's whether finished clauses graduate
# to bold/permanent history mid-utterance, or the whole thing stays one mutable
# provisional line until VAD closes it (or it force-flushes at
# MAX_SEGMENT_SECONDS). Below the threshold VAD is expected to close cleanly on
# its own, so one whole-segment entry is enough. Pinned at half the force-flush
# cap: a purely pipeline timing choice, unrelated to any backend's
# transcription cost.
LIVE_ENTRIES_MIN_DURATION = LIVE_MAX_SEGMENT_SECONDS / 2

# Minimum gap between committed lines that land together (the entries of one
# streamed translate_entries reply arrive within milliseconds of each other):
# each commit after the first waits until this long after the previous one,
# so the viewer can read them as they appear. The first line of a call is
# never delayed.
LIVE_COMMIT_MIN_INTERVAL_SECONDS = 0.5

# ASR-prompt history buffer trim policy. Entries accumulate for the whole
# session; once the buffer's span exceeds HISTORY_TRIM_AFTER_SECONDS, drop
# everything older than HISTORY_KEEP_SECONDS measured from the newest entry.
# This caps memory in long sessions while keeping enough context for any
# reasonable --history-seconds window.
HISTORY_TRIM_AFTER_SECONDS = 7200.0
HISTORY_KEEP_SECONDS = 3600.0
from history import compose_prompt, select_history
from live_vad import LiveVAD, SegmentEvent
from llm import (
    TRANSLATE_BATCH_MAX,
    TRANSLATE_HISTORY_LEN,
    TRANSLATE_HISTORY_LEN_UPPER_BOUND_MULTIPLIER,
    romanize_ja_fix,
    translate,
    translate_entries,
)
from utils.language import is_spaceless, to_iso_code
from utils.romanize import make_romanizer
from render import LiveRenderer
from transcribe import AsrRoute, live_transcribe, max_input_seconds

log = logging.getLogger("subsvibe.pipeline")


@dataclass
class _Job:
    """An ASR (and optionally translate) job for one segment event."""
    event: SegmentEvent
    enqueued_at: float           # monotonic time
    utt_start_mono: float = 0.0  # monotonic time corresponding to this event/slice's audio START — drift-free anchor for the displayed wall-clock label
    transcript: str = ""         # filled in by ASR worker
    asr_done_at: float = 0.0     # monotonic time
    meta: dict = field(default_factory=dict)


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


def _split_entries(
    ev: SegmentEvent,
    entries: list[dict],
    *,
    hold_last_on_final: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Positional split. Provisionals commit all entries except the trailing
    one (held as the new tail preview); finals commit all entries unless
    `hold_last_on_final` is set (force-flush case — see below).

    Entries here are 0-based on the audio actually sent to the ASR — the
    caller trims `ev.pcm` at `committed_until` before transcribing, so the
    entries already cover only the uncovered residue. Boundary safety comes
    from the entries themselves: faster-whisper segments and `entries_from_words`
    both carve at aligner-chosen breaks (silence gaps, sentence-end punctuation,
    soft-break punctuation, line budget), so trusting them positionally is
    correct without any silence-tail or text-comparison heuristic.

    The trailing entry is held even when its text ends with sentence-end
    punctuation: per-cycle entry *boundaries* are unstable (next cycle's
    longer audio may merge or re-cut the trailing words), regardless of
    surface punctuation.

    `hold_last_on_final`: on a force-flush final (VAD didn't close on silence;
    it chopped at LIVE_MAX_SEGMENT_SECONDS), the trailing entry's right edge
    is the chop boundary — likely mid-word. Hold it so the caller can drop
    it from this final and call LiveVAD.request_splice() with the held
    entry's absolute start; live_vad then prepends the corresponding range
    of the just-flushed PCM (stashed at flush time) to the now-open next
    utterance, so the same audio is re-transcribed with full context on
    the next cycle (lossless, no duplication)."""
    if not entries:
        return [], []
    if ev.final and not hold_last_on_final:
        return list(entries), []
    if len(entries) <= 1:
        return [], list(entries)
    return list(entries[:-1]), [entries[-1]]


def _log_promotion(
    ev: SegmentEvent,
    entries: list[dict],
    commits: list[dict],
    holds: list[dict],
    fmt_ts,
) -> None:
    log.debug(
        "promote [%s-%s] kind=%s entries=%d commit=%d hold=%d | %s",
        fmt_ts(ev.start), fmt_ts(ev.end),
        "final" if ev.final else "prov ",
        len(entries), len(commits), len(holds),
        " || ".join(
            f"[{e['start']:.2f}-{e['end']:.2f}] {e['text'][:30]!r}"
            for e in entries
        ),
    )


def _drain_stale(
    q: "queue.Queue[_Job | None]", current: _Job, *, max_age: float, label: str, fmt_ts,
    lock: "threading.Lock | None" = None,
    dropped_out: "list[_Job] | None" = None,
) -> _Job:
    """If `current` is a stale provisional, drain forward to the freshest
    item. Returns the (possibly newer) job to process.

    Finals are never dropped — they're immutable history and a missed final
    is a permanent gap in the user's transcript. Better late than lost.

    `lock` serialises the drain against a concurrent producer that also drains
    the same queue (translate_q's `_enqueue_translate_with_backoff`); pass it
    for translate_q, omit it for asr_q (single producer/consumer). Only the
    bounded `get_nowait`/`put` region is held under the lock — never a blocking
    get — so it cannot deadlock the producer.

    `dropped_out`, if given, is appended with every job dropped here (the stale
    predecessors AND the surviving job is NOT included). The translate worker
    uses this to discard renderer tail slots that were dropped before they
    could render — otherwise a tail prov whose successor was dropped stale
    leaks on screen forever (its slot key never matches a newer utterance)."""
    if current.event.final:
        return current
    dropped = 0
    if lock is not None:
        lock.acquire()
    try:
        while time.monotonic() - current.enqueued_at > max_age:
            try:
                newer = q.get_nowait()
            except queue.Empty:
                break
            if newer is None:
                q.put(None)
                break
            dropped += 1
            if dropped_out is not None:
                dropped_out.append(current)
            current = newer
            if current.event.final:
                break
    finally:
        if lock is not None:
            lock.release()
    if dropped:
        log.warning(
            "%s stale > %.1fs - dropped %d jobs, jumped to [%s-%s]",
            label, max_age, dropped,
            fmt_ts(current.event.start), fmt_ts(current.event.end),
        )
    return current


def live_capture(
    *,
    asr_client: OpenAI,
    asr_base_url: str,
    model: str,
    language: str | None,
    prompt: str | None,
    history: int = 0,
    history_seconds: float = 0.0,
    translate_target: str | None,
    translate_prompt: str | None = None,
    translate_system: str | None = None,
    translate_history_seconds: float | None = None,
    translate_history_multiplier: int | None = None,
    translate_temperature: float = 0,
    romanize: bool = True,
    theme: str = "dark",
    commit_route: AsrRoute | None = None,
) -> None:
    # Hybrid live: with `commit_route`, finals transcribe on that
    # backend/model (e.g. openrouter grok-stt) and provisionals on the
    # session's (e.g. local faster-whisper medium). Provisionals then only
    # PREVIEW: they never commit sliced entries or advance the cursor, so
    # every committed line comes from the commit route. The cost is that a
    # long utterance commits nothing until the VAD closes it or the
    # force-flush cap fires — no mid-utterance clause commits.
    hybrid = commit_route is not None
    mic = get_loopback_mic()

    # Romanizer backend is picked from the language hint (ja->cutlet,
    # zh->pypinyin, ko->korean-romanizer, else->anyascii). Display-only: never
    # feeds the LLM or ASR prompt. None disables the renderer's romaji line
    # entirely. This rule-based romanizer drives every PROVISIONAL line and any
    # committed line without a precomputed override (i.e. every language that
    # has no async romanization corrector — see romaji_corrector below).
    romanizer = make_romanizer(language) if romanize else None

    # Async romanization corrector — a generic per-language hook: a callable
    # `(source, draft, *, timeout) -> str` that refines the rule-based
    # romanizer's draft on COMMITTED lines and takes real time to run (an LLM
    # round-trip), hence the dedicated worker below instead of a call inside
    # the commit path. Selection is per language; currently only Japanese has
    # an implementation: romanize_ja_fix edits cutlet's draft to fix its few
    # predictable mis-soundings (lexicalized particles こんにちは->...ha,
    # context kanji readings, bad splits — see client/llm.py romanize_ja_fix
    # and tests/test_ja_romaji_llm.py). Corrections land in place on held
    # blocks via renderer.settle_held_romaji. Languages without a corrector
    # keep their pure rule-based romanizer untouched — no LLM call is ever
    # made for zh/ko/auto-detect. Gated on `romanize` being on (so a draft
    # exists to correct).
    romaji_corrector = None
    if romanize and romanizer is not None and to_iso_code(language) == "ja":
        romaji_corrector = romanize_ja_fix

    def _romaji_draft(transcript: str) -> str | None:
        """Rule-based draft romaji for a committed transcript, or None to let
        the renderer derive romaji on the fly (no async corrector for this
        language, or romaji disabled). Returns "" for empty/pure-ASCII input
        (the rule-based draft is ""), pinned so the renderer skips its own
        derivation.

        Synchronous half of corrected romaji: the commit renders with this
        draft immediately; `_queue_romaji_fix` then hands the slow correction
        to the romaji worker to land in place afterwards."""
        if romaji_corrector is None or not transcript:
            return None
        return romanizer(transcript)  # rule-based draft; "" for pure-ASCII

    # Async romaji correction queue. The corrector used to run synchronously
    # inside _emit, so a committed line's render was gated on a SECOND LLM
    # round-trip after translation (and on the no-translate path it blocked the
    # ASR worker outright). Now the commit renders with the rule-based draft
    # (marked `romaji_pending`, dim romaji) and the correction lands eagerly
    # when ready via settle_held_romaji. A pending block will not scroll to
    # scrollback until its own correction settles (renderer FIFO, bounded by
    # render.HELD_MAX_BLOCKS), so corrections land even when several lines
    # commit in quick succession; only a cap-forced flush keeps the draft.
    romaji_q: "queue.Queue[tuple[str, str, object] | None]" = queue.Queue()

    def _queue_romaji_fix(transcript: str, draft: str | None, key: object) -> None:
        """Queue the async correction for a just-rendered committed line.

        Must be called AFTER the renderer call that placed `key` in the held
        region — queueing first would let the worker's settle race the commit
        and no-op, leaving the block pending until the cap forces it out."""
        if draft:  # None (no corrector / disabled) and "" (pure-ASCII): nothing to fix
            romaji_q.put((transcript, draft, key))

    def _romaji_worker() -> None:
        # Jobs are processed strictly FIFO and every job is settled: each held
        # block waits for its OWN settle before it may scroll, so skipping or
        # reordering jobs would pin blocks until the renderer's cap forces
        # them out with the draft.
        while True:
            item = romaji_q.get()
            if item is None:
                break
            transcript, draft, key = item
            try:
                fixed = romaji_corrector(transcript, draft, timeout=float(LIVE_LAG_TOLERANCE_SECONDS))
            except Exception:
                # Best-effort: server down / timeout — settle on the draft
                # already on screen so the block isn't left pending.
                log.debug("romaji correction failed for %r; settling on draft", transcript[:40], exc_info=True)
                fixed = None
            # ALWAYS settle, even when the correction is a no-change (None
            # keeps the draft): the settle is what releases the block for
            # scrollback and flips its romaji to the committed color.
            renderer.settle_held_romaji(
                fixed if (fixed and fixed != draft) else None,
                key=key,
            )

    # Disable the SDK's built-in retries: by the time a retry would land, the
    # audio is stale and the staleness drop in _drain_stale is already moving
    # us to the next segment. Retries just waste server cycles.
    asr_client = asr_client.with_options(max_retries=0)

    asr_q: "queue.Queue[_Job | None]" = queue.Queue()
    translate_q: "queue.Queue[_Job | None]" = queue.Queue()
    # `translate_q` has two threads racing its non-atomic drain-all-then-reput
    # operations: the ASR thread (`_enqueue_translate_with_backoff`) is a
    # concurrent producer that also drains during its refill, while the translate
    # thread runs `_drain_stale` + `_has_pending_tail`. Without serialisation
    # they can interleave mid-drain and reorder a final behind a provisional or
    # drop a job. This lock guards every drain/refill region. It is NEVER held
    # across the worker's blocking `translate_q.get()` (that would deadlock the
    # producer); only the bounded `get_nowait` drains are wrapped. `asr_q` needs
    # no equivalent — it has a single producer and single consumer.
    translate_lock = threading.Lock()
    stop_event = threading.Event()
    # Set while the ASR worker is free (blocked on asr_q.get); cleared while it
    # is transcribing. LiveVAD's provisional emit gate reads this so a slow
    # backend never has fresh provisionals queued on top of the one in flight —
    # the next prov fires the instant ASR frees up. Starts set (idle).
    asr_idle = threading.Event()
    asr_idle.set()
    # `SegmentEvent.start`/`.end` are PCM seconds from the first pulled chunk:
    # fine for ASR cursor / VAD timing, but cumulative sample counts drift vs.
    # the system clock (~70-180ms/hr crystal drift) and MUST NOT feed displayed
    # wall-clock time or lag. The capture worker latches `_Job.utt_start_mono`
    # in monotonic space so only the short within-utterance offset is sample-
    # derived (sub-ms). Lag uses `monotonic() - job.enqueued_at` for the same
    # reason.
    capture_start: float = 0.0
    capture_start_wall: datetime = datetime.now()
    # Separate, never-re-snapshotted wall anchor for `_audio_wall`. The
    # `(capture_start, capture_start_wall)` pair is re-anchored on every VAD
    # speech-start (see the capture worker) so `_mono_wall` absorbs NTP slew /
    # suspend skew — but that re-anchor moves `capture_start_wall` to the
    # detection wall time, and `_audio_wall` adds raw audio_seconds (a
    # session-relative offset) to it, so the apparent wall-time would drift by
    # the cumulative re-anchor delta. A dedicated anchor pinned at session start
    # keeps `_audio_wall` consistent with the audio timeline.
    _audio_anchor_wall: datetime = datetime.now()
    _clock_ready = threading.Event()
    # Guards the (capture_start, capture_start_wall) pair. Writer: the capture
    # worker (first-chunk init + per-VAD-start re-anchor). Readers: `_audio_wall`
    # / `_mono_wall`. Held across two assignments / two reads only.
    _anchor_lock = threading.Lock()

    def _audio_wall(audio_seconds: float) -> str:
        """HH:MM:SS.mmm for an audio-clock offset. Anchored at session start
        and not re-snapshotted, so the displayed time tracks the audio
        timeline regardless of commit-time anchor updates. Crystal-drift
        exposed; only for infrequent, non-re-rendered log lines (VAD
        recovery, segment finalised). UI emit sites use `_mono_wall`."""
        return (_audio_anchor_wall + timedelta(seconds=audio_seconds)).strftime("%H:%M:%S.%f")[:-3]

    def _mono_wall(mono: float) -> str:
        """HH:MM:SS.mmm for a `time.monotonic()` value. Drift-free, and stable
        across re-emits when the caller latches `mono` at the utterance/slice's
        audio-start arrival."""
        with _anchor_lock:
            anchor_wall, anchor_mono = capture_start_wall, capture_start
        return (anchor_wall + timedelta(seconds=mono - anchor_mono)).strftime("%H:%M:%S.%f")[:-3]

    # --- render emitters (defined before workers so closures resolve cleanly) ---
    # Stable identifier for an utterance across its provisional updates and
    # final commit. SegmentEvent.start is monotonic per utterance and reset
    # by the VAD between utterances, so it makes a natural key.
    #
    # Tails sit in a distinct namespace ((ev.start, "tail")) so a sliced
    # final commit whose absolute start equals the utterance's own start
    # doesn't share a key with the tail prov — otherwise the commit() call
    # would clear the tail along with the matching pending slot.
    def _utt_key(ev: SegmentEvent, meta: dict | None = None) -> object:
        if meta and meta.get("tail"):
            return (ev.start, "tail")
        return ev.start

    def _slice_tag(job: _Job) -> str | None:
        """'tail' if this is the held tail provisional, 'sliced' if it's a
        committed sub-job. Every emit now flows the unified slicing loop, so
        one of these always applies; None is retained only as a defensive
        default."""
        if job.meta.get("tail"):
            return "tail"
        if job.meta.get("sliced"):
            return "sliced"
        return None

    def _emit(job: _Job, *, translation: str | None) -> None:
        """Final commit: transcript + optional translation as one atomic line.

        When the language has an async romanization corrector, the committed
        line renders with the rule-based draft immediately (dim romaji,
        `romaji_pending`); the correction is queued to the romaji worker AFTER
        the commit and lands in place via settle_held_romaji when ready — the
        commit never waits on the corrector's round-trip, and the block won't
        scroll to scrollback until its correction settles. `_romaji_draft`
        returns None when no corrector is active, in which case the renderer
        derives the romaji on the fly from its rule-based romanizer."""
        ev = job.event
        lag = time.monotonic() - job.enqueued_at
        key = _utt_key(ev, job.meta)
        romaji_draft = _romaji_draft(job.transcript)
        # No anchor re-snapshot here: the pair is re-anchored at every VAD
        # speech-start (capture worker), so NTP slew / suspend skew is already
        # absorbed once per utterance. `job.utt_start_mono` was latched at this
        # utterance's start against that fresh anchor, so `_mono_wall` formats it
        # against the same (wall, mono) frame it was latched in — drift-free and
        # stable across all re-emits / sub-finals of the utterance.
        renderer.commit(
            job.transcript, translation, key=key, lag=lag,
            entries=job.meta.get("entries"),
            tag=_slice_tag(job),
            ts=_mono_wall(job.utt_start_mono),
            duration=ev.end - ev.start,
            gain_db=job.meta.get("gain_db"),
            romaji=romaji_draft,
            # Truthiness mirrors _queue_romaji_fix's gate exactly: a block is
            # pending iff a correction job is actually queued for it.
            romaji_pending=bool(romaji_draft),
        )
        # AFTER the commit so the held block already carries `key` when the
        # worker's settle lands (see _queue_romaji_fix).
        _queue_romaji_fix(job.transcript, romaji_draft, key)
        _log_emit(job, lag, kind="final")

    def _emit_transcript(job: _Job) -> None:
        """Provisional transcript only — translation line stays as-is until the LLM lands."""
        ev = job.event
        lag = time.monotonic() - job.enqueued_at
        key = _utt_key(ev, job.meta)
        # Provs are always tail-keyed (ev.start, "tail") now — the unified
        # slicing loop emits every in-progress preview through the `holds`
        # tail branch, so successive cycles share the same key and the
        # renderer preserves the translation as a same-key refinement. The
        # `inherit_from=ev.start` hint is retained defensively (it carried the
        # translation across the old bare-key → tail-key transition, which no
        # longer occurs); it's a no-op while the key is stable.
        renderer.provisional_transcript(
            job.transcript, key=key, lag=lag,
            entries=job.meta.get("entries"),
            tag=_slice_tag(job),
            ts=_mono_wall(job.utt_start_mono),
            duration=ev.end - ev.start,
            gain_db=job.meta.get("gain_db"),
            inherit_from=ev.start if isinstance(key, tuple) else None,
        )
        _log_emit(job, lag, kind="prov ")

    def _emit_translation(job: _Job, *, translation: str | None) -> None:
        """Provisional translation update — leaves transcript untouched."""
        ev = job.event
        lag = time.monotonic() - job.enqueued_at
        if translation:
            renderer.provisional_translation(translation, key=_utt_key(ev, job.meta), lag=lag)
        _log_emit(job, lag, kind="prov ")

    def _log_emit(job: _Job, lag: float, *, kind: str) -> None:
        ev = job.event
        log.debug(
            "%s [%s-%s] dur=%.2fs asr=%.2fs%s lag=%.2fs",
            kind, _audio_wall(ev.start), _audio_wall(ev.end),
            job.meta.get("duration", 0.0),
            job.meta.get("asr_elapsed", 0.0),
            f" tr={job.meta['translate_elapsed']:.2f}s" if "translate_elapsed" in job.meta else "",
            lag,
        )

    # --- capture + VAD thread ---
    def _enqueue_with_backoff(new_job: _Job) -> None:
        """Push `new_job` onto asr_q, collapsing any stale same-utterance prov.

        Primary backpressure now lives in LiveVAD's emit gate (provs fire only
        while ASR is idle), so this is a secondary net for the narrow window
        where a prov enqueues just as ASR goes idle, or a final lands behind a
        sitting prov. A newer prov covers [open_start .. now] and strictly
        contains any older one for the same `start`, so dropping the stale
        predecessor is lossless. Finals (and provs younger than the backoff,
        or for a different segment) are always preserved."""
        ev = new_job.event
        if ev.final:
            asr_q.put(new_job)
            return
        now = time.monotonic()
        drained: list[_Job | None] = []
        while True:
            try:
                drained.append(asr_q.get_nowait())
            except queue.Empty:
                break
        dropped = 0
        for item in drained:
            if item is None:
                asr_q.put(item)
                continue
            same_segment = (
                not item.event.final
                and item.event.start == ev.start
                and (now - item.enqueued_at) > LIVE_PROVISIONAL_BACKOFF_SECONDS
            )
            if same_segment:
                dropped += 1
                continue
            asr_q.put(item)
        if dropped:
            log.debug(
                "capture backoff: collapsed %d stale provisionals for [%s-]",
                dropped, _audio_wall(ev.start),
            )
        asr_q.put(new_job)

    def _has_pending_tail(parent_start: float) -> bool:
        """True iff translate_q currently holds a tail prov for `parent_start`.

        Peeks via drain-all-then-restore. This races a concurrent producer —
        the ASR thread's `_enqueue_translate_with_backoff` also drains/refills
        translate_q — so the whole drain/restore runs under `translate_lock`.
        Used by the sub-final commit path to skip the post-commit tail discard
        when a successor tail is already queued: it will overwrite the slot when
        it renders, so an eager discard would only create a blank gap during the
        successor's LLM round-trip."""
        drained: list[_Job | None] = []
        found = False
        with translate_lock:
            while True:
                try:
                    drained.append(translate_q.get_nowait())
                except queue.Empty:
                    break
            for item in drained:
                if (
                    item is not None
                    and not item.event.final
                    and item.meta.get("tail")
                    and item.event.start == parent_start
                ):
                    found = True
                translate_q.put(item)
        return found

    def _enqueue_translate_with_backoff(new_job: _Job, *more: _Job) -> None:
        """Mirror of `_enqueue_with_backoff` for translate_q. Provs are now
        queue-first too, so without this guard translate_q would accrue
        nested provs for the same open utterance whenever the LLM trails
        ASR cadence (~1Hz). Collapse stale provs for the same `ev.start`
        before pushing `new_job`. Finals are always preserved EXCEPT for
        sliced sub-finals, which also evict pending tail provs for their
        parent utterance — the tail's audio range overlaps the just-
        promoted entry, so leaving it queued would render the same line
        twice (committed bold above, stale dim below).

        `more`: further sub-finals of the same parent utterance, queued right
        behind `new_job` under the same lock hold so the translate worker
        finds them together and translates them in one call."""
        ev = new_job.event
        now = time.monotonic()
        parent_start = new_job.meta.get("parent_start") if new_job.meta else None
        # Plain (non-sliced) finals short-circuit: nothing to evict. The `put`
        # is still under the lock so it can't interleave with a concurrent
        # drain on the translate thread (which would let a final slip behind a
        # provisional being restored).
        dropped = 0
        with translate_lock:
            if ev.final and parent_start is None:
                for j in (new_job, *more):
                    translate_q.put(j)
                return
            drained: list[_Job | None] = []
            while True:
                try:
                    drained.append(translate_q.get_nowait())
                except queue.Empty:
                    break
            for item in drained:
                if item is None:
                    translate_q.put(item)
                    continue
                # Drop stale same-utterance provs (existing backoff).
                if (
                    not ev.final
                    and not item.event.final
                    and item.event.start == ev.start
                    and (now - item.enqueued_at) > LIVE_PROVISIONAL_BACKOFF_SECONDS
                ):
                    dropped += 1
                    continue
                # Drop pending tail provs that share this sliced sub-final's
                # parent utterance — they were generated before the cursor
                # advanced past the entry we're about to commit.
                if (
                    parent_start is not None
                    and not item.event.final
                    and item.meta.get("tail")
                    and item.event.start == parent_start
                ):
                    dropped += 1
                    continue
                translate_q.put(item)
            for j in (new_job, *more):
                translate_q.put(j)
        if dropped:
            log.debug(
                "translate backoff: collapsed %d stale provisionals for [%s-]",
                dropped, _audio_wall(ev.start),
            )

    # Latched utterance-start monotonic, keyed by `ev.start`. Set on the first
    # VAD event for an utterance and reused on every subsequent event with the
    # same `ev.start`, so the displayed `ts` is bit-stable across provisional
    # refreshes. Dropped on the utterance's final.
    utt_start_mono_by_utt: dict[float, float] = {}

    # Shared handle to the LiveVAD instance owned by the capture worker. The
    # ASR worker reads it to post force-flush splice requests
    # (request_splice). Only the capture worker writes it (once at startup);
    # ASR reads after the first force-flush event lands in its queue, which
    # is necessarily after init — no race.
    vad_ref: list[LiveVAD] = []

    def _capture_worker() -> None:
        nonlocal capture_start, capture_start_wall, _audio_anchor_wall
        vad = LiveVAD(audio_wall_fn=_audio_wall, asr_idle=asr_idle.is_set)
        vad_ref.append(vad)
        try:
            with mic.recorder(samplerate=LIVE_SAMPLE_RATE, channels=1) as recorder:
                while not stop_event.is_set():
                    chunk = recorder.record(numframes=LIVE_VAD_CHUNK_FRAMES).reshape(-1).astype(np.float32)
                    if not _clock_ready.is_set():
                        # Anchor both clocks at the first chunk's arrival so
                        # WASAPI warmup + Silero load aren't baked into either
                        # the lag readings (monotonic) or the displayed wall
                        # time (wall + monotonic offset).
                        with _anchor_lock:
                            capture_start = time.monotonic()
                            capture_start_wall = datetime.now()
                        # _audio_anchor_wall is pinned here too; never updated
                        # again. Used only by `_audio_wall`.
                        _audio_anchor_wall = capture_start_wall
                        _clock_ready.set()
                    for ev in vad.feed(chunk):
                        now_mono = time.monotonic()
                        # Latch the utterance's START at the FIRST event we see
                        # for it. Re-anchor the (mono, wall) pair to *now* so NTP
                        # slew / suspend skew is re-absorbed once per utterance,
                        # then latch `utt_mono` back to the segment's true open
                        # via `now_mono - (ev.end - ev.start)`. At first sighting
                        # ev.end ≈ now and ev.start is the open position, so this
                        # resolves to the real wall-clock onset — correct for the
                        # primary VAD AND for recovery (whose ~1s onset backtrack
                        # is already baked into ev.start). Later events for the
                        # same utterance reuse the latched value so the displayed
                        # ts stays bit-stable across provisional refreshes. The
                        # segment END is NOT wall-anchored: it stays sample/VAD-
                        # derived via `duration = ev.end - ev.start` at emit sites.
                        utt_mono = utt_start_mono_by_utt.get(ev.start)
                        if utt_mono is None:
                            with _anchor_lock:
                                capture_start = now_mono
                                capture_start_wall = datetime.now()
                            utt_mono = now_mono - (ev.end - ev.start)
                            utt_start_mono_by_utt[ev.start] = utt_mono
                        if ev.final:
                            utt_start_mono_by_utt.pop(ev.start, None)
                        _enqueue_with_backoff(_Job(
                            event=ev,
                            enqueued_at=now_mono,
                            utt_start_mono=utt_mono,
                        ))
        except Exception:
            log.exception("capture worker died - shutting down pipeline")
        finally:
            # Wake the main thread's stop_event.wait() whether we exited
            # normally or crashed; its teardown pushes the None sentinel
            # through asr_q -> translate_q so the workers drain cleanly.
            stop_event.set()
            vad.close()

    # Per-VAD-utterance commit cursor: segment-relative end of the last entry
    # committed from that utterance. Keyed by ev.start (the utterance key).
    # Lets subsequent provisional re-transcriptions and the eventual VAD final
    # skip entries that overlap the already-committed prefix. Entries here are
    # cleared when the VAD final for that utterance is fully processed.
    committed_until_by_utt: dict[float, float] = {}

    # Rolling ASR-prompt history buffer (only finalised transcripts enter;
    # provisionals never do, so prompt context never drifts on mid-sentence
    # noise). Mirrors file mode's --history / --history-seconds semantics.
    history_enabled = history > 0 or history_seconds > 0
    history_buf: list[tuple[float, str]] = []  # (ev.end seconds, text)

    def _trim_history() -> None:
        """Cap history_buf growth. Trims once the buffer spans more than
        HISTORY_TRIM_AFTER_SECONDS of audio, keeping HISTORY_KEEP_SECONDS
        of the most recent entries."""
        if len(history_buf) < 2:
            return
        span = history_buf[-1][0] - history_buf[0][0]
        if span < HISTORY_TRIM_AFTER_SECONDS:
            return
        cutoff = history_buf[-1][0] - HISTORY_KEEP_SECONDS
        i = 0
        while i < len(history_buf) and history_buf[i][0] < cutoff:
            i += 1
        if i:
            dropped = i
            del history_buf[:i]
            log.debug("history buffer trimmed: dropped %d entries, kept %d", dropped, len(history_buf))

    # --- ASR worker ---
    def _transcribe_worker() -> None:
        while True:
            job = asr_q.get()
            if job is None:
                if translate_target:
                    translate_q.put(None)
                break

            # Mark busy for the whole transcribe cycle so LiveVAD holds new
            # provisionals for this open segment until we free up. The finally
            # restores idle on every exit path (commit, skip-continue, error).
            asr_idle.clear()
            try:
                _transcribe_one(job)
            except Exception:
                # Catch-all so a single bad job (entry-dict KeyError, NumPy
                # error, encode failure, ...) can't kill the worker thread
                # and freeze the pipeline. The job is dropped; the next
                # provisional cycle re-covers its audio for open utterances.
                log.exception(
                    "asr worker error on [%s-%s] - job dropped",
                    _audio_wall(job.event.start), _audio_wall(job.event.end),
                )
            finally:
                asr_idle.set()

    def _transcribe_one(job: _Job) -> None:
        job = _drain_stale(asr_q, job, max_age=LIVE_LAG_TOLERANCE_SECONDS, label="asr", fmt_ts=_audio_wall)
        ev = job.event
        duration = ev.end - ev.start

        # Cursor-trim audio: send only the residue past the last commit.
        # Eliminates the cross-cycle duplication class by construction —
        # text and entries returned cover the uncovered tail only, so
        # there's no overlap with prior sliced commits to reconcile.
        # Cost (loss of acoustic context across the cursor) is offset by
        # `seg_prompt` carrying prior committed text when history is on.
        committed_until = committed_until_by_utt.get(ev.start, 0.0)
        if committed_until > 0.0:
            trim_samples = int(round(committed_until * LIVE_SAMPLE_RATE))
            pcm_to_send = ev.pcm[trim_samples:]
        else:
            pcm_to_send = ev.pcm
        # Empty tail on a final: cursor already covered everything. No
        # audio to send and no held tail can exist (slicer would have
        # committed it on the cycle that advanced the cursor this far).
        # Clean up cursor state and move on.
        if ev.final and len(pcm_to_send) == 0:
            committed_until_by_utt.pop(ev.start, None)
            renderer.discard_provisional((ev.start, "tail"))
            return

        history_texts = select_history(
            history_buf, count=history, seconds=history_seconds, now=ev.start,
        ) if history_enabled else []
        seg_prompt = compose_prompt(prompt, " ".join(history_texts) if history_texts else None)

        t0 = time.monotonic()
        # Ask for multiple entries once the open utterance is long enough
        # that waiting for VAD to close it would leave finished clauses
        # stuck in the mutable provisional line instead of committed to
        # scrollback (half the force-flush cap), or whenever we're past the
        # first cursor advance (continuation cycles must keep draining the
        # residue). Pipeline timing intent only — the backend cost of
        # honouring it (aligner pass or not) is transcribe.py's concern.
        want_segments = duration >= LIVE_ENTRIES_MIN_DURATION or committed_until > 0.0
        if hybrid and not ev.final:
            # Preview-only provisional: one whole-tail entry is all it shows.
            want_segments = False
        tail_start_abs = ev.start + committed_until
        # Defence-in-depth: never POST audio longer than the server's input
        # cap. The VAD's force-flush ceiling keeps segments well under this,
        # so this firing means a VAD regression — trim to the cap's head and
        # warn rather than losing the whole segment to a server 500.
        max_input_samples = int(max_input_seconds() * LIVE_SAMPLE_RATE)
        if len(pcm_to_send) > max_input_samples:
            log.warning(
                "segment [%s-%s] is %.1fs, past the server input cap of %.0fs - trimming before POST",
                _audio_wall(tail_start_abs), _audio_wall(ev.end),
                len(pcm_to_send) / LIVE_SAMPLE_RATE, max_input_seconds(),
            )
            pcm_to_send = pcm_to_send[:max_input_samples]
        # Independent peak-normalise of the segment before ASR (chunks
        # were normalised individually for VAD; this scales the assembled
        # segment to its own peak so ASR sees a consistent level).
        pcm_to_send, gain_db = peak_normalize(pcm_to_send, LIVE_ASR_PEAK_NORMALIZE_MAX_DB)
        job.meta["gain_db"] = gain_db
        wav_bytes = encode_wav(pcm_to_send)

        def _asr(route: AsrRoute | None) -> tuple[str, list[dict]]:
            return live_transcribe(
                asr_client, model,
                wav_bytes,
                f"{_fmt_ts(tail_start_abs)}-{_fmt_ts(ev.end)}.wav",
                language=language,
                prompt=seg_prompt,
                timeout=LIVE_LAG_TOLERANCE_SECONDS,
                # Fallback span for the synthetic whole-segment entry: the
                # trimmed tail audio actually sent, 0-based to match the
                # aligner's own entry frame.
                segment_duration=len(pcm_to_send) / LIVE_SAMPLE_RATE,
                want_segments=want_segments,
                route=route,
            )

        def _drop_failed_final() -> None:
            # A final that could not be transcribed at all: close out the
            # cursor and the tail preview so a stale provisional isn't left
            # on screen for an utterance that will never commit.
            if ev.final:
                committed_until_by_utt.pop(ev.start, None)
                renderer.discard_provisional((ev.start, "tail"))

        try:
            try:
                text, entries = _asr(commit_route if ev.final else None)
            except (APIConnectionError, APIStatusError) as exc:
                if not (hybrid and ev.final):
                    raise
                # Hybrid: provisionals never commit, so a failed commit-route
                # final would lose the line outright. Retry it once on the
                # session (provisional) backend instead.
                log.warning(
                    "commit ASR %s for final [%s-%s] - retrying on the session backend",
                    "timeout" if isinstance(exc, APITimeoutError) else f"error ({exc})",
                    _audio_wall(tail_start_abs), _audio_wall(ev.end),
                )
                text, entries = _asr(None)
        except APITimeoutError:
            log.error(
                "ASR timeout for [%s-%s] after %.2fs - dropping",
                _audio_wall(tail_start_abs), _audio_wall(ev.end), LIVE_LAG_TOLERANCE_SECONDS,
            )
            _drop_failed_final()
            return
        except APIConnectionError:
            log.error("could not connect to transcription backend at %s", asr_base_url)
            _drop_failed_final()
            return
        except APIStatusError as exc:
            log.error("server error %s: %s", exc.status_code, exc.message)
            _drop_failed_final()
            return
        elapsed = time.monotonic() - t0

        if not text:
            # Final with no text: close out cursor + tail prov so they
            # don't leak. Most common path is tiny-tail finals where the
            # server returns empty for sub-second audio. Provisionals
            # leave state alone — a future cycle will refresh.
            if ev.final:
                committed_until_by_utt.pop(ev.start, None)
                renderer.discard_provisional((ev.start, "tail"))
            return

        job.asr_done_at = time.monotonic()
        # `entries` is the ASR's segment count for THIS provisional cycle —
        # surfaced in the live header (`n=3`) so the viewer can see how the
        # utterance was carved. live_transcribe guarantees the
        # `text ⇒ entries` invariant, so this is always >= 1; n=1 means the
        # cheap-JSON request or a single whole-segment slice.
        job.meta = {
            "asr_elapsed": elapsed,
            "duration": duration,
            "gain_db": gain_db,
            "entries": len(entries),
        }

        # Force-flush detection. VAD chops at LIVE_MAX_SEGMENT_SECONDS
        # mid-utterance when speech keeps going past the cap; the trailing
        # entry of such a final sits at the chop boundary and is likely
        # mid-word. Drop the trailing entry and call
        # LiveVAD.request_splice() so the dropped audio is prepended to
        # the next utterance and re-transcribed cleanly next cycle
        # (lossless, no duplication). When n==1 with the cursor advanced,
        # the single entry IS the trailing residue past prior commits —
        # also splice it (clamp below keeps audio disjoint).
        is_force_flush = ev.final and duration >= LIVE_MAX_SEGMENT_SECONDS
        # Splice when n>=2 (clear leading + trailing entries), OR when
        # n==1 and the cursor has already advanced — the n=1 entry IS
        # the trailing residue past the cursor, and the splice clamp
        # below (against `ev.start + committed_until`) keeps the re-fed
        # audio range provably disjoint from previously committed entries'
        # source audio. The earlier looser-gate experiment (issue #29)
        # caused visible duplication because it lacked this clamp; with
        # the sample-accurate audio-end authority the n=1 path is safe.
        # The un-spliceable cases commit at the chop boundary and warn:
        # - n==1 with committed_until==0: a whole-segment entry at the start
        #   of an utterance (a monologue the aligner couldn't split, or a
        #   cheap-JSON synthetic entry). No aligner-reported internal
        #   boundary to anchor on — splicing would loop forever dropping
        #   every cycle.
        # - ev.spliceable False: the VAD didn't stash carryover PCM for
        #   this flush (recovery-driven force-flush). Holding a trailing
        #   entry would rely on a splice that _apply_pending_splice
        #   silently drops — the held entry and its audio would be lost
        #   (issue #37).
        can_splice = is_force_flush and ev.spliceable and (
            len(entries) >= 2 or (len(entries) == 1 and committed_until > 0.0)
        )
        # Force-flush without splice eligibility: trailing audio commits
        # at the chop boundary — likely mid-word for long unbroken speech.
        if is_force_flush and not can_splice:
            log.debug(
                "force-flush with %d entries (spliceable=%s): committing potential "
                "mid-word chop, residue=[%s-%s] full-utt=[%s-%s] dur=%.2fs",
                len(entries), ev.spliceable,
                _audio_wall(tail_start_abs), _audio_wall(ev.end),
                _audio_wall(ev.start), _audio_wall(ev.end),
                duration,
            )
        # When the open utterance has crossed MAX (force-flush or about to
        # be), anchor the next ASR call at the held entry's *start* instead
        # of the prior entry's end — closes the silence-gap window where
        # the next cycle could re-cut a segment that this cycle already
        # carved cleanly.
        anchor_to_hold_start = duration >= LIVE_MAX_SEGMENT_SECONDS
        # Single unified promotion path. live_transcribe guarantees
        # non-empty text yields >= 1 entry, so entries is always non-empty
        # here (empty text already returned above). Entries are 0-based
        # on the trimmed tail audio we sent, so commits/holds split
        # positionally and we shift back into the utterance's absolute
        # frame by adding `tail_start_abs` (== ev.start + committed_until).
        # A cheap-JSON or unsplittable utterance is just the n==1 case:
        # finals commit the lone entry, provs hold it as the tail preview.
        commits, holds = _split_entries(ev, entries, hold_last_on_final=can_splice)
        if hybrid and not ev.final:
            commits, holds = [], list(entries)
        _log_promotion(ev, entries, commits, holds, fmt_ts=_audio_wall)
        sub_jobs: list[_Job] = []
        for idx, entry in enumerate(commits):
            sub_ev = SegmentEvent(
                pcm=ev.pcm,  # PCM is shared; downstream doesn't re-use it
                start=tail_start_abs + float(entry["start"]),
                end=tail_start_abs + float(entry["end"]),
                final=True,
            )
            sub_job = _Job(
                event=sub_ev,
                enqueued_at=job.enqueued_at,
                # Shift the parent's anchor by the sub-slice's offset
                # (sample-derived but bounded by MAX_SEGMENT_SECONDS,
                # so sub-ms drift).
                utt_start_mono=job.utt_start_mono + (sub_ev.start - ev.start),
                transcript=str(entry["text"]).strip(),
                asr_done_at=job.asr_done_at,
                meta={
                    **job.meta,
                    "sliced": True,
                    "slice_idx": idx,
                    # Parent utterance's `ev.start` so the translate
                    # worker can discard the matching tail prov slot
                    # `(parent_start, "tail")` after this sub-final
                    # commits — its content was emitted under the
                    # tail before being promoted, so leaving it in
                    # place would show the same line twice (once
                    # bold as commit, once dim as stale prov).
                    "parent_start": ev.start,
                },
            )
            if history_enabled and sub_job.transcript:
                history_buf.append((sub_ev.end, sub_job.transcript))
                _trim_history()
            if not translate_target:
                _emit(sub_job, translation=None)
            else:
                sub_jobs.append(sub_job)
        if sub_jobs:
            # Queue-first: don't flash the transcripts to the renderer until
            # their translations land. The prior committed line stays held on
            # screen during the LLM round-trip, and each new line arrives as
            # a single transcript+translation commit. Queued together, so the
            # translate worker sends them in one call.
            _enqueue_translate_with_backoff(*sub_jobs)
        if commits:
            if ev.final:
                committed_until_by_utt.pop(ev.start, None)
            else:
                if anchor_to_hold_start and holds:
                    # Past-MAX provisional: anchor to the held entry's
                    # *start* so the next ASR call re-feeds that audio
                    # rather than starting from the prior entry's end.
                    # Closes the silence-gap window where the aligner
                    # could re-cut the boundary between cycles.
                    new_committed_until = committed_until + float(holds[0]["start"])
                else:
                    new_committed_until = committed_until + float(commits[-1]["end"])
                committed_until_by_utt[ev.start] = new_committed_until
        elif ev.final:
            # VAD closed the utterance with nothing new to commit.
            committed_until_by_utt.pop(ev.start, None)
        # Force-flush carryover: the held trailing entry was dropped
        # (its right edge is the chop boundary, likely mid-word). Ask
        # LiveVAD to splice the exact audio range [held_start_abs, ev.end)
        # from the just-flushed segment into the now-open next utterance,
        # so the next ASR cycle re-transcribes it with full context.
        #
        # Clamp the splice start to be at-or-past the prior commits'
        # sample-accurate audio end so the spliced range is guaranteed
        # audio-disjoint from already-committed entries' source audio.
        # Without the clamp, an aligner-reported held-entry start that
        # lands marginally earlier than the trim boundary (float / aligner
        # drift) would re-feed audio the prior commit already consumed,
        # producing visible duplication after re-transcription.
        # `prior_audio_end` is computed from the locally-captured
        # `committed_until` rather than a dict because the cursor pops
        # above have already cleared the per-utterance state on this
        # final path — recomputing here is authoritative and pop-order
        # independent. Tiny splice ranges are still forwarded: the
        # spliced PCM is prepended to the next utterance's accumulating
        # audio before ASR runs, so Whisper sees the combined length —
        # not the snippet in isolation.
        if can_splice and holds and vad_ref:
            held_start_abs_samples = int(round(
                (tail_start_abs + float(holds[0]["start"])) * LIVE_SAMPLE_RATE
            ))
            # Floor at the end of THIS cycle's last committed entry,
            # not the cycle's starting cursor — otherwise n>=2 with
            # aligner drift between commits[-1].end and holds[0].start
            # would let the splice overlap audio we just committed.
            # For n==1 `commits` is empty and the floor degenerates
            # to the starting cursor (`tail_start_abs`), which is
            # correct since nothing was committed this cycle.
            if commits:
                floor_seconds = tail_start_abs + float(commits[-1]["end"])
            else:
                floor_seconds = tail_start_abs
            prior_audio_end = int(round(floor_seconds * LIVE_SAMPLE_RATE))
            splice_start_samples = max(held_start_abs_samples, prior_audio_end)
            if splice_start_samples < int(round(ev.end * LIVE_SAMPLE_RATE)):
                vad_ref[0].request_splice(splice_start_samples)
        if holds and not ev.final:
            # Keep ev.start as the tail's key — across successive
            # provisional cycles the tail is the same in-progress
            # utterance, so its key must not shift. Only `end` and
            # `transcript` change as the tail grows.
            # CJK / SE-Asian scripts don't use word-separating spaces.
            # Auto-detect (language=None) falls back to space — safe
            # for Latin scripts, mildly wrong for CJK if the user
            # didn't pass --language.
            tail_joiner = "" if is_spaceless(language) else " "
            tail_text = tail_joiner.join(str(e["text"]).strip() for e in holds).strip()
            tail_ev = SegmentEvent(
                pcm=ev.pcm,
                start=ev.start,
                end=ev.end,
                final=False,
            )
            tail_job = _Job(
                event=tail_ev,
                enqueued_at=job.enqueued_at,
                # Tail starts at ev.start, so the parent's anchor applies.
                utt_start_mono=job.utt_start_mono,
                transcript=tail_text,
                asr_done_at=job.asr_done_at,
                meta={
                    **job.meta, "sliced": True, "tail": True,
                    # Session audio time the preview's text starts at (its
                    # first held entry): lets the translate worker tell a
                    # preview of not-yet-committed speech from a stale one.
                    "content_start": tail_start_abs + float(holds[0]["start"]),
                },
            )
            if translate_target:
                # Queue-first prov: the tail preview waits for its
                # own translation before rendering, so the viewer
                # never sees a translation-blank prov line.
                _enqueue_translate_with_backoff(tail_job)
            else:
                _emit_transcript(tail_job)
        # No new tail this cycle: we *could* discard the stale prior
        # tail here, but doing so blanks the live region until either
        # the next utterance's prov arrives or the queued sub-final
        # commits (with no successor tail in queue, the post-commit
        # gate will fire its own discard). Skipping the eager discard
        # here lets the prior tail keep lingering during that gap —
        # the post-commit path in the translate worker handles the
        # cleanup once the sub-final lands, and the discard below
        # still runs for the ev.final + cursor-cleanup case where
        # no sub-final is queued.
        #
        # No-translate exception: there's no translate-worker post-
        # commit gate, and _emit above ran synchronously, so a stale
        # tail prov would leak past the parent utterance's close.
        # When no new tail was emitted this cycle (or the utterance
        # just closed), discard the slot now. The synchronous _emit
        # means there's no LLM round-trip gap to mask — the #22
        # tradeoff doesn't apply.
        if not translate_target and commits and (not holds or ev.final):
            renderer.discard_provisional((ev.start, "tail"))

    # --- translate worker ---
    # When --history / --history-seconds are passed, the translate worker
    # mirrors those semantics on its own (transcript, translation) buffer:
    # the flags fully override TRANSLATE_HISTORY_LEN. Without flags, the
    # buffer caps at TRANSLATE_HISTORY_LEN with no time window. With ONLY a
    # time window set (no --history), time is the SOLE bound: the default count
    # cap is lifted so e.g. --translate-history-seconds 300 really yields 300s
    # of context instead of silently capping at 10 lines. Both bounds are
    # applied by _trim_translate_history, never per call — see the sawtooth note
    # below.
    # --translate-history-seconds further overrides the translator's time
    # window independently of --history-seconds (None = inherit).
    effective_history_seconds = translate_history_seconds if translate_history_seconds is not None else history_seconds
    if translate_history_multiplier is None:
        translate_history_multiplier = TRANSLATE_HISTORY_LEN_UPPER_BOUND_MULTIPLIER
    # Sawtooth history trim (see TRANSLATE_HISTORY_LEN_UPPER_BOUND_MULTIPLIER in
    # client/llm.py). `*_target` is the floor the buffer trims back DOWN to;
    # `*_ceiling` is what it's allowed to grow to before a trim fires. --history
    # overrides the count target outright — including below
    # TRANSLATE_HISTORY_LEN, so an explicit small window is honoured.
    #
    # The multiplier applies to BOTH axes. The time window gets the same
    # treatment as the count for the same reason: a window that erodes
    # continuously (dropping whatever has aged past `seconds` on every call)
    # rebuilds the history block constantly and diverges the prompt prefix, which
    # is exactly what the sawtooth exists to avoid. Letting the span stretch to
    # multiplier x seconds and then snapping back to seconds converts that
    # continuous erosion into the same periodic step the count trim makes.
    if history > 0:
        translate_history_target: int | None = history
    elif effective_history_seconds > 0:
        translate_history_target = None  # time window is the sole bound
    else:
        translate_history_target = TRANSLATE_HISTORY_LEN
    translate_history_ceiling = (
        None
        if translate_history_target is None
        else max(
            translate_history_target,
            translate_history_target * translate_history_multiplier,
        )
    )
    translate_history_seconds_ceiling = effective_history_seconds * translate_history_multiplier

    def _translate_worker() -> None:
        # (start, end, source, translation); source is the commit LLM's
        # corrected text when it came through translate_entries.
        buf: list[tuple[float, float, str, str]] = []

        def _hist_pairs() -> list[tuple[float, float, str, str]]:
            # NOTE: no windowing happens here — neither by count nor by time.
            # BOTH bounds are enforced by _trim_translate_history on a sawtooth
            # (grow to the ceiling, drop back to the target), and that overgrowth
            # is the whole point: re-windowing per call would rebuild the history
            # block every time and forfeit the prompt-cache reuse the sawtooth
            # exists to buy. What `buf` holds is exactly what the LLM sees, so
            # the prefix only changes when a trim actually fires.
            return list(buf)

        def _trim_translate_history() -> None:
            """Sawtooth trim on both axes: once `buf` exceeds a ceiling, drop
            back to the corresponding target in one step. Between trims the
            buffer grows append-only, so the history block in the LLM prompt
            keeps a stable prefix and the backend reuses its KV cache instead of
            re-prefilling every call.

            Time is measured against the NEWEST entry rather than wall clock, so
            the span only shrinks when a commit actually lands — a silent gap
            never erodes the window (and never invalidates the cache) on its
            own."""
            if translate_history_ceiling is not None and len(buf) > translate_history_ceiling:
                del buf[: len(buf) - translate_history_target]
            if effective_history_seconds > 0 and buf:
                newest = buf[-1][1]
                if newest - buf[0][1] > translate_history_seconds_ceiling:
                    cutoff = newest - effective_history_seconds
                    keep = next((i for i, w in enumerate(buf) if w[1] >= cutoff), len(buf) - 1)
                    del buf[:keep]

        def _take_queued_finals(limit: int) -> list[_Job]:
            """Pop the finals queued directly behind the job in hand (at most
            `limit`), so they translate in the same call: the sibling entries
            of one ASR result, which `_enqueue_translate_with_backoff` queues
            together, plus any finals that piled up behind a slow call. Stops
            at the first provisional or the shutdown sentinel; everything
            after stays queued in order."""
            if limit <= 0:
                return []
            taken: list[_Job] = []
            with translate_lock:
                drained: list[_Job | None] = []
                while True:
                    try:
                        drained.append(translate_q.get_nowait())
                    except queue.Empty:
                        break
                i = 0
                while i < len(drained) and len(taken) < limit:
                    item = drained[i]
                    if item is None or not item.event.final:
                        break
                    taken.append(item)
                    i += 1
                for item in drained[i:]:
                    translate_q.put(item)
            return taken

        def _commit_final(job: _Job, translation: str | None, elapsed: float) -> None:
            job.meta["translate_elapsed"] = elapsed
            # Only *final* segments enter translation history. Provisional
            # outputs are throwaway previews. _trim_translate_history bounds
            # the buffer on a sawtooth so it can't grow unbounded over long
            # sessions while still serving the override path.
            if translation:
                buf.append((job.event.start, job.event.end, job.transcript, translation))
                _trim_translate_history()
            _emit(job, translation=translation)

        def _job_from_entry(jobs: list[_Job], entry: dict) -> tuple[_Job, int]:
            """A committed line for one reply entry of translate_entries, and
            the index of the input line it is anchored on. The model may merge
            or split lines, so the anchor is the input line containing (else
            nearest before) the entry's start: its timing anchor, enqueue time
            and meta carry over; times, text and key (ev.start) come from the
            entry."""
            start, end = entry["start"], entry["end"]
            anchor = 0
            for i, j in enumerate(jobs):
                if j.event.start <= start + 1e-6:
                    anchor = i
            base = jobs[anchor]
            ev = SegmentEvent(pcm=base.event.pcm, start=start, end=end, final=True)
            return _Job(
                event=ev,
                enqueued_at=base.enqueued_at,
                utt_start_mono=base.utt_start_mono + (start - base.event.start),
                transcript=entry["source"],
                asr_done_at=base.asr_done_at,
                meta=dict(base.meta),
            ), anchor

        def _translate_finals(jobs: list[_Job]) -> None:
            """Commit consecutive finals through one streamed translate_entries
            call: each reply entry (ASR-corrected source + translation, with
            its own times) commits as soon as it arrives, and its romaji
            correction queues then, against the corrected source.

            A final is never lost. Every input line no reply entry covers
            (none is anchored on it and none spans its midpoint: skipped by
            the model, or past where the reply stopped) is finished
            afterwards: when the call failed (API error, broken stream,
            timeout) it commits transcript-only, as a failed translation
            always has; when the reply simply left it out, it gets a plain
            per-line `translate` call. Such a line lands after the entries
            that did arrive, so it can sit below a later line."""
            t0 = time.monotonic()
            covered: set[int] = set()  # indices into `jobs`
            last_commit: float | None = None

            def commit(job: _Job, translation: str | None) -> None:
                nonlocal last_commit
                if last_commit is None:
                    # First line of the call: clear the preview on screen. It
                    # is this utterance's stale tail, which the renderer's
                    # commit would leave up (the entries' keys never match
                    # it), and no provisional renders while the reply streams.
                    renderer.clear_provisional()
                else:
                    # Pace lines landing together, always (even with more
                    # finals queued behind this call): only when this line
                    # comes less than LIVE_COMMIT_MIN_INTERVAL_SECONDS after
                    # the previous commit, and only for the remainder, so
                    # the gap is stretched to the interval and never past
                    # it. A line already that far apart is not delayed.
                    gap = time.monotonic() - last_commit
                    if gap < LIVE_COMMIT_MIN_INTERVAL_SECONDS:
                        time.sleep(LIVE_COMMIT_MIN_INTERVAL_SECONDS - max(gap, 0.0))
                # Stamped as the line goes up, not after the render
                # returns, so the render's own time counts toward the gap.
                last_commit = time.monotonic()
                _commit_final(job, translation, last_commit - t0)

            def on_entry(entry: dict) -> None:
                entry_job, anchor = _job_from_entry(jobs, entry)
                # Marked covered before the commit: if the commit itself
                # raises, the line must not be committed a second time.
                covered.add(anchor)
                for i, j in enumerate(jobs):
                    mid = (j.event.start + j.event.end) / 2
                    if entry["start"] <= mid <= entry["end"]:
                        covered.add(i)
                commit(entry_job, entry["translation"] or None)

            failure: Exception | None = None
            try:
                translate_entries(
                    [(j.event.start, j.event.end, j.transcript) for j in jobs],
                    _hist_pairs(),
                    on_entry=on_entry,
                    target=translate_target,
                    extra_context=translate_prompt,
                    system_override=translate_system,
                    temperature=translate_temperature,
                    timeout=float(LIVE_LAG_TOLERANCE_SECONDS),
                    final=True,
                )
            except Exception as exc:
                # Anything, not just the API error classes: a stream can also
                # break with a raw transport error or a mid-stream provider
                # error, and the lines it had not reached must still commit.
                failure = exc
                if not isinstance(exc, (APIConnectionError, APIStatusError, TimeoutError)):
                    log.exception("translate entries failed")
            rest = [j for i, j in enumerate(jobs) if i not in covered]
            if rest and failure is not None:
                # Still commit the transcripts so the viewer doesn't lose
                # committed history just because translation was slow or
                # (remote commit LLM) refused / rate-limited / unreachable.
                log.warning(
                    "translate %s for %d final(s) [%s-%s] - committing transcript only",
                    "timeout" if isinstance(failure, (APITimeoutError, TimeoutError))
                    else f"error ({failure})",
                    len(rest), _audio_wall(rest[0].event.start), _audio_wall(rest[-1].event.end),
                )
                for j in rest:
                    commit(j, None)
            elif rest:
                log.debug("translate entries left %d of %d final(s) uncovered - "
                          "translating them one by one", len(rest), len(jobs))
                for j in rest:
                    try:
                        tr = translate(
                            j.transcript, _hist_pairs(),
                            target=translate_target,
                            extra_context=translate_prompt,
                            system_override=translate_system,
                            temperature=translate_temperature,
                            timeout=float(LIVE_LAG_TOLERANCE_SECONDS),
                            final=True,
                        )
                    except Exception as exc:
                        log.warning("translate error (%s) for final [%s-%s] - committing transcript only",
                                    exc, _audio_wall(j.event.start), _audio_wall(j.event.end))
                        tr = None
                    commit(j, tr or None)
            if len(jobs) > 1:
                log.debug("translate entries: %d final(s) in one call (%.2fs)",
                          len(jobs), time.monotonic() - t0)
            # Sliced sub-finals: the parent's tail prov may still be on screen
            # (rendered in the prior cycle, before these sub-finals committed).
            # Discard it only if no successor tail is already queued for the
            # same parent — if one IS queued, it will overwrite the slot
            # naturally when it renders, and skipping the discard avoids the
            # LLM-round-trip blank gap that an eager clear would create. If no
            # successor is queued (e.g., VAD-final closed the utterance,
            # leaving no new tail to refresh), the discard fires so the stale
            # tail doesn't leak into the next utterance's frame.
            for parent_start in dict.fromkeys(
                j.meta.get("parent_start") for j in jobs if j.meta.get("parent_start") is not None
            ):
                if not _has_pending_tail(parent_start):
                    renderer.discard_provisional((parent_start, "tail"))
            if last_commit is not None:
                # The first commit cleared the preview slot (see commit()).
                # Of the previews that queued while the reply streamed, keep
                # only the newest one that starts at or after this batch's
                # end: the slot then refills after one preview translation,
                # with the speech that follows the committed lines, instead
                # of replaying the older previews one by one.
                _keep_newest_provisional(after=max(j.event.end for j in jobs))

        def _keep_newest_provisional(*, after: float) -> None:
            """Drop every queued provisional but the newest one whose text
            starts at or past `after` (the end of the audio just committed);
            a preview of speech before that point shows lines already
            committed. Finals and the shutdown sentinel keep their order. The
            dropped ones were never rendered (provisionals render from this
            thread), so no slot needs discarding."""
            def fresh(j: _Job) -> bool:
                return j.meta.get("content_start", j.event.start) >= after - 0.05

            with translate_lock:
                drained: list[_Job | None] = []
                while True:
                    try:
                        drained.append(translate_q.get_nowait())
                    except queue.Empty:
                        break
                newest = next(
                    (j for j in reversed(drained) if j is not None and not j.event.final and fresh(j)),
                    None,
                )
                dropped = 0
                for item in drained:
                    if item is not None and not item.event.final and item is not newest:
                        dropped += 1
                        continue
                    translate_q.put(item)
            if dropped:
                log.debug("after commit stream (ends %s): dropped %d provisional(s), kept %s",
                          _audio_wall(after), dropped,
                          "none" if newest is None else f"[{_audio_wall(newest.meta.get('content_start', newest.event.start))}-]")

        def _translate_one(job: _Job) -> None:
            dropped_jobs: list[_Job] = []
            job = _drain_stale(
                translate_q, job, max_age=LIVE_LAG_TOLERANCE_SECONDS, label="translate",
                fmt_ts=_audio_wall, lock=translate_lock, dropped_out=dropped_jobs,
            )
            ev = job.event
            # Discard renderer tail slots for any dropped tail prov whose parent
            # differs from the surviving job's: those slots were rendered in a
            # prior cycle and their refresher was just dropped stale, so without
            # this they'd linger on screen forever (the slot key (start,"tail")
            # never matches a newer utterance). A dropped tail for the SAME
            # parent as the survivor is left alone — the survivor refreshes it.
            for d in dropped_jobs:
                if d.event.final or not d.meta.get("tail"):
                    continue
                if d.event.start == ev.start:
                    continue
                renderer.discard_provisional((d.event.start, "tail"))

            if ev.final:
                _translate_finals([job, *_take_queued_finals(TRANSLATE_BATCH_MAX - 1)])
                return

            t0 = time.monotonic()
            try:
                translation = translate(
                    job.transcript, _hist_pairs(),
                    target=translate_target,
                    extra_context=translate_prompt,
                    system_override=translate_system,
                    temperature=translate_temperature,
                    timeout=float(LIVE_LAG_TOLERANCE_SECONDS),
                    final=False,
                )
            except (APIConnectionError, APIStatusError) as exc:
                # Provisionals: skip — the next one will catch up.
                log.error(
                    "translate %s for [%s-%s] - dropping",
                    "timeout" if isinstance(exc, APITimeoutError) else f"error ({exc})",
                    _audio_wall(ev.start), _audio_wall(ev.end),
                )
                return
            job.meta["translate_elapsed"] = time.monotonic() - t0

            if translation:
                # Queue-first prov. Render transcript and translation in a
                # single atomic step. `_emit_transcript` sets the prov slot
                # (and, with the deferred-held-flush gate in render.py,
                # keeps any prior held visible until the translation lands);
                # `_emit_translation` follows immediately and flushes held.
                # The viewer sees the new prov appear already paired.
                # Skip the prov render entirely on empty translation — a
                # transcript-only prov would re-introduce the blank-line
                # flicker this queue-first design exists to avoid.
                _emit_transcript(job)
                _emit_translation(job, translation=translation)

        while True:
            job = translate_q.get()
            if job is None:
                break
            try:
                _translate_one(job)
            except Exception:
                # Catch-all so one bad job can't kill the worker thread and
                # silently stop translations for the rest of the session.
                log.exception(
                    "translate worker error on [%s-%s] - job dropped",
                    _audio_wall(job.event.start), _audio_wall(job.event.end),
                )

    with LiveRenderer(romanizer=romanizer, theme=theme) as renderer:
        capture_thread = threading.Thread(target=_capture_worker, daemon=True)
        transcribe_thread = threading.Thread(target=_transcribe_worker, daemon=True)
        translate_thread: threading.Thread | None = None
        romaji_thread: threading.Thread | None = None
        capture_thread.start()
        transcribe_thread.start()
        if translate_target:
            translate_thread = threading.Thread(target=_translate_worker, daemon=True)
            translate_thread.start()
        if romaji_corrector is not None:
            romaji_thread = threading.Thread(target=_romaji_worker, daemon=True)
            romaji_thread.start()

        log.info("live capture started - commit-on-silence (Ctrl+C to stop)")

        try:
            # Main thread idles until interrupted; all real work is in the workers.
            stop_event.wait()
        finally:
            stop_event.set()
            asr_q.put(None)
            capture_thread.join(timeout=2)
            transcribe_thread.join(timeout=10)
            if translate_thread is not None:
                translate_thread.join(timeout=10)
            if romaji_thread is not None:
                # Sentinel goes in only after the producers (ASR / translate
                # workers) have drained, so no fix job can slip in behind it.
                romaji_q.put(None)
                # Short join: the worker may be mid-LLM-call; it's a daemon and
                # any late settle no-ops against the already-flushed renderer.
                romaji_thread.join(timeout=5)
