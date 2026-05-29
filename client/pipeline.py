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

# ASR-prompt history buffer trim policy. Entries accumulate for the whole
# session; once the buffer's span exceeds HISTORY_TRIM_AFTER_SECONDS, drop
# everything older than HISTORY_KEEP_SECONDS measured from the newest entry.
# This caps memory in long sessions while keeping enough context for any
# reasonable --history-seconds window.
HISTORY_TRIM_AFTER_SECONDS = 7200.0
HISTORY_KEEP_SECONDS = 3600.0
from history import compose_prompt, select_history
from live_vad import LiveVAD, SegmentEvent
from llm import TRANSLATE_HISTORY_LEN, translate, translate_pair
from utils.language import is_spaceless
from render import LiveRenderer
from transcribe import live_transcribe

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


def _drain_stale(q: "queue.Queue[_Job | None]", current: _Job, *, max_age: float, label: str, fmt_ts) -> _Job:
    """If `current` is a stale provisional, drain forward to the freshest
    item. Returns the (possibly newer) job to process.

    Finals are never dropped — they're immutable history and a missed final
    is a permanent gap in the user's transcript. Better late than lost."""
    if current.event.final:
        return current
    dropped = 0
    while time.monotonic() - current.enqueued_at > max_age:
        try:
            newer = q.get_nowait()
        except queue.Empty:
            break
        if newer is None:
            q.put(None)
            break
        dropped += 1
        current = newer
        if current.event.final:
            break
    if dropped:
        log.warning(
            "%s stale > %.1fs - dropped %d job(s), jumped to [%s-%s]",
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
    translate_temperature: float = 0,
) -> None:
    mic = get_loopback_mic()

    # Disable the SDK's built-in retries: by the time a retry would land, the
    # audio is stale and the staleness drop in _drain_stale is already moving
    # us to the next segment. Retries just waste server cycles.
    asr_client = asr_client.with_options(max_retries=0)

    asr_q: "queue.Queue[_Job | None]" = queue.Queue()
    translate_q: "queue.Queue[_Job | None]" = queue.Queue()
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
        """Final commit: transcript + optional translation as one atomic line."""
        ev = job.event
        lag = time.monotonic() - job.enqueued_at
        # No anchor re-snapshot here: the pair is re-anchored at every VAD
        # speech-start (capture worker), so NTP slew / suspend skew is already
        # absorbed once per utterance. `job.utt_start_mono` was latched at this
        # utterance's start against that fresh anchor, so `_mono_wall` formats it
        # against the same (wall, mono) frame it was latched in — drift-free and
        # stable across all re-emits / sub-finals of the utterance.
        renderer.commit(
            job.transcript, translation, key=_utt_key(ev, job.meta), lag=lag,
            entries=job.meta.get("entries"),
            tag=_slice_tag(job),
            ts=_mono_wall(job.utt_start_mono),
            duration=ev.end - ev.start,
            gain_db=job.meta.get("gain_db"),
        )
        _log_emit(job, lag, kind="final")

    def _emit_transcript(job: _Job, *, keep_held: bool = False) -> None:
        """Provisional transcript only — translation line stays as-is until the LLM lands.

        `keep_held` pins the committed line in the held slot (array-form
        translation pairs a tail prov with the revisable committed line above)."""
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
            keep_held=keep_held,
        )
        _log_emit(job, lag, kind="prov ")

    def _emit_translation(job: _Job, *, translation: str | None, keep_held: bool = False) -> None:
        """Provisional translation update — leaves transcript untouched.

        `keep_held` pins the committed line in the held slot (array-form
        translation; see _emit_transcript)."""
        ev = job.event
        lag = time.monotonic() - job.enqueued_at
        if translation:
            renderer.provisional_translation(translation, key=_utt_key(ev, job.meta), lag=lag, keep_held=keep_held)
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
                "capture backoff: collapsed %d stale provisional(s) for [%s-]",
                dropped, _audio_wall(ev.start),
            )
        asr_q.put(new_job)

    def _has_pending_tail(parent_start: float) -> bool:
        """True iff translate_q currently holds a tail prov for `parent_start`.

        Drains + restores the queue under the assumption that the translate
        worker is the only consumer (single-thread) — so peeking via drain
        won't race a concurrent get. Used by the sub-final commit path to
        skip the post-commit tail discard when a successor tail is already
        queued: it will overwrite the slot when it renders, so an eager
        discard would only create a blank gap during the successor's LLM
        round-trip."""
        drained: list[_Job | None] = []
        found = False
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

    def _enqueue_translate_with_backoff(new_job: _Job) -> None:
        """Mirror of `_enqueue_with_backoff` for translate_q. Provs are now
        queue-first too, so without this guard translate_q would accrue
        nested provs for the same open utterance whenever the LLM trails
        ASR cadence (~1Hz). Collapse stale provs for the same `ev.start`
        before pushing `new_job`. Finals are always preserved EXCEPT for
        sliced sub-finals, which also evict pending tail provs for their
        parent utterance — the tail's audio range overlaps the just-
        promoted entry, so leaving it queued would render the same line
        twice (committed bold above, stale dim below)."""
        ev = new_job.event
        now = time.monotonic()
        parent_start = new_job.meta.get("parent_start") if new_job.meta else None
        # Plain (non-sliced) finals short-circuit: nothing to evict.
        if ev.final and parent_start is None:
            translate_q.put(new_job)
            return
        drained: list[_Job | None] = []
        while True:
            try:
                drained.append(translate_q.get_nowait())
            except queue.Empty:
                break
        dropped = 0
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
        if dropped:
            log.debug(
                "translate backoff: collapsed %d stale provisional(s) for [%s-]",
                dropped, _audio_wall(ev.start),
            )
        translate_q.put(new_job)

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
        finally:
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
        seg_prompt = compose_prompt(prompt, "\n".join(history_texts) if history_texts else None)

        t0 = time.monotonic()
        # Ask for multiple entries once the open utterance is long enough
        # that waiting for VAD to close it would leave finished clauses
        # stuck in the mutable provisional line instead of committed to
        # scrollback (half the force-flush cap), or whenever we're past the
        # first cursor advance (continuation cycles must keep draining the
        # residue). Pipeline timing intent only — the backend cost of
        # honouring it (aligner pass or not) is transcribe.py's concern.
        want_segments = duration >= LIVE_ENTRIES_MIN_DURATION or committed_until > 0.0
        tail_start_abs = ev.start + committed_until
        # Independent peak-normalise of the segment before ASR (chunks
        # were normalised individually for VAD; this scales the assembled
        # segment to its own peak so ASR sees a consistent level).
        pcm_to_send, gain_db = peak_normalize(pcm_to_send)
        job.meta["gain_db"] = gain_db
        try:
            text, entries = live_transcribe(
                asr_client, model,
                encode_wav(pcm_to_send),
                f"{_fmt_ts(tail_start_abs)}-{_fmt_ts(ev.end)}.wav",
                language=language,
                prompt=seg_prompt,
                timeout=LIVE_LAG_TOLERANCE_SECONDS,
                # Fallback span for the synthetic whole-segment entry: the
                # trimmed tail audio actually sent, 0-based to match the
                # aligner's own entry frame.
                segment_duration=len(pcm_to_send) / LIVE_SAMPLE_RATE,
                want_segments=want_segments,
            )
        except APITimeoutError:
            log.error(
                "ASR timeout for [%s-%s] after %.2fs - dropping",
                _audio_wall(tail_start_abs), _audio_wall(ev.end), LIVE_LAG_TOLERANCE_SECONDS,
            )
            return
        except APIConnectionError:
            log.error("could not connect to transcription backend at %s", asr_base_url)
            return
        except APIStatusError as exc:
            log.error("server error %s: %s", exc.status_code, exc.message)
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
        # The remaining un-spliceable case is a force-flush with a SINGLE
        # whole-segment entry at the start of an utterance (n==1,
        # committed_until==0): a monologue the aligner couldn't split at
        # all (or a cheap-JSON synthetic entry). There's no aligner-reported
        # internal boundary to anchor on, so commit at the chop boundary
        # and warn — splicing would loop forever dropping every cycle.
        can_splice = is_force_flush and (
            len(entries) >= 2 or (len(entries) == 1 and committed_until > 0.0)
        )
        # Force-flush without splice eligibility: trailing audio commits
        # at the chop boundary — likely mid-word for long unbroken speech.
        if is_force_flush and not can_splice:
            log.warning(
                "force-flush with %d entries: committing potential mid-word chop, "
                "residue=[%s-%s] full-utt=[%s-%s] dur=%.2fs",
                len(entries),
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
        _log_promotion(ev, entries, commits, holds, fmt_ts=_audio_wall)
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
                # Queue-first: don't flash the transcript to the
                # renderer until its translation lands. The prior
                # committed line stays held on screen during the LLM
                # round-trip, and the new line arrives as a single
                # transcript+translation commit.
                _enqueue_translate_with_backoff(sub_job)
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
                meta={**job.meta, "sliced": True, "tail": True},
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
    # buffer caps at TRANSLATE_HISTORY_LEN with no time window.
    # --translate-history-seconds further overrides the translator's time
    # window independently of --history-seconds (None = inherit).
    effective_history_seconds = translate_history_seconds if translate_history_seconds is not None else history_seconds
    translate_history_override = history > 0 or effective_history_seconds > 0
    def _translate_worker() -> None:
        buf: list[tuple[float, str, str, object]] = []  # (ev.end, transcript, translation, utt_key)
        # The most recently committed line, still sitting in the renderer's
        # held slot. Its translation stays REVISABLE: we re-translate this
        # committed line together with the following utterance in one array call
        # so the model can refine it in light of the continuation, landing the
        # refined text via revise_held_translation. This fires for the next
        # in-progress tail prov (same growing utterance) AND for the next
        # separate utterance's final (cross-VAD — refine A just before B's
        # commit flushes A to scrollback, so the refined text is what scrolls).
        # (transcript, held_key, parent_start). Cleared when the held slot has
        # flushed (the revise no-ops) so we never feed an off-screen line into
        # translate_pair.
        last_committed: tuple[str, object, float] | None = None

        def _hist_pairs(at_start: float, *, exclude_last: int = 0) -> list[tuple[str, str]]:
            # `exclude_last` drops that many trailing committed entries before
            # building the window. The array path re-translates the held line
            # (which is buf[-1]) inside its line-list, so it must NOT also appear
            # in the immutable "do not re-translate" history block — feeding both
            # makes the model defer to the stale committed copy instead of
            # refining it. Excluding it leaves only strictly-prior context.
            source = buf[: len(buf) - exclude_last] if exclude_last else buf
            if translate_history_override:
                window = source
                if effective_history_seconds > 0:
                    cutoff = at_start - effective_history_seconds
                    window = [w for w in window if w[0] >= cutoff]
                if history > 0:
                    window = window[-history:]
                return [(raw, tr) for _, raw, tr, _ in window]
            return [(raw, tr) for _, raw, tr, _ in source[-TRANSLATE_HISTORY_LEN:]]

        while True:
            job = translate_q.get()
            if job is None:
                break

            job = _drain_stale(translate_q, job, max_age=LIVE_LAG_TOLERANCE_SECONDS, label="translate", fmt_ts=_audio_wall)
            ev = job.event

            # Array path: an in-progress prov (every prov is tail-keyed — the
            # holds branch in _transcribe_worker is the only prov source) while
            # a committed line is still held in the renderer. Re-translate
            # [committed, prov] together so the model can refine the committed
            # line's translation in light of the continuation, keeping it
            # revisable on screen until it flushes to scrollback. On any non-2
            # result (paired is None) we fall through to the per-line path,
            # leaving the held line's translation untouched.
            is_tail = not ev.final and bool(job.meta.get("tail"))
            pair_with = last_committed if is_tail else None

            t0 = time.monotonic()
            if pair_with is not None:
                # The held line we're re-translating is buf[-1] (it was the most
                # recent committed entry). Drop it from the history block so the
                # model refines it rather than echoing its stale committed copy.
                exclude_held = 1 if buf and buf[-1][3] == pair_with[1] else 0
                try:
                    paired = translate_pair(
                        [pair_with[0], job.transcript], _hist_pairs(ev.start, exclude_last=exclude_held),
                        target=translate_target,
                        extra_context=translate_prompt,
                        system_override=translate_system,
                        temperature=translate_temperature,
                        timeout=float(LIVE_LAG_TOLERANCE_SECONDS),
                    )
                except APITimeoutError:
                    # Tail prov: skip — the next cycle catches up. The held
                    # line keeps its prior translation.
                    log.error(
                        "translate_pair timeout for [%s-%s] after %.2fs - dropping",
                        _audio_wall(ev.start), _audio_wall(ev.end), LIVE_LAG_TOLERANCE_SECONDS,
                    )
                    continue
                if paired is not None and len(paired) == 2:
                    # Two translations: refine the held (committed) line, then
                    # render the tail prov below it. keep_held pins the committed
                    # line on screen as the revisable upper line instead of
                    # flushing it to scrollback when the tail gains its
                    # translation. (A length-1 result here means the model merged
                    # the committed line + tail into one rendering — for a tail
                    # PROV that's not actionable as a positional pair, so we fall
                    # through to per-line and let the prov render on its own; the
                    # squash logic only applies to committed finals below.)
                    applied = renderer.revise_held_translation(paired[0], key=pair_with[1])
                    if not applied:
                        # The held line already scrolled to scrollback (it can't
                        # be revised once flushed) — our pairing target is stale.
                        # Drop it so the next prov stops feeding an off-screen
                        # line into translate_pair. We still have THIS prov's own
                        # translation from the array call (paired[1]), so render
                        # it directly rather than re-translating: keep_held is
                        # omitted (no live held line left to pin above it).
                        last_committed = None
                        job.meta["translate_elapsed"] = time.monotonic() - t0
                        if paired[1]:
                            _emit_transcript(job)
                            _emit_translation(job, translation=paired[1])
                        continue
                    # Keep the translation-history buffer consistent with what
                    # the viewer now sees: the last committed entry (matched by
                    # utterance KEY, not text — two identical short lines must
                    # not cross-rewrite) carries the refined translation so
                    # future context isn't built on the superseded text.
                    if paired[0] and buf and buf[-1][3] == pair_with[1]:
                        end, raw, _, k = buf[-1]
                        buf[-1] = (end, raw, paired[0], k)
                    job.meta["translate_elapsed"] = time.monotonic() - t0
                    _emit_transcript(job, keep_held=True)
                    # Guard empty translation: a transcript-only prov would
                    # reintroduce the blank-line flicker the queue-first design
                    # exists to avoid (the per-line path guards this the same
                    # way with `elif translation`).
                    if paired[1]:
                        _emit_translation(job, translation=paired[1], keep_held=True)
                    continue

            # Cross-VAD pair path: a NEW utterance B finalizing while the prior
            # committed line A is still held. Re-translate [A, B] together; the
            # COUNT the model returns is the boundary signal:
            #   - 1 translation  -> the model rendered A+B as ONE utterance:
            #       SQUASH. Replace A's held line in place with merged "A B" text
            #       carrying that one translation; B gets no separate line. A
            #       keeps its key so the NEXT utterance can squash again.
            #   - 2 translations -> distinct utterances: refine A (paired[0])
            #       BEFORE committing B (paired[1]) so A scrolls refined.
            #   - None (refusal/error/timeout/empty) -> per-line commit for B.
            # The parent-start guard excludes sub-finals of the SAME long
            # utterance (the tail path's job); only a genuinely new utterance
            # (different parent) pairs here.
            b_parent = job.meta.get("parent_start", ev.start)
            same_utterance = last_committed is not None and b_parent == last_committed[2]
            if ev.final and last_committed is not None and not same_utterance:
                a_text, a_key, a_parent = last_committed
                exclude_held = 1 if (buf and buf[-1][3] == a_key) else 0
                try:
                    paired = translate_pair(
                        [a_text, job.transcript], _hist_pairs(ev.start, exclude_last=exclude_held),
                        target=translate_target,
                        extra_context=translate_prompt,
                        system_override=translate_system,
                        temperature=translate_temperature,
                        timeout=float(LIVE_LAG_TOLERANCE_SECONDS),
                    )
                except APITimeoutError:
                    # Treat as a non-pair result: fall through to the per-line
                    # path, which still commits B (and its own final-timeout
                    # branch handles a second timeout). A keeps its translation.
                    paired = None

                if paired is not None and len(paired) == 1 and paired[0]:
                    # SQUASH: merge B into A's held line in place (no flush), so
                    # A keeps its slot and start-ts and grows into the combined
                    # utterance. Skip squash on empty paired[0] (would blank A).
                    joiner = "" if is_spaceless(language) else " "
                    merged_text = f"{a_text}{joiner}{job.transcript}".strip()
                    merged_trans = paired[0]
                    job.meta["translate_elapsed"] = time.monotonic() - t0
                    a_start = a_key if isinstance(a_key, (int, float)) else ev.start
                    if renderer.replace_held(
                        merged_text, merged_trans, key=a_key,
                        duration=ev.end - a_start,
                        gain_db=job.meta.get("gain_db"),
                    ):
                        # Replace A's buf entry with the merged line (key-matched);
                        # B is NOT a separate buf entry. Keep A's key + parent so
                        # the next utterance squashes/pairs against the merged line.
                        if buf and buf[-1][3] == a_key:
                            buf[-1] = (ev.end, merged_text, merged_trans, a_key)
                        last_committed = (merged_text, a_key, a_parent)
                        # B never gets its own held line; retire its tail prov.
                        parent_start = job.meta.get("parent_start")
                        if parent_start is not None and not _has_pending_tail(parent_start):
                            renderer.discard_provisional((parent_start, "tail"))
                        continue
                    # replace_held no-op (A already gone): fall through to
                    # commit B standalone per-line below.

                elif paired is not None and len(paired) == 2:
                    # SEPARATE: refine A in place (no-op if A already scrolled
                    # off; guard empty paired[0] so we never blank A) BEFORE B's
                    # commit flushes A, so A scrolls refined.
                    if paired[0] and renderer.revise_held_translation(paired[0], key=a_key) \
                            and buf and buf[-1][3] == a_key:
                        end, raw, _, k = buf[-1]
                        buf[-1] = (end, raw, paired[0], k)
                    if paired[1]:
                        # Commit B as its own line.
                        b_key = _utt_key(ev, job.meta)
                        b_trans = paired[1]
                        buf.append((ev.end, job.transcript, b_trans, b_key))
                        cap = max(TRANSLATE_HISTORY_LEN, history) if translate_history_override else TRANSLATE_HISTORY_LEN
                        if len(buf) > cap:
                            del buf[: len(buf) - cap]
                        job.meta["translate_elapsed"] = time.monotonic() - t0
                        # Commit B: flushes A (now carrying its refinement) to
                        # scrollback and takes the held slot.
                        _emit(job, translation=b_trans)
                        last_committed = (job.transcript, b_key, b_parent)
                        parent_start = job.meta.get("parent_start")
                        if parent_start is not None and not _has_pending_tail(parent_start):
                            renderer.discard_provisional((parent_start, "tail"))
                        continue
                    # paired[1] empty: fall through to per-line for B (A refined).

                # paired is None, or a guarded empty result: fall through to
                # per-line translate() + _emit for B. Any A refinement already
                # landed above.

            try:
                translation = translate(
                    job.transcript, _hist_pairs(ev.start),
                    target=translate_target,
                    extra_context=translate_prompt,
                    system_override=translate_system,
                    temperature=translate_temperature,
                    timeout=float(LIVE_LAG_TOLERANCE_SECONDS),
                )
            except APITimeoutError:
                # Provisionals: skip — the next one will catch up.
                # Finals: still commit the transcript so the viewer doesn't
                # lose committed history just because translation was slow.
                if ev.final:
                    log.warning(
                        "translate timeout for final [%s-%s] after %.2fs - committing transcript only",
                        _audio_wall(ev.start), _audio_wall(ev.end), LIVE_LAG_TOLERANCE_SECONDS,
                    )
                    _emit(job, translation=None)
                    last_committed = (job.transcript, _utt_key(ev, job.meta), job.meta.get("parent_start", ev.start))
                else:
                    log.error(
                        "translate timeout for [%s-%s] after %.2fs - dropping",
                        _audio_wall(ev.start), _audio_wall(ev.end), LIVE_LAG_TOLERANCE_SECONDS,
                    )
                continue
            t_translate = time.monotonic() - t0
            job.meta["translate_elapsed"] = t_translate

            # Only commit *final* segments to translation history. Provisional
            # outputs are throwaway previews. Hard cap the buffer at the
            # larger of the two windows so it can't grow unbounded over long
            # sessions while still serving the override path.
            if ev.final and translation:
                buf.append((ev.end, job.transcript, translation, _utt_key(ev, job.meta)))
                cap = max(TRANSLATE_HISTORY_LEN, history) if translate_history_override else TRANSLATE_HISTORY_LEN
                if len(buf) > cap:
                    del buf[: len(buf) - cap]

            if ev.final:
                _emit(job, translation=translation)
                # This commit now owns the held slot; pair the NEXT tail OR the
                # next separate utterance's final against it so its translation
                # stays revisable.
                last_committed = (job.transcript, _utt_key(ev, job.meta), job.meta.get("parent_start", ev.start))
                # Sliced sub-final: the parent's tail prov may still be on
                # screen (rendered in the prior cycle, before this sub-final
                # committed). Discard it only if no successor tail is already
                # queued for the same parent — if one IS queued, it will
                # overwrite the slot naturally when it renders, and skipping
                # the discard avoids the LLM-round-trip blank gap that an
                # eager clear would create. The brief on-screen "duplicate"
                # (sub-final bold above + stale tail dim below) resolves
                # itself within the queued tail's LLM time. If no successor
                # is queued (e.g., VAD-final closed the utterance, leaving
                # no new tail to refresh), the discard fires so the stale
                # tail doesn't leak into the next utterance's frame.
                parent_start = job.meta.get("parent_start")
                if parent_start is not None and not _has_pending_tail(parent_start):
                    renderer.discard_provisional((parent_start, "tail"))
            elif translation:
                # Queue-first prov (per-line path: no committed line was held
                # to pair against, or the array call fell back to per-line).
                # Render transcript and translation in a
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

    with LiveRenderer() as renderer:
        capture_thread = threading.Thread(target=_capture_worker, daemon=True)
        transcribe_thread = threading.Thread(target=_transcribe_worker, daemon=True)
        translate_thread: threading.Thread | None = None
        capture_thread.start()
        transcribe_thread.start()
        if translate_target:
            translate_thread = threading.Thread(target=_translate_worker, daemon=True)
            translate_thread.start()

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
