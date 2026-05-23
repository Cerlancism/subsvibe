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

import numpy as np
from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError

from capture import (
    LIVE_LAG_TOLERANCE_SECONDS,
    LIVE_MAX_SEGMENT_SECONDS,
    LIVE_MIN_SILENCE_MS,
    LIVE_PROVISIONAL_BACKOFF_SECONDS,
    LIVE_SAMPLE_RATE,
    LIVE_VAD_CHUNK_FRAMES,
    encode_wav,
    get_loopback_mic,
)

# Only request word/segment granularity once the open utterance has crossed
# this threshold. Below it, VAD is expected to close the segment cleanly on
# its own, so the cheaper plain-JSON path is enough. Above it, we're at risk
# of a force-flush at MAX_SEGMENT_SECONDS — start asking the server for
# entries so we can promote completed pieces early.
LIVE_ENTRIES_MIN_DURATION = LIVE_MAX_SEGMENT_SECONDS / 2
from live_vad import LiveVAD, SegmentEvent
from llm import TRANSLATE_HISTORY_LEN, translate
from render import LiveRenderer
from transcribe import live_transcribe

log = logging.getLogger("subsvibe.pipeline")


@dataclass
class _Job:
    """An ASR (and optionally translate) job for one segment event."""
    event: SegmentEvent
    enqueued_at: float           # monotonic time
    transcript: str = ""         # filled in by ASR worker
    asr_done_at: float = 0.0     # monotonic time
    meta: dict = field(default_factory=dict)


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


# Float-rounding tolerance for segment-relative entry boundary comparisons.
# Entries from the server are already rounded to 3 decimals (1ms); this
# absorbs add/subtract noise without letting overlapping entries through.
_TIME_EPS = 0.001


def _split_entries(
    ev: SegmentEvent,
    entries: list[dict],
    *,
    silence_tail_s: float,
    committed_until: float,
) -> tuple[list[dict], list[dict]]:
    """Decide which entries to promote to final and which to hold as the
    new provisional tail.

    `committed_until` is the segment-relative end of the last entry already
    committed for this VAD utterance. Entries that start before that cut
    are dropped (they overlap the committed prefix) — never re-emitted.

    Final VAD events: commit everything past the cut (the utterance is closed).
    Provisionals: commit only entries whose end is at least silence_tail_s
    before the segment's audio end AND whose start is past committed_until."""
    if not entries:
        return [], []

    # Drop entries that overlap the already-committed prefix.
    fresh = [e for e in entries if float(e["start"]) >= committed_until - _TIME_EPS]
    if not fresh:
        return [], []

    if ev.final:
        return fresh, []

    audio_end_rel = ev.end - ev.start
    cutoff = audio_end_rel - silence_tail_s
    commits: list[dict] = []
    holds: list[dict] = []
    for e in fresh:
        # Strict: both endpoints must be in the safe zone. e.end <= cutoff is
        # the silence-tail rule; e.start >= committed_until is the no-overlap
        # rule (already enforced by `fresh` but kept here for clarity).
        e_start = float(e["start"])
        e_end = float(e["end"])
        if e_end <= cutoff + _TIME_EPS and e_start >= committed_until - _TIME_EPS:
            commits.append(e)
        else:
            holds.append(e)
    # Only commit a contiguous leading prefix — a hold-then-commit pattern
    # would mean we're skipping an entry, which is never what we want.
    while commits and holds and float(commits[-1]["start"]) > float(holds[0]["start"]):
        holds.insert(0, commits.pop())
    return commits, holds


def _log_promotion(
    ev: SegmentEvent,
    entries: list[dict],
    commits: list[dict],
    holds: list[dict],
) -> None:
    log.debug(
        "promote [%s-%s] kind=%s entries=%d commit=%d hold=%d | %s",
        _fmt_ts(ev.start), _fmt_ts(ev.end),
        "final" if ev.final else "prov ",
        len(entries), len(commits), len(holds),
        " || ".join(
            f"[{e['start']:.2f}-{e['end']:.2f}] {e['text'][:30]!r}"
            for e in entries
        ),
    )


def _drain_stale(q: "queue.Queue[_Job | None]", current: _Job, *, max_age: float, label: str) -> _Job:
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
            _fmt_ts(current.event.start), _fmt_ts(current.event.end),
        )
    return current


def live_capture(
    *,
    asr_client: OpenAI,
    asr_base_url: str,
    model: str,
    language: str | None,
    prompt: str | None,
    translate_target: str | None,
    translate_prompt: str | None = None,
) -> None:
    mic = get_loopback_mic()

    # Disable the SDK's built-in retries: by the time a retry would land, the
    # audio is stale and the staleness drop in _drain_stale is already moving
    # us to the next segment. Retries just waste server cycles.
    asr_client = asr_client.with_options(max_retries=0)

    asr_q: "queue.Queue[_Job | None]" = queue.Queue()
    translate_q: "queue.Queue[_Job | None]" = queue.Queue()
    stop_event = threading.Event()
    capture_start = time.monotonic()
    # `start`/`end` on SegmentEvent are PCM seconds. Wall-clock lag of an output
    # against real-time audio is: monotonic() - (capture_start + event.end).

    # --- render emitters (defined before workers so closures resolve cleanly) ---
    # Stable identifier for an utterance across its provisional updates and
    # final commit. SegmentEvent.start is monotonic per utterance and reset
    # by the VAD between utterances, so it makes a natural key.
    def _utt_key(ev: SegmentEvent) -> float:
        return ev.start

    def _slice_tag(job: _Job) -> str | None:
        """'tail' if this is the held tail provisional, 'sliced' if it's a
        sub-job from the slicing path, None for the cheap whole-utterance path."""
        if job.meta.get("tail"):
            return "tail"
        if job.meta.get("sliced"):
            return "sliced"
        return None

    def _emit(job: _Job, *, translation: str | None) -> None:
        """Final commit: transcript + optional translation as one atomic line."""
        ev = job.event
        lag = time.monotonic() - (capture_start + ev.end)
        renderer.commit(
            job.transcript, translation, key=_utt_key(ev), lag=lag,
            entries=job.meta.get("entries"),
            tag=_slice_tag(job),
        )
        _log_emit(job, lag, kind="final")

    def _emit_transcript(job: _Job) -> None:
        """Provisional transcript only — translation line stays as-is until the LLM lands."""
        ev = job.event
        lag = time.monotonic() - (capture_start + ev.end)
        renderer.provisional_transcript(
            job.transcript, key=_utt_key(ev), lag=lag,
            entries=job.meta.get("entries"),
            tag=_slice_tag(job),
        )
        _log_emit(job, lag, kind="prov ")

    def _emit_translation(job: _Job, *, translation: str | None) -> None:
        """Provisional translation update — leaves transcript untouched."""
        ev = job.event
        lag = time.monotonic() - (capture_start + ev.end)
        if translation:
            renderer.provisional_translation(translation, key=_utt_key(ev), lag=lag)
        _log_emit(job, lag, kind="prov ")

    def _emit_pending_final(job: _Job) -> None:
        """Park a final's transcript on screen while its translation is
        being computed. The renderer keeps it visible until commit, and the
        next utterance's provisional renders alongside rather than over it."""
        ev = job.event
        lag = time.monotonic() - (capture_start + ev.end)
        renderer.pending_final(
            job.transcript, key=_utt_key(ev), lag=lag,
            entries=job.meta.get("entries"),
            tag=_slice_tag(job),
        )
        _log_emit(job, lag, kind="pend ")

    def _log_emit(job: _Job, lag: float, *, kind: str) -> None:
        ev = job.event
        log.debug(
            "%s [%s-%s] dur=%.2fs asr=%.2fs%s lag=%.2fs",
            kind, _fmt_ts(ev.start), _fmt_ts(ev.end),
            job.meta.get("duration", 0.0),
            job.meta.get("asr_elapsed", 0.0),
            f" tr={job.meta['translate_elapsed']:.2f}s" if "translate_elapsed" in job.meta else "",
            lag,
        )

    # --- capture + VAD thread ---
    def _enqueue_with_backoff(new_job: _Job) -> None:
        """Push `new_job` onto asr_q, first collapsing any stale provisional
        for the same open utterance.

        A provisional SegmentEvent always covers [open_start .. now], so a
        newer provisional for the same `start` strictly contains any older
        one. When the ASR backend can't keep up, the queue accrues these
        nested provisionals; dropping the stale predecessors here keeps work
        from piling up while preserving the same audio range for the next
        cycle. Finals (and provisionals younger than the backoff, or for a
        different open segment) are always preserved."""
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
                dropped, _fmt_ts(ev.start),
            )
        asr_q.put(new_job)

    def _capture_worker() -> None:
        vad = LiveVAD()
        with mic.recorder(samplerate=LIVE_SAMPLE_RATE, channels=1) as recorder:
            while not stop_event.is_set():
                chunk = recorder.record(numframes=LIVE_VAD_CHUNK_FRAMES).reshape(-1).astype(np.float32)
                for ev in vad.feed(chunk):
                    _enqueue_with_backoff(_Job(event=ev, enqueued_at=time.monotonic()))

    # Per-VAD-utterance commit cursor: segment-relative end of the last entry
    # committed from that utterance. Keyed by ev.start (the utterance key).
    # Lets subsequent provisional re-transcriptions and the eventual VAD final
    # skip entries that overlap the already-committed prefix. Entries here are
    # cleared when the VAD final for that utterance is fully processed.
    committed_until_by_utt: dict[float, float] = {}

    # --- ASR worker ---
    def _transcribe_worker() -> None:
        while True:
            job = asr_q.get()
            if job is None:
                if translate_target:
                    translate_q.put(None)
                break

            job = _drain_stale(asr_q, job, max_age=LIVE_LAG_TOLERANCE_SECONDS, label="asr")
            ev = job.event
            duration = ev.end - ev.start

            t0 = time.monotonic()
            with_entries = duration >= LIVE_ENTRIES_MIN_DURATION
            try:
                text, entries = live_transcribe(
                    asr_client, model,
                    encode_wav(ev.pcm),
                    f"{_fmt_ts(ev.start)}-{_fmt_ts(ev.end)}.wav",
                    language=language,
                    prompt=prompt,
                    timeout=LIVE_LAG_TOLERANCE_SECONDS,
                    with_entries=with_entries,
                )
            except APITimeoutError:
                log.error(
                    "ASR timeout for [%s-%s] after %.2fs - dropping",
                    _fmt_ts(ev.start), _fmt_ts(ev.end), LIVE_LAG_TOLERANCE_SECONDS,
                )
                continue
            except APIConnectionError:
                log.error("could not connect to transcription backend at %s", asr_base_url)
                continue
            except APIStatusError as exc:
                log.error("server error %s: %s", exc.status_code, exc.message)
                continue
            elapsed = time.monotonic() - t0

            if not text:
                continue

            job.asr_done_at = time.monotonic()
            # `entries` is the ASR's segment count for THIS provisional cycle —
            # surfaced in the live header (`n=3`) so the viewer can see when
            # the slicing path is actually engaging. Omitted on the cheap path
            # where the server returned no granularity.
            job.meta = {
                "asr_elapsed": elapsed,
                "duration": duration,
                **({"entries": len(entries)} if entries else {}),
            }

            # Entry-driven promotion path: when we asked the server for word/
            # segment timestamps and got >1 entry back, treat the leading
            # entries that are followed by silence_tail_s of quiet as final
            # commits, and the trailing entry (if any) as the new provisional.
            # Each committed entry becomes its own _Job with absolute timing
            # and a distinct key so the renderer/translator treat them as
            # independent utterances.
            committed_until = committed_until_by_utt.get(ev.start, 0.0)
            commits, holds = _split_entries(
                ev, entries,
                silence_tail_s=LIVE_MIN_SILENCE_MS / 1000.0,
                committed_until=committed_until,
            )

            # Slicing only takes over when we'd actually move the cursor or
            # we've already moved it for this utterance. Single-entry results
            # with nothing committed yet fall through to the cheap whole-
            # utterance path.
            sliced = (commits or holds) and (
                len(entries) > 1 or committed_until > 0.0
            )
            if sliced:
                _log_promotion(ev, entries, commits, holds)
                for idx, entry in enumerate(commits):
                    sub_ev = SegmentEvent(
                        pcm=ev.pcm,  # PCM is shared; downstream doesn't re-use it
                        start=ev.start + float(entry["start"]),
                        end=ev.start + float(entry["end"]),
                        final=True,
                    )
                    sub_job = _Job(
                        event=sub_ev,
                        enqueued_at=job.enqueued_at,
                        transcript=str(entry["text"]).strip(),
                        asr_done_at=job.asr_done_at,
                        meta={**job.meta, "sliced": True, "slice_idx": idx},
                    )
                    if not translate_target:
                        _emit(sub_job, translation=None)
                    else:
                        _emit_pending_final(sub_job)
                        translate_q.put(sub_job)
                if commits:
                    new_cursor = max(committed_until, float(commits[-1]["end"]))
                    if ev.final:
                        committed_until_by_utt.pop(ev.start, None)
                    else:
                        committed_until_by_utt[ev.start] = new_cursor
                elif ev.final:
                    # VAD closed the utterance with nothing new to commit.
                    committed_until_by_utt.pop(ev.start, None)
                if holds and not ev.final:
                    # Keep ev.start as the tail's key — across successive
                    # provisional cycles the tail is the same in-progress
                    # utterance, so its key must not shift. Only `end` and
                    # `transcript` change as the tail grows.
                    tail_text = " ".join(str(e["text"]).strip() for e in holds).strip()
                    tail_ev = SegmentEvent(
                        pcm=ev.pcm,
                        start=ev.start,
                        end=ev.end,
                        final=False,
                    )
                    tail_job = _Job(
                        event=tail_ev,
                        enqueued_at=job.enqueued_at,
                        transcript=tail_text,
                        asr_done_at=job.asr_done_at,
                        meta={**job.meta, "sliced": True, "tail": True},
                    )
                    _emit_transcript(tail_job)
                    if translate_target:
                        translate_q.put(tail_job)
                continue

            # Fell through to whole-utterance path. If we've already committed
            # some prefix for this utterance, the cheap-path text would duplicate
            # it — drop the cursor entry and emit nothing (the prior commits
            # already covered everything; the residue isn't safely recoverable
            # without timestamps).
            if committed_until_by_utt.get(ev.start, 0.0) > 0.0:
                if ev.final:
                    committed_until_by_utt.pop(ev.start, None)
                log.debug(
                    "skip whole-utterance emit for [%s-%s] kind=%s - already partly committed",
                    _fmt_ts(ev.start), _fmt_ts(ev.end),
                    "final" if ev.final else "prov",
                )
                continue

            # Whole-utterance path (cheap json transcription, or aligned
            # transcription that produced 0/1 entry — nothing to split).
            job.transcript = text

            if not translate_target:
                _emit(job, translation=None)
                continue

            # Show the transcript right away — don't wait on the LLM. For
            # finals: park in the pending slot so the next utterance's
            # provisional can render alongside (not over) it. For provisionals:
            # the normal in-place preview line.
            if ev.final:
                _emit_pending_final(job)
            else:
                _emit_transcript(job)
            translate_q.put(job)

    # --- translate worker ---
    def _translate_worker() -> None:
        history: list[tuple[str, str]] = []
        while True:
            job = translate_q.get()
            if job is None:
                break

            job = _drain_stale(translate_q, job, max_age=LIVE_LAG_TOLERANCE_SECONDS, label="translate")
            ev = job.event

            t0 = time.monotonic()
            try:
                translation = translate(
                    job.transcript, history,
                    target=translate_target,
                    extra_context=translate_prompt,
                    timeout=float(LIVE_LAG_TOLERANCE_SECONDS),
                )
            except APITimeoutError:
                # Provisionals: skip — the next one will catch up.
                # Finals: still commit the transcript so the viewer doesn't
                # lose committed history just because translation was slow.
                if ev.final:
                    log.warning(
                        "translate timeout for final [%s-%s] after %.2fs - committing transcript only",
                        _fmt_ts(ev.start), _fmt_ts(ev.end), LIVE_LAG_TOLERANCE_SECONDS,
                    )
                    _emit(job, translation=None)
                else:
                    log.error(
                        "translate timeout for [%s-%s] after %.2fs - dropping",
                        _fmt_ts(ev.start), _fmt_ts(ev.end), LIVE_LAG_TOLERANCE_SECONDS,
                    )
                continue
            t_translate = time.monotonic() - t0
            job.meta["translate_elapsed"] = t_translate

            # Only commit *final* segments to translation history. Provisional
            # outputs are throwaway previews.
            if ev.final and translation:
                history.append((job.transcript, translation))
                if len(history) > TRANSLATE_HISTORY_LEN:
                    history.pop(0)

            if ev.final:
                _emit(job, translation=translation)
            else:
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
