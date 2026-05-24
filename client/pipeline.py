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
)

# Only request word/segment granularity once the open utterance has crossed
# this threshold. Below it, VAD is expected to close the segment cleanly on
# its own, so the cheaper plain-JSON path is enough. Above it, we're at risk
# of a force-flush at MAX_SEGMENT_SECONDS — start asking the server for
# entries so we can promote completed pieces early.
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
from llm import TRANSLATE_HISTORY_LEN, translate
from utils.language import is_spaceless
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

# Minimum residue length (seconds) past `committed_until` that warrants
# another ASR cycle. Below this, the tail is too short for reliable
# transcription (Whisper-family backends produce gibberish or hallucinate
# on sub-100ms clips); we skip and let the next cycle accumulate audio.
_MIN_TAIL_SECONDS = 0.1


def _split_entries(
    ev: SegmentEvent,
    entries: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Positional split. Provisionals commit all entries except the trailing
    one (held as the new tail preview); finals commit all entries.

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
    surface punctuation."""
    if not entries:
        return [], []
    if ev.final:
        return list(entries), []
    if len(entries) <= 1:
        return [], list(entries)
    return list(entries[:-1]), [entries[-1]]


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
    history: int = 0,
    history_seconds: float = 0.0,
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
    capture_start_wall = datetime.now()
    # `start`/`end` on SegmentEvent are PCM seconds. Wall-clock lag of an output
    # against real-time audio is: monotonic() - (capture_start + event.end).
    # Wall-clock of an audio position is: capture_start_wall + ev.start seconds.

    def _audio_wall(audio_seconds: float) -> str:
        """Wall-clock HH:MM:SS.mmm for an audio-relative time. Stable across
        cycles: a slice at audio_seconds=K always renders the same string,
        regardless of when it's first emitted / re-emitted / committed."""
        return (capture_start_wall + timedelta(seconds=audio_seconds)).strftime("%H:%M:%S.%f")[:-3]

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
            job.transcript, translation, key=_utt_key(ev, job.meta), lag=lag,
            entries=job.meta.get("entries"),
            tag=_slice_tag(job),
            ts=_audio_wall(ev.start),
        )
        _log_emit(job, lag, kind="final")

    def _emit_transcript(job: _Job) -> None:
        """Provisional transcript only — translation line stays as-is until the LLM lands."""
        ev = job.event
        lag = time.monotonic() - (capture_start + ev.end)
        renderer.provisional_transcript(
            job.transcript, key=_utt_key(ev, job.meta), lag=lag,
            entries=job.meta.get("entries"),
            tag=_slice_tag(job),
            ts=_audio_wall(ev.start),
        )
        _log_emit(job, lag, kind="prov ")

    def _emit_translation(job: _Job, *, translation: str | None) -> None:
        """Provisional translation update — leaves transcript untouched."""
        ev = job.event
        lag = time.monotonic() - (capture_start + ev.end)
        if translation:
            renderer.provisional_translation(translation, key=_utt_key(ev, job.meta), lag=lag)
        _log_emit(job, lag, kind="prov ")

    def _emit_pending_final(job: _Job) -> None:
        """Park a final's transcript on screen while its translation is
        being computed. The renderer keeps it visible until commit, and the
        next utterance's provisional renders alongside rather than over it."""
        ev = job.event
        lag = time.monotonic() - (capture_start + ev.end)
        renderer.pending_final(
            job.transcript, key=_utt_key(ev, job.meta), lag=lag,
            entries=job.meta.get("entries"),
            tag=_slice_tag(job),
            ts=_audio_wall(ev.start),
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

            job = _drain_stale(asr_q, job, max_age=LIVE_LAG_TOLERANCE_SECONDS, label="asr")
            ev = job.event
            duration = ev.end - ev.start

            # Cursor-trim audio: send only the residue past the last commit.
            # Eliminates the cross-cycle duplication class by construction —
            # text and entries returned cover the uncovered tail only, so
            # there's no overlap with prior sliced commits to reconcile.
            # Cost (loss of acoustic context across the cursor) is offset by
            # `seg_prompt` carrying prior committed text when history is on.
            committed_until = committed_until_by_utt.get(ev.start, 0.0)
            tail_seconds = duration - committed_until
            # Skip ASR only for provisionals with nothing meaningful past the
            # cursor — next provisional cycle will accumulate more audio.
            # Finals always proceed: a prior cycle may have held a tail entry
            # whose text is rendered as the tail prov but never committed,
            # and only this final's ASR pass can promote it. Empty server
            # text is still handled below by `if not text: continue`.
            if tail_seconds < _MIN_TAIL_SECONDS and not ev.final:
                continue
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
                continue

            history_texts = select_history(
                history_buf, count=history, seconds=history_seconds, now=ev.start,
            ) if history_enabled else []
            seg_prompt = compose_prompt(prompt, "\n".join(history_texts) if history_texts else None)

            t0 = time.monotonic()
            # Request entries once the utterance is long enough to be at risk
            # of force-flush, or whenever we're past the first cursor advance
            # (continuation cycles should keep slicing even on short tails).
            with_entries = duration >= LIVE_ENTRIES_MIN_DURATION or committed_until > 0.0
            tail_start_abs = ev.start + committed_until
            try:
                text, entries = live_transcribe(
                    asr_client, model,
                    encode_wav(pcm_to_send),
                    f"{_fmt_ts(tail_start_abs)}-{_fmt_ts(ev.end)}.wav",
                    language=language,
                    prompt=seg_prompt,
                    timeout=LIVE_LAG_TOLERANCE_SECONDS,
                    with_entries=with_entries,
                )
            except APITimeoutError:
                log.error(
                    "ASR timeout for [%s-%s] after %.2fs - dropping",
                    _fmt_ts(tail_start_abs), _fmt_ts(ev.end), LIVE_LAG_TOLERANCE_SECONDS,
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
                # Final with no text: close out cursor + tail prov so they
                # don't leak. Most common path is tiny-tail finals where the
                # server returns empty for sub-second audio. Provisionals
                # leave state alone — a future cycle will refresh.
                if ev.final:
                    committed_until_by_utt.pop(ev.start, None)
                    renderer.discard_provisional((ev.start, "tail"))
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

            # Entry-driven promotion path: entries are 0-based on the trimmed
            # tail audio we sent, so commits/holds split positionally and
            # we shift back into the utterance's absolute frame by adding
            # `tail_start_abs` (== ev.start + committed_until).
            commits, holds = _split_entries(ev, entries)
            sliced = bool(commits or holds) and (
                len(entries) > 1 or committed_until > 0.0
            )
            if sliced:
                _log_promotion(ev, entries, commits, holds)
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
                        transcript=str(entry["text"]).strip(),
                        asr_done_at=job.asr_done_at,
                        meta={**job.meta, "sliced": True, "slice_idx": idx},
                    )
                    if history_enabled and sub_job.transcript:
                        history_buf.append((sub_ev.end, sub_job.transcript))
                        _trim_history()
                    if not translate_target:
                        _emit(sub_job, translation=None)
                    else:
                        _emit_pending_final(sub_job)
                        translate_q.put(sub_job)
                if commits:
                    if ev.final:
                        committed_until_by_utt.pop(ev.start, None)
                    else:
                        committed_until_by_utt[ev.start] = (
                            committed_until + float(commits[-1]["end"])
                        )
                elif ev.final:
                    # VAD closed the utterance with nothing new to commit.
                    committed_until_by_utt.pop(ev.start, None)
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
                        transcript=tail_text,
                        asr_done_at=job.asr_done_at,
                        meta={**job.meta, "sliced": True, "tail": True},
                    )
                    _emit_transcript(tail_job)
                    if translate_target:
                        translate_q.put(tail_job)
                else:
                    # No new tail this cycle. A prior cycle's tail is now
                    # stale (either the utterance just closed, or every
                    # entry was promoted with nothing left to preview).
                    # Retire it explicitly — keys are namespaced so the
                    # sub-commits above won't have cleared it on their own.
                    renderer.discard_provisional((ev.start, "tail"))
                continue

            # Whole-utterance path: cheap json transcription, or aligned
            # transcription that produced 0/1 entry. `text` covers only the
            # trimmed tail audio when `committed_until > 0`.
            #   Final  : emit as a sliced commit at [tail_start_abs, ev.end].
            #            Shifts job.event so the new key/wall-clock reflect
            #            the slice position in scrollback.
            #   Prov   : emit as a tail prov keyed on the OPEN utterance
            #            (ev.start) so successive cycles overwrite the same
            #            preview line in place. Shifting ev.start here would
            #            move the tail key per cycle and stack stale previews.
            job.transcript = text
            if committed_until > 0.0:
                if ev.final:
                    job.event = SegmentEvent(
                        pcm=ev.pcm,
                        start=tail_start_abs,
                        end=ev.end,
                        final=True,
                    )
                    job.meta = {**job.meta, "sliced": True}
                    committed_until_by_utt.pop(ev.start, None)
                    renderer.discard_provisional((ev.start, "tail"))
                else:
                    job.meta = {**job.meta, "sliced": True, "tail": True}

            if history_enabled and ev.final and job.transcript:
                history_buf.append((ev.end, job.transcript))
                _trim_history()

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
    # When --history / --history-seconds are passed, the translate worker
    # mirrors those semantics on its own (transcript, translation) buffer:
    # the flags fully override TRANSLATE_HISTORY_LEN. Without flags, the
    # buffer caps at TRANSLATE_HISTORY_LEN with no time window.
    translate_history_override = history > 0 or history_seconds > 0
    def _translate_worker() -> None:
        buf: list[tuple[float, str, str]] = []  # (ev.end, transcript, translation)
        while True:
            job = translate_q.get()
            if job is None:
                break

            job = _drain_stale(translate_q, job, max_age=LIVE_LAG_TOLERANCE_SECONDS, label="translate")
            ev = job.event

            if translate_history_override:
                window = buf
                if history_seconds > 0:
                    cutoff = ev.start - history_seconds
                    window = [w for w in window if w[0] >= cutoff]
                if history > 0:
                    window = window[-history:]
                hist_pairs = [(raw, tr) for _, raw, tr in window]
            else:
                hist_pairs = [(raw, tr) for _, raw, tr in buf[-TRANSLATE_HISTORY_LEN:]]

            t0 = time.monotonic()
            try:
                translation = translate(
                    job.transcript, hist_pairs,
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
            # outputs are throwaway previews. Hard cap the buffer at the
            # larger of the two windows so it can't grow unbounded over long
            # sessions while still serving the override path.
            if ev.final and translation:
                buf.append((ev.end, job.transcript, translation))
                cap = max(TRANSLATE_HISTORY_LEN, history) if translate_history_override else TRANSLATE_HISTORY_LEN
                if len(buf) > cap:
                    del buf[: len(buf) - cap]

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
