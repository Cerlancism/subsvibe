"""Online VAD: PCM stream -> segment events.

Wraps Silero VADIterator and exposes a single `feed(chunk_float32)` call that
returns a list of zero or more events. The pipeline drives this from the
capture thread.

Events
------
- ProvisionalSegment(pcm, start, end):
    emitted while a speech segment is open, at most every
    LIVE_PROVISIONAL_INTERVAL_SECONDS. Lets the renderer show a mid-sentence
    preview. May be superseded by a later provisional or by the final.
- FinalSegment(pcm, start, end):
    emitted on confirmed end-of-speech (silence >= LIVE_MIN_SILENCE_MS) OR on
    force-flush when an open segment exceeds LIVE_MAX_SEGMENT_SECONDS. Final
    segments are immutable history.

Time base
---------
`start`/`end` are seconds since `feed` was first called. We maintain our own
monotonic sample counter (`_samples_seen`) and never use Silero's internal
`current_sample` for absolute time — VADIterator.reset_states() would reset
that counter and shift the timeline. Boundary timestamps are therefore
quantised to chunk granularity (~32 ms), which is well below human-perceptible
subtitle accuracy.
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from capture import (
    LIVE_MAX_SEGMENT_SECONDS,
    LIVE_MIN_SILENCE_MS,
    LIVE_PROVISIONAL_INTERVAL_SECONDS,
    LIVE_SAMPLE_RATE,
    LIVE_VAD_CHUNK_FRAMES,
    peak_normalize,
)

log = logging.getLogger("subsvibe.live_vad")

VAD_THRESHOLD = 0.5
# Recovery uses a more permissive threshold than the primary: it only runs
# after the primary VADIterator already missed the speech in question, so
# the whole point is to catch quieter / less confident content that fell
# below 0.5. Matches the file-input primary pass in vad.py — both are
# 'second-chance, prefer recall over precision' passes.
RECOVERY_VAD_THRESHOLD = 0.1
SPEECH_PAD_MS = 30
# On a fresh primary-VAD speech start, prepend up to this much audio captured
# since the previous VAD finding. Silero's threshold-crossing chunk often
# lands a beat after the actual word onset; without pre-roll, ASR sees a
# clipped opening syllable. Sourced from the recovery accumulator, which is
# purged on every VAD finding — so this can never reach back past audio
# already sent to ASR via a prior segment.
PRESPEECH_PAD_SECONDS = 1.0
# Fallback recovery: a sidecar accumulator buffers raw audio from the moment
# of the last primary-VAD finding onward. When it holds at least
# RECOVERY_MIN_SECONDS of silence-since-finding, each subsequent chunk runs
# a stateless VAD pass over the peak-normalised accumulator. A hit opens a
# segment retroactively at the recovery-detected onset — recovers quiet
# speech the primary VADIterator missed, without disturbing its RNN state.
# The accumulator is purged on any VAD finding (primary start/end or
# recovery hit) and slides forward at RECOVERY_MAX_SECONDS to bound memory.
RECOVERY_MIN_SECONDS = 1.0
RECOVERY_MAX_SECONDS = 30.0
# Recovery runs in a sidecar thread because get_speech_timestamps over a 30s
# window can take ~150ms on CPU — far past the 32ms chunk budget the capture
# thread runs on. Letting that block capture overflows the WASAPI loopback
# ring buffer and drifts audio-clock behind wall-clock permanently. The
# sidecar thread paces itself; capture only pays a non-blocking queue poll.
RECOVERY_PASS_INTERVAL_SECONDS = 0.5
# When recovery opens a segment, the primary VADIterator is still in 'not
# triggered' state — it never saw the speech, so it will never emit an 'end'
# event to close the segment. To prevent an indefinitely-open segment that
# just keeps feeding silence to ASR, the sidecar keeps running normalised
# low-threshold passes over the post-onset buffer. When the trailing edge of
# the latest detected speech sits >= RECOVERY_END_SILENCE_MS behind the
# buffer's tail, the sidecar tells capture to finalise. The segment closes
# at the last-speech edge and capture does NOT auto re-open it — control
# returns to the primary VAD.
RECOVERY_END_SILENCE_MS = LIVE_MIN_SILENCE_MS


@dataclass(frozen=True)
class SegmentEvent:
    pcm: np.ndarray   # float32, mono, 16 kHz
    start: float      # seconds from capture start
    end: float        # seconds from capture start
    final: bool       # True = immutable; False = preview, will be superseded


class LiveVAD:
    """Drive Silero VADIterator over a real-time PCM stream."""

    def __init__(self, audio_wall_fn=None) -> None:
        """`audio_wall_fn(audio_seconds: float) -> str` formats an audio-clock
        offset as HH:MM:SS.mmm for VAD log lines. The pipeline's implementation
        is audio-clock based and crystal-drifts on long sessions, but these
        lines fire infrequently and aren't re-rendered, so drift is acceptable.
        Falls back to `{s:.2f}s` for tests / standalone use."""
        import torch
        from silero_vad import VADIterator, load_silero_vad
        self._audio_wall = audio_wall_fn or (lambda s: f"{s:.2f}s")

        self._torch = torch
        self._model = load_silero_vad(onnx=True)
        self._iter = VADIterator(
            self._model,
            threshold=VAD_THRESHOLD,
            sampling_rate=LIVE_SAMPLE_RATE,
            min_silence_duration_ms=LIVE_MIN_SILENCE_MS,
            speech_pad_ms=SPEECH_PAD_MS,
        )
        # Our own absolute timeline. Never touched by VADIterator.reset_states.
        self._samples_seen = 0
        self._pad_samples = int(SPEECH_PAD_MS * LIVE_SAMPLE_RATE / 1000)
        self._prespeech_pad_samples = int(PRESPEECH_PAD_SECONDS * LIVE_SAMPLE_RATE)
        self._max_samples = int(LIVE_MAX_SEGMENT_SECONDS * LIVE_SAMPLE_RATE)
        self._provisional_samples = int(LIVE_PROVISIONAL_INTERVAL_SECONDS * LIVE_SAMPLE_RATE)
        # Open segment state. _open_start_sample is in our timeline.
        self._open_start_sample: int | None = None
        self._open_pcm: list[np.ndarray] = []
        self._last_provisional_sample: int = 0
        # Force-flush carryover: one-slot stash of the just-flushed segment's
        # full PCM + absolute start sample. Populated on every primary-driven
        # force-flush; consumed (or overwritten) when the pipeline calls
        # request_splice() with the held entry's absolute start. Lets the
        # pipeline retroactively prepend the dropped trailing entry's audio
        # to the now-open utterance once ASR has identified its exact range —
        # no fixed back-off, no duplicate audio, lossless.
        self._flush_stash_pcm: np.ndarray | None = None
        self._flush_stash_start_sample: int = 0
        # Splice requests posted by the pipeline thread; applied on the next
        # feed() so only the capture thread mutates _open_pcm / _open_start_sample.
        # Holds absolute start samples (int); the freshest is honoured per cycle.
        self._splice_q: "queue.Queue[int]" = queue.Queue()
        # True while the currently-open segment was opened by recovery rather
        # than the primary VADIterator. The primary's state machine is NOT
        # triggered for such a segment, so it will never emit an 'end' — the
        # sidecar's recovery-end advisories are the only thing that closes
        # it. Also suppresses force-flush auto re-open, to avoid the loop
        # where a quiet noise floor keeps re-opening every MAX_SEGMENT_SECONDS.
        self._open_via_recovery: bool = False
        # Capture-thread silence ring. Holds raw chunks captured since the last
        # VAD finding. Used by primary-start to back-date PCM by up to
        # PRESPEECH_PAD_SECONDS. Bounded by RECOVERY_MAX_SECONDS so memory is
        # capped during long silences. Cheap append/slide — stays on capture
        # thread because primary-start needs synchronous access to it.
        self._silence_ring_max_samples = int(RECOVERY_MAX_SECONDS * LIVE_SAMPLE_RATE)
        self._silence_ring: list[np.ndarray] = []
        self._silence_ring_start_sample: int = 0
        # Recovery sidecar: stateless VAD passes over the silence accumulator
        # run on a dedicated thread because get_speech_timestamps over a long
        # window can take ~150ms — far past the capture thread's 32ms chunk
        # budget. Capture posts silence chunks + purge advisories on the in
        # queue and polls the out queue non-blocking each chunk.
        self._recovery_in_q: "queue.Queue" = queue.Queue()
        self._recovery_out_q: "queue.Queue" = queue.Queue()
        self._recovery_stop = threading.Event()
        # Monotonic generation counter. Bumped on every primary VAD finding
        # (start/end/force-flush) when capture posts a purge. Recovery stamps
        # each hit it emits with the generation it observed; capture ignores
        # hits whose generation is stale (i.e., a primary finding happened
        # between the recovery decision and the hit being polled). Prevents
        # a late hit from re-opening a segment after a primary boundary.
        self._recovery_gen: int = 0
        self._recovery_thread = threading.Thread(
            target=self._recovery_loop, daemon=True, name="LiveVAD-recovery",
        )
        self._recovery_thread.start()

    def request_splice(self, absolute_start_sample: int) -> None:
        """Pipeline-side hook for force-flush carryover. Posts a request that
        the next feed() applies: the open utterance's start is rewound to
        `absolute_start_sample`, and the PCM range
        [absolute_start_sample, current open start) is prepended from the
        just-flushed segment's stash.

        Called from the ASR worker thread once the held trailing entry's
        absolute start is known. Non-blocking — capture applies the request
        on the next chunk so only one thread ever mutates open-segment state.

        If the splice request can't be honoured (stash overwritten by a newer
        force-flush, requested start outside the stash range, or no open
        utterance), it is silently dropped — the dropped held entry stays
        dropped. Splicing failure is a degraded path, not a correctness bug:
        a subsequent provisional cycle will eventually catch the audio if any
        VAD opens over it."""
        self._splice_q.put(int(absolute_start_sample))

    def close(self) -> None:
        """Stop the recovery sidecar thread. Idempotent."""
        if self._recovery_stop.is_set():
            return
        self._recovery_stop.set()
        # Wake the recovery loop's blocking get() so it can notice the stop flag.
        self._recovery_in_q.put(("stop", 0, None))
        self._recovery_thread.join(timeout=2)

    @property
    def now(self) -> float:
        """Seconds of audio consumed so far."""
        return self._samples_seen / LIVE_SAMPLE_RATE

    def feed(self, chunk: np.ndarray) -> Iterable[SegmentEvent]:
        """Feed one PCM chunk (float32, length LIVE_VAD_CHUNK_FRAMES)."""
        if len(chunk) != LIVE_VAD_CHUNK_FRAMES:
            raise ValueError(
                f"LiveVAD expects {LIVE_VAD_CHUNK_FRAMES}-sample chunks, got {len(chunk)}"
            )
        events: list[SegmentEvent] = []

        # Apply any pending splice request from the pipeline (force-flush
        # carryover). Always handled on the capture thread, so no lock is
        # needed around _open_pcm / _open_start_sample.
        self._apply_pending_splice()

        # Mark the absolute sample index AT the start of this chunk (before
        # incrementing) — useful for back-dating speech_start across the
        # speech_pad lookback.
        chunk_start_sample = self._samples_seen
        chunk_end_sample = chunk_start_sample + LIVE_VAD_CHUNK_FRAMES

        tensor = self._torch.from_numpy(chunk)
        flag = self._iter(tensor, return_seconds=False)
        self._samples_seen = chunk_end_sample

        # Accumulate audio for any open segment.
        if self._open_start_sample is not None:
            self._open_pcm.append(chunk)
            # If recovery owns the open segment, keep feeding the sidecar so
            # it can watch for end-of-speech. The sidecar's buffer for an
            # open segment is the post-onset audio; it runs the same low-
            # threshold normalised pass and tells us when to finalise.
            if self._open_via_recovery:
                self._recovery_in_q.put(("chunk", chunk_start_sample, chunk))
        else:
            # No segment open: this chunk is candidate silence. Push to the
            # capture-side silence ring (for primary-start pre-roll) and
            # forward a copy to the recovery thread's accumulator.
            self._push_silence_chunk(chunk, chunk_start_sample)
            self._recovery_in_q.put(("chunk", chunk_start_sample, chunk))

        # --- handle Silero boundary events ---
        if flag is not None and "start" in flag:
            # If recovery already opened a segment, the primary VAD's later
            # threshold crossing is just confirmation — keep recovery's
            # earlier start (more accurate) and ignore this 'start'. Flip
            # _open_via_recovery off so the segment is now owned by the
            # primary: Silero is now in 'triggered' state and will emit
            # 'end' naturally, and the sidecar's end-watch is no longer
            # needed (recovery-end advisories from before the takeover are
            # stale-filtered by the gen bump in _purge_silence below).
            if self._open_start_sample is not None:
                self._open_via_recovery = False
                self._purge_silence(chunk_end_sample)
                return events
            # Pre-roll: take up to PRESPEECH_PAD_SECONDS from the silence
            # ring's tail. The ring is purged on every VAD finding, so its
            # contents are exclusively audio captured since the last segment
            # was sent to ASR — pre-roll cannot overlap prior transcription
            # input. The threshold-crossing chunk itself was just appended
            # to the ring, so excluding the last entry here avoids double-
            # counting before we seed _open_pcm with the chunk below.
            preroll: list[np.ndarray] = []
            preroll_samples = 0
            if len(self._silence_ring) > 1:
                for prev in reversed(self._silence_ring[:-1]):
                    preroll.insert(0, prev)
                    preroll_samples += len(prev)
                    if preroll_samples >= self._prespeech_pad_samples:
                        break
                if preroll_samples > self._prespeech_pad_samples:
                    excess = preroll_samples - self._prespeech_pad_samples
                    preroll[0] = preroll[0][excess:]
                    preroll_samples -= excess
            preroll_start = chunk_start_sample - preroll_samples
            # Silero pads its reported start back by speech_pad_samples on
            # top of the pre-roll — keeps the original 30ms timestamp
            # behavior for downstream consumers that already account for it.
            self._open_start_sample = max(0, preroll_start - self._pad_samples)
            # Seed _open_pcm with pre-roll + the threshold-crossing chunk.
            self._open_pcm = [*preroll, chunk]
            self._last_provisional_sample = self._open_start_sample
            self._purge_silence(chunk_end_sample)
            return events

        if flag is not None and "end" in flag and self._open_start_sample is not None:
            # Silero adds speech_pad on the trailing side too. End just past
            # current chunk minus a chunk's worth of lookahead.
            end_sample = chunk_end_sample + self._pad_samples - LIVE_VAD_CHUNK_FRAMES
            events.append(self._flush(end_sample, final=True))
            return events

        # --- no boundary this chunk: maybe poll recovery, maybe provisional ---
        if self._open_start_sample is None:
            # No segment open. Drain any pending recovery hits — but only
            # honour them if their generation matches the current one (i.e.
            # no primary VAD finding has happened since the recovery thread
            # made its decision). Stale hits are silently dropped.
            self._consume_recovery_hits(chunk_end_sample)
            return events

        # Recovery-owned open segment: the primary VAD will never emit 'end'
        # for it, so poll the sidecar for an end-of-speech advisory. If one
        # arrives, finalise at the recovery-detected silence boundary and
        # return — do NOT re-open. Control hands back to the primary VAD.
        if self._open_via_recovery:
            end_at = self._consume_recovery_end(chunk_end_sample)
            if end_at is not None:
                events.append(self._flush(end_at, final=True))
                return events

        open_samples = chunk_end_sample - self._open_start_sample
        if open_samples >= self._max_samples:
            was_recovery = self._open_via_recovery
            # Stash the full just-flushed PCM before _flush wipes _open_pcm.
            # The pipeline will (a few hundred ms later, once ASR is done)
            # call request_splice() with the held trailing entry's absolute
            # start, and the next feed() will prepend the corresponding
            # range of this stash to the new utterance. Skip the stash for
            # recovery-driven flush (no re-open follows; nothing to splice
            # into).
            if not was_recovery and self._open_pcm:
                self._flush_stash_pcm = np.concatenate(self._open_pcm)
                self._flush_stash_start_sample = int(self._open_start_sample)
            events.append(self._flush(chunk_end_sample, final=True))
            if was_recovery:
                # Recovery-driven force-flush: do NOT auto re-open. The
                # primary VAD isn't triggered, and blindly re-opening just
                # keeps feeding silence to ASR every MAX_SEGMENT_SECONDS
                # (the loop this guard exists to prevent). If quiet speech
                # continues, recovery will catch it on the next pass.
                return events
            # Primary-driven force-flush: Silero is still legitimately in
            # 'triggered' state. Re-open immediately at the current cursor
            # so the next chunk continues to accumulate audio toward the
            # next final. The stash above lets request_splice() retroactively
            # prepend the dropped trailing entry's audio once the pipeline
            # knows its exact range.
            self._open_start_sample = chunk_end_sample
            self._open_pcm = []
            self._last_provisional_sample = chunk_end_sample
            return events

        if chunk_end_sample - self._last_provisional_sample >= self._provisional_samples:
            self._last_provisional_sample = chunk_end_sample
            pcm = np.concatenate(self._open_pcm)
            events.append(SegmentEvent(
                pcm=pcm,
                start=self._open_start_sample / LIVE_SAMPLE_RATE,
                end=chunk_end_sample / LIVE_SAMPLE_RATE,
                final=False,
            ))

        return events

    def _apply_pending_splice(self) -> None:
        """Drain splice_q and honour the freshest request, if any.

        A splice request rewinds the open utterance's start to
        `absolute_start_sample` and prepends the PCM range
        [absolute_start_sample, current open start) sourced from the
        force-flush stash. Used by the pipeline to retroactively carry the
        dropped trailing entry of a force-flush final into the new utterance.

        Drops the request silently if:
        - no open utterance (recovery never opened the post-flush segment)
        - stash empty (next force-flush hasn't populated it, or already
          consumed)
        - requested start lies outside the stash range
        - requested start is at or past the current open start (nothing to
          prepend; ignore to avoid going backwards on a stale request that
          arrived after VAD already moved past)
        """
        latest: int | None = None
        try:
            while True:
                latest = self._splice_q.get_nowait()
        except queue.Empty:
            pass
        if latest is None:
            return
        if self._open_start_sample is None:
            return
        if self._flush_stash_pcm is None:
            return
        stash_start = self._flush_stash_start_sample
        stash_end = stash_start + len(self._flush_stash_pcm)
        if latest < stash_start or latest >= stash_end:
            return
        if latest >= self._open_start_sample:
            return
        slice_offset = latest - stash_start
        carryover = self._flush_stash_pcm[slice_offset:].copy()
        # Prepend the carryover; current _open_pcm is whatever has been
        # captured since the force-flush boundary (could be empty if the
        # splice arrived before the first post-flush chunk).
        self._open_pcm = [carryover, *self._open_pcm]
        prior_start = self._open_start_sample
        self._open_start_sample = latest
        # Anchor provisional-cadence to the new start so the next emit
        # interval is measured from there.
        self._last_provisional_sample = min(self._last_provisional_sample, latest)
        log.debug(
            "force-flush carryover: spliced %.2fs of audio, new start=%s (was %s)",
            len(carryover) / LIVE_SAMPLE_RATE,
            self._audio_wall(latest / LIVE_SAMPLE_RATE),
            self._audio_wall(prior_start / LIVE_SAMPLE_RATE),
        )
        # One-shot consumption: clear the stash so the next force-flush
        # starts with a fresh slot.
        self._flush_stash_pcm = None

    def _push_silence_chunk(self, chunk: np.ndarray, chunk_start_sample: int) -> None:
        """Append to the capture-side silence ring. Slides the window forward
        when it exceeds RECOVERY_MAX_SECONDS so memory is bounded during very
        long silences."""
        if not self._silence_ring:
            self._silence_ring_start_sample = chunk_start_sample
        self._silence_ring.append(chunk)
        total = sum(len(c) for c in self._silence_ring)
        while total > self._silence_ring_max_samples and len(self._silence_ring) > 1:
            dropped = self._silence_ring.pop(0)
            total -= len(dropped)
            self._silence_ring_start_sample += len(dropped)

    def _purge_silence(self, chunk_end_sample: int) -> None:
        """Empty the capture-side silence ring AND notify the recovery thread
        to purge its own accumulator. Bumps the generation counter so any
        recovery hit emitted before the purge is recognised as stale.

        Called on every primary VAD finding (start, end, force-flush) and on
        accepted recovery hits — i.e., whenever 'silence since last finding'
        should restart at zero."""
        self._silence_ring = []
        self._silence_ring_start_sample = chunk_end_sample
        self._recovery_gen += 1
        self._recovery_in_q.put(("purge", chunk_end_sample, self._recovery_gen))

    def _consume_recovery_hits(self, chunk_end_sample: int) -> None:
        """Poll the recovery thread's output queue (non-blocking) and act on
        any hits that arrived since the last poll. Only the freshest matching
        hit is honoured; older queued hits are discarded.

        A hit is accepted iff its stamped generation matches the current
        _recovery_gen — which guarantees no primary VAD finding happened
        between recovery's decision and our poll.

        On accept, synthesises an open segment at the recovery-detected onset
        and triggers a purge (which also bumps the gen, so the recovery
        thread restarts its accumulator)."""
        accepted: tuple[int, np.ndarray, float] | None = None
        try:
            while True:
                kind, gen, payload = self._recovery_out_q.get_nowait()
                if kind != "hit":
                    continue
                if gen != self._recovery_gen:
                    continue  # stale
                accepted = payload  # keep the most recent matching hit
        except queue.Empty:
            pass
        if accepted is None:
            return
        recovery_start_abs, raw_tail, gain_db = accepted
        self._open_start_sample = int(recovery_start_abs)
        self._open_pcm = [raw_tail]
        self._last_provisional_sample = int(recovery_start_abs)
        self._open_via_recovery = True
        log.info(
            "recovery: missed speech at [%s-%s] in %+.1fdB normalised window",
            self._audio_wall(recovery_start_abs / LIVE_SAMPLE_RATE),
            self._audio_wall(chunk_end_sample / LIVE_SAMPLE_RATE),
            gain_db,
        )
        # Restart the silence cycle: clears capture ring, bumps gen, tells
        # recovery thread to drop its accumulator and switch into 'watch
        # for end-of-speech' mode for this open segment.
        self._purge_silence(chunk_end_sample)
        # The 'open' advisory uses the freshly-bumped generation so the
        # sidecar correlates subsequent chunks with the right segment.
        self._recovery_in_q.put(("open", self._recovery_gen, int(recovery_start_abs)))

    def _consume_recovery_end(self, chunk_end_sample: int) -> int | None:
        """Poll the recovery sidecar for an end-of-speech advisory on the
        currently-open recovery segment. Returns the absolute sample at
        which to finalise, or None.

        Only advisories with the current generation are honoured. If the
        sidecar reports an end-sample before the segment's start (shouldn't
        happen, defensive), it is ignored."""
        accepted: int | None = None
        try:
            while True:
                kind, gen, payload = self._recovery_out_q.get_nowait()
                if kind != "end":
                    continue
                if gen != self._recovery_gen:
                    continue
                accepted = int(payload)
        except queue.Empty:
            pass
        if accepted is None:
            return None
        assert self._open_start_sample is not None
        if accepted <= self._open_start_sample:
            return None
        # Don't extend past where capture has actually consumed audio.
        return min(accepted, chunk_end_sample)

    def _recovery_loop(self) -> None:
        """Sidecar that runs stateless VAD passes over a normalised audio
        buffer. Operates in one of two modes per generation:

        - 'watch_onset' (default after every 'purge'): buffer holds audio
          captured since the last primary-VAD finding. When it reaches
          RECOVERY_MIN_SECONDS, each pass looks for an onset that the
          primary VADIterator missed. On a hit, posts ('hit', gen, ...) and
          waits for the inevitable 'purge' from capture (capture bumps the
          gen when it accepts the hit, which resets us).

        - 'watch_end' (entered via 'open' advisory from capture, used while
          a recovery-opened segment is live): buffer holds post-onset audio
          for that segment. Each pass looks for the trailing silence — the
          gap between the latest detected speech edge and the buffer tail.
          When that gap reaches RECOVERY_END_SILENCE_MS, posts ('end', gen,
          end_sample_abs) so capture can finalise. The primary VAD never
          fires 'end' for a recovery-opened segment (it's not triggered),
          so without this the segment would only ever close on force-flush
          and immediately loop.

        Lifecycle: every 'purge' resets to 'watch_onset' and bumps observed
        gen. An 'open' advisory after a purge switches us to 'watch_end'
        for that gen. A subsequent 'purge' (from capture finalising the
        recovery segment, or any primary VAD finding) returns us to onset
        mode.

        Each pass runs at most every RECOVERY_PASS_INTERVAL_SECONDS so a
        long-running pass over a 30s window can't backlog the queue.
        """
        from silero_vad import get_speech_timestamps
        import time as _time

        buf: list[np.ndarray] = []
        buf_start_sample = 0
        observed_gen = 0
        # 'watch_onset' or 'watch_end'. Reset to 'watch_onset' on every purge.
        mode = "watch_onset"
        last_pass_at = 0.0
        idle_timeout = RECOVERY_PASS_INTERVAL_SECONDS

        def apply_msg(m: tuple) -> bool:
            """Mutate buf/mode/gen for one queued message. Returns True if
            'stop' was seen so the caller can exit."""
            nonlocal buf, buf_start_sample, observed_gen, mode
            kind = m[0]
            if kind == "stop":
                return True
            if kind == "purge":
                _, end_sample, gen = m
                buf = []
                buf_start_sample = int(end_sample)
                observed_gen = int(gen)
                mode = "watch_onset"
                return False
            if kind == "open":
                _, gen, start_sample = m
                # Only honour 'open' for the current generation (purge
                # always precedes it in capture; if a later purge has
                # already bumped past it, this advisory is stale).
                if int(gen) != observed_gen:
                    return False
                buf = []
                buf_start_sample = int(start_sample)
                mode = "watch_end"
                return False
            if kind == "chunk":
                _, start, chunk = m
                if not buf:
                    buf_start_sample = int(start)
                buf.append(chunk)
                return False
            return False

        while not self._recovery_stop.is_set():
            try:
                msg = self._recovery_in_q.get(timeout=idle_timeout)
            except queue.Empty:
                msg = None

            if msg is not None:
                if apply_msg(msg):
                    return
                # Drain any further queued messages in this tick so buf is
                # aligned with the latest generation/mode before we run a
                # pass. Without this, a 'purge' followed by 'open' followed
                # by chunks could run an onset-mode pass over post-open audio.
                while True:
                    try:
                        extra = self._recovery_in_q.get_nowait()
                    except queue.Empty:
                        break
                    if apply_msg(extra):
                        return

            now = _time.monotonic()
            if now - last_pass_at < RECOVERY_PASS_INTERVAL_SECONDS:
                continue
            total = sum(len(c) for c in buf)
            if total < int(RECOVERY_MIN_SECONDS * LIVE_SAMPLE_RATE):
                continue
            last_pass_at = now

            window = np.concatenate(buf)
            normalised, gain_db = peak_normalize(window)
            try:
                speech = get_speech_timestamps(
                    normalised,
                    self._model,
                    sampling_rate=LIVE_SAMPLE_RATE,
                    threshold=RECOVERY_VAD_THRESHOLD,
                    return_seconds=False,
                )
            except Exception:
                log.exception("recovery: get_speech_timestamps failed")
                continue

            if mode == "watch_onset":
                if not speech:
                    continue
                local_start = int(speech[0]["start"])
                # Pre-roll: back up by PRESPEECH_PAD_SECONDS, clamped to the
                # start of buf. buf was emptied at the last VAD finding so
                # nothing before its origin can overlap audio already sent to
                # ASR — clamping to 0 is the only bound we need.
                preroll_samples = int(PRESPEECH_PAD_SECONDS * LIVE_SAMPLE_RATE)
                slice_start = max(0, local_start - preroll_samples)
                recovery_start_abs = buf_start_sample + slice_start
                raw_tail = window[slice_start:].copy()
                self._recovery_out_q.put((
                    "hit",
                    observed_gen,
                    (recovery_start_abs, raw_tail, float(gain_db)),
                ))
            else:  # 'watch_end'
                # End-of-speech detection: the gap between the trailing edge
                # of the last detected speech and the buffer tail. If no
                # speech is found at all in the post-onset buffer, the gap
                # is the entire buffer length. If the gap is long enough,
                # tell capture to finalise at the last-speech edge.
                end_silence_samples = int(RECOVERY_END_SILENCE_MS * LIVE_SAMPLE_RATE / 1000)
                if speech:
                    last_speech_end = int(speech[-1]["end"])
                else:
                    last_speech_end = 0
                trailing_silence = len(window) - last_speech_end
                if trailing_silence < end_silence_samples:
                    continue
                # Finalise at the speech edge plus a chunk's worth of pad,
                # mirroring the primary's _flush math which adds speech_pad
                # on the trailing side. Clamp to buffer length.
                end_local = min(len(window), last_speech_end + self._pad_samples)
                end_abs = buf_start_sample + end_local
                self._recovery_out_q.put(("end", observed_gen, end_abs))


    def _flush(self, end_sample: int, *, final: bool) -> SegmentEvent:
        assert self._open_start_sample is not None
        pcm = np.concatenate(self._open_pcm) if self._open_pcm else np.zeros(0, dtype=np.float32)
        if final:
            start_s = self._open_start_sample / LIVE_SAMPLE_RATE
            end_s = end_sample / LIVE_SAMPLE_RATE
            forced = (end_sample - self._open_start_sample) >= self._max_samples
            log.debug(
                "segment finalised [%s-%s] dur=%.2fs%s",
                self._audio_wall(start_s),
                self._audio_wall(end_s),
                end_s - start_s,
                " (force-flush: exceeded MAX_SEGMENT_SECONDS)" if forced else "",
            )
        ev = SegmentEvent(
            pcm=pcm,
            start=self._open_start_sample / LIVE_SAMPLE_RATE,
            end=end_sample / LIVE_SAMPLE_RATE,
            final=final,
        )
        self._open_start_sample = None
        self._open_pcm = []
        self._last_provisional_sample = 0
        self._open_via_recovery = False
        # The flush counts as a VAD finding: purge the silence ring (and
        # signal recovery to do the same) so silence-since-finding restarts
        # at this boundary, and so we don't immediately re-fire on the
        # just-finalised speech's tail.
        self._purge_silence(end_sample)
        return ev
