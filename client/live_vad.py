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


@dataclass(frozen=True)
class SegmentEvent:
    pcm: np.ndarray   # float32, mono, 16 kHz
    start: float      # seconds from capture start
    end: float        # seconds from capture start
    final: bool       # True = immutable; False = preview, will be superseded


class LiveVAD:
    """Drive Silero VADIterator over a real-time PCM stream."""

    def __init__(self) -> None:
        import torch
        from silero_vad import VADIterator, load_silero_vad

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
            # earlier start (more accurate) and ignore this 'start'.
            if self._open_start_sample is not None:
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

        open_samples = chunk_end_sample - self._open_start_sample
        if open_samples >= self._max_samples:
            # Force-finalise overlong utterance. Do NOT reset Silero — it's
            # still legitimately in `triggered` state. Keep the segment open
            # under a fresh start so the next chunk continues to accumulate
            # audio toward the next final.
            events.append(self._flush(chunk_end_sample, final=True))
            # Re-open immediately at the current cursor.
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
        log.info(
            "recovery: missed speech at [%.2f-%.2f] in %+.1fdB normalised window",
            recovery_start_abs / LIVE_SAMPLE_RATE,
            chunk_end_sample / LIVE_SAMPLE_RATE,
            gain_db,
        )
        # Restart the silence cycle: clears capture ring, bumps gen, tells
        # recovery thread to drop its accumulator.
        self._purge_silence(chunk_end_sample)

    def _recovery_loop(self) -> None:
        """Sidecar that watches the silence accumulator and runs stateless
        VAD passes when it holds >= RECOVERY_MIN_SECONDS of audio.

        Lifecycle: ticks only while accumulating silence. A 'purge' message
        empties the accumulator and bumps the observed generation; a 'chunk'
        message extends it. Between purges the loop is dormant (waits on
        queue.get with a short timeout to bound pacing).

        Each pass runs at most every RECOVERY_PASS_INTERVAL_SECONDS so a
        long-running pass over a 30s window can't backlog the queue or
        starve other work. On a hit, posts (start_sample, raw_tail, gain_db)
        to the output queue stamped with the generation the pass observed.
        """
        from silero_vad import get_speech_timestamps
        import time as _time

        buf: list[np.ndarray] = []
        buf_start_sample = 0
        observed_gen = 0
        last_pass_at = 0.0
        # Block up to this long when idle; short enough to react to the
        # accumulator crossing RECOVERY_MIN_SECONDS shortly after it does.
        idle_timeout = RECOVERY_PASS_INTERVAL_SECONDS

        while not self._recovery_stop.is_set():
            try:
                msg = self._recovery_in_q.get(timeout=idle_timeout)
            except queue.Empty:
                msg = None

            if msg is not None:
                kind = msg[0]
                if kind == "stop":
                    return
                if kind == "purge":
                    _, end_sample, gen = msg
                    buf = []
                    buf_start_sample = int(end_sample)
                    observed_gen = int(gen)
                    # Drain any further queued messages so the next iteration
                    # only sees post-purge chunks — keeps buf aligned with
                    # the new generation.
                    while True:
                        try:
                            extra = self._recovery_in_q.get_nowait()
                        except queue.Empty:
                            break
                        if extra[0] == "stop":
                            return
                        if extra[0] == "purge":
                            _, end_sample2, gen2 = extra
                            buf = []
                            buf_start_sample = int(end_sample2)
                            observed_gen = int(gen2)
                        elif extra[0] == "chunk":
                            _, start, chunk = extra
                            if not buf:
                                buf_start_sample = int(start)
                            buf.append(chunk)
                    continue
                if kind == "chunk":
                    _, start, chunk = msg
                    if not buf:
                        buf_start_sample = int(start)
                    buf.append(chunk)

            # Pacing: only run a pass if enough time has elapsed since the
            # last one. Cheap to skip; the next chunk message will retry.
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


    def _flush(self, end_sample: int, *, final: bool) -> SegmentEvent:
        assert self._open_start_sample is not None
        pcm = np.concatenate(self._open_pcm) if self._open_pcm else np.zeros(0, dtype=np.float32)
        if final:
            log.debug(
                "segment finalised [%.2f-%.2f] dur=%.2fs%s",
                self._open_start_sample / LIVE_SAMPLE_RATE,
                end_sample / LIVE_SAMPLE_RATE,
                (end_sample - self._open_start_sample) / LIVE_SAMPLE_RATE,
                " (force-flush: exceeded MAX_SEGMENT_SECONDS)"
                if (end_sample - self._open_start_sample) >= self._max_samples else "",
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
        # The flush counts as a VAD finding: purge the silence ring (and
        # signal recovery to do the same) so silence-since-finding restarts
        # at this boundary, and so we don't immediately re-fire on the
        # just-finalised speech's tail.
        self._purge_silence(end_sample)
        return ev
