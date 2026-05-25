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
SPEECH_PAD_MS = 30
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
        self._max_samples = int(LIVE_MAX_SEGMENT_SECONDS * LIVE_SAMPLE_RATE)
        self._provisional_samples = int(LIVE_PROVISIONAL_INTERVAL_SECONDS * LIVE_SAMPLE_RATE)
        # Open segment state. _open_start_sample is in our timeline.
        self._open_start_sample: int | None = None
        self._open_pcm: list[np.ndarray] = []
        self._last_provisional_sample: int = 0
        # Fallback recovery accumulator. Buffers raw audio captured since the
        # last primary-VAD finding (start/end) or recovery hit. Length-in-time
        # directly encodes "silence since last finding" — no separate tracker.
        # Soft cap at RECOVERY_MAX_SECONDS via a sliding window.
        self._recovery_min_samples = int(RECOVERY_MIN_SECONDS * LIVE_SAMPLE_RATE)
        self._recovery_max_samples = int(RECOVERY_MAX_SECONDS * LIVE_SAMPLE_RATE)
        self._recovery_buffer: list[np.ndarray] = []
        self._recovery_buffer_start_sample: int = 0

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
            # No segment open: this chunk is candidate silence. Append it to
            # the recovery accumulator. Buffer is purged on any VAD finding
            # below so its length always equals 'silence since last finding'.
            self._push_recovery_chunk(chunk)

        # --- handle Silero boundary events ---
        if flag is not None and "start" in flag:
            # If recovery already opened a segment, the primary VAD's later
            # threshold crossing is just confirmation — keep recovery's
            # earlier start (more accurate) and ignore this 'start'.
            if self._open_start_sample is not None:
                self._purge_recovery(chunk_end_sample)
                return events
            # Silero pads its reported start back by speech_pad_samples. Mirror
            # that in our timeline by anchoring just before this chunk.
            self._open_start_sample = max(0, chunk_start_sample - self._pad_samples)
            # The chunk itself wasn't appended above (open_start was None).
            self._open_pcm = [chunk]
            self._last_provisional_sample = self._open_start_sample
            self._purge_recovery(chunk_end_sample)
            return events

        if flag is not None and "end" in flag and self._open_start_sample is not None:
            # Silero adds speech_pad on the trailing side too. End just past
            # current chunk minus a chunk's worth of lookahead.
            end_sample = chunk_end_sample + self._pad_samples - LIVE_VAD_CHUNK_FRAMES
            events.append(self._flush(end_sample, final=True))
            return events

        # --- no boundary this chunk: maybe provisional, maybe force-flush ---
        if self._open_start_sample is None:
            # No segment open. Try the recovery pass: if the accumulator
            # holds >= RECOVERY_MIN_SECONDS of silence-since-last-finding,
            # peak-normalise it and re-run a stateless VAD. A hit opens a
            # segment retroactively at the recovery-detected onset.
            self._maybe_recover(chunk_end_sample)
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

    def _push_recovery_chunk(self, chunk: np.ndarray) -> None:
        """Append the current chunk to the recovery accumulator. Slides the
        window forward when the buffer exceeds RECOVERY_MAX_SECONDS so memory
        is bounded during very long silences."""
        self._recovery_buffer.append(chunk)
        total = sum(len(c) for c in self._recovery_buffer)
        while total > self._recovery_max_samples and len(self._recovery_buffer) > 1:
            dropped = self._recovery_buffer.pop(0)
            total -= len(dropped)
            self._recovery_buffer_start_sample += len(dropped)

    def _purge_recovery(self, chunk_end_sample: int) -> None:
        """Drop everything from the recovery accumulator and re-anchor its
        start to `chunk_end_sample`. Called on any VAD finding (primary
        start/end or recovery hit) so the accumulator's length always
        equals 'silence since last finding'."""
        self._recovery_buffer = []
        self._recovery_buffer_start_sample = chunk_end_sample

    def _maybe_recover(self, chunk_end_sample: int) -> None:
        """Run the fallback recovery pass if eligible.

        Eligibility: no segment open (caller guarantees) AND the accumulator
        holds at least RECOVERY_MIN_SECONDS of audio (which by construction
        equals 'silence since last VAD finding'). Recovery considers the
        whole accumulator on each chunk, so it can look back across long
        silences if the primary VAD never fired.

        On a hit, synthesises an open segment retroactively at the
        recovery-detected onset and purges the accumulator.
        """
        total = sum(len(c) for c in self._recovery_buffer)
        if total < self._recovery_min_samples:
            return

        window = np.concatenate(self._recovery_buffer)
        normalised, gain_db = peak_normalize(window)
        # Stateless pass: keeps the primary VADIterator's RNN state untouched.
        from silero_vad import get_speech_timestamps
        speech = get_speech_timestamps(
            normalised,
            self._model,
            sampling_rate=LIVE_SAMPLE_RATE,
            threshold=VAD_THRESHOLD,
            return_seconds=False,
        )
        if not speech:
            return
        # Earliest detected onset in the window.
        local_start = int(speech[0]["start"])
        recovery_start_abs = self._recovery_buffer_start_sample + local_start
        # Slice the RAW (un-normalised) audio from the recovery onset to now —
        # ASR gets its own per-segment normalisation downstream.
        raw_tail = window[local_start:]
        self._open_start_sample = recovery_start_abs
        self._open_pcm = [raw_tail]
        self._last_provisional_sample = recovery_start_abs
        self._purge_recovery(chunk_end_sample)
        log.info(
            "recovery: missed speech at [%.2f-%.2f] in %.1fdB normalised window",
            recovery_start_abs / LIVE_SAMPLE_RATE,
            chunk_end_sample / LIVE_SAMPLE_RATE,
            gain_db,
        )

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
        # The flush counts as a VAD finding: purge the recovery accumulator
        # so silence-since-finding restarts at this boundary, and so we
        # don't immediately re-fire on the just-finalised speech's tail.
        self._purge_recovery(end_sample)
        return ev
