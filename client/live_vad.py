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
)

log = logging.getLogger("subsvibe.live_vad")

VAD_THRESHOLD = 0.5
SPEECH_PAD_MS = 30


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

        # --- handle Silero boundary events ---
        if flag is not None and "start" in flag:
            # Silero pads its reported start back by speech_pad_samples. Mirror
            # that in our timeline by anchoring just before this chunk.
            self._open_start_sample = max(0, chunk_start_sample - self._pad_samples)
            # The chunk itself wasn't appended above (open_start was None).
            self._open_pcm = [chunk]
            self._last_provisional_sample = self._open_start_sample
            return events

        if flag is not None and "end" in flag and self._open_start_sample is not None:
            # Silero adds speech_pad on the trailing side too. End just past
            # current chunk minus a chunk's worth of lookahead.
            end_sample = chunk_end_sample + self._pad_samples - LIVE_VAD_CHUNK_FRAMES
            events.append(self._flush(end_sample, final=True))
            return events

        # --- no boundary this chunk: maybe provisional, maybe force-flush ---
        if self._open_start_sample is None:
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
        return ev
