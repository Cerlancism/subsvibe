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
`start`/`end` are seconds since `feed` was first called (i.e. since capture
started). They are wall-clock-aligned with the recorder's PCM clock, not the
host monotonic clock — useful for lag measurement.
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
            speech_pad_ms=30,
        )
        self._samples_seen = 0
        self._open_pcm: list[np.ndarray] = []   # chunks since current speech_start
        self._open_start: float | None = None   # seconds
        self._last_provisional_at: float = 0.0  # seconds (segment timeline)
        self._max_samples = int(LIVE_MAX_SEGMENT_SECONDS * LIVE_SAMPLE_RATE)
        self._provisional_step = int(LIVE_PROVISIONAL_INTERVAL_SECONDS * LIVE_SAMPLE_RATE)

    @property
    def now(self) -> float:
        """Seconds of audio consumed so far (PCM-clock time)."""
        return self._samples_seen / LIVE_SAMPLE_RATE

    def feed(self, chunk: np.ndarray) -> Iterable[SegmentEvent]:
        """Feed one PCM chunk (float32, length LIVE_VAD_CHUNK_FRAMES) and yield events."""
        if len(chunk) != LIVE_VAD_CHUNK_FRAMES:
            raise ValueError(
                f"LiveVAD expects {LIVE_VAD_CHUNK_FRAMES}-sample chunks, got {len(chunk)}"
            )
        events: list[SegmentEvent] = []

        tensor = self._torch.from_numpy(chunk)
        flag = self._iter(tensor, return_seconds=False)
        self._samples_seen += LIVE_VAD_CHUNK_FRAMES

        if self._open_start is not None:
            self._open_pcm.append(chunk)

        if flag is not None and "start" in flag:
            # New speech onset — Silero pads slightly, so trust its sample index.
            start_sample = int(flag["start"])
            self._open_start = start_sample / LIVE_SAMPLE_RATE
            # If Silero looked back across the chunk boundary, capture this chunk
            # as the first speech audio (we already appended above when
            # _open_start was None — fix that here).
            if not self._open_pcm:
                self._open_pcm.append(chunk)
            self._last_provisional_at = self._open_start
            return events

        if flag is not None and "end" in flag and self._open_start is not None:
            end_sample = int(flag["end"])
            events.append(self._flush(end_sample / LIVE_SAMPLE_RATE, final=True))
            return events

        # No boundary on this chunk; maybe emit a provisional or force-flush.
        if self._open_start is not None:
            open_samples = sum(len(c) for c in self._open_pcm)
            if open_samples >= self._max_samples:
                # Force-finalise on overlong utterance.
                end_seconds = self._open_start + open_samples / LIVE_SAMPLE_RATE
                log.warning(
                    "segment exceeded %.1fs - force-finalising at %.2fs",
                    LIVE_MAX_SEGMENT_SECONDS, end_seconds,
                )
                events.append(self._flush(end_seconds, final=True))
                self._iter.reset_states()
                return events

            now = self._open_start + open_samples / LIVE_SAMPLE_RATE
            if now - self._last_provisional_at >= LIVE_PROVISIONAL_INTERVAL_SECONDS:
                self._last_provisional_at = now
                pcm = np.concatenate(self._open_pcm)
                events.append(SegmentEvent(pcm=pcm, start=self._open_start, end=now, final=False))

        return events

    def _flush(self, end_seconds: float, *, final: bool) -> SegmentEvent:
        assert self._open_start is not None
        pcm = np.concatenate(self._open_pcm) if self._open_pcm else np.zeros(0, dtype=np.float32)
        ev = SegmentEvent(pcm=pcm, start=self._open_start, end=end_seconds, final=final)
        self._open_pcm = []
        self._open_start = None
        self._last_provisional_at = 0.0
        return ev
