from __future__ import annotations

from typing import Protocol

import numpy as np


class TranscriptionResult(Protocol):
    """
    Dict shape returned by transcribe_result:
      {
        "text": str,
        "language": str | None,
        "words": list[{"word": str, "start": float, "end": float}],
        "segments": list[{"text": str, "start": float, "end": float}],
      }
    Note: `words` uses the OpenAI `TranscriptionWord` schema ("word"),
    while `segments` uses "text" — same as the Whisper API.
    """


class Backend(Protocol):
    """Constructed with the model id to serve: ``Backend(model_id)``."""

    def set_model_id(self, model_id: str) -> None:
        """Select the model the next load() uses. The caller unloads first;
        a running worker keeps the model it was spawned with."""
        ...

    def load(self) -> None:
        """Load the primary model (lazy on the next request otherwise)."""
        ...

    def load_aligner(self) -> None:
        """Load the secondary/aligner model. No-op if the backend has no aligner."""
        ...

    def max_concurrency(self) -> int:
        """How many transcribe_result calls may run at once. The server runs
        up to this many in parallel and queues the rest in arrival order."""
        ...

    def resident_model(self) -> bool:
        """True when one model is loaded at a time: a call for another id
        needs set_model_id first, which the server does with nothing in
        flight, and every call then runs with `model_id` None. False when
        each call carries its own model (the remote backend) and
        set_model_id only chooses the model used when a call names none."""
        ...

    def transcribe_result(
        self,
        audio: np.ndarray,
        language: str | None,
        prompt: str | None,
        want_words: bool,
        model_id: str | None = None,
    ) -> dict:
        """Transcribe audio and return a normalised result dict.

        `want_words=True` requests word-level timestamps (the expensive bit on
        every backend). Segment-level timestamps are returned whenever the
        backend can produce them cheaply (always on faster-whisper; only
        alongside words on qwen).

        `model_id` names the model for this call on a backend without a
        resident model; None means the id set_model_id chose. A backend
        with a resident model is always called with None, after the
        switch."""
        ...

    def align(
        self,
        audio: np.ndarray,
        text: str,
        language: str | None,
    ) -> list[dict]:
        """Align externally-provided text against audio. Returns
        [{"word": str, "start": float, "end": float}, ...]."""
        ...

    def is_loaded(self) -> bool:
        """Return True if the primary model is currently loaded."""
        ...

    def unload(self) -> None:
        """Unload the primary model and free its memory."""
        ...

    def has_secondary(self) -> bool:
        """Return True if this backend has a secondary model that can be unloaded."""
        ...

    def unload_secondary(self) -> None:
        """Unload the secondary model (e.g. aligner). No-op if none loaded."""
        ...

    def profile(self, model_id: str | None = None) -> dict:
        """How a client should feed this backend, for its *current* model id
        or, on a backend without a resident model, for `model_id`.
        Must not load anything. Keys:
          max_input_seconds: float - longest audio one request may carry.
          long_form: bool - the model transcribes minutes to an hour per
            request (client cuts evenly near its cap) rather than Whisper's
            30 s windows (client cuts at the latest speech onset in a short
            window).
          entries_from: "segments" | "words" - whether the backend's own
            `segments` are subtitle-shaped as returned, or entries must be
            built from `words`.
          max_lossless_bytes: int | None - optional; the largest FLAC
            encoding of a request's audio the backend forwards losslessly
            (it re-encodes lossy past it). A long-form client cuts chunks
            to stay under it. None or absent: no byte bound."""
        ...
