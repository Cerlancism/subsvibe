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
        """How many transcribe_result calls may run at once. 1 means one
        model is resident and calls are serialised: the server switches the
        active model (set_model_id) before a call that names another one.
        Above 1 the backend holds no model state per call: the server runs
        up to this many calls in parallel, queues the rest, and passes each
        call's model as `model_id` without switching anything."""
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

        `model_id` names the model for this call on a backend whose
        max_concurrency() is above 1; None means the id set_model_id chose.
        A serial backend is always called with None, after the switch."""
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
        or, on a backend whose max_concurrency() is above 1, for `model_id`.
        Must not load anything. Keys:
          max_input_seconds: float - longest audio one request may carry.
          long_form: bool - the model transcribes minutes to an hour per
            request (client cuts evenly near its cap) rather than Whisper's
            30 s windows (client cuts at the latest speech onset in a short
            window).
          entries_from: "segments" | "words" - whether the backend's own
            `segments` are subtitle-shaped as returned, or entries must be
            built from `words`."""
        ...
