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

    def transcribe_result(
        self,
        audio: np.ndarray,
        language: str | None,
        prompt: str | None,
        want_words: bool,
    ) -> dict:
        """Transcribe audio and return a normalised result dict.

        `want_words=True` requests word-level timestamps (the expensive bit on
        every backend). Segment-level timestamps are returned whenever the
        backend can produce them cheaply (always on faster-whisper; only
        alongside words on qwen)."""
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
