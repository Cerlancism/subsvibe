from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np


class TranscriptionResult(Protocol):
    """
    Dict shape returned by transcribe_result:
      {
        "text": str,
        "language": str | None,
        "words": list[{"text": str, "start": float, "end": float}],
        "segments": list[{"text": str, "start": float, "end": float}],
      }
    """


class Backend(Protocol):
    def load(self) -> None:
        """Load the primary model. Called once at server startup."""
        ...

    def load_aligner(self) -> None:
        """Load the secondary/aligner model. No-op if the backend has no aligner."""
        ...

    def transcribe_result(
        self,
        audio: np.ndarray,
        language: str | None,
        prompt: str | None,
        return_timestamps: bool,
    ) -> dict:
        """Transcribe audio and return a normalised result dict."""
        ...

    def transcribe_stream(
        self,
        audio: np.ndarray,
        language: str | None,
        prompt: str | None,
        return_timestamps: bool,
    ) -> Iterator[tuple]:
        """
        Yield (chunk_text, chunk_audio, offset, lang) per chunk.
        Final item: (None, words, segments, full_text).
        """
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
