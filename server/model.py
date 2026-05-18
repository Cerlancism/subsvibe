from __future__ import annotations

import os

import numpy as np

from backends.base import Backend

TRANSCRIPT_BACKEND = os.environ.get("TRANSCRIPT_BACKEND", "qwen")

_backend: Backend | None = None


def _get_backend() -> Backend:
    global _backend
    if _backend is not None:
        return _backend
    if TRANSCRIPT_BACKEND == "qwen":
        from backends.qwen import QwenBackend
        _backend = QwenBackend()
    elif TRANSCRIPT_BACKEND in {"faster-whisper", "faster_whisper"}:
        from backends.faster_whisper import FasterWhisperBackend
        _backend = FasterWhisperBackend()
    else:
        raise ValueError(
            f"unknown TRANSCRIPT_BACKEND: {TRANSCRIPT_BACKEND!r} (supported: 'qwen', 'faster-whisper')"
        )
    return _backend


def load_model() -> None:
    _get_backend().load()


def unload_model() -> None:
    _get_backend().unload()


def is_model_loaded() -> bool:
    return _get_backend().is_loaded()


def transcribe_result(
    audio: np.ndarray,
    language: str | None = None,
    prompt: str | None = None,
    return_timestamps: bool = False,
) -> dict:
    return _get_backend().transcribe_result(audio, language, prompt, return_timestamps)


def has_secondary() -> bool:
    return _get_backend().has_secondary()


def unload_secondary() -> None:
    _get_backend().unload_secondary()


def load_aligner() -> None:
    _get_backend().load_aligner()


def align(
    audio: np.ndarray,
    text: str,
    language: str | None = None,
) -> list[dict]:
    return _get_backend().align(audio, text, language)
