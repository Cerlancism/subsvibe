from __future__ import annotations

import os

import numpy as np

from backends.base import Backend

TRANSCRIPT_BACKEND = os.environ.get("TRANSCRIPT_BACKEND", "qwen")

_BACKEND_DEFAULT_MODEL_IDS = {
    "qwen": "Qwen/Qwen3-ASR-1.7B",
    "faster-whisper": "Systran/faster-whisper-large-v3",
    "anime-whisper": "litagin/anime-whisper",
}


def default_model_id() -> str:
    """Default TRANSCRIPT_MODEL_ID for the configured backend."""
    return _BACKEND_DEFAULT_MODEL_IDS.get(TRANSCRIPT_BACKEND, "")


def resolved_model_id() -> str:
    return os.environ.get("TRANSCRIPT_MODEL_ID") or default_model_id()


_backend: Backend | None = None


def _get_backend() -> Backend:
    global _backend
    if _backend is not None:
        return _backend
    if TRANSCRIPT_BACKEND == "qwen":
        from backends.qwen import QwenBackend
        _backend = QwenBackend()
    elif TRANSCRIPT_BACKEND == "faster-whisper":
        from backends.faster_whisper import FasterWhisperBackend
        _backend = FasterWhisperBackend()
    elif TRANSCRIPT_BACKEND == "anime-whisper":
        from backends.anime_whisper import AnimeWhisperBackend
        _backend = AnimeWhisperBackend()
    else:
        raise ValueError(
            f"unknown TRANSCRIPT_BACKEND: {TRANSCRIPT_BACKEND!r} "
            "(supported: 'qwen', 'faster-whisper', 'anime-whisper')"
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
    want_words: bool = False,
) -> dict:
    return _get_backend().transcribe_result(audio, language, prompt, want_words)


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
