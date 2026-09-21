from __future__ import annotations

import logging
import os

import numpy as np

from backends.base import Backend
from utils.backend import DEFAULT_BACKEND

log = logging.getLogger("subsvibe.model")

# DEFAULT_BACKEND lives in utils/ because the client needs the same fallback
# and cannot import from server/. Sharing it moved the server's
# unset-TRANSCRIPT_BACKEND default from "qwen" to faster-whisper, which is what
# scripts/env.example.sh and server/README.md always said.

_BACKEND_DEFAULT_MODEL_IDS = {
    "qwen": "Qwen/Qwen3-ASR-1.7B",
    "faster-whisper": "Systran/faster-whisper-large-v3",
    "anime-whisper": "litagin/anime-whisper",
}

# Derived rather than listed separately: a name accepted here that _get_backend
# cannot construct would pass validation, dispose the outgoing backend, and
# then fail every later request with no way back.
SUPPORTED_BACKENDS = tuple(_BACKEND_DEFAULT_MODEL_IDS)

# Startup configuration, captured once at import. TRANSCRIPT_MODEL_ID is
# rewritten on every switch (worker children read their id from it at spawn
# time), so these are the only surviving record of how the process was launched.
_INITIAL_BACKEND = os.environ.get("TRANSCRIPT_BACKEND", DEFAULT_BACKEND)
_INITIAL_MODEL_ID = os.environ.get("TRANSCRIPT_MODEL_ID", "")

_active_backend: str = _INITIAL_BACKEND
_active_model_id: str | None = None
_backend: Backend | None = None

# Model id last selected on each backend, so switching away and back restores
# what was in use rather than snapping to the backend's default. An unvisited
# backend falls back to the id a freshly-started server would have used (see
# model_id_for).
_last_model_ids: dict[str, str] = {}


def active_backend() -> str:
    """The backend currently serving requests. Starts at TRANSCRIPT_BACKEND
    and changes when a client calls switch_backend."""
    return _active_backend


def default_model_id(backend: str | None = None) -> str:
    """Default TRANSCRIPT_MODEL_ID for a backend (the active one by default)."""
    return _BACKEND_DEFAULT_MODEL_IDS.get(backend or _active_backend, "")


def resolved_model_id() -> str:
    """The active model id. Starts at TRANSCRIPT_MODEL_ID (or the backend
    default) and changes when a client requests a different model."""
    global _active_model_id
    if _active_model_id is None:
        _active_model_id = _INITIAL_MODEL_ID or default_model_id()
    return _active_model_id


def _publish_model_id(model_id: str) -> None:
    """Record the active model id and mirror it into the environment, which is
    where worker children read it from at spawn time."""
    global _active_model_id
    _active_model_id = model_id
    if model_id:
        os.environ["TRANSCRIPT_MODEL_ID"] = model_id
    else:
        os.environ.pop("TRANSCRIPT_MODEL_ID", None)


def switch_model(model_id: str) -> None:
    """Switch the active model within the current backend.

    Unloads the current model if loaded; the new one lazy-loads on the next
    transcription. The choice is remembered for this backend, so a later
    switch away and back returns to it."""
    backend = _get_backend()
    if backend.is_loaded():
        backend.unload()
    _publish_model_id(model_id)
    _last_model_ids[_active_backend] = model_id


def model_id_for(backend: str) -> str:
    """The model id to select when switching to `backend`: whatever was last
    in use there this run, else the id a freshly-started server would have
    used. TRANSCRIPT_MODEL_ID counts only for the backend it was configured
    alongside; every other backend falls back to its own default."""
    last = _last_model_ids.get(backend)
    if last:
        return last
    if backend == _INITIAL_BACKEND and _INITIAL_MODEL_ID:
        return _INITIAL_MODEL_ID
    return default_model_id(backend)


def switch_backend(name: str, model_id: str | None = None) -> None:
    """Swap the active ASR backend at runtime, raising ValueError on an
    unknown name before any state is touched.

    Disposes the outgoing backend exactly as POST /v1/model/unload does —
    aligner first, then the ASR worker — so the OS reclaims its VRAM before
    the new backend spawns its own. Disposal is unconditional rather than
    guarded by is_loaded(): a worker can outlive a failed load, and every
    unload path bottoms out in ModelWorker.stop(), a no-op when idle.

    `model_id` overrides the selection; without one see model_id_for. The new
    backend's model lazy-loads on the next request."""
    global _backend, _active_backend
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unknown backend: {name!r} (supported: {', '.join(SUPPORTED_BACKENDS)})"
        )

    # Remember the outgoing backend and its model before either is
    # overwritten, so switching back to it later restores this selection.
    previous = _active_backend
    outgoing_model = resolved_model_id()
    if outgoing_model:
        _last_model_ids[previous] = outgoing_model

    # Publish the new identity before disposing the old backend, holding the
    # outgoing object in a local. Not every lazy reader holds the lifecycle
    # lock (GET /v1/health does not), so clearing _backend while
    # _active_backend still named the old one would let such a reader rebuild
    # the outgoing backend under the incoming name - every later transcription
    # silently running the old model. This order can only rebuild the incoming
    # one, and construction is parent-side, so no worker spawns before the
    # disposal below frees the outgoing VRAM.
    outgoing_backend = _backend
    _active_backend = name
    _backend = None
    # Not mirrored into os.environ the way the model id is: the child entry
    # point is chosen parent-side in _get_backend, so nothing downstream reads
    # TRANSCRIPT_BACKEND at spawn time, and writing it would only create a
    # second apparent source of truth beside _active_backend.
    selected = model_id or model_id_for(name)
    _publish_model_id(selected)
    _last_model_ids[name] = selected

    if outgoing_backend is not None:
        log.info("disposing %s backend workers", previous)
        # try/finally because _backend no longer references this object: a
        # throw from the aligner unload would otherwise strand the ASR worker
        # child with no handle left to kill it, holding its VRAM for the rest
        # of the run.
        try:
            outgoing_backend.unload_secondary()
        finally:
            outgoing_backend.unload()

    log.info("active backend is now %s (model %s)", name, resolved_model_id() or "<default>")


def _get_backend() -> Backend:
    global _backend
    if _backend is not None:
        return _backend
    if _active_backend == "qwen":
        from backends.qwen import QwenBackend
        _backend = QwenBackend()
    elif _active_backend == "faster-whisper":
        from backends.faster_whisper import FasterWhisperBackend
        _backend = FasterWhisperBackend()
    elif _active_backend == "anime-whisper":
        from backends.anime_whisper import AnimeWhisperBackend
        _backend = AnimeWhisperBackend()
    else:
        raise ValueError(
            f"unknown TRANSCRIPT_BACKEND: {_active_backend!r} "
            f"(supported: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)})"
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
