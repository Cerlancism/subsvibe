from __future__ import annotations

import importlib
import logging
import os

import numpy as np

from backends.base import Backend
from backends.config import DEFAULT_MODEL_IDS
from utils.backend import DEFAULT_BACKEND

log = logging.getLogger("subsvibe.model")

# DEFAULT_BACKEND lives in utils/ because the client needs the same fallback
# and cannot import from server/. Sharing it moved the server's
# unset-TRANSCRIPT_BACKEND default from "qwen" to faster-whisper, which is what
# scripts/env.example.sh and server/README.md always said.

# Per-backend default model ids (and their <BACKEND>_MODEL_ID overrides) live
# in backends/config.py beside the rest of the backend tuning.
#
# SUPPORTED_BACKENDS is derived from that table rather than listed separately:
# a name accepted here that _get_backend cannot construct would pass
# validation, dispose the outgoing backend, and then fail every later request
# with no way back.
SUPPORTED_BACKENDS = tuple(DEFAULT_MODEL_IDS)

# Startup backend, captured once at import; the model it starts on is that
# backend's <BACKEND>_MODEL_ID from backends/config.py.
_INITIAL_BACKEND = os.environ.get("TRANSCRIPT_BACKEND", DEFAULT_BACKEND)

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
    """The <BACKEND>_MODEL_ID a backend starts on (the active one by default)."""
    return DEFAULT_MODEL_IDS.get(backend or _active_backend, "")


def resolved_model_id() -> str:
    """The active model id. Starts at the backend default and changes when a
    client requests a different model."""
    global _active_model_id
    if _active_model_id is None:
        _active_model_id = default_model_id()
    return _active_model_id


def switch_model(model_id: str) -> None:
    """Switch the active model within the current backend.

    Unloads the current model if loaded; the new one lazy-loads on the next
    transcription. The choice is remembered for this backend, so a later
    switch away and back returns to it. On a backend that takes the model
    per request (max_concurrency above 1) this only sets the model used by
    requests that name none; calls in flight keep theirs."""
    global _active_model_id
    backend = _get_backend()
    if backend.is_loaded():
        backend.unload()
    _active_model_id = model_id
    backend.set_model_id(model_id)
    _last_model_ids[_active_backend] = model_id


def model_id_for(backend: str) -> str:
    """The model id to select when switching to `backend`: whatever was last
    in use there this run, else its <BACKEND>_MODEL_ID default."""
    return _last_model_ids.get(backend) or default_model_id(backend)


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
    global _backend, _active_backend, _active_model_id
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
    # Neither value is mirrored into os.environ: the child entry point is
    # chosen parent-side in _get_backend and receives its model id as an
    # argument at spawn, so the environment holds only the startup values.
    selected = model_id or model_id_for(name)
    _active_model_id = selected
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


# backend name -> (module, class) constructing it. Imported lazily: each
# module pulls in its own framework, and only the active one should load.
_BACKEND_CLASSES: dict[str, tuple[str, str]] = {
    "faster-whisper": ("backends.faster_whisper", "FasterWhisperBackend"),
    "qwen": ("backends.qwen", "QwenBackend"),
    "anime-whisper": ("backends.anime_whisper", "AnimeWhisperBackend"),
    "openrouter": ("backends.remote", "RemoteBackend"),
}


def _get_backend() -> Backend:
    global _backend
    if _backend is not None:
        return _backend
    entry = _BACKEND_CLASSES.get(_active_backend)
    if entry is None:
        raise ValueError(
            f"unknown TRANSCRIPT_BACKEND: {_active_backend!r} "
            f"(supported: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)})"
        )
    module_name, class_name = entry
    cls = getattr(importlib.import_module(module_name), class_name)
    _backend = cls(resolved_model_id())
    return _backend


def profile(model_id: str | None = None) -> dict:
    """The active backend's request profile for its current model, or for
    `model_id` where the backend takes the model per request: see
    Backend.profile. Constructs the backend object but loads nothing."""
    return _get_backend().profile(model_id)


def max_concurrency() -> int:
    """How many transcriptions the active backend runs at once: see
    Backend.max_concurrency. Constructs the backend object but loads
    nothing."""
    return _get_backend().max_concurrency()


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
    model_id: str | None = None,
) -> dict:
    """See Backend.transcribe_result. `model_id` only means something on a
    backend whose max_concurrency is above 1; a serial backend is switched
    with switch_model first and called with None."""
    return _get_backend().transcribe_result(audio, language, prompt, want_words, model_id)


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
