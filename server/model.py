"""Backend registry: every supported ASR backend, side by side.

A request names its backend (and model); nothing is "active" server-wide.
Each backend object is built on first use and kept for the run, holding
whatever it has loaded: a local backend one resident model (a worker child)
plus possibly its aligner, the remote backend nothing. The model a backend
uses when a request names none is its <BACKEND>_MODEL_ID from
backends/config.py until a request or a load selects another.

Every function takes the backend name first; `resolve` turns the client's
optional name into a known one (the server default when absent).
"""
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
# and cannot import from server/.

# Per-backend default model ids (and their <BACKEND>_MODEL_ID overrides) live
# in backends/config.py beside the rest of the backend tuning.
#
# SUPPORTED_BACKENDS is derived from that table rather than listed separately:
# a name accepted here that _get_backend cannot construct would pass
# validation and then fail every request naming it.
SUPPORTED_BACKENDS = tuple(DEFAULT_MODEL_IDS)

# The backend a request gets when it names none, captured once at import.
DEFAULT_BACKEND_NAME = os.environ.get("TRANSCRIPT_BACKEND", DEFAULT_BACKEND)

# backend name -> (module, class) constructing it. Imported lazily: each
# module pulls in its own framework, and only the ones in use should load.
_BACKEND_CLASSES: dict[str, tuple[str, str]] = {
    "faster-whisper": ("backends.faster_whisper", "FasterWhisperBackend"),
    "qwen": ("backends.qwen", "QwenBackend"),
    "anime-whisper": ("backends.anime_whisper", "AnimeWhisperBackend"),
    "openrouter": ("backends.remote", "RemoteBackend"),
}

_backends: dict[str, Backend] = {}
# Model each backend currently uses for requests that name none. Starts at
# the backend's default and changes on switch_model.
_model_ids: dict[str, str] = {}


def resolve(name: str | None) -> str:
    """The backend a request means: `name` when given and known, else the
    server default. Raises ValueError on an unknown name."""
    chosen = (name or "").strip() or DEFAULT_BACKEND_NAME
    if chosen not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unknown backend: {chosen!r} (supported: {', '.join(SUPPORTED_BACKENDS)})"
        )
    return chosen


def default_model_id(backend: str) -> str:
    """The <BACKEND>_MODEL_ID a backend starts on."""
    return DEFAULT_MODEL_IDS.get(backend, "")


def resolved_model_id(backend: str) -> str:
    """The model `backend` uses for requests that name none."""
    return _model_ids.get(backend) or default_model_id(backend)


def switch_model(backend: str, model_id: str) -> None:
    """Select `model_id` on `backend`.

    On a backend with a resident model (every local one) this unloads the
    current model if loaded; the new one lazy-loads on the next request.
    The caller makes sure nothing is in flight on that backend. On a backend
    that takes the model per request (the remote one) this only changes the
    model used by requests that name none; calls in flight keep theirs."""
    obj = _get_backend(backend)
    if obj.resident_model() and obj.is_loaded():
        obj.unload()
    _model_ids[backend] = model_id
    obj.set_model_id(model_id)


def _get_backend(backend: str) -> Backend:
    obj = _backends.get(backend)
    if obj is not None:
        return obj
    entry = _BACKEND_CLASSES.get(backend)
    if entry is None:
        raise ValueError(
            f"unknown backend: {backend!r} (supported: {', '.join(SUPPORTED_BACKENDS)})"
        )
    module_name, class_name = entry
    cls = getattr(importlib.import_module(module_name), class_name)
    obj = cls(resolved_model_id(backend))
    _backends[backend] = obj
    return obj


def constructed() -> list[str]:
    """Backends built so far this run, the only ones that can hold anything."""
    return list(_backends)


def profile(backend: str, model_id: str | None = None) -> dict:
    """The backend's request profile for its current model, or for
    `model_id` where it takes the model per request: see Backend.profile.
    Constructs the backend object but loads nothing."""
    return _get_backend(backend).profile(model_id)


def max_concurrency(backend: str) -> int:
    """How many transcriptions the backend runs at once: see
    Backend.max_concurrency. Constructs the backend object but loads
    nothing."""
    return _get_backend(backend).max_concurrency()


def resident_model(backend: str) -> bool:
    """Whether the backend holds one model at a time (switch_model before a
    request naming another) or takes the model per call: see
    Backend.resident_model."""
    return _get_backend(backend).resident_model()


def load_model(backend: str) -> None:
    _get_backend(backend).load()


def unload_model(backend: str) -> None:
    _get_backend(backend).unload()


def is_model_loaded(backend: str) -> bool:
    return _get_backend(backend).is_loaded()


def transcribe_result(
    backend: str,
    audio: np.ndarray,
    language: str | None = None,
    prompt: str | None = None,
    want_words: bool = False,
    model_id: str | None = None,
) -> dict:
    """See Backend.transcribe_result. `model_id` only means something on a
    backend without a resident model; one with a resident model is switched
    with switch_model first and called with None."""
    return _get_backend(backend).transcribe_result(audio, language, prompt, want_words, model_id)


def has_secondary(backend: str) -> bool:
    return _get_backend(backend).has_secondary()


def unload_secondary(backend: str) -> None:
    _get_backend(backend).unload_secondary()


def load_aligner(backend: str) -> None:
    _get_backend(backend).load_aligner()


def align(
    backend: str,
    audio: np.ndarray,
    text: str,
    language: str | None = None,
) -> list[dict]:
    return _get_backend(backend).align(audio, text, language)
