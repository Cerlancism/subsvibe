"""Per-backend file-mode chunking profiles.

A chunk is one ASR request's worth of audio (see CoarseChunker in
./client/vad.py). How much audio one request should carry is a property of
the model serving it: Whisper-family models were trained on 30 s windows and
degrade past them, where a long-form backend may take minutes per request.
So the bounds are a table keyed by the server's backend name, each entry a
constant default overridable by a backend-prefixed variable
(``<BACKEND>_CHUNK_MAX_SECONDS`` / ``<BACKEND>_CHUNK_MIN_SECONDS``), matching
the convention of ./server/backends/config.py.

The session learns the backend from the server (sync_backend_with_server in
./client/transcribe.py) and looks its profile up here; a backend this table
does not know, and the --llm-asr path, get the Whisper-shaped default.
"""
from __future__ import annotations

from dataclasses import dataclass

from utils.env import env_float

# Whisper-shaped bounds: 30 s is the training window; below ~5 s the ASR
# starves for context and hallucinates. Chunks average well above the minimum
# since the chunker takes the latest boundary that fits.
DEFAULT_CHUNK_MAX_SECONDS = 30.0
DEFAULT_CHUNK_MIN_SECONDS = 5.0


@dataclass(frozen=True)
class ChunkProfile:
    """Bounds on one coarse chunk, in seconds from the cursor.

    `anchor_first_entry` says whether a chunk's first entry start is moved
    onto the VAD speech onset (anchor_first_entry in ./client/vad.py). The
    anchor exists because Whisper pulls a chunk's first cue early; it only
    ever applies on a backend whose own segment timings become entries."""
    max_seconds: float = DEFAULT_CHUNK_MAX_SECONDS
    min_seconds: float = DEFAULT_CHUNK_MIN_SECONDS
    anchor_first_entry: bool = True


def _profile(prefix: str, **overrides) -> ChunkProfile:
    base = ChunkProfile(**overrides)
    return ChunkProfile(
        max_seconds=env_float(f"{prefix}_CHUNK_MAX_SECONDS", base.max_seconds),
        min_seconds=env_float(f"{prefix}_CHUNK_MIN_SECONDS", base.min_seconds),
        anchor_first_entry=base.anchor_first_entry,
    )


DEFAULT_PROFILE = ChunkProfile()

# backend name (as GET /v1/backend reports it) -> profile.
CHUNK_PROFILES: dict[str, ChunkProfile] = {
    "faster-whisper": _profile("FASTER_WHISPER"),
    "qwen": _profile("QWEN"),
    "anime-whisper": _profile("ANIME_WHISPER"),
}


def chunk_profile(backend: str | None) -> ChunkProfile:
    """The profile for a backend name; the Whisper-shaped default for None or
    a name the table does not know."""
    if backend is None:
        return DEFAULT_PROFILE
    return CHUNK_PROFILES.get(backend, DEFAULT_PROFILE)
