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

Two cutting strategies exist (see CoarseChunker in ./client/vad.py). The
short form cuts at the latest speech onset inside the bounds, so a 30 s
model gets as much audio per request as fits. The long form is for a model
that takes up to an hour per request: the remaining audio is split into
equal parts no longer than the maximum, so 61 minutes become two of ~30
rather than 60 + 1, and only the tail of each part is scanned for a seam.
The server reports which one its active model needs (`long_form` on
GET /v1/backend) and, for a backend that re-encodes, the FLAC size a
request stays under to be forwarded losslessly (`max_lossless_bytes`),
which the long form also cuts by; a backend's row here sets the default
and the bounds.
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
    # Even split near the maximum (long form) rather than the latest onset
    # inside the bounds (short form); see the module docstring.
    long_form: bool = False
    # Long form only: the shortest part worth keeping lossless. When the
    # server's FLAC budget would cut a part shorter than this, the part is
    # cut by time instead and the server sends it lossy.
    lossless_min_seconds: float = 0.0


# Long-form bounds. The maximum sits a couple of minutes under the hour the
# remote backend accepts so that VBR Opus at 80 kbps stays inside its 50 MB
# request body; the lossless budget the server reports usually binds first
# (about 27 min of speech as FLAC). The minimum only matters for a file
# barely over one chunk.
LONG_FORM_CHUNK_MAX_SECONDS = 3480.0
LONG_FORM_CHUNK_MIN_SECONDS = 600.0
# Below this, the request overhead of many small lossless parts (a re-heard
# trailing entry and a round trip each) costs more than Opus at 80 kbps,
# which measured within noise of lossless. FLAC on 16 kHz mono cannot
# compress worse than raw PCM (~19 min per budget), so at the default cap
# this only binds when OPENROUTER_JSON_CAP_BYTES is lowered.
LONG_FORM_LOSSLESS_MIN_SECONDS = 900.0


def _profile(prefix: str, **overrides) -> ChunkProfile:
    base = ChunkProfile(**overrides)
    return ChunkProfile(
        max_seconds=env_float(f"{prefix}_CHUNK_MAX_SECONDS", base.max_seconds),
        min_seconds=env_float(f"{prefix}_CHUNK_MIN_SECONDS", base.min_seconds),
        anchor_first_entry=base.anchor_first_entry,
        long_form=base.long_form,
        lossless_min_seconds=env_float(
            f"{prefix}_CHUNK_LOSSLESS_MIN_SECONDS", base.lossless_min_seconds),
    )


DEFAULT_PROFILE = ChunkProfile()
LONG_FORM_PROFILE = ChunkProfile(
    max_seconds=LONG_FORM_CHUNK_MAX_SECONDS,
    min_seconds=LONG_FORM_CHUNK_MIN_SECONDS,
    anchor_first_entry=False,
    long_form=True,
    lossless_min_seconds=LONG_FORM_LOSSLESS_MIN_SECONDS,
)

# backend name (as GET /v1/backend reports it) -> profile.
CHUNK_PROFILES: dict[str, ChunkProfile] = {
    "faster-whisper": _profile("FASTER_WHISPER"),
    "qwen": _profile("QWEN"),
    "anime-whisper": _profile("ANIME_WHISPER"),
    # Remote, model per request: mai-transcribe-2 is long-form; a whisper id
    # on the same backend is reported short-form by the server and takes the
    # Whisper-shaped default below.
    "openrouter": _profile(
        "OPENROUTER",
        max_seconds=LONG_FORM_CHUNK_MAX_SECONDS,
        min_seconds=LONG_FORM_CHUNK_MIN_SECONDS,
        anchor_first_entry=False,
        long_form=True,
        lossless_min_seconds=LONG_FORM_LOSSLESS_MIN_SECONDS,
    ),
}


def chunk_profile(backend: str | None, long_form: bool | None = None) -> ChunkProfile:
    """The profile for a backend name; the Whisper-shaped default for None or
    a name the table does not know.

    `long_form` is what the server reported for its active model, when it
    did: it overrides the row's strategy, so a long-form backend serving a
    short-form model falls back to the default bounds and the other way round
    takes the generic long-form bounds."""
    if backend is None:
        return DEFAULT_PROFILE
    profile = CHUNK_PROFILES.get(backend, DEFAULT_PROFILE)
    if long_form is None or long_form == profile.long_form:
        return profile
    return LONG_FORM_PROFILE if long_form else DEFAULT_PROFILE
