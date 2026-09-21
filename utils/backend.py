"""The ASR backend name shared by client and server.

The server owns the active backend and swaps it at runtime; the client asks
which one is live before a session, because entry post-processing differs per
backend (segment-trust vs word-aligner). Both sides still need a fallback for
the case where that question cannot be asked - the server before any switch,
the client when the server is unreachable - and a disagreement there silently
mismatches post-processing against the model that produced the text. So the
fallback lives here, imported by both, rather than as a literal in each.

faster-whisper is the value because it is what scripts/env.example.sh exports
and what server/README.md documents: the backend that runs without a GPU, so
an unconfigured install starts on the one that cannot fail for lack of VRAM.
"""
from __future__ import annotations

DEFAULT_BACKEND = "faster-whisper"
