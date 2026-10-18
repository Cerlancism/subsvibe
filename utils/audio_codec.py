"""Lossless and lossy encoders for 16 kHz mono PCM, through PyAV's bundled
FFmpeg (libopus and flac ship in the wheel; no ffmpeg binary is needed).

Shared on purpose: the remote backend (``./server/backends/remote.py``)
sends what `encode_flac` / `encode_opus` produce, and the client's long-form
chunker (``./client/vad.py``) measures `flac_byte_timeline` on the audio it
is about to cut so that a chunk stays under the request budget the server
reports. Both sides therefore see the same bytes for the same samples.
"""
from __future__ import annotations

import io

import av
import numpy as np

SAMPLE_RATE = 16000


def to_int16(pcm: np.ndarray) -> np.ndarray:
    """float32 in [-1, 1] -> int16; int16 passes through."""
    if pcm.dtype == np.int16:
        return pcm
    return (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)


def _encode(
    pcm: np.ndarray,
    container: str,
    codec: str,
    timeline: list[tuple[float, int]] | None = None,
    **stream_attrs,
) -> bytes:
    """Encode the whole of `pcm` as one frame; PyAV re-frames it to the
    codec's block size. With `timeline`, append (seconds encoded so far,
    bytes written so far) after every packet, so the caller can read how
    much audio fits in a byte budget."""
    buf = io.BytesIO()
    samples = 0
    with av.open(buf, "w", format=container) as out:
        stream = out.add_stream(codec, rate=SAMPLE_RATE)
        stream.layout = "mono"
        stream.format = "s16"
        for name, value in stream_attrs.items():
            setattr(stream, name, value)
        frame = av.AudioFrame.from_ndarray(to_int16(pcm).reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = SAMPLE_RATE
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            if not packet.size:
                continue  # the encoder's end-of-stream marker carries no audio
            out.mux(packet)
            if timeline is not None:
                samples += int(packet.duration or 0)
                timeline.append((samples / SAMPLE_RATE, buf.tell()))
    return buf.getvalue()


def encode_flac(pcm: np.ndarray) -> bytes:
    return _encode(pcm, "flac", "flac")


def encode_opus(pcm: np.ndarray, bitrate: int) -> bytes:
    """Opus VBR in an Ogg container, tuned for speech (`voip`)."""
    return _encode(pcm, "ogg", "libopus", bit_rate=bitrate,
                   options={"application": "voip", "vbr": "on"})


def flac_byte_timeline(pcm: np.ndarray) -> list[tuple[float, int]]:
    """(seconds, cumulative FLAC bytes) after each encoded block, rising in
    both. FLAC blocks are independent, so the bytes a prefix of `pcm`
    encodes to is the entry at that prefix's end, give or take one block
    (~72 ms) and the fixed stream header."""
    timeline: list[tuple[float, int]] = []
    _encode(pcm, "flac", "flac", timeline=timeline)
    return timeline


def seconds_within(timeline: list[tuple[float, int]], budget_bytes: int) -> float:
    """Longest prefix, in seconds, whose FLAC encoding stays under
    `budget_bytes`; 0.0 when not even the first block fits."""
    fit = 0.0
    for seconds, total in timeline:
        if total > budget_bytes:
            break
        fit = seconds
    return fit
