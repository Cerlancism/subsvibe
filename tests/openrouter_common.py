"""Shared helpers for the manual OpenRouter probes in this directory.

Not a test. Imported by test_openrouter_whisper.py (30 s window probes +
chunked file mode) and test_openrouter_longform.py (mai-transcribe-2 one-shot /
split). Everything that talks to OpenRouter reads its key and base URL from
the env that scripts/env.sh exports (OPENROUTER_* via
scripts/backends/env.openrouter.sh).
"""
from __future__ import annotations

import io
import os
import sys
import wave
from pathlib import Path

import av
import numpy as np
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "client")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.env import env_float, env_str  # noqa: E402
from utils.text import fold_word_marks  # noqa: E402

SAMPLE_RATE = 16000

OPENROUTER_BASE_URL = env_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
# Upstream providers time out at 60 s per request. Default sits above that so
# a request that dies upstream comes back as OpenRouter's error body rather
# than our own timeout; never retry (a retry re-uploads and re-bills).
OPENROUTER_TIMEOUT_SECONDS = env_float("OPENROUTER_TIMEOUT_SECONDS", 180.0)

DEFAULT_INPUT = ROOT / "tests" / "data" / "sample_15mins_ja.mp4"
# Per-request timeout for ~30 s chunks in chunked file mode.
CHUNK_TIMEOUT_SECONDS = 90.0


def require_key() -> None:
    if not OPENROUTER_API_KEY:
        raise SystemExit(
            "error: OPENROUTER_API_KEY is empty "
            "(scripts/backends/env.openrouter.sh maps it from OPENROUTER_KEY_GENERIC)"
        )


def make_client(timeout: float = OPENROUTER_TIMEOUT_SECONDS) -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, timeout=timeout, max_retries=0)


def out_base(path: Path) -> Path:
    """`tests/data/<stem>` - outputs are named `<stem>.out.openrouter.<...>`."""
    return path.with_suffix("")


# --- audio ---------------------------------------------------------------


def _iter_frames(path: Path):
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
        for packet in container.demux(stream):
            for frame in packet.decode():
                for resampled in resampler.resample(frame):
                    yield resampled.to_ndarray()[0].astype(np.float32)
        for resampled in resampler.resample(None):
            yield resampled.to_ndarray()[0].astype(np.float32)


def decode_file(path: Path) -> np.ndarray:
    """Whole file as mono float32 16 kHz PCM."""
    chunks = list(_iter_frames(path))
    if not chunks:
        raise SystemExit(f"error: no audio decoded from {path}")
    return np.concatenate(chunks)


def decode_window(path: Path, offset: float, seconds: float) -> np.ndarray:
    """Mono float32 16 kHz PCM for [offset, offset + seconds) of the file."""
    start = int(offset * SAMPLE_RATE)
    end = start + int(seconds * SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    have = 0
    for arr in _iter_frames(path):
        chunks.append(arr)
        have += len(arr)
        if have >= end:
            break
    pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    window = pcm[start:end]
    if len(window) == 0:
        raise SystemExit(f"error: no audio at offset {offset}s in {path}")
    return window


def encode_wav(pcm: np.ndarray) -> bytes:
    pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def encode_flac(pcm: np.ndarray) -> bytes:
    """FLAC via PyAV: 16-bit mono, the same samples the WAV carries."""
    pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with av.open(buf, "w", format="flac") as container:
        stream = container.add_stream("flac", rate=SAMPLE_RATE)
        stream.layout = "mono"
        stream.format = "s16"
        frame = av.AudioFrame.from_ndarray(pcm_int16.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = SAMPLE_RATE
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buf.getvalue()


def quiet_seam(pcm: np.ndarray, target_s: float, radius_s: float = 20.0, win_s: float = 0.5) -> float:
    """Centre of the quietest `win_s` window within +-radius_s of target_s."""
    hop = int(0.02 * SAMPLE_RATE)
    lo = max(0, int((target_s - radius_s) * SAMPLE_RATE))
    hi = min(len(pcm), int((target_s + radius_s) * SAMPLE_RATE))
    frames = (hi - lo) // hop
    if frames < 2:
        return target_s
    seg = pcm[lo : lo + frames * hop].reshape(frames, hop)
    rms = np.sqrt(np.mean(seg * seg, axis=1))
    k = max(1, int(win_s / 0.02))
    moving = np.convolve(rms, np.ones(k) / k, mode="valid")
    best = int(np.argmin(moving))
    return (lo + (best + k / 2) * hop) / SAMPLE_RATE


# --- word normalisation ---------------------------------------------------


def _is_punctuation(token: str) -> bool:
    import unicodedata

    return all(unicodedata.category(c).startswith(("P", "Z", "S")) for c in token)


def normalize_words(words: list) -> list[dict]:
    """fold_word_marks (./utils/text.py) over SDK word objects or dicts:
    letters only in `word`, every mark in the previous word's `trailing`.
    What the `openrouter` server backend does before handing words to the
    client. Returns plain dicts."""
    def as_dict(w) -> dict:
        if isinstance(w, dict):
            return dict(w)
        if hasattr(w, "model_dump"):
            return w.model_dump()
        return dict(vars(w))

    return fold_word_marks([as_dict(w) for w in words])


# Below this much audio a request is pointless; whisper providers answer an
# empty WAV with a bare 400. transcribe_file can emit a 0.0 s file-tail chunk
# (seen 2026-10-11 on sample_15mins_ja: "chunk 48 [15:18.29-15:18.29] 0.0s
# file-tail"), which the local server tolerates and a remote one does not.
MIN_REQUEST_SECONDS = 0.1


def _wav_seconds(file_arg) -> float | None:
    """Duration of the WAV bytes transcribe_file passes as `file`, else None."""
    data = file_arg[1] if isinstance(file_arg, tuple) and len(file_arg) >= 2 else file_arg
    if not isinstance(data, (bytes, bytearray)):
        return None
    try:
        with wave.open(io.BytesIO(bytes(data)), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate() or SAMPLE_RATE)
    except Exception:
        return None


class NormalizingTranscriptions:
    """`client.audio.transcriptions` stand-in whose create() normalises words,
    skips empty audio, and tallies OpenRouter's per-request `usage`."""

    def __init__(self, inner, stats: dict) -> None:
        self._inner = inner
        self._stats = stats

    def create(self, **kwargs):
        seconds = _wav_seconds(kwargs.get("file"))
        if seconds is not None and seconds < MIN_REQUEST_SECONDS:
            self._stats["skipped_empty"] += 1
            from openai.types.audio import Transcription

            return Transcription(text="", words=[], segments=[])
        result = self._inner.create(**kwargs)
        self._stats["requests"] += 1
        usage = getattr(result, "usage", None)
        if usage is not None:
            get = usage.get if isinstance(usage, dict) else lambda k, d=None: getattr(usage, k, d)
            self._stats["cost"] += float(get("cost", 0) or 0)
            self._stats["seconds"] += float(get("seconds", 0) or 0)
        words = getattr(result, "words", None)
        if words:
            result.words = normalize_words(list(words))
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


class NormalizingAudio:
    def __init__(self, inner, stats: dict) -> None:
        self._inner = inner
        self.transcriptions = NormalizingTranscriptions(inner.transcriptions, stats)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class NormalizingClient:
    """OpenAI client proxy: same object, but audio.transcriptions.create()
    returns aligner-shaped words. What the `openrouter` server backend would
    do before handing words to the client."""

    def __init__(self, inner: OpenAI) -> None:
        self._inner = inner
        self.stats: dict = {"requests": 0, "skipped_empty": 0, "cost": 0.0, "seconds": 0.0}
        self.audio = NormalizingAudio(inner.audio, self.stats)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# --- chunked file mode ----------------------------------------------------


def run_chunked_file_mode(*, model: str, client: OpenAI, path: Path, language: str | None, srt_path: Path) -> dict:
    """File mode as the client runs it today, with the ASR client pointed at
    OpenRouter: `transcribe_file` (./client/client.py) with the coarse
    chunker and the word-aligner SRT path. `openrouter` is not a known
    backend name yet, so it routes to the word path (with a warning) and takes
    the default 30/5 s chunk profile. Words are normalised (see
    normalize_words) on the way back, as the server backend would."""
    import logging
    import time

    from client import transcribe_file
    from transcribe import set_active_backend

    set_active_backend("openrouter")
    # Chunks are 30 s, so a retry re-uploads little: unlike the long-form
    # runs, allow the SDK's retries and a tighter per-request timeout here
    # (a provider stall of 180 s with no retry killed a run on 2026-10-11).
    wrapped = NormalizingClient(client.with_options(timeout=CHUNK_TIMEOUT_SECONDS, max_retries=2))
    print(f"-> {model} chunked via transcribe_file ...")
    t0 = time.perf_counter()
    ok = True
    try:
        transcribe_file(path, asr_client=wrapped, model=model, language=language, prompt=None, output=srt_path)
    except Exception as e:  # keep going with other runs; the log has the trace
        ok = False
        logging.getLogger("subsvibe").exception("chunked run failed")
        print(f"   error: {type(e).__name__}: {e}")
    latency = round(time.perf_counter() - t0, 3)
    st = wrapped.stats
    print(f"   {'wrote' if ok else 'FAILED'} {srt_path.name} in {latency:.1f}s - "
          f"{st['requests']} requests, {st['skipped_empty']} empty skipped, "
          f"{st['seconds']:.0f} s billed, ${st['cost']:.4f}")
    return {"model": model, "mode": "chunked", "ok": ok, "latency_s": latency, "srt": str(srt_path), **st}


def print_table(cols: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [max(len(c), *(len(row[i]) for row in rows)) for i, c in enumerate(cols)]
    line = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
