"""Remote ASR: forward each request to an OpenAI-compatible transcription
endpoint instead of running a model in-process.

One class serves one endpoint per backend *name*; `openrouter` is the first.
The model varies per request on an aggregator, and each model has its own
shape - how long a request may be, which codec and transport the provider
accepts, whether its `segments` are usable - so those live in a spec table
keyed by model id (`MODEL_SPECS`), with a Whisper-shaped default for ids the
table does not know. Adding a model is one row there.

Nothing loads: `load`/`unload` only flip a flag so the lifecycle endpoints
and idle unload keep their meaning. The audio arrives from our own client as
WAV, is re-encoded here to what the spec says (Opus in an Ogg container for
mai-transcribe-2: the provider rejects every AAC form, and Opus at 80 kbps
measured within noise of lossless), and goes out base64-encoded in a JSON
body, whose cap (50 MB) is twice the multipart one.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from dataclasses import dataclass

import av
import numpy as np

from backends import config
from backends._qwen_aligner import segments_from_words
from backends.base import Backend
from utils.text import fold_word_marks

log = logging.getLogger("subsvibe.remote")

SAMPLE_RATE = 16000

# Below this much audio a request is pointless, and the provider answers a
# bare 400 to an empty file. The client's chunker can emit a 0.0 s file-tail
# chunk; answer it with nothing rather than an error.
MIN_REQUEST_SECONDS = 0.1

# Pause before a retry, multiplied by the attempt number.
RETRY_BACKOFF_SECONDS = 2.0


class RemoteError(RuntimeError):
    """The remote endpoint refused or failed the request. `status` is the
    HTTP status when there was one (a provider error wrapped in a 200 keeps
    200), and `retryable` says whether another attempt could succeed."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass(frozen=True)
class RemoteModelSpec:
    """How one remote model is fed and read. See the module docstring."""
    # Reported to the client: minutes-to-an-hour per request (client cuts
    # evenly near the cap) rather than Whisper's 30 s windows.
    long_form: bool
    # Longest audio one request may carry, seconds.
    max_input_seconds: float
    # "opus" (Ogg container), "flac" or "wav" - what the audio is re-encoded
    # to before it leaves. Lossy only where measured to cost nothing.
    codec: str
    # "json" sends base64 `input_audio` (50 MB cap); "multipart" is the plain
    # OpenAI form (25 MB cap).
    transport: str
    # Whether the model's own `segments` are subtitle-shaped. mai returns
    # segments with zeroed timings; the client builds entries from words.
    trust_segments: bool
    # Whether `prompt` reaches the provider. OpenRouter accepts the field for
    # every model and only Groq-hosted ones read it; not forwarded by default.
    send_prompt: bool = False
    # Whether a client-given `language` is forwarded. Absent, the provider
    # detects it.
    send_language: bool = True

    @property
    def entries_from(self) -> str:
        return "segments" if self.trust_segments else "words"


# model id -> spec. Measured 2026-10-11/12 against OpenRouter; the memory note
# ./.claude/memory/file-chunk-scan-window.md holds the numbers behind each
# value. A whisper row would be: long_form False, max_input_seconds
# config.MAX_INPUT_SECONDS, codec "flac", transport "json", trust_segments
# True (its segments are Whisper's own), send_prompt False.
MODEL_SPECS: dict[str, RemoteModelSpec] = {
    "microsoft/mai-transcribe-2": RemoteModelSpec(
        long_form=True,
        max_input_seconds=config.OPENROUTER_MAX_INPUT_SECONDS,
        codec="opus",
        transport="json",
        trust_segments=False,
    ),
}

# Ids the table does not know are fed like a Whisper endpoint: short
# requests, lossless audio, entries from words. Safe for any model; only
# slower than a tuned row.
DEFAULT_SPEC = RemoteModelSpec(
    long_form=False,
    max_input_seconds=config.MAX_INPUT_SECONDS,
    codec="flac",
    transport="json",
    trust_segments=False,
)


def spec_for(model_id: str) -> RemoteModelSpec:
    return MODEL_SPECS.get(model_id, DEFAULT_SPEC)


# ---------------------------------------------------------------------------
# Audio encoders: float32 mono 16 kHz in, container bytes out
# ---------------------------------------------------------------------------


def _to_int16(pcm: np.ndarray) -> np.ndarray:
    return (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)


def encode_wav(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(_to_int16(pcm).tobytes())
    return buf.getvalue()


def _encode_av(pcm: np.ndarray, container: str, codec: str, **stream_attrs) -> bytes:
    """Encode through PyAV's bundled FFmpeg (libopus and flac ship in the
    wheel, so no ffmpeg binary is needed). The whole chunk goes in as one
    frame; PyAV re-frames it to the codec's frame size."""
    buf = io.BytesIO()
    with av.open(buf, "w", format=container) as out:
        stream = out.add_stream(codec, rate=SAMPLE_RATE)
        stream.layout = "mono"
        stream.format = "s16"
        for name, value in stream_attrs.items():
            setattr(stream, name, value)
        frame = av.AudioFrame.from_ndarray(_to_int16(pcm).reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = SAMPLE_RATE
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return buf.getvalue()


def encode_opus(pcm: np.ndarray, bitrate: int = config.OPENROUTER_OPUS_BITRATE) -> bytes:
    """Opus VBR in an Ogg container, tuned for speech (`voip`)."""
    return _encode_av(pcm, "ogg", "libopus", bit_rate=bitrate,
                      options={"application": "voip", "vbr": "on"})


def encode_flac(pcm: np.ndarray) -> bytes:
    return _encode_av(pcm, "flac", "flac")


# codec -> (encoder, JSON `input_audio.format`, multipart filename, mime)
_CODECS = {
    "opus": (encode_opus, "ogg", "audio.ogg", "audio/ogg"),
    "flac": (encode_flac, "flac", "audio.flac", "audio/flac"),
    "wav": (encode_wav, "wav", "audio.wav", "audio/wav"),
}


# ---------------------------------------------------------------------------
# Response normalisation
# ---------------------------------------------------------------------------


def normalize_words(raw_words: list) -> list[dict]:
    """The provider's `words` as aligner-shaped words that already carry
    their punctuation (fold_word_marks): letters in `word`, marks in the
    previous word's `trailing`, timings rounded. Whisper providers embed
    marks in the token ("さい。"); mai-transcribe-2 emits each mark as a word
    of its own; both fold the same way. The client's attach_punctuation
    leaves words with `trailing` as they are, so the response text is never
    searched."""
    timed = [
        w for w in raw_words
        if isinstance(w, dict) and w.get("start") is not None and w.get("end") is not None
    ]
    return [
        {
            "word": w["word"],
            "start": round(float(w["start"]), 3),
            "end": round(float(w["end"]), 3),
            "trailing": w["trailing"],
        }
        for w in fold_word_marks(timed)
        if w["word"]
    ]


def _map_segments(raw_segments: list) -> list[dict]:
    segments: list[dict] = []
    for s in raw_segments:
        if not isinstance(s, dict):
            continue
        text = str(s.get("text") or "").strip()
        if not text:
            continue
        segments.append({
            "text": text,
            "start": round(float(s.get("start") or 0.0), 3),
            "end": round(float(s.get("end") or 0.0), 3),
        })
    return segments


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class RemoteBackend(Backend):
    """Forwards to `config.OPENROUTER_BASE_URL` with the OpenRouter key. The
    endpoint and key are per backend name; everything model-specific comes
    from `spec_for(model_id)`."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._loaded = False
        self._lock = threading.Lock()
        self._base_url = config.OPENROUTER_BASE_URL.rstrip("/")
        self._api_key = config.OPENROUTER_API_KEY
        self._timeout = config.OPENROUTER_TIMEOUT_SECONDS
        self._max_retries = max(0, config.OPENROUTER_MAX_RETRIES)
        self._warned_prompt = False

    # -- lifecycle (nothing to load; the flag keeps the endpoints honest) --

    def set_model_id(self, model_id: str) -> None:
        self._model_id = model_id

    def load(self) -> None:
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY is empty; see scripts/backends/env.openrouter.sh")
        if not self._loaded:
            log.info("remote backend ready: %s -> %s", self._model_id, self._base_url)
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False

    def load_aligner(self) -> None:
        return None

    def has_secondary(self) -> bool:
        return False

    def unload_secondary(self) -> None:
        return None

    def profile(self) -> dict:
        spec = spec_for(self._model_id)
        return {
            "max_input_seconds": spec.max_input_seconds,
            "long_form": spec.long_form,
            "entries_from": spec.entries_from,
        }

    def align(
        self,
        audio: np.ndarray,
        text: str,
        language: str | None,
    ) -> list[dict]:
        raise NotImplementedError(
            "remote backend does not support standalone alignment; "
            "request word timestamps via /v1/audio/transcriptions instead"
        )

    # -- transcription --

    def transcribe_result(
        self,
        audio: np.ndarray,
        language: str | None,
        prompt: str | None,
        want_words: bool,
    ) -> dict:
        del want_words  # words come back with every verbose_json response
        empty = {"text": "", "language": None, "words": [], "segments": []}
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = audio.size / SAMPLE_RATE
        if duration < MIN_REQUEST_SECONDS:
            return empty

        spec = spec_for(self._model_id)
        if duration > spec.max_input_seconds:
            raise ValueError(
                f"audio is {duration:.1f}s, exceeds {self._model_id} max "
                f"{spec.max_input_seconds:.0f}s - split on the client"
            )
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY is empty; see scripts/backends/env.openrouter.sh")
        self._loaded = True

        if prompt and not spec.send_prompt and not self._warned_prompt:
            log.info("prompt dropped: %s does not read it", self._model_id)
            self._warned_prompt = True

        encoder, json_format, filename, mime = _CODECS[spec.codec]
        t0 = time.monotonic()
        payload = encoder(audio)
        t_encode = time.monotonic() - t0

        fields: dict = {
            "model": self._model_id,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment", "word"],
            "temperature": 0.0,
        }
        if language and spec.send_language:
            fields["language"] = language
        if prompt and spec.send_prompt:
            fields["prompt"] = prompt

        if spec.transport == "json":
            fields["input_audio"] = {"data": base64.b64encode(payload).decode("ascii"), "format": json_format}
            body = json.dumps(fields).encode("utf-8")
            content_type = "application/json"
        else:
            body, content_type = _multipart(fields, filename, payload, mime)

        t1 = time.monotonic()
        result = self._post(body, content_type)
        t_request = time.monotonic() - t1

        text = str(result.get("text") or "").strip()
        words = normalize_words(list(result.get("words") or []))
        if spec.trust_segments:
            segments = _map_segments(list(result.get("segments") or []))
        else:
            segments = segments_from_words(words, text)
        if text and not words and not segments:
            log.warning("%s returned text without timings for %.1fs of audio", self._model_id, duration)

        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        cost = usage.get("cost")
        log.info(
            "%s: %.1fs audio as %s %.1f MB, encode %.1fs, request %.1fs, %d words%s",
            self._model_id, duration, spec.codec, len(payload) / 1e6, t_encode, t_request,
            len(words), f", cost ${float(cost):.4f}" if isinstance(cost, (int, float)) else "",
        )
        return {
            "text": text,
            "language": result.get("language") or language,
            "words": words,
            "segments": segments,
        }

    # -- HTTP --

    def _post(self, body: bytes, content_type: str) -> dict:
        """POST with bounded retries on timeouts, connection errors and 5xx.
        A 4xx, and a 200 whose body is an error (a provider failure wrapped
        by the aggregator), are final."""
        url = f"{self._base_url}/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": content_type,
            "Accept": "application/json",
        }
        last: RemoteError | None = None
        for attempt in range(self._max_retries + 1):
            if attempt:
                pause = RETRY_BACKOFF_SECONDS * attempt
                log.warning("retrying %s in %.0fs (attempt %d/%d): %s",
                            self._model_id, pause, attempt + 1, self._max_retries + 1, last)
                time.sleep(pause)
            try:
                return self._post_once(url, body, headers)
            except RemoteError as exc:
                last = exc
                if not exc.retryable:
                    raise
        assert last is not None
        raise last

    def _post_once(self, url: str, body: bytes, headers: dict) -> dict:
        req = urllib.request.Request(url, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
                status = resp.status
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RemoteError(f"{self._model_id}: HTTP {exc.code}: {detail}",
                              status=exc.code, retryable=exc.code >= 500) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RemoteError(f"{self._model_id}: {exc}", retryable=True) from exc

        try:
            result = json.loads(raw)
        except ValueError as exc:
            raise RemoteError(f"{self._model_id}: non-JSON response: {raw[:200]!r}", status=status) from exc
        if not isinstance(result, dict):
            raise RemoteError(f"{self._model_id}: unexpected response shape", status=status)
        if "error" in result:
            raise RemoteError(f"{self._model_id}: provider error: {json.dumps(result['error'])[:500]}",
                              status=status)
        return result


def _multipart(fields: dict, filename: str, payload: bytes, mime: str) -> tuple[bytes, str]:
    boundary = "----subsvibe-remote-" + uuid.uuid4().hex
    crlf = b"\r\n"
    parts: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        parts.append(f"--{boundary}".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(value.encode("utf-8"))

    for name, value in fields.items():
        if isinstance(value, list):
            for item in value:
                add_field(f"{name}[]", str(item))
        else:
            add_field(name, str(value))
    parts.append(f"--{boundary}".encode())
    parts.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode())
    parts.append(f"Content-Type: {mime}".encode())
    parts.append(b"")
    parts.append(payload)
    parts.append(f"--{boundary}--".encode())
    parts.append(b"")
    return crlf.join(parts), f"multipart/form-data; boundary={boundary}"
