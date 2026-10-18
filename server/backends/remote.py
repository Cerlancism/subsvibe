"""Remote ASR: forward each request to an OpenAI-compatible transcription
endpoint instead of running a model in-process.

One class serves one endpoint per backend *name*; `openrouter` is the first.
The model varies per request on an aggregator, and each model has its own
shape - how long a request may be, which codec and transport the provider
accepts, whether its `segments` are usable - so those live in a spec table
keyed by model id (`MODEL_SPECS`), with a Whisper-shaped default for ids the
table does not know. Adding a model is one row there.

Nothing loads: `load`/`unload` only flip a flag so the lifecycle endpoints
and idle unload keep their meaning. With no model state per call, requests
run in parallel (up to `config.OPENROUTER_MAX_CONCURRENT`; the server queues
the rest) and each names its own model: `transcribe_result` takes the model
id per call, falling back to the one `set_model_id` chose, so two clients on
different models never switch anything under each other. Timeouts and
retries are per call. The audio arrives from our own client as
WAV, is re-encoded here to what the spec says, and goes out base64-encoded
in a JSON body, whose cap (50 MB) is twice the multipart one. The default
codec is "auto": lossless FLAC whenever the request fits under the
transport's cap, else Opus in an Ogg container (the provider rejects every
AAC form, and Opus at 80 kbps measured within noise of lossless). The codec
that went out and its size are in each request's log line.
"""
from __future__ import annotations

import base64
import http.client
import io
import json
import logging
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
from utils.language import to_iso_code
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
    # "auto", "opus" (Ogg container), "flac" or "wav" - what the audio is
    # re-encoded to before it leaves. "auto" is FLAC when the request fits
    # the transport cap, else Opus (lossy only where measured to cost
    # nothing).
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

# Minutes-to-an-hour per request in a JSON body, per-character words with
# marks as words of their own and one untimed segment: mai, Grok STT and
# ElevenLabs Scribe all measured this shape (100 min in one 24 MB Opus
# request). FLAC fits roughly the first 35 min; longer falls back to Opus.
LONG_FORM_SPEC = RemoteModelSpec(
    long_form=True,
    max_input_seconds=config.OPENROUTER_MAX_INPUT_SECONDS,
    codec="auto",
    transport="json",
    trust_segments=False,
)

MODEL_SPECS: dict[str, RemoteModelSpec] = {
    "microsoft/mai-transcribe-2": LONG_FORM_SPEC,
    "x-ai/grok-stt-1.0": LONG_FORM_SPEC,
    "elevenlabs/scribe-v2": LONG_FORM_SPEC,
}

# Ids the table does not know are fed like a Whisper endpoint: short
# requests, lossless audio, entries from words. Safe for any model; only
# slower than a tuned row.
DEFAULT_SPEC = RemoteModelSpec(
    long_form=False,
    max_input_seconds=config.MAX_INPUT_SECONDS,
    codec="auto",
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

# Request body caps per transport (config.OPENROUTER_*_CAP_BYTES), less a
# margin for the other fields.
_TRANSPORT_CAP_BYTES = {
    "json": config.OPENROUTER_JSON_CAP_BYTES,
    "multipart": config.OPENROUTER_MULTIPART_CAP_BYTES,
}
_CAP_MARGIN_BYTES = 256_000


def _wire_size(payload_len: int, transport: str) -> int:
    """Bytes the audio occupies in the request body: base64 inflates 4/3
    on the JSON transport, multipart carries it raw."""
    return (payload_len + 2) // 3 * 4 if transport == "json" else payload_len


def encode_for(pcm: np.ndarray, spec: RemoteModelSpec) -> tuple[str, bytes, str]:
    """Encode `pcm` per the spec's codec; "auto" tries FLAC first and falls
    back to Opus when the request would not fit the transport's cap.
    Returns (codec used, payload, note for the log - empty unless auto
    fell back, then the FLAC size and time it discarded)."""
    if spec.codec != "auto":
        return spec.codec, _CODECS[spec.codec][0](pcm), ""
    budget = _TRANSPORT_CAP_BYTES[spec.transport] - _CAP_MARGIN_BYTES
    t0 = time.monotonic()
    flac = encode_flac(pcm)
    t_flac = time.monotonic() - t0
    if _wire_size(len(flac), spec.transport) <= budget:
        return "flac", flac, ""
    note = f" (flac {len(flac) / 1e6:.1f} MB in {t_flac:.1f}s over the {spec.transport} cap)"
    return "opus", encode_opus(pcm), note


# ---------------------------------------------------------------------------
# Response normalisation
# ---------------------------------------------------------------------------


def _is_bracketed(token: str) -> bool:
    t = token.strip()
    return len(t) > 1 and t[0] in "[(（【" and t[-1] in "])）】"


def normalize_words(raw_words: list) -> list[dict]:
    """The provider's `words` as aligner-shaped words that already carry
    their punctuation (fold_word_marks): letters in `word`, marks in the
    previous word's `trailing`, timings rounded. Whisper providers embed
    marks in the token ("さい。"); mai-transcribe-2 emits each mark as a word
    of its own; both fold the same way. The client's attach_punctuation
    leaves words with `trailing` as they are, so the response text is never
    searched."""
    # Scribe tags non-speech as `type: audio_event` ("[音楽]"); a bracketed
    # token is the same thing from a provider that does not set the type.
    timed = [
        w for w in raw_words
        if isinstance(w, dict) and w.get("start") is not None and w.get("end") is not None
        and w.get("type") != "audio_event" and not _is_bracketed(str(w.get("word") or ""))
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
        self._base_url = config.OPENROUTER_BASE_URL.rstrip("/")
        self._api_key = config.OPENROUTER_API_KEY
        self._timeout = config.OPENROUTER_TIMEOUT_SECONDS
        self._max_retries = max(0, config.OPENROUTER_MAX_RETRIES)
        # Models already warned about dropping `prompt`. Concurrent calls may
        # both add the same id; the worst case is one repeated log line.
        self._warned_prompt: set[str] = set()

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

    def max_concurrency(self) -> int:
        return config.OPENROUTER_MAX_CONCURRENT

    def resident_model(self) -> bool:
        return False  # every call carries its model; nothing is loaded

    def profile(self, model_id: str | None = None) -> dict:
        spec = spec_for(model_id or self._model_id)
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
        model_id: str | None = None,
    ) -> dict:
        del want_words  # words come back with every verbose_json response
        # Everything model-specific below reads this local, never
        # self._model_id: a concurrent set_model_id must not change the model
        # a call in flight is building its request for.
        model_id = model_id or self._model_id
        empty = {"text": "", "language": None, "words": [], "segments": []}
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = audio.size / SAMPLE_RATE
        if duration < MIN_REQUEST_SECONDS:
            return empty

        spec = spec_for(model_id)
        if duration > spec.max_input_seconds:
            raise ValueError(
                f"audio is {duration:.1f}s, exceeds {model_id} max "
                f"{spec.max_input_seconds:.0f}s - split on the client"
            )
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY is empty; see scripts/backends/env.openrouter.sh")
        self._loaded = True

        if prompt and not spec.send_prompt and model_id not in self._warned_prompt:
            log.info("prompt dropped: %s does not read it", model_id)
            self._warned_prompt.add(model_id)

        t0 = time.monotonic()
        codec, payload, codec_note = encode_for(audio, spec)
        t_encode = time.monotonic() - t0
        _, json_format, filename, mime = _CODECS[codec]

        fields: dict = {
            "model": model_id,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment", "word"],
            "temperature": 0.0,
        }
        if language and spec.send_language:
            # The bundled client sends canonical names ("Japanese"); providers
            # take ISO-639-1 codes.
            fields["language"] = to_iso_code(language)
        if prompt and spec.send_prompt:
            fields["prompt"] = prompt

        if spec.transport == "json":
            fields["input_audio"] = {"data": base64.b64encode(payload).decode("ascii"), "format": json_format}
            body = json.dumps(fields).encode("utf-8")
            content_type = "application/json"
        else:
            body, content_type = _multipart(fields, filename, payload, mime)

        t1 = time.monotonic()
        result = self._post(model_id, body, content_type)
        t_request = time.monotonic() - t1

        text = str(result.get("text") or "").strip()
        words = normalize_words(list(result.get("words") or []))
        if spec.trust_segments:
            segments = _map_segments(list(result.get("segments") or []))
        else:
            segments = segments_from_words(words, text)
        if text and not words and not segments:
            log.warning("%s returned text without timings for %.1fs of audio", model_id, duration)

        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        cost = usage.get("cost")
        log.info(
            "%s: %.1fs audio as %s %.1f MB%s, encode %.1fs, request %.1fs, %d words%s",
            model_id, duration, codec, len(payload) / 1e6, codec_note, t_encode, t_request,
            len(words), f", cost ${float(cost):.4f}" if isinstance(cost, (int, float)) else "",
        )
        return {
            "text": text,
            "language": to_iso_code(str(result.get("language") or "")) or language,
            "words": words,
            "segments": segments,
        }

    # -- HTTP --

    def _post(self, model_id: str, body: bytes, content_type: str) -> dict:
        """POST with bounded retries on timeouts, connection errors and 5xx.
        A 4xx is final, as is a wrapped provider error whose code is not a
        5xx. `model_id` only names the model in log and error text; the body
        already carries it."""
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
                            model_id, pause, attempt + 1, self._max_retries + 1, last)
                time.sleep(pause)
            try:
                return self._post_once(model_id, url, body, headers)
            except RemoteError as exc:
                last = exc
                if not exc.retryable:
                    raise
        assert last is not None
        raise last

    def _post_once(self, model_id: str, url: str, body: bytes, headers: dict) -> dict:
        req = urllib.request.Request(url, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
                status = resp.status
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RemoteError(f"{model_id}: HTTP {exc.code}: {detail}",
                              status=exc.code, retryable=exc.code >= 500) from exc
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            # http.client.HTTPException covers a connection that drops while
            # the body is still streaming (IncompleteRead on a chunked reply)
            # and other malformed replies; none of them subclass OSError. Each
            # is as transient as a timeout, and like one may already have been
            # billed upstream.
            raise RemoteError(f"{model_id}: {type(exc).__name__}: {exc}", retryable=True) from exc

        try:
            result = json.loads(raw)
        except ValueError as exc:
            raise RemoteError(f"{model_id}: non-JSON response: {raw[:200]!r}", status=status) from exc
        if not isinstance(result, dict):
            raise RemoteError(f"{model_id}: unexpected response shape", status=status)
        if "error" in result:
            # The aggregator wraps an upstream failure in a 200; its `code`
            # is the upstream status (524 edge timeout, 529 overloaded) and
            # a 5xx there is as transient as a bare one.
            error = result["error"]
            code = error.get("code") if isinstance(error, dict) else None
            raise RemoteError(f"{model_id}: provider error: {json.dumps(error)[:500]}",
                              status=status, retryable=isinstance(code, int) and code >= 500)
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
