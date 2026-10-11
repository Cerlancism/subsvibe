from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request

from openai import OpenAI

from llm import LLM_BASE_URL, llm_client
from utils.backend import DEFAULT_BACKEND
from utils.language import to_canonical_name
from utils.text import attach_punctuation

log = logging.getLogger("subsvibe.transcribe")

TRANSCRIPT_HOST = os.environ.get("TRANSCRIPT_HOST", "127.0.0.1")
TRANSCRIPT_PORT = os.environ.get("TRANSCRIPT_PORT", "8000")
# The `/v1` suffix is part of the base URL, not of the paths built from it:
# it is what the OpenAI client appends `/audio/transcriptions` to, and what
# align_words and server_json append their own paths to.
TRANSCRIPT_BASE_URL = os.environ.get("TRANSCRIPT_BASE_URL", f"http://{TRANSCRIPT_HOST}:{TRANSCRIPT_PORT}/v1")
TRANSCRIPT_API_KEY = os.environ.get("TRANSCRIPT_API_KEY", "not-needed-locally")
# The server's input cap, adopted from GET /v1/backend at session start (see
# sync_backend_with_server); this constant only covers a server that could not
# be asked, and matches the server's own default. The live pipeline's
# force-flush ceiling keeps segments far below it, so the guard firing at all
# means a VAD regression — the pipeline trims to the cap and warns instead of
# eating a server 500 that loses the whole segment.
DEFAULT_MAX_INPUT_SECONDS = 180.0
# Selects how the client turns ASR output into SRT entries.
#   - "qwen" / "anime-whisper": request word-level timestamps and run the
#     attach_punctuation + entries_from_words post-processor.
#   - "faster-whisper": trust the model's own segmentation and skip the
#     word-level pass.
# Only the fallback: the server owns the active backend, so a session syncs the
# real value from GET /backend (see sync_backend_with_server). This is reached
# when that query fails, and a wrong guess there mismatches entry
# post-processing against the model that produced the text - hence the shared
# DEFAULT_BACKEND, which the server falls back to as well.
TRANSCRIPT_BACKEND = os.environ.get("TRANSCRIPT_BACKEND", DEFAULT_BACKEND)

# Backends whose returned `segments` already match what we'd produce by aligning
# and slicing words. faster-whisper gives clean silence-bounded segments natively;
# qwen/anime-whisper return one segment covering the whole utterance (see
# segments_from_words in server/backends/_qwen_aligner.py), so they need the
# word -> entries_from_words path instead.
# openrouter: the remote models' segments are either zeroed (mai) or whisper's
# own; the server says which per model (`entries_from` on GET /v1/backend) and
# that report wins over this table whenever a session has it.
_BACKENDS_USE_SEGMENTS = frozenset({"faster-whisper"})
_BACKENDS_USE_WORDS = frozenset({"qwen", "anime-whisper", "openrouter"})
# Every name the client recognizes, default first, in a stable order so the
# --backend help text can be built from it rather than repeating the names.
# Deliberately not a validation list for --backend: what the server can
# actually construct is the server's own SUPPORTED_BACKENDS, and it rejects
# anything else with a 400. The two lists overlap but answer different
# questions - this one is "which post-processing path does the entry take".
KNOWN_BACKENDS = tuple(sorted(_BACKENDS_USE_SEGMENTS)) + tuple(sorted(_BACKENDS_USE_WORDS))


# The backend the server is currently running, as last synced. Code that
# branches on backend behaviour asks backend_returns_segments() rather than
# reading the TRANSCRIPT_BACKEND constant, which is only the fallback.
_active_backend = TRANSCRIPT_BACKEND
_max_input_seconds = DEFAULT_MAX_INPUT_SECONDS
# The server's per-model report (GET /v1/backend), None until synced: which
# chunking strategy the active model needs and where entries come from.
_long_form: bool | None = None
_entries_from: str | None = None

# File mode: one request may carry an hour of audio on a long-form backend,
# and the server's own outbound timeout is 600 s. No SDK retries - a retry
# re-uploads the whole chunk, and the server retries upstream itself.
FILE_ASR_TIMEOUT_SECONDS = 900.0


def active_backend() -> str:
    """The backend this session post-processes for: the server's, once
    sync_backend_with_server has run, else the TRANSCRIPT_BACKEND fallback."""
    return _active_backend


def max_input_seconds() -> float:
    """Longest audio the server accepts per request, as it reported at
    session start; DEFAULT_MAX_INPUT_SECONDS when it could not be asked."""
    return _max_input_seconds


def set_active_backend(name: str) -> None:
    """Publish the backend a session has adopted. The unknown-name warning
    lives here, at the one point a backend is adopted, rather than in the
    per-segment routing that would repeat it for every utterance."""
    global _active_backend
    if name not in KNOWN_BACKENDS:
        log.warning("unknown ASR backend %r; using the word-aligner SRT path", name)
    _active_backend = name


def backend_returns_segments() -> bool:
    """True when the active backend's own `segments` can become SRT entries
    as they are; False when entries must be built from aligned words, which
    is also where an unrecognized backend lands. The server's per-model
    report wins over the client table whenever the session has one."""
    if _entries_from is not None:
        return _entries_from == "segments"
    return _active_backend in _BACKENDS_USE_SEGMENTS


def adopt_backend_report(report: dict) -> None:
    """Take a GET /v1/backend body as this session's truth: the backend name
    and, when present, the active model's request cap, chunking strategy and
    entry source. Shared by the live sync and the in-process bench, which
    reads the same fields straight from the server module."""
    global _max_input_seconds, _long_form, _entries_from
    set_active_backend(report.get("backend") or TRANSCRIPT_BACKEND)
    cap = report.get("max_input_seconds")
    if isinstance(cap, (int, float)) and cap > 0:
        _max_input_seconds = float(cap)
    long_form = report.get("long_form")
    if isinstance(long_form, bool):
        _long_form = long_form
    entries_from = report.get("entries_from")
    if entries_from in ("segments", "words"):
        _entries_from = entries_from


def server_long_form() -> bool | None:
    """Whether the server's active model is long-form (minutes to an hour per
    request), as it reported at session start; None when it could not be
    asked or did not say."""
    return _long_form


LLM_ASR_MODEL_ID = os.environ.get("LLM_ASR_MODEL_ID", "gemma4:e4b")
LLM_ASR_MAX_TOKENS = 512

transcribe_client = OpenAI(api_key=TRANSCRIPT_API_KEY, base_url=TRANSCRIPT_BASE_URL)


def get_asr_client(use_llm: bool, model: str | None) -> tuple[OpenAI, str, str]:
    """Pick the (client, model, base_url) triple for ASR requests.

    For the FastAPI backend the caller supplies the model id - normally the
    server's own, via sync_backend_with_server. An empty string means "use
    whatever model is active" - the server owns the model selection, so the
    client never pins one from its own environment. The LLM backend has no
    such server-side state and still falls back to LLM_ASR_MODEL_ID.
    base_url is returned only for diagnostic log/error messages."""
    if use_llm:
        return llm_client, model or LLM_ASR_MODEL_ID, LLM_BASE_URL
    return transcribe_client, model or "", TRANSCRIPT_BASE_URL

# Either form (ISO code or canonical name) is acceptable on the wire; the
# server backend translates as needed. We keep the client-side helper for
# early CLI validation.
normalize_language = to_canonical_name


def build_llm_asr_system_prompt(
    *,
    language: str | None,
    base_prompt: str | None,
    history: str | None,
    reference: str | None,
) -> str:
    parts: list[str] = []
    if base_prompt:
        parts.append(base_prompt)
    if language:
        parts.append(f"The audio is in {language}.")
    if history:
        parts.append(f"History (recent transcriptions, for context):\n{history}")
    if reference:
        parts.append(
            "Reference (existing subtitle for this segment, may be inaccurate "
            f"but use as a guide):\n{reference}"
        )
    return "\n\n".join(parts)


def llm_asr_chat_transcribe(
    asr_client: OpenAI,
    model: str,
    wav_bytes: bytes,
    *,
    system_prompt: str,
) -> str:
    """Send audio to a chat-completions endpoint as an `input_audio` content
    part. Returns the assistant's plain-text reply, stripped."""
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    # Per-request nonce in the system prompt forces a cache-slot miss on
    # Ollama (see ollama#15333). Without it, repeated audio requests share
    # the system-prompt prefix and the runner reuses a slot whose tensor
    # state was sized for the previous audio batch, occasionally tripping
    # `data_size + view_offs <= ggml_nbytes(view_src)` in ggml.
    nonce = os.urandom(8).hex()
    # Penalties pinned to 0: a transcript must reproduce whatever the speaker
    # said, repetition included. reasoning_effort="none" keeps the multimodal
    # model from spending the token budget thinking before it transcribes.
    response = asr_client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=LLM_ASR_MAX_TOKENS,
        reasoning_effort="none",
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
            {"role": "system", "content": f"{system_prompt}\n\n[request_id:{nonce}]"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                    {"type": "text", "text": "Transcribe this audio."},
                ],
            },
        ],
    )
    text = (response.choices[0].message.content or "").strip()

    # Previous workaround: a 1-token text request after each audio request
    # to reset Ollama's audio-tensor state. Replaced by the per-request
    # nonce above (forces a cache miss). Re-enable if the assert returns.
    # try:
    #     asr_client.chat.completions.create(
    #         model=model,
    #         temperature=0,
    #         max_tokens=1,
    #         messages=[{"role": "user", "content": "reply: hi"}],
    #     )
    # except Exception as exc:
    #     log.debug("llm-asr post-transcribe reset failed (ignored): %s", exc)

    return text


SERVER_BACKEND_TIMEOUT_SECONDS = 30.0
# Loading a cold backend can mean a HuggingFace download; give the eager-load
# path room rather than failing a switch that is still making progress.
SERVER_BACKEND_LOAD_TIMEOUT_SECONDS = 900.0


def _json_object(raw: bytes | str) -> dict | None:
    """Parse `raw` as a JSON object, or None if it is neither."""
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def server_json(
    path: str,
    payload: dict | None = None,
    *,
    method: str | None = None,
    timeout: float = SERVER_BACKEND_TIMEOUT_SECONDS,
) -> dict:
    """GET or POST JSON to a path on the transcription server.

    Public because client.py's lifecycle flags share this transport, and with
    it one error convention: every failure - unreachable, HTTP error, or a
    reply that is not a JSON object - surfaces as RuntimeError. `path` is
    relative to TRANSCRIPT_BASE_URL, which already includes `/v1`. The method
    defaults to POST when a payload is given and GET otherwise; pass `method`
    explicitly for bodyless POSTs such as /model/load."""
    url = TRANSCRIPT_BASE_URL.rstrip("/") + path
    verb = method or ("POST" if payload is not None else "GET")
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    if data is None and verb == "POST":
        data = b""
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, method=verb, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        parsed = _json_object(detail)
        raise RuntimeError(
            f"{path} returned {exc.code}: {parsed.get('detail', detail) if parsed else detail}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"cannot reach transcription server at {url}: {exc}") from exc

    # A 200 that is not a JSON object means something other than our server
    # answered (a proxy error page, say) - as much a transport failure as a
    # refused connection, so it raises the same way.
    parsed = _json_object(body)
    if parsed is None:
        raise RuntimeError(f"{path} returned a non-JSON response: {body[:200]!r}")
    return parsed


def get_server_health() -> dict:
    """Query the server's health probe. Raises RuntimeError when the server
    cannot be reached, which is itself the answer --health is asking for."""
    return server_json("/health")


def get_server_backend() -> dict:
    """Query the server's active ASR backend.
    Returns {backend, model, supported, model_loaded, aligner_loaded}."""
    return server_json("/backend")


def set_server_backend(
    backend: str,
    model: str | None = None,
    *,
    load: bool = False,
) -> dict:
    """Ask the server to switch its ASR backend - the live alternative to
    editing scripts/env.sh and restarting. The server disposes the outgoing
    backend's worker processes before the new one spawns, so call this between
    runs rather than during one.
    Returns {status, backend, previous_backend, model, model_loaded}."""
    payload: dict = {"backend": backend, "load": load}
    if model:
        payload["model"] = model
    timeout = SERVER_BACKEND_LOAD_TIMEOUT_SECONDS if load else SERVER_BACKEND_TIMEOUT_SECONDS
    return server_json("/backend", payload, timeout=timeout)


def sync_backend_with_server(*, load: bool = False) -> str:
    """Adopt the server's active backend before a session, so the SRT/entry
    post-processing path matches the model that actually produced the text.
    A session only ever reads; changing the backend is set_server_backend.

    `load` warms the server's current model so the session does not pay the
    load on its first segment. That load is the one part of this call that can
    raise: a server that answers but cannot load is a hard failure, where a
    server that cannot be reached at all is not.

    Returns the server's model id (empty when it could not be asked) for the
    caller to send on each request; the backend is published via
    set_active_backend. A failed backend query only warns and falls back to
    the TRANSCRIPT_BACKEND environment value.
    """
    try:
        result = get_server_backend()
    except RuntimeError as exc:
        # Send an empty model field: it means "use the active model", so the
        # server's own selection stays in force whichever backend it is in
        # fact running.
        log.warning(
            "could not read server backend (%s); assuming TRANSCRIPT_BACKEND=%r",
            exc, TRANSCRIPT_BACKEND,
        )
        set_active_backend(TRANSCRIPT_BACKEND)
        return ""

    backend = result.get("backend") or TRANSCRIPT_BACKEND
    if backend != TRANSCRIPT_BACKEND:
        log.info(
            "server is running the %r backend (local TRANSCRIPT_BACKEND=%r); following the server",
            backend, TRANSCRIPT_BACKEND,
        )
    adopt_backend_report(result)

    if load and not result.get("model_loaded"):
        server_json(
            "/model/load", method="POST", timeout=SERVER_BACKEND_LOAD_TIMEOUT_SECONDS,
        )

    return result.get("model") or ""


def align_words(
    base_url: str,
    wav_bytes: bytes,
    text: str,
    language: str | None,
    *,
    timeout: float = 120.0,
) -> list[dict]:
    """POST audio + text to /audio/align on the transcription server.
    `base_url` is expected to already include the `/v1` suffix (the OpenAI
    convention used throughout SubsVibe). Returns
    [{"text": str, "start": float, "end": float}, ...]."""
    url = base_url.rstrip("/") + "/audio/align"

    boundary = "----subsvibe-align-" + os.urandom(8).hex()
    crlf = b"\r\n"
    parts: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        parts.append(f"--{boundary}".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(value.encode("utf-8"))

    def add_file(name: str, filename: str, content: bytes, content_type: str) -> None:
        parts.append(f"--{boundary}".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode()
        )
        parts.append(f"Content-Type: {content_type}".encode())
        parts.append(b"")
        parts.append(content)

    add_file("file", "segment.wav", wav_bytes, "audio/wav")
    add_field("text", text)
    if language:
        add_field("language", language)

    parts.append(f"--{boundary}--".encode())
    parts.append(b"")
    body = crlf.join(parts)

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"align endpoint returned {exc.code}: {detail}") from exc

    return list(payload.get("words", []))


def live_transcribe(
    asr_client: OpenAI,
    model: str,
    wav_bytes: bytes,
    filename: str,
    *,
    language: str | None,
    prompt: str | None,
    timeout: float,
    segment_duration: float,
    want_segments: bool,
) -> tuple[str, list[dict]]:
    """Transcribe one segment and return (text, entries).

    Contract: output is always entries. **Non-empty text always yields at
    least one entry** — when the aligner produces none (empty words,
    punctuation rejected all, or no timestamps were requested at all) we
    synthesise a single whole-segment entry `[0, segment_duration]`, mirroring
    file mode's `_words_to_entries` fallback. The caller can therefore trust
    "text ⇒ entries" unconditionally and drive one uniform code path.
    Entries are in audio-relative seconds.

    `want_segments` is the caller's *intent*: does it want this segment
    broken into multiple subtitle-quality entries this cycle (so the pipeline
    can promote completed pieces to the live display before VAD closes the
    segment)? It is NOT a backend concern — how the request is satisfied is
    decided here, per backend:
      - False: the caller only needs the whole segment as one unit (short
        utterance VAD will close on its own). Plain-JSON request + the single
        synthetic entry. As a side effect this also skips qwen/anime-whisper's
        forced-aligner model pass, but that's an implementation detail of
        honouring the single-segment intent, not its purpose.
      - True: request timestamps so multiple entries can be produced on
        subtitle-quality boundaries:
          - faster-whisper: request segment timestamps; pass through directly.
          - qwen / anime-whisper: request word timestamps, reattach
            punctuation from the full text, then run entries_from_words to
            split on word/punctuation boundaries."""
    if not want_segments:
        result = asr_client.audio.transcriptions.create(
            model=model,
            file=(filename, wav_bytes, "audio/wav"),
            response_format="json",
            timeout=timeout,
            **({"language": language} if language else {}),
            **({"prompt": prompt} if prompt else {}),
        )
        text = (result if isinstance(result, str) else getattr(result, "text", "") or "").strip()
        return text, _ensure_entries([], text, segment_duration)

    # Local import: client/subtitle.py pulls utils.text which is heavy at
    # import time on cold start; keep transcribe.py importable without it.
    from subtitle import entries_from_words

    use_segments = backend_returns_segments()
    granularity = "segment" if use_segments else "word"

    result = asr_client.audio.transcriptions.create(
        model=model,
        file=(filename, wav_bytes, "audio/wav"),
        response_format="verbose_json",
        timestamp_granularities=[granularity],
        timeout=timeout,
        **({"language": language} if language else {}),
        **({"prompt": prompt} if prompt else {}),
    )

    text = (getattr(result, "text", "") or "").strip()
    if not text:
        return "", []

    entries: list[dict] = []
    if use_segments:
        for seg in (getattr(result, "segments", None) or []):
            seg_text = (getattr(seg, "text", "") or "").strip()
            if not seg_text:
                continue
            entries.append({
                "start": round(float(getattr(seg, "start", 0.0)), 3),
                "end": round(float(getattr(seg, "end", 0.0)), 3),
                "text": seg_text,
            })
    else:
        raw_words = getattr(result, "words", None) or []
        words = [
            {"word": getattr(w, "word", "") or "", "start": float(getattr(w, "start", 0.0)),
             "end": float(getattr(w, "end", 0.0))}
            for w in raw_words
        ]
        if words:
            enriched = attach_punctuation(words, text)
            entries = entries_from_words(enriched)

    return text, _ensure_entries(entries, text, segment_duration)


def _ensure_entries(entries: list[dict], text: str, segment_duration: float) -> list[dict]:
    """Guarantee the `text ⇒ entries` invariant. Returns `entries` unchanged
    when it already has content; otherwise synthesises one whole-segment entry
    covering `[0, segment_duration]` so non-empty text never leaks as zero
    entries. Empty text returns `[]` (the caller drops it)."""
    if entries:
        return entries
    if text:
        log.debug("synthetic whole-segment entry: aligner returned no entries for %d-char text", len(text))
        return [{"start": 0.0, "end": round(float(segment_duration), 3), "text": text}]
    return []
