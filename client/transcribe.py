from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request

from openai import OpenAI

from llm import LLM_BASE_URL, llm_client
from utils.language import to_canonical_name
from utils.text import attach_punctuation

log = logging.getLogger("subsvibe.transcribe")

TRANSCRIPT_HOST = os.environ.get("TRANSCRIPT_HOST", "127.0.0.1")
TRANSCRIPT_PORT = os.environ.get("TRANSCRIPT_PORT", "8000")
# Empty by default: the server fills in whichever backend's model id it resolved
# (see _model.resolved_model_id in server/server.py). Override only if you need
# to pin a specific id from the client side.
TRANSCRIPT_MODEL_ID = os.environ.get("TRANSCRIPT_MODEL_ID", "")
TRANSCRIPT_BASE_URL = os.environ.get("TRANSCRIPT_BASE_URL", f"http://{TRANSCRIPT_HOST}:{TRANSCRIPT_PORT}")
TRANSCRIPT_API_KEY = os.environ.get("TRANSCRIPT_API_KEY", "not-needed-locally")
# Client-side mirror of the server's input cap (see server/README.md). The
# live pipeline's force-flush ceiling keeps segments far below this, so the
# guard firing at all means a VAD regression — the pipeline trims to the cap
# and warns instead of eating a server 500 that loses the whole segment.
TRANSCRIPT_MAX_INPUT_SECONDS = float(os.environ.get("TRANSCRIPT_MAX_INPUT_SECONDS", "180"))
# Selects how the client turns ASR output into SRT entries.
#   - "qwen" / "anime-whisper": request word-level timestamps and run the
#     attach_punctuation + entries_from_words post-processor.
#   - "faster-whisper": trust the model's own segmentation and skip the
#     word-level pass.
# Must match the server's TRANSCRIPT_BACKEND. Default matches scripts/env.example.sh
# (faster-whisper: CPU-friendly, no GPU required, native segment timestamps).
TRANSCRIPT_BACKEND = os.environ.get("TRANSCRIPT_BACKEND", "faster-whisper")

# Backends whose returned `segments` already match what we'd produce by aligning
# and slicing words. faster-whisper gives clean silence-bounded segments natively;
# qwen/anime-whisper return one segment covering the whole utterance (see
# segments_from_words in server/backends/_qwen_aligner.py), so they need the
# word -> entries_from_words path instead.
_BACKENDS_USE_SEGMENTS = frozenset({"faster-whisper"})

LLM_ASR_MODEL_ID = os.environ.get("LLM_ASR_MODEL_ID", "gemma4:e4b")
LLM_ASR_MAX_TOKENS = 512

transcribe_client = OpenAI(api_key=TRANSCRIPT_API_KEY, base_url=TRANSCRIPT_BASE_URL)


def get_asr_client(use_llm: bool, model: str | None) -> tuple[OpenAI, str, str]:
    """Pick the (client, model, base_url) triple for ASR requests.

    For the FastAPI backend an empty model string means "let the server use
    its configured TRANSCRIPT_MODEL_ID". The LLM backend always needs a real
    model name, so falls back to LLM_ASR_MODEL_ID.
    base_url is returned only for diagnostic log/error messages."""
    if use_llm:
        return llm_client, model or LLM_ASR_MODEL_ID, LLM_BASE_URL
    return transcribe_client, model or TRANSCRIPT_MODEL_ID, TRANSCRIPT_BASE_URL

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

    backend_returns_segments = TRANSCRIPT_BACKEND in _BACKENDS_USE_SEGMENTS
    granularity = "segment" if backend_returns_segments else "word"

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
    if backend_returns_segments:
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
