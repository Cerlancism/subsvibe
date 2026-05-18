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

log = logging.getLogger("subsvibe.transcribe")

TRANSCRIPT_HOST = os.environ.get("TRANSCRIPT_HOST", "127.0.0.1")
TRANSCRIPT_PORT = os.environ.get("TRANSCRIPT_PORT", "8000")
# Empty by default: the server fills in whichever backend's model id it resolved
# (see _model.resolved_model_id in server/server.py). Override only if you need
# to pin a specific id from the client side.
TRANSCRIPT_MODEL_ID = os.environ.get("TRANSCRIPT_MODEL_ID", "")
TRANSCRIPT_BASE_URL = os.environ.get("TRANSCRIPT_BASE_URL", f"http://{TRANSCRIPT_HOST}:{TRANSCRIPT_PORT}")
TRANSCRIPT_API_KEY = os.environ.get("TRANSCRIPT_API_KEY", "not-needed-locally")

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
    response = asr_client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=LLM_ASR_MAX_TOKENS,
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
