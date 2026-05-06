from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request

from openai import OpenAI

from llm import LLM_BASE_URL, llm_client

log = logging.getLogger("subsvibe.transcribe")

TRANSCRIPT_HOST = os.environ.get("TRANSCRIPT_HOST", "127.0.0.1")
TRANSCRIPT_PORT = os.environ.get("TRANSCRIPT_PORT", "8000")
TRANSCRIPT_MODEL_NAME = os.environ.get("TRANSCRIPT_MODEL_NAME", "qwen3-asr")
TRANSCRIPT_BASE_URL = os.environ.get("TRANSCRIPT_BASE_URL", f"http://{TRANSCRIPT_HOST}:{TRANSCRIPT_PORT}")
TRANSCRIPT_API_KEY = os.environ.get("TRANSCRIPT_API_KEY", "not-needed-locally")

LLM_ASR_MODEL_NAME = os.environ.get("LLM_ASR_MODEL_NAME", "gemma4:e4b")
LLM_ASR_MAX_TOKENS = 512

transcribe_client = OpenAI(api_key=TRANSCRIPT_API_KEY, base_url=TRANSCRIPT_BASE_URL)


def get_asr_client(use_llm: bool, model: str | None) -> tuple[OpenAI, str, str]:
    """Pick the (client, model, base_url) triple for ASR requests.

    base_url is returned only for diagnostic log/error messages."""
    if use_llm:
        return llm_client, model or LLM_ASR_MODEL_NAME, LLM_BASE_URL
    return transcribe_client, model or TRANSCRIPT_MODEL_NAME, TRANSCRIPT_BASE_URL

# Qwen3-ASR accepts only canonical English language names. Map ISO-639-1
# codes (and a few common aliases) to those names so the user can pass either.
LANGUAGE_NONE_VALUES = {"", "auto", "detect", "none"}
_LANGUAGE_ALIASES = {
    "zh": "Chinese", "zh-cn": "Chinese", "zh-tw": "Chinese", "cmn": "Chinese", "mandarin": "Chinese",
    "en": "English",
    "yue": "Cantonese", "zh-yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino", "tl": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}
SUPPORTED_LANGUAGES = {
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish",
    "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai",
    "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish",
    "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek",
    "Romanian", "Hungarian", "Macedonian",
}


def normalize_language(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    lowered = text.lower()
    if lowered in LANGUAGE_NONE_VALUES:
        return None
    if lowered in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[lowered]
    canonical = text[:1].upper() + text[1:].lower()
    if canonical in SUPPORTED_LANGUAGES:
        return canonical
    raise ValueError(
        f"unsupported language {value!r}; pass an ISO-639-1 code "
        f"(e.g. ja, zh, en) or a canonical name like {sorted(SUPPORTED_LANGUAGES)}"
    )


LLM_ASR_SYSTEM_PROMPT = (
    "You are a transcription engine. Output only the literal transcription "
    "of the audio, with no commentary, prefixes, or formatting."
)


def build_llm_asr_system_prompt(
    *,
    language: str | None,
    base_prompt: str | None,
    history: str | None,
    reference: str | None,
) -> str:
    parts: list[str] = [LLM_ASR_SYSTEM_PROMPT]
    if language:
        parts.append(f"The audio is in {language}.")
    if base_prompt:
        parts.append(base_prompt)
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
    response = asr_client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=LLM_ASR_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio."},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                ],
            },
        ],
    )
    text = (response.choices[0].message.content or "").strip()

    # Workaround for an Ollama bug: after sending audio, the next audio
    # request can fail with a ggml assert ("data_size + view_offs <=
    # ggml_nbytes(view_src)"). Sending a minimal text-only request between
    # audio requests resets the state so the next audio request works.
    try:
        asr_client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=1,
            messages=[{"role": "user", "content": "reply: hi"}],
        )
    except Exception as exc:
        log.debug("llm-asr post-transcribe reset failed (ignored): %s", exc)

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
