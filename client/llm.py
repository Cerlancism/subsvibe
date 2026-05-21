from __future__ import annotations

import logging
import os

from openai import OpenAI
from pydantic import BaseModel

log = logging.getLogger("subsvibe.llm")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "qwen3.5-instruct:4b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

# max_retries=0: this client is used for live translation. Retries are stale
# work — the staleness drop in pipeline._drain_stale moves on instead.
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=0)

TRANSLATE_HISTORY_LEN = 6
TRANSLATE_MAX_TOKENS = 256


class Translation(BaseModel):
    translation: str


def _translate_system(target: str) -> str:
    return (
        "You are a real-time subtitle translator. "
        "Speech is segmented by voice activity detection, so each input is a complete "
        "utterance (possibly mid-sentence if the speaker paused). Recent committed "
        "utterances are provided as context — already shown to the viewer and "
        "immutable. Do not re-translate the history; only translate the current "
        "utterance.\n"
        f"Target language: {target}. "
        f"Output natural, fluent {target} that reads well as a subtitle line. "
        "Preserve proper nouns and technical terms when surrounding context makes them "
        "unambiguous. If the input is a short fragment, translate it as a fragment — "
        "do not invent continuations."
    )


def translate(
    text: str,
    history: list[tuple[str, str]],
    *,
    target: str = "English",
    timeout: float | None = None,
) -> str:
    """Translate one utterance.

    `history` is an ordered list of (source, translation) tuples for previously
    *committed* utterances (oldest first). It must NOT contain provisional
    output, which would otherwise pollute future calls.
    """
    messages: list[dict] = [{"role": "system", "content": _translate_system(target)}]
    if history:
        context_lines = "\n".join(
            f"- {raw}\n  -> {tr}" for raw, tr in history
        )
        messages.append({
            "role": "user",
            "content": f"Recent committed utterances (oldest to newest):\n{context_lines}",
        })
        messages.append({"role": "assistant", "content": "Understood."})
    messages.append({"role": "user", "content": f"Current utterance: {text}"})
    completion = llm_client.chat.completions.parse(
        model=LLM_MODEL_ID,
        messages=messages,
        response_format=Translation,
        temperature=0,
        max_tokens=TRANSLATE_MAX_TOKENS,
        **({"timeout": timeout} if timeout is not None else {}),
    )
    message = completion.choices[0].message
    if message.refusal:
        log.warning("translate refusal: %s", message.refusal)
        return ""
    if message.parsed is None:
        log.warning("translate returned no parsed output")
        return ""
    return message.parsed.translation.strip()
