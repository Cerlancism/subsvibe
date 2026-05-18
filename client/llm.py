from __future__ import annotations

import logging
import os

from openai import OpenAI

from capture import LIVE_TICK_SECONDS, LIVE_WINDOW_SECONDS

log = logging.getLogger("subsvibe.llm")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "qwen3.5-instruct:4b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

TRANSLATE_HISTORY_LEN = 10

_TRANSLATE_SYSTEM = (
    "You are a real-time subtitle translator working with a sliding window ASR system. "
    f"Every {LIVE_TICK_SECONDS} second(s) you receive a new {LIVE_WINDOW_SECONDS}-second transcript window. "
    "Each window heavily overlaps with the previous one - only the last second or so is genuinely new. "
    "The transcript is raw ASR output and may contain mid-sentence fragments, repeated phrases, "
    "or mis-heard words that get corrected in later windows. "
    "Your job: translate the complete thought visible in the current window into natural English. "
    "Use the history to understand context and spot ASR corrections "
    "(e.g. a word that was wrong before now appears correctly - prefer the corrected form). "
    "Focus on what is new or corrected compared to the previous window. "
    "Output only the English translation of the current window, no explanations."
)


def translate(text: str, history: list[tuple[str, str]]) -> str:
    messages: list[dict] = [{"role": "system", "content": _TRANSLATE_SYSTEM}]
    if history:
        context_lines = "\n".join(
            f"transcript: {raw}\ntranslation: {tr}" for raw, tr in history
        )
        messages.append({
            "role": "user",
            "content": f"Recent context (oldest to newest):\n{context_lines}",
        })
        messages.append({"role": "assistant", "content": "Understood."})
    messages.append({"role": "user", "content": f"Current window transcript: {text}"})
    resp = llm_client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=messages,
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()
