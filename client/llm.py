from __future__ import annotations

import logging
import os
import re

from openai import LengthFinishReasonError, OpenAI
from pydantic import BaseModel, ValidationError

log = logging.getLogger("subsvibe.llm")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "qwen3.5-instruct:4b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

# max_retries=0: this client is used for live translation. Retries are stale
# work — the staleness drop in pipeline._drain_stale moves on instead.
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=0)

TRANSLATE_HISTORY_LEN = 10
TRANSLATE_MAX_TOKENS = 256


class Translation(BaseModel):
    translation: str


# Matches the "translation" field's string value in a (possibly truncated) JSON
# object. Stops at the first unescaped quote OR at end-of-string. Handles
# escaped quotes within the value.
_TRANSLATION_FIELD_RE = re.compile(
    r'"translation"\s*:\s*"((?:[^"\\]|\\.)*)',
    re.DOTALL,
)


def _salvage_translation(raw: str | None) -> str:
    """Best-effort extraction of the `translation` field from a truncated /
    malformed JSON blob. Returns "" if nothing usable is found.

    Decodes the basic JSON string escapes (\\", \\\\, \\n, \\t) — anything more
    exotic is left as-is; live subtitles tolerate the occasional stray escape
    better than a dropped line.
    """
    if not raw:
        return ""
    m = _TRANSLATION_FIELD_RE.search(raw)
    if not m:
        return ""
    value = m.group(1)
    # Cheap unescape for the common cases.
    value = (
        value.replace(r"\"", '"')
             .replace(r"\\", "\\")
             .replace(r"\n", "\n")
             .replace(r"\t", "\t")
    )
    return value.strip()


def _translate_system(target: str, extra_context: str | None = None) -> str:
    base = (
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
    if extra_context:
        base += f"\n\nAdditional context from the user:\n{extra_context}"
    return base


def translate(
    text: str,
    history: list[tuple[str, str]],
    *,
    target: str = "English",
    extra_context: str | None = None,
    timeout: float | None = None,
) -> str:
    """Translate one utterance.

    `history` is an ordered list of (source, translation) tuples for previously
    *committed* utterances (oldest first). It must NOT contain provisional
    output, which would otherwise pollute future calls.
    """
    messages: list[dict] = [{"role": "system", "content": _translate_system(target, extra_context)}]
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
    try:
        completion = llm_client.chat.completions.parse(
            model=LLM_MODEL_ID,
            messages=messages,
            response_format=Translation,
            temperature=0,
            max_tokens=TRANSLATE_MAX_TOKENS,
            **({"timeout": timeout} if timeout is not None else {}),
        )
    except LengthFinishReasonError as exc:
        # max_tokens hit before the JSON object closed. Try to recover the
        # partial translation field — better a clipped subtitle than nothing.
        raw = exc.completion.choices[0].message.content if exc.completion.choices else None
        salvaged = _salvage_translation(raw)
        if salvaged:
            log.warning("translate hit max_tokens - returning salvaged partial (raw_len=%d)", len(raw or ""))
            return salvaged
        log.warning("translate hit max_tokens before closing JSON - dropping (raw_len=%d)", len(raw or ""))
        return ""
    except ValidationError as exc:
        # Malformed structured output. The bad input is in errors()[0]["input"];
        # try the same regex recovery before giving up.
        errors = exc.errors()
        raw = errors[0].get("input") if errors else None
        raw = raw if isinstance(raw, str) else None
        salvaged = _salvage_translation(raw)
        if salvaged:
            log.warning("translate produced invalid JSON - returning salvaged partial (raw_len=%d)", len(raw or ""))
            return salvaged
        log.warning("translate produced invalid structured output (raw_len=%d): %s", len(raw or ""), exc)
        return ""
    message = completion.choices[0].message
    if message.refusal:
        log.warning("translate refusal: %s", message.refusal)
        return ""
    if message.parsed is None:
        log.warning("translate returned no parsed output")
        return ""
    return message.parsed.translation.strip()
