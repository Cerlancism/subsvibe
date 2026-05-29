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


class Translations(BaseModel):
    # Ordered translations, one per input line, in the same order. The caller
    # validates the count matches the input and falls back to per-line
    # translation when it doesn't.
    translations: list[str]


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


def _translate_system(
    target: str,
    extra_context: str | None = None,
    override: str | None = None,
    *,
    multi: bool = False,
) -> str:
    if override is not None:
        # Full replacement: caller takes responsibility for telling the model
        # what to do (target language, style, etc.). extra_context is ignored.
        return override
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
    if multi:
        # Array form: the "current utterance" is instead an ordered list of
        # consecutive lines from the same ongoing speech. Override the
        # single-utterance instruction with the per-line contract.
        base += (
            "\n\nThis input is an ORDERED list of consecutive lines from the same "
            "ongoing speech (the last is the latest, in-progress). Translate every "
            "line — seeing the continuation lets you refine an earlier line committed "
            "mid-thought — and return one translation per input line, in the SAME "
            "ORDER and SAME COUNT. Do not merge, split, drop, or add lines."
        )
    if extra_context:
        base += f"\n\nAdditional context from the user:\n{extra_context}"
    return base


def _base_messages(
    history: list[tuple[str, str]],
    target: str,
    extra_context: str | None,
    system_override: str | None,
    *,
    multi: bool,
) -> list[dict]:
    """System prompt + optional committed-history context block, shared by the
    single-utterance and array forms. The caller appends the final user turn
    (the utterance or the line list)."""
    messages: list[dict] = [
        {"role": "system", "content": _translate_system(target, extra_context, system_override, multi=multi)}
    ]
    if history:
        context_lines = "\n".join(f"- {raw}\n  -> {tr}" for raw, tr in history)
        messages.append({
            "role": "user",
            "content": f"Recent committed utterances (oldest to newest):\n{context_lines}",
        })
        messages.append({"role": "assistant", "content": "Understood."})
    return messages


def translate(
    text: str,
    history: list[tuple[str, str]],
    *,
    target: str = "English",
    extra_context: str | None = None,
    system_override: str | None = None,
    temperature: float = 0,
    timeout: float | None = None,
) -> str:
    """Translate one utterance.

    `history` is an ordered list of (source, translation) tuples for previously
    *committed* utterances (oldest first). It must NOT contain provisional
    output, which would otherwise pollute future calls.
    """
    messages = _base_messages(history, target, extra_context, system_override, multi=False)
    messages.append({"role": "user", "content": f"Current utterance: {text}"})
    try:
        completion = llm_client.chat.completions.parse(
            model=LLM_MODEL_ID,
            messages=messages,
            response_format=Translation,
            temperature=temperature,
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


def translate_pair(
    lines: list[str],
    history: list[tuple[str, str]],
    *,
    target: str = "English",
    extra_context: str | None = None,
    system_override: str | None = None,
    temperature: float = 0,
    timeout: float | None = None,
) -> list[str] | None:
    """Translate an ordered list of consecutive lines in ONE call so the model
    can refine an earlier line in light of the continuation.

    Returns a list of stripped translations, OR `None` on refusal / parse error /
    a count that's neither 1 nor `len(lines)`. The accepted lengths carry meaning
    for the caller:

    - `len == len(lines)`: one translation per line, positionally paired —
      the model kept the lines distinct.
    - `len == 1` (when `len(lines) > 1`): the model rendered all the input lines
      as ONE continuous utterance. This is a SIGNAL, not a failure: the caller
      may squash the lines into a single merged commit carrying this translation.

    Any other count (e.g. 3 for a 2-line input) is ambiguous to pair positionally
    → `None`, and the caller falls back to per-line `translate()`.
    """
    messages = _base_messages(history, target, extra_context, system_override, multi=True)
    numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
    messages.append({
        "role": "user",
        "content": f"Translate these {len(lines)} lines, one translation each, in order:\n{numbered}",
    })
    try:
        completion = llm_client.chat.completions.parse(
            model=LLM_MODEL_ID,
            messages=messages,
            response_format=Translations,
            temperature=temperature,
            max_tokens=TRANSLATE_MAX_TOKENS,
            **({"timeout": timeout} if timeout is not None else {}),
        )
    except (LengthFinishReasonError, ValidationError) as exc:
        log.warning("translate_pair failed (%s) - falling back to per-line", type(exc).__name__)
        return None
    message = completion.choices[0].message
    if message.refusal:
        log.warning("translate_pair refusal: %s", message.refusal)
        return None
    if message.parsed is None:
        log.warning("translate_pair returned no parsed output")
        return None
    got = len(message.parsed.translations)
    # Accept the full per-line count (distinct lines) OR exactly 1 (the model
    # merged the lines into one utterance — a squash signal for the caller).
    # Any other count is ambiguous to pair positionally → fall back to per-line.
    if got != len(lines) and got != 1:
        log.debug("translate_pair count ambiguous (want=%d got=%d) - falling back", len(lines), got)
        return None
    return [t.strip() for t in message.parsed.translations]
