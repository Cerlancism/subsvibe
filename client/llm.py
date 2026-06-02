from __future__ import annotations

import logging
import os

from openai import OpenAI

log = logging.getLogger("subsvibe.llm")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "qwen3.5-instruct:4b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

# max_retries=0: this client is used for live translation. Retries are stale
# work — the staleness drop in pipeline._drain_stale moves on instead.
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=0)

TRANSLATE_HISTORY_LEN = 10
TRANSLATE_MAX_TOKENS = 256


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
        "do not invent continuations.\n"
        "Output ONLY the translation as plain text, with no quotes, labels, or "
        "explanation."
    )
    if multi:
        # List form: the "current utterance" is instead an ordered list of
        # consecutive lines from the same ongoing speech. Override the
        # single-utterance instruction with the per-line contract — EXCEPT the
        # collapse rule, which lets the caller squash fragments that are really
        # one clause (e.g. a bare subject followed by its predicate).
        base += (
            "\n\nThis input is an ORDERED list of consecutive lines from the same "
            "ongoing speech (the last is the latest, in-progress). Normally return "
            "one translation per input line, in the SAME ORDER and SAME COUNT — "
            "seeing the continuation lets you refine an earlier line committed "
            "mid-thought. EXCEPTION: if the lines together form a SINGLE clause or "
            "sentence (e.g. a line is just a sentence subject/fragment and the next "
            "completes it), return ONE combined translation of the whole instead of "
            "a stilted word-by-word rendering. So return either N translations (the "
            "lines are distinct) or exactly 1 (they are one continuous sentence) — "
            "never any other count, and never split, drop, or add.\n"
            "Output one translation per line, separated by a single newline, in "
            "order. No numbering, quotes, labels, or blank lines."
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
    single-utterance and list forms. The caller appends the final user turn
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


def _content(completion) -> str | None:
    """Extract the assistant message text, or None if absent."""
    if not completion.choices:
        return None
    message = completion.choices[0].message
    if message.refusal:
        log.warning("translate refusal: %s", message.refusal)
        return None
    return message.content


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
    completion = llm_client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=messages,
        temperature=temperature,
        max_tokens=TRANSLATE_MAX_TOKENS,
        reasoning_effort="none",
        **({"timeout": timeout} if timeout is not None else {}),
    )
    content = _content(completion)
    if content is None:
        log.warning("translate returned no output")
        return ""
    # Plain-text mode: the whole response is the translation. Collapse any stray
    # newlines (the model may wrap) into spaces so it stays a single line.
    return " ".join(content.split())


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

    Returns a list of stripped translations, OR `None` on refusal / empty output /
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
    completion = llm_client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=messages,
        temperature=temperature,
        max_tokens=TRANSLATE_MAX_TOKENS,
        reasoning_effort="none",
        **({"timeout": timeout} if timeout is not None else {}),
    )
    content = _content(completion)
    if content is None:
        log.warning("translate_pair returned no output")
        return None
    # One translation per non-blank line, in order.
    out = [ln.strip() for ln in content.splitlines() if ln.strip()]
    got = len(out)
    # Accept the full per-line count (distinct lines) OR exactly 1 (the model
    # merged the lines into one utterance — a squash signal for the caller).
    # Any other count is ambiguous to pair positionally → fall back to per-line.
    if got != len(lines) and got != 1:
        log.debug("translate_pair count ambiguous (want=%d got=%d) - falling back", len(lines), got)
        return None
    return out
