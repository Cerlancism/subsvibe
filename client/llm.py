from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Callable

from openai import BadRequestError, OpenAI

log = logging.getLogger("subsvibe.llm")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "qwen3.5:4b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

# max_retries=0: this client is used for live translation. Retries are stale
# work — the staleness drop in pipeline._drain_stale moves on instead.
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=0)

# Commit LLM: an optional second LLM for COMMITTED (final) live lines, set up
# per live session by configure_commit_llm. When configured, translation of
# finals and the romaji correction (unless a romaji LLM is configured, see
# below) go there (e.g. OpenRouter) while
# provisional previews stay on the LLM_* client above (e.g. local Ollama).
# Not configured: every call uses LLM_*. The SDK's httpx client honours
# HTTPS_PROXY / NO_PROXY from the environment, so a remote base URL goes
# through the proxy and 127.0.0.1 (Ollama) does not.
commit_llm_client: OpenAI | None = None
commit_llm_model_id = ""
# `reasoning_effort` sent on commit-LLM calls (LIVE_COMMIT_LLM_REASONING_EFFORT).
# "none" keeps calls fast, but some models refuse it (z-ai/glm-5.3-flash: 400
# "Reasoning is mandatory"); "minimal" or "low" then costs ~1 reasoning token
# and ~1 s. Reasoning tokens count against each call's max_tokens.
commit_llm_reasoning_effort = "none"
# One id per call purpose, fixed for the session, sent as OpenRouter's
# `x-session-id` header on commit-LLM calls: OpenRouter keeps every call with
# an id on the same provider endpoint, so its prompt cache stays warm, and
# groups the id's calls in its activity log (separate ids split translation
# and romaji spend). Without one OpenRouter keys stickiness on a hash of the
# first non-system message, which for translation is the history block — it
# changes on every commit, so the sticky key would too. Other endpoints
# ignore the header.
_commit_llm_session_ids: dict[str, str] = {}

# Romaji LLM: an optional third LLM dedicated to the Japanese romaji
# correction (romanize_ja_fix), set up per live session by
# configure_romaji_llm. It takes precedence over the commit LLM for that one
# purpose, so e.g. committed lines can translate on OpenRouter while romaji
# is corrected on a local Ollama model. Not configured: romaji follows the
# commit LLM, else LLM_*.
romaji_llm_client: OpenAI | None = None
romaji_llm_model_id = ""
romaji_llm_reasoning_effort = "none"
_romaji_llm_session_id = ""


def _env(prefix: str, name: str) -> str:
    return os.environ.get(f"{prefix}_{name}", "").strip()


def _open_route(prefix: str, model: str | None, label: str, fallback: str) -> tuple[OpenAI, str, str, str] | None:
    """(client, model_id, base_url, reasoning_effort) for an optional live LLM
    route configured by `<prefix>_MODEL_ID` / `_BASE_URL` / `_API_KEY` /
    `_REASONING_EFFORT`, or None when off. `model` (a CLI flag) overrides
    `<prefix>_MODEL_ID`; base URL and key each fall back to their LLM_*
    value; reasoning effort defaults to "none". A route on its own endpoint
    with no key of its own is refused with a warning: falling back to
    LLM_API_KEY would send e.g. "ollama" to OpenRouter and 401 every call."""
    model_id = (model or _env(prefix, "MODEL_ID")).strip()
    if not model_id:
        return None
    base_url = _env(prefix, "BASE_URL") or LLM_BASE_URL
    own_key = _env(prefix, "API_KEY")
    if not own_key and base_url != LLM_BASE_URL:
        log.warning(
            "%s_API_KEY is empty - %s %s at %s is disabled; %s",
            prefix, label, model_id, base_url, fallback,
        )
        return None
    effort = _env(prefix, "REASONING_EFFORT") or "none"
    client = OpenAI(api_key=own_key or LLM_API_KEY, base_url=base_url, max_retries=0)
    return client, model_id, base_url, effort


def configure_commit_llm(model: str | None = None) -> bool:
    """Set up the commit LLM for a live session. `model` (--commit-llm-model)
    overrides LIVE_COMMIT_LLM_MODEL_ID; with neither set there is no commit
    LLM and every call stays on LLM_*. The endpoint and key come from
    LIVE_COMMIT_LLM_BASE_URL / LIVE_COMMIT_LLM_API_KEY, each falling back to
    its LLM_* counterpart; LIVE_COMMIT_LLM_REASONING_EFFORT sets its
    reasoning effort (default "none"). Returns whether a commit LLM is now
    active."""
    global commit_llm_client, commit_llm_model_id, commit_llm_reasoning_effort, _commit_llm_session_ids
    route = _open_route("LIVE_COMMIT_LLM", model, "commit LLM",
                        f"committed lines use {LLM_MODEL_ID} instead")
    if route is None:
        return False
    client, model_id, base_url, effort = route
    session_uuid = uuid.uuid4()
    _commit_llm_session_ids = {
        "translate": f"subsvibe-translate-{session_uuid}",
        "romaji": f"subsvibe-romaji-{session_uuid}",
    }
    commit_llm_client, commit_llm_model_id, commit_llm_reasoning_effort = client, model_id, effort
    log.info("live finals translate on %s at %s (reasoning %s)", model_id, base_url, effort)
    if "openrouter.ai" in base_url:
        log.info("openrouter session_id: %s", ", ".join(_commit_llm_session_ids.values()))
    return True


def configure_romaji_llm(model: str | None = None) -> bool:
    """Set up the romaji LLM for a live session: the model the ja romaji
    correction runs on, ahead of the commit LLM. `model` (--romaji-llm-model)
    overrides LIVE_ROMAJI_LLM_MODEL_ID; LIVE_ROMAJI_LLM_BASE_URL / _API_KEY
    fall back to LLM_* (not to the commit LLM), _REASONING_EFFORT defaults to
    "none". Returns whether a romaji LLM is now active."""
    global romaji_llm_client, romaji_llm_model_id, romaji_llm_reasoning_effort, _romaji_llm_session_id
    route = _open_route("LIVE_ROMAJI_LLM", model, "romaji LLM",
                        "romaji falls back to the commit LLM, else LLM_*")
    if route is None:
        return False
    client, model_id, base_url, effort = route
    romaji_llm_client, romaji_llm_model_id, romaji_llm_reasoning_effort = client, model_id, effort
    _romaji_llm_session_id = f"subsvibe-romaji-{uuid.uuid4()}"
    log.info("live romaji corrects on %s at %s (reasoning %s)", model_id, base_url, effort)
    if "openrouter.ai" in base_url:
        log.info("openrouter session_id: %s", _romaji_llm_session_id)
    return True


def _route_for(final: bool, purpose: str) -> tuple[OpenAI, str, str, str | None]:
    """(client, model, reasoning_effort, x-session-id or None) for a call: the
    romaji LLM for romaji when configured, the commit LLM for finals when
    configured, else the LLM_* default."""
    if purpose == "romaji" and romaji_llm_client is not None:
        return romaji_llm_client, romaji_llm_model_id, romaji_llm_reasoning_effort, _romaji_llm_session_id
    if final and commit_llm_client is not None:
        return (commit_llm_client, commit_llm_model_id, commit_llm_reasoning_effort,
                _commit_llm_session_ids[purpose])
    return llm_client, LLM_MODEL_ID, "none", None


def _llm_for(final: bool, purpose: str = "translate") -> tuple[OpenAI, str]:
    """(client, model) for a call — see _route_for."""
    client, model, _, _ = _route_for(final, purpose)
    return client, model

TRANSLATE_HISTORY_LEN = 10

# History is trimmed on a sawtooth rather than on every commit: the buffer is
# allowed to grow to TRANSLATE_HISTORY_LEN * this multiplier, then trims back
# down to TRANSLATE_HISTORY_LEN in one step. The point is prompt-cache reuse.
# Trimming on every append rewrites the history block in _base_messages each
# call, so the prompt prefix diverges a few tokens after the system prompt and
# the backend re-prefills nearly everything. Growing append-only keeps the
# prefix byte-identical between trims — only the trim invalidates the cache, so
# the re-prefill cost is paid once per TRANSLATE_HISTORY_LEN commits instead of
# every commit. The tradeoff is that the effective history window now varies
# between 1x and Nx the configured length rather than sitting at a fixed size.
# Set to 1 to restore trim-on-every-append (fixed window, no cache reuse).
TRANSLATE_HISTORY_LEN_UPPER_BOUND_MULTIPLIER = 2

TRANSLATE_MAX_TOKENS = 256


def _translate_system(
    target: str,
    extra_context: str | None = None,
    override: str | None = None,
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
    if extra_context:
        base += f"\n\nAdditional context from the user:\n{extra_context}"
    return base


# One committed line of translation history: (start, end, source, translation),
# start/end in seconds on the session's audio clock.
HistoryItem = tuple[float, float, str, str]


def _ts(seconds: float) -> float:
    """Timestamps as the LLM sees them: seconds, 2 decimals."""
    return round(float(seconds), 2)


def _base_messages(
    history: list[HistoryItem],
    target: str,
    extra_context: str | None,
    system_override: str | None,
) -> list[dict]:
    """System prompt + optional committed-history context block. The caller
    appends the final user turn (the utterance or utterances).

    The history is a JSON array of {start, end, source, translation} objects,
    the structure `translate_entries` replies in, and byte-identical for
    `translate` and `translate_entries`, so both reuse one cached prompt
    prefix. Rendered one object per line, append-only between sawtooth trims
    (see TRANSLATE_HISTORY_LEN_UPPER_BOUND_MULTIPLIER), so the prefix stays
    stable."""
    messages: list[dict] = [
        {"role": "system", "content": _translate_system(target, extra_context, system_override)}
    ]
    if history:
        rows = ",\n".join(
            "  " + json.dumps(
                {"start": _ts(start), "end": _ts(end), "source": raw, "translation": tr},
                ensure_ascii=False,
            )
            for start, end, raw, tr in history
        )
        messages.append({
            "role": "user",
            "content": (
                "Recent committed utterances (JSON, oldest to newest; start/end in "
                f"seconds):\n[\n{rows}\n]"
            ),
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


def _request(messages: list[dict], *, max_tokens: int, temperature: float,
             timeout: float | None, final: bool, purpose: str,
             response_format: dict | None = None, stream: bool = False):
    """One chat-completion request with the settings shared across all LLM
    stages (deterministic, no retries, no reasoning unless the commit or
    romaji LLM is configured with some); returns the completion, or the chunk stream when
    `stream`.

    Both penalties are pinned to 0: every stage here is a faithful rendering of
    its input (translation, romaji correction), so repetition present in the
    source must survive into the output. A non-zero penalty taxes tokens for
    having already appeared, which on short subtitle lines pushes the model off
    the correct word toward a synonym.
    """
    client, model, reasoning_effort, session_id = _route_for(final, purpose)
    session_header = {"extra_headers": {"x-session-id": session_id}} if session_id else {}
    # OpenAI's own API rejects `max_tokens` on its reasoning-capable models
    # (gpt-5 onward) in favour of `max_completion_tokens`; Ollama and
    # OpenRouter keep the classic name.
    token_cap = ("max_completion_tokens" if "api.openai.com" in str(client.base_url)
                 else "max_tokens")
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **{token_cap: max_tokens},
        reasoning_effort=reasoning_effort,
        frequency_penalty=0,
        presence_penalty=0,
        **({"timeout": timeout} if timeout is not None else {}),
        **({"response_format": response_format} if response_format is not None else {}),
        **({"stream": True} if stream else {}),
        **session_header,
    )


def _complete(messages: list[dict], *, max_tokens: int, temperature: float,
              timeout: float | None, final: bool = False,
              purpose: str = "translate") -> str | None:
    """One plain-text completion: the assistant text, or None on refusal /
    empty output. Callers own message construction and post-processing."""
    return _content(_request(messages, max_tokens=max_tokens, temperature=temperature,
                             timeout=timeout, final=final, purpose=purpose))


def translate(
    text: str,
    history: list[HistoryItem],
    *,
    target: str = "English",
    extra_context: str | None = None,
    system_override: str | None = None,
    temperature: float = 0,
    timeout: float | None = None,
    final: bool = False,
) -> str:
    """Translate one utterance. `final` routes to the commit LLM when one is
    configured (see configure_commit_llm).

    `history` is an ordered list of (start, end, source, translation) tuples for
    previously *committed* utterances (oldest first). It must NOT contain provisional
    output, which would otherwise pollute future calls.
    """
    messages = _base_messages(history, target, extra_context, system_override)
    messages.append({"role": "user", "content": f"Current utterance: {text}"})
    content = _complete(messages, max_tokens=TRANSLATE_MAX_TOKENS,
                        temperature=temperature, timeout=timeout, final=final)
    if content is None:
        log.warning("translate returned no output")
        return ""
    # Plain-text mode: the whole response is the translation. Collapse any stray
    # newlines (the model may wrap) into spaces so it stays a single line.
    return " ".join(content.split())


# Committed lines translate through `translate_entries`: the lines one commit
# step produced (the entries of one ASR result, or finals queued behind a slow
# call; at most TRANSLATE_BATCH_MAX) go out in one request as timed JSON, and
# the reply is a JSON array of timed entries, streamed and handed over one by
# one as each entry closes. Besides translating, the model corrects the ASR
# text, and it decides the entry count itself: it may merge or split lines,
# giving each entry the times it covers.
TRANSLATE_BATCH_MAX = 8
# Per input line: the source rewrite plus the translation plus JSON framing.
TRANSLATE_ENTRY_MAX_TOKENS = 384

# Whether translate_entries still sends a json_schema response_format. Cleared
# the first time an endpoint rejects it (400): later calls then ask for the
# JSON by prompt alone, which the stream parser reads just the same.
_json_schema_ok = True

_ENTRIES_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "entries",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                            "source": {"type": "string"},
                            "translation": {"type": "string"},
                        },
                        "required": ["start", "end", "source", "translation"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["entries"],
            "additionalProperties": False,
        },
    },
}


class _EntryStreamParser:
    """Pulls each complete object out of the first JSON array in a reply as
    the text streams in: {"entries": [{...}, {...}]} (or a bare [...]), with
    code fences or prose around it tolerated. Tracks strings and escapes so
    braces inside text don't count."""

    def __init__(self) -> None:
        self.text = ""
        self._i = 0
        self._in_array = False
        self._done = False
        self._depth = 0
        self._in_str = False
        self._esc = False
        self._obj_start = -1

    def feed(self, chunk: str) -> list[dict]:
        self.text += chunk
        out: list[dict] = []
        t = self.text
        while self._i < len(t) and not self._done:
            c = t[self._i]
            if not self._in_array:
                if c == "[":
                    self._in_array = True
            elif self._in_str:
                if self._esc:
                    self._esc = False
                elif c == "\\":
                    self._esc = True
                elif c == '"':
                    self._in_str = False
            elif c == '"':
                self._in_str = True
            elif c == "{":
                if self._depth == 0:
                    self._obj_start = self._i
                self._depth += 1
            elif c == "}" and self._depth > 0:
                self._depth -= 1
                if self._depth == 0:
                    try:
                        obj = json.loads(t[self._obj_start:self._i + 1])
                    except ValueError:
                        obj = None
                    if isinstance(obj, dict):
                        out.append(obj)
            elif c == "]" and self._depth == 0:
                self._done = True
            self._i += 1
        return out


def _num(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN -> None


def translate_entries(
    lines: list[tuple[float, float, str]],
    history: list[HistoryItem],
    *,
    on_entry: Callable[[dict], None],
    target: str = "English",
    extra_context: str | None = None,
    system_override: str | None = None,
    temperature: float = 0,
    timeout: float | None = None,
    final: bool = True,
) -> int:
    """Correct and translate consecutive committed lines, (start, end, asr_text)
    oldest first, in one streamed request. Each reply entry is handed to
    `on_entry` as soon as it closes, as {start, end, source, translation}:
    `source` is the ASR text with recognition errors corrected, `translation`
    its single-line translation. Returns the number of entries handed over.

    The model picks the entry count: normally one per line with the line's
    times, but it may merge or split lines. Times are kept inside the input
    span and starts strictly increasing (they key the committed lines); a
    missing time follows the previous entry. The prompt prefix (system +
    history) is `translate`'s, so both share a cached prompt; the
    instructions ride the last user turn and so also apply under
    `system_override`.

    `timeout` bounds the whole stream, not counting time spent inside
    `on_entry`: TimeoutError past it, after whatever entries already arrived.
    API and transport errors propagate, also after partial delivery. The
    caller finishes the lines the reply left uncovered."""
    global _json_schema_ok
    lo, hi = float(lines[0][0]), float(lines[-1][1])
    payload = json.dumps(
        [{"start": _ts(a), "end": _ts(b), "text": t} for a, b, t in lines],
        ensure_ascii=False,
    )
    messages = _base_messages(history, target, extra_context, system_override)
    messages.append({
        "role": "user",
        "content": (
            "Current utterances (JSON, consecutive, oldest first; start/end in seconds; "
            "text is raw speech recognition output and may contain recognition errors):\n"
            f"{payload}\n\n"
            "Reply with ONLY a JSON object {\"entries\": [...]}, each entry "
            "{\"start\", \"end\", \"source\", \"translation\"}, in time order:\n"
            "- source: the utterance in its original language with clear speech "
            "recognition errors fixed (misheard words, wrong homophones or characters, "
            "names made consistent with the history and context). Otherwise keep the "
            "speaker's wording exactly; never paraphrase, summarise, or add words.\n"
            f"- translation: source translated into {target} as one subtitle line.\n"
            "- start/end: the times of the utterance the entry covers, copied exactly. "
            "Keep one entry per utterance unless an utterance boundary clearly cuts a "
            "sentence badly (merge them: first start, last end) or one utterance holds "
            "clearly separate sentences (split it within its times). Cover every "
            "utterance."
        ),
    })
    kwargs = dict(max_tokens=TRANSLATE_ENTRY_MAX_TOKENS * len(lines), temperature=temperature,
                  timeout=timeout, final=final, purpose="translate", stream=True)
    deadline = time.monotonic() + timeout if timeout is not None else None
    stream = None
    if _json_schema_ok:
        try:
            stream = _request(messages, response_format=_ENTRIES_FORMAT, **kwargs)
        except BadRequestError as exc:
            log.warning("translate entries: endpoint rejected json_schema (%s) - asking by "
                        "prompt only from now on", exc)
            _json_schema_ok = False
    if stream is None:
        stream = _request(messages, **kwargs)

    parser = _EntryStreamParser()
    emitted = 0
    prev_start = lo - 1.0
    prev_end = lo
    try:
        for chunk in stream:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"translate stream past {timeout:.0f}s")
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            for raw in parser.feed(delta):
                source = raw.get("source")
                translation = raw.get("translation")
                source = " ".join(source.split()) if isinstance(source, str) else ""
                translation = " ".join(translation.split()) if isinstance(translation, str) else ""
                if not source and not translation:
                    continue
                start = _num(raw.get("start"))
                end = _num(raw.get("end"))
                start = min(max(prev_end if start is None else start, lo), hi)
                start = max(start, prev_start + 0.001)
                end = min(max(start if end is None else end, start), max(hi, start))
                prev_start, prev_end = start, end
                emitted += 1
                handed_at = time.monotonic()
                on_entry({"start": start, "end": end, "source": source,
                          "translation": translation})
                # Time spent in the callback (rendering, commit pacing) is
                # not the model's: keep it off the stream's deadline.
                if deadline is not None:
                    deadline += time.monotonic() - handed_at
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    if not emitted:
        log.debug("translate entries: no entry parsed from reply: %r", parser.text[:300])
    return emitted


ROMANIZE_JA_MAX_TOKENS = 256

# Japanese romaji corrector (the hybrid path). cutlet (utils.romanize) produces a
# mostly-correct Hepburn draft but mis-sounds on a few predictable cases:
# lexicalized particles (こんにちは->...ha), context-dependent kanji readings
# (お兄ちゃん->oanichan), and bad compound splits (月曜日->getsuyou hi). Rather than
# re-romanize from scratch — which makes small models invent phantom words — we
# hand the LLM cutlet's draft and ask it to EDIT only what mis-sounds. The draft
# anchors the model: the job shrinks to a few tokens, leaving little room to
# fabricate. Evaluated against from-scratch romanization in tests/test_ja_romaji_llm.py;
# the corrector won at 4b. The prompt's examples are deliberately minimal: at 4b
# a word-example fixes exactly the word it names (no generalization — an お姉ちゃん
# example did not fix お兄ちゃん) and examples compete for attention (adding one
# broke a previously-working fix), while a PATTERN statement (曜日 words read
# '...youbi') does generalize across words. Any example added here leaks the
# matching eval row in tests/test_ja_romaji_llm.py — keep its leak annotations in
# sync. Intended for committed (final) lines only — provisional
# previews stay pure cutlet so the ~1Hz refresh path takes no LLM call.
_ROMANIZE_JA_FIX_SYSTEM = (
    "You proofread a Hepburn romaji draft of Japanese text. The draft is produced "
    "by a dictionary tool and is mostly correct; your job is to fix ONLY the parts "
    "that mis-sound when read aloud, keeping everything else exactly as given.\n"
    "Common things the draft gets wrong:\n"
    "- Lexicalized particles in set phrases romanized by spelling instead of "
    "sound.\n"
    "- A kanji read with the wrong reading for its context: 私 in plain "
    "conversation reads 'watashi', not 'watakushi'.\n"
    "- A compound word wrongly split into separate pieces: 曜日 words read "
    "'...youbi', so a draft like 'nan'you hi' should be 'nan youbi'.\n"
    "Rules: change a word ONLY if it genuinely mis-sounds against the source. "
    "Copy every other word from the draft character-for-character — never respell "
    "a word in a different romanization style. The draft deliberately writes long "
    "vowels as doubled letters; that is correct, keep every doubled vowel exactly "
    "as written. Do NOT add, drop, or reorder words. Match the source "
    "mora-for-mora. Keep the draft's spacing and casing.\n"
    "Output ONLY the corrected romaji as plain text — no quotes, labels, or notes. "
    "Use plain ASCII letters only: no macrons or accented characters anywhere. "
    "If the draft is already correct, output it unchanged."
)


def romanize_ja_fix(
    source: str,
    draft: str,
    *,
    temperature: float = 0,
    timeout: float | None = None,
) -> str:
    """Correct cutlet's Japanese romaji `draft` against its `source` via the LLM
    (the hybrid path). Anchored on the draft so a small model edits the few
    mis-sounding tokens instead of re-romanizing — minimizing hallucination.

    Returns the corrected romaji, or the original `draft` unchanged on empty
    input / no LLM output (best-effort: never worse than cutlet alone)."""
    if not source or not source.strip():
        return draft
    messages = [
        {"role": "system", "content": _ROMANIZE_JA_FIX_SYSTEM},
        {"role": "user", "content": f"Source: {source}\nDraft: {draft}"},
    ]
    # Committed lines only: rides the romaji LLM when configured, else the
    # commit LLM, else LLM_*.
    content = _complete(messages, max_tokens=ROMANIZE_JA_MAX_TOKENS,
                        temperature=temperature, timeout=timeout, final=True,
                        purpose="romaji")
    if content is None:
        log.warning("romanize_ja_fix returned no output; keeping draft")
        return draft
    return " ".join(content.split())
