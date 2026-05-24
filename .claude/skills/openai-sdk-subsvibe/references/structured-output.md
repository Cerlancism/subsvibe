# Structured Output (Pydantic) for SubsVibe

Use this when the LLM stage must return data your code can branch on — not free-form prose. The transcription refinement stage in `./client/llm.py` currently returns a plain translation string, but anything that needs fields (confidence scores, multiple speakers, segment classification, glossary suggestions, dictionary lookups) should use structured output instead of asking the model nicely and then string-parsing the reply.

The SDK helper is `client.chat.completions.parse(...)`. It takes a Pydantic `BaseModel` as `response_format`, converts it to a JSON schema with `strict: True`, and parses the response back into a typed instance. You get `message.parsed` (a real Pydantic object) or `message.refusal` (the model declined and explained why).

## Why parse() instead of create() + json.loads

Three things `parse()` gives you that the manual route doesn't:

1. **Schema generation is automatic.** No hand-written JSON Schema, no drift between the schema and the Pydantic model.
2. **Strict mode is on.** The server-side decoder is constrained to your schema — it cannot emit malformed JSON or extra fields. For local backends (Ollama, vLLM, llama.cpp) this depends on backend support; see the compatibility note below.
3. **Refusals are first-class.** `message.refusal` is `None` on success and a string on failure. You don't have to guess from a malformed body why parsing failed.

## Basic shape

```python
from pydantic import BaseModel
from openai import OpenAI

class SubtitleRefinement(BaseModel):
    refined_text: str
    confidence: float  # 0.0-1.0
    had_corrections: bool

client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")

completion = client.chat.completions.parse(
    model="qwen3.5-instruct:4b",
    messages=[
        {"role": "system", "content": "Refine the ASR output. Flag low-confidence outputs."},
        {"role": "user", "content": "Current window: the quick brown focks jumped"},
    ],
    response_format=SubtitleRefinement,
    temperature=0,
)

message = completion.choices[0].message
if message.parsed:
    result: SubtitleRefinement = message.parsed
    if result.confidence < 0.5:
        log.warning("low confidence refinement: %s", result.refined_text)
else:
    log.warning("refusal: %s", message.refusal)
```

Note the type annotation on `result` — `message.parsed` is statically typed to your model, which means IDE completion and pyright catch field typos for free.

## Async version (preferred for pipeline stages)

The pipeline stages in SubsVibe run on their own threads and queues, but if a stage moves to `asyncio` you want the async client. Same call, just `await`:

```python
from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")

async def refine(text: str) -> SubtitleRefinement | None:
    completion = await aclient.chat.completions.parse(
        model="qwen3.5-instruct:4b",
        messages=[
            {"role": "system", "content": "..."},
            {"role": "user", "content": text},
        ],
        response_format=SubtitleRefinement,
        temperature=0,
    )
    return completion.choices[0].message.parsed
```

## Designing the schema

A few patterns that work well for subtitle-style outputs.

**Use `Literal` (or `Enum`) for finite choices.** When the model is supposed to pick from a known set — language code, speaker label, severity — give it an enum. Free-form strings invite drift.

```python
from typing import Literal
from pydantic import BaseModel

class SegmentClassification(BaseModel):
    kind: Literal["speech", "music", "silence", "noise"]
    language: Literal["en", "ja", "zh", "unknown"]
    refined_text: str
```

**Lists for repeating structure.** If the window contains multiple speakers or sentences, model that as a list rather than a single concatenated string.

```python
from pydantic import BaseModel

class Utterance(BaseModel):
    speaker: str          # e.g. "speaker_1"
    text: str

class WindowRefinement(BaseModel):
    utterances: list[Utterance]
    needs_more_context: bool
```

**Optional fields for things the model may not know.** Use `... | None` (or `Optional[...]`) and default to `None`. The model is allowed to leave them out only if you mark them optional; otherwise strict mode requires them.

```python
class GlossaryHit(BaseModel):
    surface_form: str
    canonical: str
    confidence: float
    note: str | None = None  # optional explanation
```

**Don't ask for free-form JSON.** If you find yourself reaching for `dict[str, Any]`, step back — the whole point of structured output is that the shape is known up front. If the shape is genuinely open-ended, structured output isn't the right tool.

## Handling refusals

`message.refusal` is set when the model decides not to comply (e.g., the system prompt asked for something it won't do, or the input is gibberish and it can't fill the schema). Treat this as a soft failure — log it and fall back to the un-refined text, don't crash the pipeline.

```python
message = completion.choices[0].message
if message.refusal:
    log.warning("LLM refused to refine: %s", message.refusal)
    return raw_text  # fall back, keep the pipeline flowing
return message.parsed.refined_text
```

For live subtitles this matters: dropping a window because the LLM refused is worse than showing the raw ASR output for one tick.

## Local backend compatibility

`chat.completions.parse` requires the server to honor `response_format={"type": "json_schema", ...}`. Compatibility as of writing:

- **Ollama** — supports JSON-schema-constrained output via its OpenAI-compatible endpoint for most local models. Older Ollama versions only honor `response_format={"type": "json_object"}` (no schema enforcement); upgrade if `parse()` returns malformed output.
- **vLLM / llama.cpp / LM Studio** — support depends on the version. vLLM and llama.cpp use grammar-constrained sampling and work well; LM Studio added schema support in mid-2024 releases.
- **OpenAI / Azure OpenAI** — fully supported on `gpt-4o-2024-08-06` and later snapshots.

If the backend doesn't honor the schema, you'll see `message.parsed = None` and a Pydantic validation error or malformed content. The fallback is to use plain `chat.completions.create(...)` with a strong system prompt asking for JSON and parse it manually with `json.loads` + `Model.model_validate(...)`. Less reliable, but works everywhere.

```python
import json
from pydantic import ValidationError

response = client.chat.completions.create(
    model="some-model-without-schema-support",
    messages=[
        {"role": "system", "content": "Reply with JSON matching: {refined_text: str, confidence: float}. No prose."},
        {"role": "user", "content": text},
    ],
    response_format={"type": "json_object"},  # weaker — only forces valid JSON, not your schema
    temperature=0,
)
try:
    result = SubtitleRefinement.model_validate_json(response.choices[0].message.content)
except ValidationError as e:
    log.warning("schema mismatch on fallback path: %s", e)
    return None
```

## Bounding output length

Pass `max_tokens=output_cap(messages, multiplier=5, floor=128)` on every `parse()` call — see [SKILL.md](../SKILL.md#bounding-output-length-with-max_tokens). The schema bounds *shape*, not *length*, so a model can still stall on a long string field. If the cap fires mid-response, the JSON truncates and `message.parsed` comes back `None`; `finish_reason="length"` is the tell.

## Why `temperature=0` for structured output

Structured output is for cases where you want a deterministic mapping from input to fields. Higher temperatures don't make the schema looser, but they do introduce variance in which valid completion you get — and for subtitle refinement, run-to-run variance is just noise. Set `temperature=0` unless you specifically want diversity.

## SubsVibe-applied example: classifying refinements

A refinement stage that decides whether to overwrite the raw ASR or keep it:

```python
import logging
import os
from typing import Literal
from pydantic import BaseModel
from openai import OpenAI

log = logging.getLogger("subsvibe.refine")

class RefinementDecision(BaseModel):
    action: Literal["replace", "keep_raw", "merge_with_previous"]
    refined_text: str
    reasoning: str  # short, for logs only

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "qwen3.5-instruct:4b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

_SYSTEM = (
    "You decide what to do with raw ASR output for a single sliding window. "
    "Action 'replace' if you can confidently fix errors. "
    "Action 'keep_raw' if the raw text is already fine or you're uncertain. "
    "Action 'merge_with_previous' if this window is a continuation of the prior line."
)

def refine_window(raw: str, prev: str | None) -> RefinementDecision | None:
    user = f"Previous line: {prev or '(none)'}\nCurrent window: {raw}"
    completion = client.chat.completions.parse(
        model=LLM_MODEL_ID,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format=RefinementDecision,
        temperature=0,
    )
    message = completion.choices[0].message
    if message.refusal:
        log.warning("refine refusal: %s", message.refusal)
        return None
    log.debug("refine: action=%s reason=%s", message.parsed.action, message.parsed.reasoning)
    return message.parsed
```

Downstream code branches on `decision.action` instead of trying to parse free-form English. That's the whole point.
