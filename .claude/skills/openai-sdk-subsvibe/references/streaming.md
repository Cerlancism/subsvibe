# Streaming Chat Completions for SubsVibe

Use streaming when you want to show partial output as it arrives, or when you need to react to tokens before the full completion is done. For SubsVibe specifically: streaming the translation/refinement stage lets the subtitle bar update mid-thought instead of going dark for the full LLM latency.

The SDK exposes two streaming APIs. Prefer the helper unless you have a reason not to.

| API | When |
|---|---|
| `client.chat.completions.stream(...)` (context manager, typed events) | First choice. Handles event dispatch, gives you a final `ChatCompletion` at the end, works with structured output and tool calls. |
| `client.chat.completions.create(..., stream=True)` (raw chunk iterator) | Fallback for backends that don't speak the helper API correctly, or when you specifically need raw `ChatCompletionChunk` objects. |

## Plain text streaming (sync)

The helper yields typed events. The one you almost always care about is `content.delta` — the new text appended this step.

```python
from openai import OpenAI

client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")

with client.chat.completions.stream(
    model="qwen3.5-instruct:4b",
    messages=[
        {"role": "system", "content": "Translate to English."},
        {"role": "user", "content": "こんにちは、元気ですか？"},
    ],
    temperature=0,
) as stream:
    for event in stream:
        if event.type == "content.delta":
            print(event.delta, end="", flush=True)

    final = stream.get_final_completion()
    # final is a ChatCompletion — final.choices[0].message.content is the full text
```

The `with` block matters: it ensures the underlying HTTP connection closes even if you break out of the loop or raise.

## Plain text streaming (async)

For pipeline stages, async is the right shape — you're already mixing I/O with queue waits.

```python
from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")

async def translate_streaming(text: str, on_delta):
    """Calls on_delta(str) for each new chunk; returns the final string."""
    async with aclient.chat.completions.stream(
        model="qwen3.5-instruct:4b",
        messages=[
            {"role": "system", "content": "Translate to English."},
            {"role": "user", "content": text},
        ],
        temperature=0,
    ) as stream:
        async for event in stream:
            if event.type == "content.delta":
                on_delta(event.delta)

        final = await stream.get_final_completion()
        return final.choices[0].message.content or ""
```

The callback pattern fits SubsVibe well — the live subtitle widget can be the `on_delta` consumer, repainting on each delta. When the stream finishes you've also got the complete final text for logging / history.

## Event types you'll actually see

The helper emits a stream of typed events. The common ones, in roughly the order they fire:

- `content.delta` — `event.delta` is the new text (str). Most code only handles this.
- `content.done` — `event.content` is the full text built so far. Fires once when text generation finishes.
- `chunk` — the raw `ChatCompletionChunk`. Use only if you need access to fields the helper doesn't surface (e.g., custom finish reasons from a non-OpenAI backend).
- `refusal.delta` / `refusal.done` — only fire when structured output is on and the model refuses; see below.
- `tool_calls.function.arguments.delta` / `.done` — only when tools are in play; see [tool-calling.md](tool-calling.md).

You can branch on `event.type` with an `if/elif` chain. Don't worry about unknown event types — just ignore anything you don't care about.

## Cancelling a stream

If the user moves the playhead (skips ahead, restarts the source), an in-flight LLM call becomes wasted work. The async helper supports cancellation via the surrounding task:

```python
import asyncio

task = asyncio.create_task(translate_streaming(text, on_delta))
# ... later, if the window changes ...
task.cancel()
try:
    await task
except asyncio.CancelledError:
    pass  # connection torn down by the context manager
```

For the sync helper, break out of the `for event in stream` loop — the `with` block will close the underlying connection on exit.

## Structured output + streaming

The same helper streams structured output too. Pass `response_format=YourModel` like with `.parse()`. You get the same `content.delta` events as text streaming, plus you can call `stream.get_final_completion()` at the end to receive a `ParsedChatCompletion` where `message.parsed` is your Pydantic instance.

```python
from pydantic import BaseModel
from openai import OpenAI

class SubtitleRefinement(BaseModel):
    refined_text: str
    confidence: float
    had_corrections: bool

client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")

with client.chat.completions.stream(
    model="qwen3.5-instruct:4b",
    messages=[
        {"role": "system", "content": "Refine the ASR output, score your confidence."},
        {"role": "user", "content": "the quick brown focks jumped"},
    ],
    response_format=SubtitleRefinement,
    temperature=0,
) as stream:
    for event in stream:
        if event.type == "content.delta":
            # Mid-stream: partial JSON, not yet parseable.
            # Show progress, but don't try to act on the fields yet.
            pass

    final = stream.get_final_completion()
    if final.choices[0].message.refusal:
        log.warning("refusal: %s", final.choices[0].message.refusal)
    else:
        result: SubtitleRefinement = final.choices[0].message.parsed
        # result is fully populated and type-checked here
```

A small gotcha: **mid-stream content is partial JSON, not a partial Pydantic instance.** The helper doesn't try to validate every delta against the schema (that wouldn't even be possible — JSON isn't valid until the last `}`). If you want to display fields as they arrive, you can either:

1. Wait for the final completion and show the whole result at once (simplest, what most code should do).
2. Maintain a running `accumulated += event.delta` and attempt `json.loads` periodically — most attempts will fail until the stream is complete. Only worth it if you're showing a single string field and want a typewriter effect.

For SubsVibe-style subtitle UI, option 1 is almost always right. The total LLM latency for a small refinement is short enough that waiting for the full structured object beats showing fragments.

## Structured output streaming (async)

Same shape:

```python
async with aclient.chat.completions.stream(
    model="qwen3.5-instruct:4b",
    messages=[...],
    response_format=SubtitleRefinement,
    temperature=0,
) as stream:
    async for event in stream:
        if event.type == "content.delta":
            update_progress_indicator()

    final = await stream.get_final_completion()
    return final.choices[0].message.parsed
```

## Handling refusals while streaming

When structured output is on, the model may emit a refusal instead of content. The helper fires `refusal.delta` events (like `content.delta` but for the refusal text) followed by `refusal.done`. After the stream completes, `final.choices[0].message.refusal` will be set and `.parsed` will be `None`.

```python
async with aclient.chat.completions.stream(
    model=...,
    messages=...,
    response_format=SubtitleRefinement,
) as stream:
    refusal_text = ""
    async for event in stream:
        if event.type == "content.delta":
            on_delta(event.delta)
        elif event.type == "refusal.delta":
            refusal_text += event.delta

    if refusal_text:
        log.warning("refusal during stream: %s", refusal_text)
        return None
    final = await stream.get_final_completion()
    return final.choices[0].message.parsed
```

## Fallback: raw `stream=True`

Some backends (older Ollama, custom servers) don't implement the helper's full event protocol. If you see the helper hanging or returning no events, drop to the raw iterator:

```python
response = client.chat.completions.create(
    model="some-old-backend",
    messages=[...],
    stream=True,
)
for chunk in response:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

`chunk` here is a `ChatCompletionChunk`. There's no `.parse()` equivalent in raw mode, and structured output is best-effort — you'll have to accumulate `.content` deltas and `json.loads` at the end yourself. Avoid raw mode for structured output if at all possible; the helper exists to manage exactly this pain.

## Bounding output length

Streaming makes runaway visible, not cheaper. Pass `max_tokens=output_cap(messages)` on `.stream(...)` the same way you would on `.create(...)` — see [SKILL.md](../SKILL.md#bounding-output-length-with-max_tokens) for the helper and multipliers. When the cap is hit mid-stream, the helper still fires `content.done` cleanly and `get_final_completion()` returns with `finish_reason="length"`.

## Why not always stream?

Streaming adds connection-management complexity (context managers, cancellation, partial state). For LLM calls that produce a short refinement and feed something offline — batch transcription, post-processing — the non-streaming `parse()` or `create()` call is simpler and just as fast end-to-end. Stream when there's a human watching and the time-to-first-token matters; otherwise don't.

## SubsVibe-applied example: streaming translation with cancellation

A translation worker that streams English text into the subtitle widget and can be cancelled when the window shifts:

```python
import asyncio
import logging
import os
from openai import AsyncOpenAI

log = logging.getLogger("subsvibe.translate")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "qwen3.5-instruct:4b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")
aclient = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


class TranslationWorker:
    def __init__(self):
        self._current: asyncio.Task | None = None

    async def translate(self, text: str, history: list[tuple[str, str]], on_delta):
        # Cancel any in-flight translation for the previous window.
        if self._current and not self._current.done():
            self._current.cancel()

        self._current = asyncio.create_task(self._stream(text, history, on_delta))
        try:
            return await self._current
        except asyncio.CancelledError:
            log.debug("translation cancelled (window shifted)")
            return None

    async def _stream(self, text, history, on_delta):
        messages = _build_messages(text, history)  # same logic as client/llm.py
        async with aclient.chat.completions.stream(
            model=LLM_MODEL_ID,
            messages=messages,
            temperature=0,
        ) as stream:
            async for event in stream:
                if event.type == "content.delta":
                    on_delta(event.delta)
            final = await stream.get_final_completion()
            return (final.choices[0].message.content or "").strip()
```

The cancellation pattern is the important part — without it, slow LLM responses pile up behind a fast-moving source and you end up displaying stale translations of windows that have already scrolled off-screen.
