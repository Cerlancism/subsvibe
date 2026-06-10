# Tool Calling for SubsVibe

Use tool calling when the LLM should decide *which* of several actions to take and supply the arguments. For SubsVibe-shaped work this fits things like: "look this term up in a user glossary," "switch source language because the speaker just changed," "flag this window for manual review." The model picks the tool; your code runs it; you feed the result back. Without tools, you'd end up parsing English-as-an-API, which is brittle.

For *forcing* the model to produce structured data with no choice involved, use `response_format` instead — see [structured-output.md](structured-output.md). Tools are for branching; structured output is for a fixed shape.

## The mental model

A round of tool calling has three steps:

1. **Call** — you send messages + a tool catalogue. The model either replies in text or returns one or more `tool_calls`.
2. **Execute** — for each tool call, your code runs the actual function and produces a result.
3. **Round-trip** — you send the original messages + the assistant message (with tool calls) + a tool message per result. The model integrates the results and replies, usually with final text.

If the model is allowed to call multiple tools in a turn (`parallel_tool_calls=True`), step 2 fans out and step 3 still happens once with all the results bundled.

## Defining tools

The SDK accepts two equivalent forms. Use the Pydantic helper unless you have a reason to hand-write JSON Schema.

### Pydantic helper (recommended)

```python
import openai
from pydantic import BaseModel, Field

class LookupGlossary(BaseModel):
    """Look up a term in the user's personal glossary for canonical spelling."""
    term: str = Field(..., description="The surface form to look up, as the model heard it.")
    language: str = Field(..., description="ISO 639-1 code of the source language, e.g. 'ja'.")

tools = [openai.pydantic_function_tool(LookupGlossary)]
```

The helper:
- Names the tool after the class (`LookupGlossary`); override with `name="lookup_glossary"` if you want snake_case.
- Uses the class docstring as the description; override with `description=...`.
- Sets `strict: True` and generates a strict JSON Schema from the model.

### Raw schema (fallback)

If you're targeting a backend that doesn't support strict mode or you need a shape Pydantic can't easily express:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "lookup_glossary",
        "description": "Look up a term in the user's personal glossary.",
        "parameters": {
            "type": "object",
            "properties": {
                "term": {"type": "string"},
                "language": {"type": "string"},
            },
            "required": ["term", "language"],
            "additionalProperties": False,
        },
    },
}]
```

Hand-written schemas drift from your actual function signature over time. The Pydantic helper keeps them in sync.

## Step 1 — call with the tool catalogue

```python
from openai import OpenAI

client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")

messages = [
    {"role": "system", "content": "You refine ASR output. Use tools when uncertain."},
    {"role": "user", "content": "Current window: 'we deployed kubarnetties yesterday'"},
]

response = client.chat.completions.create(
    model="frob/qwen3.5-instruct:4b",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # let the model decide; "required" forces a tool call
    temperature=0,
)

choice = response.choices[0]
```

`tool_choice` options:
- `"auto"` (default) — model decides between calling tools and replying in text.
- `"required"` — model must call at least one tool.
- `"none"` — disable tool use for this turn.
- `{"type": "function", "function": {"name": "lookup_glossary"}}` — force a specific tool.

`parallel_tool_calls=True` (the default for capable models) lets the model emit multiple tool calls in a single response, e.g., looking up several terms at once.

## Step 2 — execute the tool calls

```python
import json

assistant_message = choice.message

if not assistant_message.tool_calls:
    # Plain text reply — model didn't need tools.
    return assistant_message.content

tool_results = []
for tool_call in assistant_message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)  # always a JSON-encoded string

    if name == "LookupGlossary":
        result = my_glossary.lookup(term=args["term"], language=args["language"])
    else:
        result = {"error": f"unknown tool: {name}"}

    tool_results.append({
        "tool_call_id": tool_call.id,
        "role": "tool",
        "content": json.dumps(result),  # tool messages must be strings
    })
```

A few things that bite people:

- `tool_call.function.arguments` is a **JSON-encoded string**, not a dict. Run it through `json.loads`. (With `parse()` you'd get `parsed_arguments` as a typed Pydantic instance instead — see below.)
- The `role: "tool"` message MUST include `tool_call_id` matching the id from the assistant's tool call. The model uses this to pair requests and results.
- Tool result `content` is a string. Encode dicts with `json.dumps`. The model parses it back.
- Unknown tool names are your responsibility — return an error result rather than raising. The model can then react.

## Step 3 — send results back

```python
messages.append(assistant_message)  # the assistant turn with tool_calls
messages.extend(tool_results)        # one tool message per call

final = client.chat.completions.create(
    model="frob/qwen3.5-instruct:4b",
    messages=messages,
    tools=tools,         # keep the catalogue available in case it wants another round
    temperature=0,
)

print(final.choices[0].message.content)
```

The model usually replies with text after seeing tool results. If it decides it needs more information, it can emit another round of `tool_calls` — loop until `assistant_message.tool_calls` is empty.

## Auto-parsed tool arguments with `parse()`

If you want typed tool arguments without the `json.loads` dance, use `client.chat.completions.parse(...)` with `pydantic_function_tool`. The SDK validates the arguments against the Pydantic model and gives you `parsed_arguments` directly.

```python
import openai
from openai import OpenAI
from pydantic import BaseModel

class LookupGlossary(BaseModel):
    """Look up a term in the user's personal glossary."""
    term: str
    language: str

client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")

completion = client.chat.completions.parse(
    model="frob/qwen3.5-instruct:4b",
    messages=[
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
    ],
    tools=[openai.pydantic_function_tool(LookupGlossary)],
)

tool_call = (completion.choices[0].message.tool_calls or [None])[0]
if tool_call:
    args: LookupGlossary = tool_call.function.parsed_arguments
    result = my_glossary.lookup(term=args.term, language=args.language)
```

`parsed_arguments` is a real Pydantic instance — IDE autocompletion works, types are checked. This is the better path when you control the tool definitions.

## Parallel tool calls

When `parallel_tool_calls=True`, treat `assistant_message.tool_calls` as a list of independent calls and ideally execute them concurrently:

```python
import asyncio
import json

async def run_one(tool_call, dispatch):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    result = await dispatch(name, args)
    return {
        "tool_call_id": tool_call.id,
        "role": "tool",
        "content": json.dumps(result),
    }

async def run_all(assistant_message, dispatch):
    return await asyncio.gather(*(
        run_one(tc, dispatch) for tc in assistant_message.tool_calls
    ))
```

For local backends, parallel tool calling support varies — Ollama supports it on recent models, smaller models often serialize even when you allow parallel. If you only ever expect one tool call, set `parallel_tool_calls=False` to be explicit.

## Streaming tool calls

When you need to surface "the model is thinking about calling X" before the call is complete (e.g., to show a spinner with the tool name), use the streaming helper. Tool-call events arrive as `tool_calls.function.arguments.delta` (raw JSON chunks) and `tool_calls.function.arguments.done` (the parsed arguments, if using `pydantic_function_tool`).

```python
with client.chat.completions.stream(
    model="frob/qwen3.5-instruct:4b",
    messages=[...],
    tools=[openai.pydantic_function_tool(LookupGlossary)],
    parallel_tool_calls=True,
) as stream:
    for event in stream:
        if event.type == "tool_calls.function.arguments.delta":
            # event.name, event.arguments (raw JSON so far), event.index
            update_ui(f"calling {event.name}...")
        elif event.type == "tool_calls.function.arguments.done":
            # event.parsed_arguments is the typed Pydantic instance
            pass

    final = stream.get_final_completion()
    for tool_call in final.choices[0].message.tool_calls or []:
        # Execute and round-trip, same as non-streaming flow.
        pass
```

Streaming tool calls is mostly a UX nicety. The execution + round-trip pattern is the same.

## Designing tools well

Keep this in mind when choosing what to expose:

- **One tool, one purpose.** A `query_database` tool with a free-form SQL string is a thin wrapper over the LLM hallucinating SQL. Split it into `lookup_term`, `find_segment`, etc., each with constrained arguments.
- **Make arguments closed.** Use enums (`Literal[...]`) or constrained types wherever the domain is finite. Open-ended strings invite the model to make up values.
- **Return structured results.** Tool result content is a string, but the convention is to encode JSON. Give the model a consistent shape (e.g., `{"found": true, "canonical": "Kubernetes"}`) so it can branch on the result.
- **Don't expose tools the model shouldn't use.** If a tool is dangerous or expensive, gate it before sending it to the model rather than relying on prompt instructions.

## Bounding output length

Tool calling has two runaway modes: a single response can stall on a long argument, and the multi-round loop can ping-pong forever. `max_rounds` in the example below handles the loop; `max_tokens=output_cap(messages, multiplier=5, floor=128)` handles each call — see [SKILL.md](../SKILL.md#bounding-output-length-with-max_tokens). Recount per round, since the message history grows with each tool result.

## Local backend compatibility

Tool calling support on local backends is uneven:

- **Ollama** — supports tool calling on models tagged as tool-capable (recent Qwen, Llama 3.1+, Mistral, etc.). Strict mode and parallel calls vary by model.
- **vLLM** — supports tool calling via its OpenAI-compatible server; quality depends on the underlying model's training.
- **llama.cpp / LM Studio** — support landed in 2024 releases; check the version.
- **OpenAI / Azure** — fully supported on all recent snapshots.

If a tool call returns malformed arguments or the model ignores tools entirely, the model probably wasn't trained for tool use. Either swap to a tool-capable model or fall back to structured output for the same task.

## SubsVibe-applied example: glossary lookup during refinement

A refinement worker that calls a local glossary tool when it sees an unfamiliar term:

```python
import json
import logging
import openai
from openai import OpenAI
from pydantic import BaseModel, Field

log = logging.getLogger("subsvibe.refine")
client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")


class LookupGlossary(BaseModel):
    """Look up the canonical spelling of a term in the user's glossary."""
    term: str = Field(..., description="The term as the ASR transcribed it.")


class FlagForReview(BaseModel):
    """Mark this window as low-confidence for later human review."""
    reason: str = Field(..., description="One short sentence on why this is uncertain.")


TOOLS = [
    openai.pydantic_function_tool(LookupGlossary),
    openai.pydantic_function_tool(FlagForReview),
]


def _dispatch(name: str, args: dict, glossary) -> dict:
    if name == "LookupGlossary":
        hit = glossary.find(args["term"])
        return {"found": hit is not None, "canonical": hit}
    if name == "FlagForReview":
        log.info("flagged: %s", args["reason"])
        return {"acknowledged": True}
    return {"error": f"unknown tool: {name}"}


def refine_with_tools(raw_text: str, glossary, max_rounds: int = 3) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Refine ASR output. Use LookupGlossary for unfamiliar names or "
                "jargon. Use FlagForReview if you can't refine confidently. "
                "Otherwise reply with just the refined text."
            ),
        },
        {"role": "user", "content": f"Window: {raw_text}"},
    ]

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model="frob/qwen3.5-instruct:4b",
            messages=messages,
            tools=TOOLS,
            temperature=0,
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return (msg.content or "").strip()

        messages.append(msg)
        for tc in msg.tool_calls:
            result = _dispatch(tc.function.name, json.loads(tc.function.arguments), glossary)
            messages.append({
                "tool_call_id": tc.id,
                "role": "tool",
                "content": json.dumps(result),
            })

    log.warning("refine: tool loop hit max_rounds")
    return raw_text  # fall back to raw if the model never settles
```

The `max_rounds` bound is important — without it, a confused model can ping-pong tool calls forever and stall the pipeline.
