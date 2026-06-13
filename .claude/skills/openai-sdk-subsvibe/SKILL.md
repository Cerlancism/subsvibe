---
name: openai-sdk-subsvibe
description: |
  Guide for using the OpenAI Python SDK in SubsVibe client code. Covers synchronous and asynchronous chat completion and audio transcription calls via OpenAI-compatible APIs (Ollama, vLLM, LM Studio, OpenAI), plus structured output with Pydantic, streaming responses, and tool / function calling. Invoke this when implementing transcription workers (`./client/transcribe.py`), LLM context refinement or translation (`./client/llm.py`), or any code that needs to call OpenAI-compatible HTTP endpoints for chat completions or Whisper-compatible transcription using the openai SDK. Also use this when adding structured / JSON-mode outputs with response_format, streaming tokens to the subtitle UI, function calling for glossary lookup or branching logic, debugging API client setup, handling streaming responses, error handling with retries, or configuring base URLs for custom backends.
---

# OpenAI SDK for SubsVibe Client Code

SubsVibe uses the OpenAI Python SDK to call two types of APIs:
1. **Transcription**: `POST /v1/audio/transcriptions` on a Whisper-compatible server
2. **Chat Completions**: `POST /v1/chat/completions` on any OpenAI-compatible LLM server (Ollama, vLLM, LM Studio, OpenAI, etc.)

Both are configured with a `base_url` and model name. The client is backend-agnostic — it just hits HTTP APIs.

## When to read each reference

This file covers the core: setup, transcription, basic chat completions, error handling, env config. For richer LLM features, jump to the matching reference:

| Topic | Read | When |
|---|---|---|
| Pydantic-typed JSON responses, `chat.completions.parse(...)`, refusals | `references/structured-output.md` | The LLM stage needs to return fields your code branches on (confidence scores, decisions, classifications) — not free-form prose. |
| Real-time token streaming, partial output, cancellation, structured output streaming | `references/streaming.md` | You want to update the subtitle UI mid-response, or you need to react to tokens as they arrive. Includes plain streaming AND streaming with `response_format`. |
| Function / tool calling, `pydantic_function_tool`, multi-round loops, parallel tool calls | `references/tool-calling.md` | The LLM should *choose* between several actions (glossary lookup, flag-for-review, language switch) and you'll execute the chosen action and feed the result back. |

If you're not sure which to read, the rule of thumb is:
- "Make the model return a specific shape" → structured output
- "Make the model produce output token-by-token" → streaming
- "Let the model decide which function to call" → tool calling

The three compose: streaming + structured output is one call; tool calling + streaming is one call. The references cross-link where it matters.

## Installation & Setup

```bash
pip install openai
```

The openai SDK (v1+) supports both sync and async clients. SubsVibe uses **sync** clients inside dedicated worker threads (the pipeline is thread + queue based, not asyncio); the async client is shown for completeness.

### Client Initialization

SubsVibe uses separate clients for transcription and chat completions, each pointing to a local backend.

```python
from openai import OpenAI, AsyncOpenAI

# Transcription client (points to Whisper-compatible server, default port 8000)
transcription_client = OpenAI(
    api_key="not-needed-locally",  # dummy key for local backends
    base_url="http://127.0.0.1:8000/v1"  # note the /v1 suffix (TRANSCRIPT_BASE_URL)
)

# Chat/LLM client (points to Ollama or compatible server, default port 11434)
llm_client = OpenAI(
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1"  # Ollama default
)

# Async versions (not used by the pipeline; shown for completeness)
aclient_transcription = AsyncOpenAI(api_key="not-needed-locally", base_url="http://127.0.0.1:8000/v1")
aclient_llm = AsyncOpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")
```

**Defaults for SubsVibe**:
- Transcription server: `http://127.0.0.1:8000/v1` (Faster Whisper, Qwen3-ASR, anime-whisper)
- LLM server: `http://127.0.0.1:11434/v1` (Ollama)
- API key: dummy value (local backends don't validate it)

**On Windows, use `127.0.0.1` not `localhost`** — `localhost` triggers an IPv6 resolution delay that adds noticeable latency on every request.

## Transcription via Whisper API

SubsVibe sends WAV-encoded speech segments to a Whisper-compatible transcription server.

### Sync Transcription

```python
client = OpenAI(api_key="not-needed-locally", base_url="http://127.0.0.1:8000")

with open("speech_segment.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="qwen3-asr",
        file=f,
        language="en",                   # optional
        prompt="Preserve proper nouns",  # optional context
        response_format="json",          # or "text", "verbose_json"
    )

print(transcript.text)
```

### Async Transcription

```python
aclient = AsyncOpenAI(api_key="not-needed-locally", base_url="http://127.0.0.1:8000")

async def transcribe(audio_data: bytes) -> str:
    transcript = await aclient.audio.transcriptions.create(
        model="qwen3-asr",
        file=("audio.wav", audio_data, "audio/wav"),
        response_format="json",
    )
    return transcript.text
```

### Response Formats

- `"json"` — `{"text": "..."}`. Default for SubsVibe.
- `"verbose_json"` — includes `segments`, `language`, `duration`. Use if you need timestamps (backend must support it).
- `"text"` — plain string body.

## Chat Completions for Context Refinement

SubsVibe uses chat completions to refine and translate transcriptions with sliding context — recent subtitle history is sent alongside the current window so the LLM can correct cross-segment errors.

### Sync Chat Completion

```python
client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")

response = client.chat.completions.create(
    model="frob/qwen3.5-instruct:4b",
    messages=[
        {"role": "system", "content": "You refine speech transcriptions..."},
        {"role": "user", "content": "Correct this: 'the quick brown fox'"},
    ],
    temperature=0,  # deterministic for refinement
)

corrected_text = response.choices[0].message.content
```

### Async Chat Completion

```python
aclient = AsyncOpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")

async def refine_subtitle(raw_text: str, context: list[str]) -> str:
    context_str = "\n".join(context[-3:])  # last 3 subtitles
    response = await aclient.chat.completions.create(
        model="frob/qwen3.5-instruct:4b",
        messages=[
            {"role": "system", "content": "Fix transcription errors using context..."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nFix: {raw_text}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content
```

### Bounding output length with `max_tokens`

Always set `max_tokens` on chat completions. Without it, a confused model can keep generating until the backend's hard limit (often thousands of tokens) — for a real-time subtitle pipeline that's a multi-second stall on a single window, which freezes the UI and lets new audio pile up behind it. The cost of a too-tight cap (a truncated reply) is much smaller than the cost of unbounded latency.

**The rule: estimate the input size, then cap output at roughly 3× that, with a floor of ~64.** For refinement/translation the output should be similar in length to the input; the 3× buffer covers languages that expand (Japanese → English often grows 1.5–2×) plus a safety margin without leaving the cap effectively unset.

SubsVibe already has the right tokenizer library: `transformers` is in the lockfile (transitively via qwen-asr / faster-whisper). Use `AutoTokenizer` loaded once at module scope — matched to the LLM you're calling, *not* `tiktoken`, which uses OpenAI's BPE and over-counts for Qwen/Llama/Mistral.

```python
import os
from functools import lru_cache
from transformers import AutoTokenizer

LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "frob/qwen3.5-instruct:4b")

# A HF repo id for the tokenizer of the model you're serving via Ollama/vLLM.
# Ollama uses model tags like "frob/qwen3.5-instruct:4b"; map to the HF repo here.
LLM_TOKENIZER_REPO = os.environ.get("LLM_TOKENIZER_REPO", "Qwen/Qwen2.5-4B-Instruct")

@lru_cache(maxsize=1)
def _tokenizer():
    return AutoTokenizer.from_pretrained(LLM_TOKENIZER_REPO)

def count_message_tokens(messages: list[dict]) -> int:
    tok = _tokenizer()
    # apply_chat_template returns the exact token sequence the model will see,
    # including role markers — more accurate than tokenizing each content string.
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    return len(ids)

def output_cap(messages: list[dict], multiplier: int = 3, floor: int = 64) -> int:
    return max(floor, count_message_tokens(messages) * multiplier)
```

Then plug it into every call:

```python
messages = [
    {"role": "system", "content": _TRANSLATE_SYSTEM},
    {"role": "user", "content": f"Current window: {text}"},
]

resp = llm_client.chat.completions.create(
    model=LLM_MODEL_ID,
    messages=messages,
    temperature=0,
    max_tokens=output_cap(messages),
)
```

Multipliers by mode (use the same `output_cap` helper, just pass the right multiplier):

| Mode | Multiplier | Floor | Why |
|---|---|---|---|
| Plain text refinement / translation | 3 | 64 | Output is roughly input-sized; buffer covers language expansion. |
| Structured output (`response_format`) | 5 | 128 | JSON overhead: field names, quotes, braces, escaping. |
| Tool calling | 5 | 128 | Tool argument JSON has the same overhead; recount per round since history grows. |

Finer points:

- The tokenizer repo doesn't have to match byte-for-byte — a same-family cousin is accurate to within a few percent, plenty for sizing a cap.
- If the HF tokenizer can't load (no network, gated model), fall back to `tokens ≈ len(text) // 3` for English, `// 2` for CJK, and widen the multiplier one step.
- `max_tokens` is a ceiling, not a target — the model still stops early when it's done. A 200-token cap on a 30-token reply costs nothing.
- When you see `finish_reason="length"` in logs, the cap was hit mid-generation. Investigate the prompt first; usually the system message asks for something the model reads as "keep going forever". Raising the cap is the wrong fix.

### Going beyond plain text

Plain text completions are fine when the LLM's job is to output one string that gets shown to the user. For anything more structured:

- Need typed fields back? → `references/structured-output.md`
- Need partial output in the UI as it arrives? → `references/streaming.md`
- Need the model to pick from several actions? → `references/tool-calling.md`

## Error Handling & Retries

The SDK raises typed exceptions. Handle them explicitly rather than catching `Exception`:

```python
import time
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError

def transcribe_with_retry(audio_data: bytes, max_retries: int = 3) -> str:
    client = OpenAI(api_key="not-needed-locally", base_url="http://127.0.0.1:8000")
    for attempt in range(max_retries):
        try:
            result = client.audio.transcriptions.create(
                model="qwen3-asr",
                file=("audio.wav", audio_data, "audio/wav"),
            )
            return result.text
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except APIConnectionError as e:
            raise RuntimeError(f"Transcription server unavailable: {e}")
        except APIStatusError as e:
            raise RuntimeError(f"Transcription failed ({e.status_code}): {e}")
    raise RuntimeError("Max retries exceeded")
```

Async version uses `asyncio.sleep` and the `AsyncOpenAI` client; the exception types are the same.

For live-pipeline stages, prefer logging and falling back to the raw input over raising — a dropped window is worse than a slightly imperfect subtitle. For batch transcription (`--input file.mp3`), it's the other way around: fail fast so the user sees the error.

## Configuration via Environment

SubsVibe uses environment variables (set in `./scripts/env.sh`) to configure backend servers. The defaults match Ollama on localhost.

```bash
# Transcription server
TRANSCRIPT_BASE_URL=http://127.0.0.1:8000
TRANSCRIPT_MODEL_ID=qwen3-asr

# LLM/Chat server
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL_ID=qwen3.5-instruct:4b
LLM_API_KEY=ollama
```

In your client code (matches `./client/llm.py`):

```python
import os
from openai import OpenAI, AsyncOpenAI

TRANSCRIPT_BASE_URL = os.environ.get("TRANSCRIPT_BASE_URL", "http://127.0.0.1:8000")
TRANSCRIPT_MODEL_ID = os.environ.get("TRANSCRIPT_MODEL_ID", "qwen3-asr")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "frob/qwen3.5-instruct:4b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

transcription_client = OpenAI(api_key="not-needed-locally", base_url=TRANSCRIPT_BASE_URL)
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

aclient_transcription = AsyncOpenAI(api_key="not-needed-locally", base_url=TRANSCRIPT_BASE_URL)
aclient_llm = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
```

## Common Patterns for SubsVibe

### Pattern 1: Transcription Queue Worker

```python
import asyncio
import os
from openai import AsyncOpenAI

class TranscriptionWorker:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key="not-needed-locally",
            base_url=os.environ.get("TRANSCRIPT_BASE_URL", "http://127.0.0.1:8000"),
        )
        self.model = os.environ.get("TRANSCRIPT_MODEL_ID", "qwen3-asr")

    async def transcribe(self, audio_segment: bytes) -> str:
        result = await self.client.audio.transcriptions.create(
            model=self.model,
            file=("audio.wav", audio_segment, "audio/wav"),
            response_format="json",
        )
        return result.text
```

### Pattern 2: Context-Aware Subtitle Refinement

```python
import os
from openai import AsyncOpenAI

class SubtitleRefinement:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.environ.get("LLM_API_KEY", "ollama"),
            base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
        )
        self.model = os.environ.get("LLM_MODEL_ID", "frob/qwen3.5-instruct:4b")

    async def refine(self, raw_text: str, context_history: list[str]) -> str:
        context_str = "\n".join(context_history[-3:])
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Refine using cross-segment context. Don't hallucinate."},
                {"role": "user", "content": f"Context:\n{context_str}\n\nNew: {raw_text}"},
            ],
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip()
```

For variations on Pattern 2 — typed output, streaming to the UI, tool-driven glossary lookup — see the matching reference file.

## Debugging

### Check if server is reachable

```python
import httpx

def is_server_alive(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/v1/models", timeout=2).status_code == 200
    except Exception:
        return False

if not is_server_alive("http://127.0.0.1:8000"):
    raise RuntimeError("Transcription server is not running")
```

### Log API calls

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# The openai SDK logs detailed request/response info at DEBUG level.
```

## Live API reference

For exact, current SDK syntax and recently-added features, use the **context7 MCP tool** — it's faster and more current than web search:

- OpenAI Python SDK: `mcp__context7__query-docs` with `libraryId: /openai/openai-python`
- OpenAI API reference: `mcp__context7__query-docs` with `libraryId: /websites/developers_openai_api`

Useful when you hit a less-common parameter, want to confirm an API still exists in the installed SDK version, or need to see how a feature changed between releases.
