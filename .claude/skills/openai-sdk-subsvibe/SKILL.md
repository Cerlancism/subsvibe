---
name: openai-sdk-subsvibe
description: |
  Guide for using the OpenAI Python SDK in SubsVibe client code. Covers synchronous and asynchronous chat completion and audio transcription calls via OpenAI-compatible APIs. Invoke this when implementing transcription workers (client/transcribe.py), LLM context refinement (client/llm.py), or any code that needs to call OpenAI-compatible HTTP endpoints for chat completions or Whisper-compatible transcription using the openai SDK. Also useful when debugging API client setup, handling streaming responses, error handling with retries, or configuring base URLs for custom backends (Ollama, vLLM, etc.).
---

# OpenAI SDK for SubsVibe Client Code

SubsVibe uses the OpenAI Python SDK to call two types of APIs:
1. **Transcription**: `POST /v1/audio/transcriptions` on a Whisper-compatible server
2. **Chat Completions**: `POST /v1/chat/completions` on any OpenAI-compatible LLM server (Ollama, vLLM, LM Studio, OpenAI, etc.)

Both are configured with a `base_url` and model name. The client is backend-agnostic — it just hits HTTP APIs.

## Installation & Setup

```bash
pip install openai
```

The openai SDK (v1+) supports both sync and async clients. SubsVibe uses async for non-blocking I/O in the pipeline.

### Client Initialization

SubsVibe uses separate clients for transcription and chat completions, each pointing to a local backend.

```python
from openai import OpenAI, AsyncOpenAI

# Transcription client (points to Whisper-compatible server, default port 8000)
transcription_client = OpenAI(
    api_key="not-needed-locally",  # dummy key for local backends
    base_url="http://localhost:8000"
)

# Chat/LLM client (points to Ollama or compatible server, default port 11434)
llm_client = OpenAI(
    api_key="not-needed-locally",
    base_url="http://localhost:11434/v1"  # Ollama default
)

# Async versions (preferred for pipeline stages)
aclient_transcription = AsyncOpenAI(
    api_key="not-needed-locally",
    base_url="http://localhost:8000"
)

aclient_llm = AsyncOpenAI(
    api_key="not-needed-locally",
    base_url="http://localhost:11434/v1"
)
```

**Defaults for SubsVibe**:
- Transcription server: `http://localhost:8000` (Qwen3-ASR or Faster Whisper)
- LLM server: `http://localhost:11434/v1` (Ollama)
- API key: dummy value like `"not-needed-locally"` (local backends don't require auth)

**Key point**: For local backends, the `api_key` can be anything — the SDK doesn't validate it. Just set `base_url` and the SDK handles the rest.

## Transcription via Whisper API

SubsVibe sends WAV-encoded speech segments to a Whisper-compatible transcription server.

### Sync Transcription

```python
client = OpenAI(api_key="not-needed-locally", base_url="http://localhost:8000")

with open("speech_segment.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="qwen3-asr",  # default SubsVibe backend
        file=f,
        language="en",           # optional
        prompt="Preserve proper nouns",  # optional context
        response_format="json",  # or "text", "verbose_json"
    )

print(transcript.text)
```

### Async Transcription

```python
aclient = AsyncOpenAI(api_key="not-needed-locally", base_url="http://localhost:8000")

async def transcribe(audio_data: bytes) -> str:
    # Audio can be passed as bytes or file-like object
    transcript = await aclient.audio.transcriptions.create(
        model="qwen3-asr",
        file=("audio.wav", audio_data, "audio/wav"),
        response_format="json",
    )
    return transcript.text
```

### Response Formats

- `"json"`: Returns `{"text": "..."}` — default for SubsVibe
- `"verbose_json"`: Includes `segments`, `language`, `duration` — use if you need timestamps (backend must support it)

## Chat Completions for Context Refinement

SubsVibe uses chat completions to refine transcriptions with sliding context — recent subtitle history provides context to correct transcription errors.

### Sync Chat Completion

```python
client = OpenAI(
    api_key="not-needed-locally",
    base_url="http://localhost:11434/v1"  # Ollama (default SubsVibe LLM backend)
)

response = client.chat.completions.create(
    model="mistral",  # or "neural-chat", "llama2", etc. (whatever you have in Ollama)
    messages=[
        {"role": "system", "content": "You refine speech transcriptions..."},
        {"role": "user", "content": "Correct this: 'the quick brown fox'"},
    ],
    temperature=0.3,  # low temp for consistent corrections
)

corrected_text = response.choices[0].message.content
```

### Async Chat Completion

```python
aclient = AsyncOpenAI(
    api_key="not-needed-locally",
    base_url="http://localhost:11434/v1"  # Ollama
)

async def refine_subtitle(raw_text: str, context: list[str]) -> str:
    context_str = "\n".join(context[-3:])  # last 3 subtitles
    
    response = await aclient.chat.completions.create(
        model="mistral",
        messages=[
            {"role": "system", "content": "Fix transcription errors using context..."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nFix: {raw_text}"},
        ],
        temperature=0.3,
    )
    
    return response.choices[0].message.content
```

### Streaming Responses

For real-time subtitle display, use the `.stream()` context manager (preferred):

```python
client = OpenAI(base_url="...")

with client.chat.completions.stream(model="...", messages=[...]) as stream:
    for event in stream:
        if event.type == "content.delta":
            print(event.delta, end="", flush=True)

final = stream.get_final_completion()
```

Async streaming:

```python
aclient = AsyncOpenAI(base_url="...")

async with aclient.chat.completions.stream(model="...", messages=[...]) as stream:
    async for event in stream:
        if event.type == "content.delta":
            print(event.content, end="", flush=True)
```

The older `stream=True` parameter also works if the backend doesn't support the helpers API — iterate `chunk.choices[0].delta.content` directly in that case.

## Error Handling & Retries

The SDK raises typed exceptions. Handle them explicitly:

```python
import openai
from openai import APIConnectionError, RateLimitError, APIStatusError
import time

def transcribe_with_retry(audio_data: bytes, max_retries: int = 3) -> str:
    client = OpenAI(base_url="http://localhost:8000")
    
    for attempt in range(max_retries):
        try:
            result = client.audio.transcriptions.create(
                model="qwen3-asr",
                file=("audio.wav", audio_data, "audio/wav"),
            )
            return result.text
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
        except APIConnectionError as e:
            raise RuntimeError(f"Transcription server unavailable: {e}")
        except APIStatusError as e:
            # Non-200 response — e.status_code and e.response available
            raise RuntimeError(f"Transcription failed ({e.status_code}): {e}")
    
    raise RuntimeError("Max retries exceeded")
```

For async:

```python
import asyncio
import openai
from openai import APIConnectionError, RateLimitError, APIStatusError

async def transcribe_with_retry_async(audio_data: bytes, max_retries: int = 3) -> str:
    aclient = AsyncOpenAI(base_url="http://localhost:8000")
    
    for attempt in range(max_retries):
        try:
            result = await aclient.audio.transcriptions.create(
                model="qwen3-asr",
                file=("audio.wav", audio_data, "audio/wav"),
            )
            return result.text
        except RateLimitError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        except (APIConnectionError, APIStatusError) as e:
            raise RuntimeError(f"Transcription failed: {e}")
    
    raise RuntimeError("Max retries exceeded")
```

## Configuration via Environment

SubsVibe uses environment variables to configure backend servers:

```bash
# Transcription server (default: http://localhost:8000)
TRANSCRIPTION_BASE_URL=http://localhost:8000
TRANSCRIPTION_MODEL=qwen3-asr

# LLM/Chat server (default: http://localhost:11434/v1 for Ollama)
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=mistral

# API key (not needed for local backends, but SDK may require a dummy value)
OPENAI_API_KEY=not-needed-locally
```

In your client code:

```python
import os
from openai import OpenAI, AsyncOpenAI

TRANSCRIPTION_BASE_URL = os.environ.get("TRANSCRIPTION_BASE_URL", "http://localhost:8000")
TRANSCRIPTION_MODEL = os.environ.get("TRANSCRIPTION_MODEL", "qwen3-asr")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "mistral")
API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed-locally")

transcription_client = OpenAI(api_key=API_KEY, base_url=TRANSCRIPTION_BASE_URL)
llm_client = OpenAI(api_key=API_KEY, base_url=LLM_BASE_URL)

# Async versions
aclient_transcription = AsyncOpenAI(api_key=API_KEY, base_url=TRANSCRIPTION_BASE_URL)
aclient_llm = AsyncOpenAI(api_key=API_KEY, base_url=LLM_BASE_URL)
```

## Common Patterns for SubsVibe

### Pattern 1: Transcription Queue Worker

```python
import asyncio
import os
from openai import AsyncOpenAI

class TranscriptionWorker:
    def __init__(self):
        base_url = os.environ.get("TRANSCRIPTION_BASE_URL", "http://localhost:8000")
        model = os.environ.get("TRANSCRIPTION_MODEL", "qwen3-asr")
        api_key = os.environ.get("OPENAI_API_KEY", "not-needed-locally")
        
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    async def transcribe(self, audio_segment: bytes) -> str:
        try:
            result = await self.client.audio.transcriptions.create(
                model=self.model,
                file=("audio.wav", audio_segment, "audio/wav"),
                response_format="json",
            )
            return result.text
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
```

### Pattern 2: Context-Aware Subtitle Refinement (Ollama)

```python
import os
from openai import AsyncOpenAI

class SubtitleRefinement:
    def __init__(self):
        base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
        model = os.environ.get("LLM_MODEL", "mistral")
        api_key = os.environ.get("OPENAI_API_KEY", "not-needed-locally")
        
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    async def refine(self, raw_text: str, context_history: list[str]) -> str:
        context_str = "\n".join(context_history[-3:])  # sliding window (last 3 subtitles)
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You refine speech transcriptions using cross-segment context. "
                               "Correct errors, preserve style, don't hallucinate.",
                },
                {
                    "role": "user",
                    "content": f"Recent subtitles:\n{context_str}\n\nNew segment: {raw_text}\n\nRefine it:",
                },
            ],
            temperature=0.3,
        )
        
        return response.choices[0].message.content.strip()
```

## Debugging

### Check if server is reachable

```python
import httpx

def is_server_alive(base_url: str) -> bool:
    try:
        response = httpx.get(f"{base_url}/v1/models", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

# Usage
if not is_server_alive("http://localhost:8000"):
    raise RuntimeError("Transcription server is not running")
```

### Log API calls

```python
import logging

logging.basicConfig(level=logging.DEBUG)
# The openai SDK will log detailed request/response info
```

## Resources

For the most current SDK documentation and API reference, use the **context7 MCP tool**:
- Query the OpenAI Python SDK docs: `/mcp__context7__query-docs --libraryId /openai/openai-python --query "your question"`
- Search OpenAI API reference: `/mcp__context7__query-docs --libraryId /websites/developers_openai_api --query "your question"`

These are faster than web search and always up-to-date with the latest SDK versions and API changes.
