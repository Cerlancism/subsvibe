---
name: transcription-ref
description: >
  Guide for navigating the SubsVibe transcription reference implementations.
  Use this skill whenever working on the transcription server (server/server.py,
  server/model.py), the client transcription worker (client/transcribe.py),
  designing API endpoints, handling audio input, WAV encoding, integrating model
  backends (Faster Whisper or Qwen3-ASR), or any question about how transcription
  should behave end-to-end. Also trigger when the user asks "how does X work in
  the reference" or "check the reference for Y".
---

# SubsVibe Transcription - Reference Guide

Reference implementations live in `./references/`. They inform the design of
both `./server/` and `./client/transcribe.py`. Read them when you need to understand
how something should work, but do not modify them.

`./references/.keep` is the source of truth for which repos belong here - one URL
per line. Before relying on the references, read `.keep` and check that each URL's
target directory exists under `./references/`. For any that's missing, `git clone`
it into `./references/` (the directory name is the repo's basename). The notes
below describe the repos this skill knew about at the time of writing; if `.keep`
has gained or dropped entries since, trust `.keep`.

---

## Client side: `client/transcribe.py`

The client transcription worker sits between the VAD queue and the LLM queue.
It has no dependency on model packages - all heavy lifting happens on the server.

**Responsibilities:**
- Read completed speech segments (raw PCM bytes) from the VAD output queue
- Encode each segment as a WAV buffer in memory (16kHz, mono, int16 LE - matching
  the fixed PCM format used everywhere in the pipeline)
- POST the WAV buffer to `POST /v1/audio/transcriptions` using the `openai` SDK's
  `client.audio.transcriptions.create()`
- Push the returned transcript text to the LLM input queue
- Configured via `transcription.base_url` and `transcription.model`

**What to look for in the references for this:**
- The multipart form fields the server expects (`file`, `model`, `language`,
  `prompt`, `response_format`) - match these exactly when constructing the client call
- The `json` response shape: `{"text": "..."}` - parse `.text` from the response
- WAV encoding: standard Python `wave` module writes 16kHz/mono/int16 into a
  `BytesIO` buffer; pass as a named tuple `("segment.wav", buffer, "audio/wav")`

---

## Server side: `server/`

### `./references/qwen3-asr-openai/`

The primary reference. A FastAPI server exposing an OpenAI Whisper-compatible API
backed by Qwen3-ASR. This is the closest model for SubsVibe's transcription server.

Key files:
- `./references/qwen3-asr-openai/server.py` - main FastAPI app: all endpoints,
  audio decoding via PyAV, model invocation, response formatting
- `./references/qwen3-asr-openai/README.md` - setup, environment variables,
  supported parameters

What to look for here:
- Endpoint signatures (`/v1/audio/transcriptions`, `/v1/models`)
- Multipart form handling for audio uploads
- PyAV decode pipeline (any format → mono 16kHz float32): `decode_audio_bytes_mono_16k()`
- `json` and `verbose_json` response shapes (with segments and optional word timestamps):
  `build_transcription_response()`
- Language, prompt, and `timestamp_granularities` parameter handling
- Model loading, inference lock, idle-unload pattern: `get_model()`, `_INFER_LOCK`
- Streaming SSE response for long audio: the `stream=true` branch in
  `create_audio_transcription()`

### `./references/faster-whisper-webui/`

Secondary reference. A Gradio web UI, not an API server - don't copy its structure
directly. Useful for understanding how Faster Whisper is loaded and called.

Key files:
- `./references/faster-whisper-webui/src/whisper/fasterWhisperContainer.py` -
  model loading, transcription call, result handling
- `./references/faster-whisper-webui/src/vad.py` - VAD integration with
  Faster Whisper (in SubsVibe, VAD is client-side; this is reference only)
- `./references/faster-whisper-webui/src/config.py` - configuration shape

What to look for here:
- How `faster-whisper` accepts audio input (numpy float32, 16kHz)
- Model size / quantization / device options
- How transcription results (segments, words) are structured

### `./references/Qwen3-ASR-Toolkit/`

Tertiary reference. A CLI toolkit that calls the hosted Qwen-ASR (DashScope) API
and works around its 3-minute audio limit by VAD-splitting long media, processing
chunks in parallel, and stitching the results into text or SRT. Not an API server,
but its splitting / aggregation logic mirrors what SubsVibe does at the segment
boundary level, so it's the best reference for those concerns.

Key files:
- `./references/Qwen3-ASR-Toolkit/qwen3_asr_toolkit/qwen3asr.py` - top-level
  pipeline: load media → VAD split → parallel API calls → aggregate → write output
- `./references/Qwen3-ASR-Toolkit/qwen3_asr_toolkit/audio_tools.py` - FFmpeg-based
  resampling to 16kHz mono, VAD-based silence splitting, chunk size targeting
- `./references/Qwen3-ASR-Toolkit/qwen3_asr_toolkit/call_api.py` - DashScope API
  call shape, retry logic, response parsing

What to look for here:
- VAD-driven chunking strategy: targeting a duration (default ~120s) while
  cutting only at silences so words/sentences aren't truncated
- Hallucination and repetition post-processing - patterns that recur across ASR
  outputs and can be filtered after transcription
- SRT timestamp generation from VAD segment boundaries (relevant if SubsVibe ever
  emits SRT alongside live subtitles)
- Parallel chunk dispatch with thread pools and ordered re-aggregation

Note: this toolkit targets the *hosted* DashScope Qwen-ASR API, not a local
OpenAI-compatible server. Use it for chunking/aggregation ideas, not for the
server's API surface - that comes from `qwen3-asr-openai/`.

---

## Where SubsVibe's code will live

```
./client/
  transcribe.py  # VAD queue → WAV encode → POST /v1/audio/transcriptions → LLM queue

./server/
  server.py      # FastAPI app - modelled on qwen3-asr-openai/server.py
  model.py       # Backend abstraction: Faster Whisper or Qwen3-ASR
```

The server exposes:
- `GET /v1/models`
- `POST /v1/audio/transcriptions` - accepts multipart audio, returns `json` or
  `verbose_json`; parameters: `file`, `model`, `language`, `prompt`,
  `response_format`, `timestamp_granularities`

Audio is decoded server-side (PyAV) so the client always sends plain WAV.
The model backend is selected via server config, not per-request.

### `./references/ollama/`

OpenAI-compatible local LLM server. In SubsVibe this is the default LLM backend
(`LLM_BASE_URL=http://localhost:11434`). Not a transcription reference — consult
it when working on the LLM API surface (`client/llm.py`) or when understanding
how Ollama exposes the OpenAI chat completions API.

Key paths:
- `./references/ollama/docs/` - API documentation and endpoint reference
- `./references/ollama/openai.go` (or similar) - OpenAI compatibility layer

What to look for here:
- Which OpenAI chat completions parameters Ollama honours vs. ignores
- Model name format and how `LLM_MODEL_ID` maps to a loaded model
- Streaming behaviour for chat completions

---

## Design doc

For the full specification of what both sides should do, read `./docs/plan.md`
(Phase 3 section). The references show *how* - the plan shows *what*.
