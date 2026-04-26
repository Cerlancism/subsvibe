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

# SubsVibe Transcription — Reference Guide

Two reference implementations live in `./references/`. They inform the design of
both `./server/` and `./client/transcribe.py`. Read them when you need to understand
how something should work, but do not modify them.

---

## Client side: `client/transcribe.py`

The client transcription worker sits between the VAD queue and the LLM queue.
It has no dependency on model packages — all heavy lifting happens on the server.

**Responsibilities:**
- Read completed speech segments (raw PCM bytes) from the VAD output queue
- Encode each segment as a WAV buffer in memory (16kHz, mono, int16 LE — matching
  the fixed PCM format used everywhere in the pipeline)
- POST the WAV buffer to `POST /v1/audio/transcriptions` using the `openai` SDK's
  `client.audio.transcriptions.create()`
- Push the returned transcript text to the LLM input queue
- Configured via `transcription.base_url` and `transcription.model`

**What to look for in the references for this:**
- The multipart form fields the server expects (`file`, `model`, `language`,
  `prompt`, `response_format`) — match these exactly when constructing the client call
- The `json` response shape: `{"text": "..."}` — parse `.text` from the response
- WAV encoding: standard Python `wave` module writes 16kHz/mono/int16 into a
  `BytesIO` buffer; pass as a named tuple `("segment.wav", buffer, "audio/wav")`

---

## Server side: `server/`

### `./references/qwen3-asr-openai/`

The primary reference. A FastAPI server exposing an OpenAI Whisper-compatible API
backed by Qwen3-ASR. This is the closest model for SubsVibe's transcription server.

Key files:
- `./references/qwen3-asr-openai/server.py` — main FastAPI app: all endpoints,
  audio decoding via PyAV, model invocation, response formatting
- `./references/qwen3-asr-openai/README.md` — setup, environment variables,
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

Secondary reference. A Gradio web UI, not an API server — don't copy its structure
directly. Useful for understanding how Faster Whisper is loaded and called.

Key files:
- `./references/faster-whisper-webui/src/whisper/fasterWhisperContainer.py` —
  model loading, transcription call, result handling
- `./references/faster-whisper-webui/src/vad.py` — VAD integration with
  Faster Whisper (in SubsVibe, VAD is client-side; this is reference only)
- `./references/faster-whisper-webui/src/config.py` — configuration shape

What to look for here:
- How `faster-whisper` accepts audio input (numpy float32, 16kHz)
- Model size / quantization / device options
- How transcription results (segments, words) are structured

---

## Where SubsVibe's code will live

```
./client/
  transcribe.py  # VAD queue → WAV encode → POST /v1/audio/transcriptions → LLM queue

./server/
  server.py      # FastAPI app — modelled on qwen3-asr-openai/server.py
  model.py       # Backend abstraction: Faster Whisper or Qwen3-ASR
```

The server exposes:
- `GET /v1/models`
- `POST /v1/audio/transcriptions` — accepts multipart audio, returns `json` or
  `verbose_json`; parameters: `file`, `model`, `language`, `prompt`,
  `response_format`, `timestamp_granularities`

Audio is decoded server-side (PyAV) so the client always sends plain WAV.
The model backend is selected via server config, not per-request.

---

## Design doc

For the full specification of what both sides should do, read `./docs/plan.md`
(Phase 3 section). The references show *how* — the plan shows *what*.
