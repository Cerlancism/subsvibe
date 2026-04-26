# SubsVibe Transcription Server

FastAPI server exposing an OpenAI Whisper-compatible API for speech-to-text transcription. Supports Qwen3-ASR as the ASR backend with optional forced-aligner for word/segment timestamps.

## Quick Start

1. **Configure environment** (first time only):
   ```bash
   cp scripts/env.example.sh scripts/env.sh
   # Edit scripts/env.sh with your settings
   ```

2. **Run the server**:
   ```bash
   bash scripts/server.sh
   ```

3. **Test the server**:
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/v1/models
   ```

## Environment Configuration

Environment variables are loaded from `scripts/env.sh`. Copy `scripts/env.example.sh` and customize:

### Transcription Backend (Qwen3-ASR)

- **`TRANSCRIPT_MODEL_NAME`** — Model ID in API responses (default: `qwen3-asr`)
- **`TRANSCRIPT_MODEL_ID`** — HuggingFace repo ID (default: `Qwen/Qwen3-ASR-1.7B`)
- **`TRANSCRIPT_MODEL_PATH`** — Local path to cached model; leave empty to auto-download
- **`TRANSCRIPT_ALIGNER_ID`** — Forced aligner for timestamps (default: `Qwen/Qwen3-ForcedAligner-0.6B`)
- **`TRANSCRIPT_ALIGNER_PATH`** — Local path to aligner; leave empty to auto-download

### Server Binding

- **`TRANSCRIPT_HOST`** — Bind address (default: `0.0.0.0`)
- **`TRANSCRIPT_PORT`** — Bind port (default: `8000`)

### Model Lifecycle

- **`IDLE_UNLOAD_SECONDS`** — Unload models after N seconds of inactivity (default: `120`)
- **`IDLE_CHECK_SECONDS`** — Check interval for idle unload (default: `10`)

## API Endpoints

### `GET /health`, `GET /healthz`

Readiness probe. Returns `{"status": "ok"}`.

### `GET /v1/models`

List available models. Returns OpenAI-compatible format:

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-asr",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

### `POST /v1/audio/transcriptions`

Transcribe an audio file. Accepts multipart form data.

**Request parameters**:

- `file` (required) — Audio file (any format; decoded server-side to mono 16kHz)
- `model` (optional) — Model identifier; must match `TRANSCRIPT_MODEL_NAME`
- `language` (optional) — ISO-639-1 language code (e.g., `en`, `zh`, `ko`); omit for auto-detection
- `prompt` (optional) — Hint text to guide transcription style or vocabulary
- `response_format` (optional) — `json` (default), `verbose_json`, or `text`
- `stream` (optional) — `true` for SSE streaming; `false` (default) for single response
- `timestamp_granularities` (optional) — `segment` or `word` for aligned timestamps

**Response formats**:

**json (default)**:
```json
{
  "text": "hello world",
  "language": "English",
  "duration": 2.5,
  "segments": [...],
  "words": [...]
}
```

**verbose_json**:
```json
{
  "task": "transcribe",
  "language": "English",
  "duration": 2.5,
  "text": "hello world",
  "segments": [...],
  "words": [...]
}
```

**text**:
Plain text response (no JSON).

**Streaming (stream=true)**:
Server-Sent Events (SSE) with incremental results:
```
data: {"type": "transcript.text.delta", "delta": "hello"}
data: {"type": "transcript.text.delta", "delta": " world"}
data: {"type": "transcript.text.done", "text": "hello world"}
data: [DONE]
```

## Architecture

The server decouples audio decoding from inference via asyncio. PyAV handles audio format conversion to mono 16kHz float32. Model inference runs in a thread pool with inference locks to prevent concurrent requests to the same model instance.

### Model Loading

Models are lazy-loaded on first request. The `_load_asr()` function loads the ASR model; `_load_timestamp()` loads the same model with a forced aligner attached for word/segment timestamps.

### Audio Decoding

Uses PyAV to decode any audio format (MP3, WAV, FLAC, etc.) to mono 16kHz float32. Audio peaks are normalized to ±1.0.

### Inference Threading

All heavy operations (transcription, alignment) run in `asyncio.to_thread()` to avoid blocking the event loop. An inference lock serializes model calls to prevent concurrent inference on the same model instance.

### Idle Unload

A background task checks every `IDLE_CHECK_SECONDS`. If the server has been idle for more than `IDLE_UNLOAD_SECONDS` since the last request, models are unloaded to free GPU/CPU memory.

## Development

See [../CLAUDE.md](../CLAUDE.md) for development setup and testing audio capture.
