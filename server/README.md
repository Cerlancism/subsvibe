# SubsVibe Transcription Server

FastAPI server exposing an OpenAI Whisper-compatible API backed by Qwen3-ASR, with an optional forced aligner for word/segment timestamps. See the root [README.md](../README.md) for setup; start the server with `scripts/server.sh`.

## Environment Variables

Configured via `scripts/env.sh`.

### Server binding

| Variable | Default | Purpose |
|---|---|---|
| `TRANSCRIPT_HOST` | `0.0.0.0` | Bind address |
| `TRANSCRIPT_PORT` | `8000` | Bind port |

### Transcription backend (Qwen3-ASR)

| Variable | Default | Purpose |
|---|---|---|
| `TRANSCRIPT_MODEL_NAME` | `qwen3-asr` | Model ID returned in API responses |
| `TRANSCRIPT_MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace repo of the ASR model |
| `TRANSCRIPT_MODEL_PATH` | *(empty)* | Local path to cached model; empty = auto-download |
| `TRANSCRIPT_ALIGNER_ID` | `Qwen/Qwen3-ForcedAligner-0.6B` | HuggingFace repo of the forced aligner (used when timestamps are requested) |
| `TRANSCRIPT_ALIGNER_PATH` | *(empty)* | Local path to cached aligner; empty = auto-download |
| `TRANSCRIPT_MAX_INPUT_SECONDS` | `180` | Reject audio longer than this; client must split |

### Model lifecycle

| Variable | Default | Purpose |
|---|---|---|
| `IDLE_UNLOAD_SECONDS` | `120` | Unload ASR + aligner from VRAM after this many idle seconds |
| `IDLE_CHECK_SECONDS` | `10` | How often the idle watcher runs |

## API Endpoints

### Health

`GET /health`, `GET /healthz`, `GET /v1/health`, `GET /v1/healthz`

```json
{ "status": "ok", "model_loaded": true }
```

### Models

`GET /v1/models` — OpenAI-compatible list of one model (the configured `TRANSCRIPT_MODEL_NAME`).

### Manual model load / unload

Models lazy-load on first transcription, but you can warm them up or free VRAM explicitly:

- `POST /v1/model/load` — load the ASR model (no-op if already loaded)
- `POST /v1/model/unload` — unload both ASR and aligner

### Transcribe

`POST /v1/audio/transcriptions` — multipart form.

| Field | Type | Notes |
|---|---|---|
| `file` | file | Any format; PyAV decodes to mono 16 kHz |
| `model` | string | Must match `TRANSCRIPT_MODEL_NAME` (404 otherwise) |
| `language` | string | ISO-639-1 (`en`, `zh`, `ja`, ...). Empty / `auto` / `detect` / `none` = auto-detect |
| `prompt` | string | Replaces the default ASR system context (transcription instructions / vocabulary hints) |
| `response_format` | string | `json` (default), `verbose_json`, `text` |
| `stream` | string | `"true"` to stream over SSE; default returns one response |
| `timestamp_granularities` | list | `segment` and/or `word`. Repeated field or bracket form `timestamp_granularities[]=word` (OpenAI SDK uses the latter) |
| `temperature`, `chunking_strategy` | — | Accepted for OpenAI compatibility, ignored |

#### Response — `json` (default)

```json
{ "text": "hello world" }
```

When timestamp granularities are requested *and* the aligner returns content, the response also includes `segments`, `words`, `language`, and `duration`.

#### Response — `verbose_json`

Always includes segment timestamps:

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 2.5,
  "text": "hello world",
  "segments": [{ "start": 0.0, "end": 2.5, "text": "hello world" }],
  "words": [{ "word": "hello", "start": 0.0, "end": 0.4 }]
}
```

`words` is only present when `timestamp_granularities` includes `word`.

#### Response — `text`

Plain text body, no JSON.

#### Response — streaming (`stream=true`)

Server-Sent Events:

```
data: {"type": "transcript.text.delta", "delta": "hello"}
data: {"type": "transcript.text.delta", "delta": " world"}
data: {"type": "transcript.text.done", "text": "hello world", "segments": [...], "words": [...]}
data: [DONE]
```

`segments` / `words` appear in the final `done` frame only when requested via `timestamp_granularities`.

## Architecture Notes

- **Audio decoding**: PyAV decodes any input format to mono 16 kHz float32, normalising peaks to ±1.0.
- **Inference threading**: ASR and alignment run via `asyncio.to_thread()` so the event loop stays responsive. A per-backend inference lock serialises calls to the same model instance.
- **Idle unload**: a background task unloads aligner first, then ASR, after `IDLE_UNLOAD_SECONDS` of no `/v1/audio/transcriptions` activity. Models reload on the next request.
- **Backends**: pluggable via `TRANSCRIPT_BACKEND` (currently only `qwen`). See `server/backends/`.
