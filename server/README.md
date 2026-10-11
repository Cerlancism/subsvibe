# SubsVibe Transcription Server

FastAPI server exposing an OpenAI Whisper-compatible API. Backend is pluggable via `TRANSCRIPT_BACKEND` and swappable at runtime via `POST /v1/backend`:

- `faster-whisper` (default) — Faster Whisper via CTranslate2; timestamps come from the model itself, no separate aligner.
- `qwen` — Qwen3-ASR plus an optional forced aligner for word/segment timestamps.
- `anime-whisper` — Japanese-only ASR fine-tuned on anime/galgame speech ([litagin/anime-whisper](https://huggingface.co/litagin/anime-whisper)); timestamps via the Qwen forced aligner.
- `openrouter` — remote: each request is re-encoded and forwarded to OpenRouter's transcription endpoint. `microsoft/mai-transcribe-2` is long-form (up to an hour of audio per request) with its own word timestamps; nothing runs locally.

Per-backend details are in each backend's section below.

See the root [README.md](../README.md) for setup; start the server with `scripts/server.sh`.

## Environment Variables

Configured via `scripts/env.sh`.

### Server binding

| Variable | Default | Purpose |
|---|---|---|
| `TRANSCRIPT_HOST` | `0.0.0.0` | Bind address |
| `TRANSCRIPT_PORT` | `8000` | Bind port |

### Transcription backend selection

| Variable | Default | Purpose |
|---|---|---|
| `TRANSCRIPT_BACKEND` | `faster-whisper` | `faster-whisper`, `qwen`, `anime-whisper` or `openrouter`. Initial selection only — `POST /v1/backend` swaps it at runtime |
| `TRANSCRIPT_MAX_INPUT_SECONDS` | `180` | Reject audio longer than this; client must split. Shared by the local backends (`openrouter` has `OPENROUTER_MAX_INPUT_SECONDS`); the active value is reported on `GET /v1/backend` |
| `TRANSCRIPT_ALIGNER_ID` | `Qwen/Qwen3-ForcedAligner-0.6B` | Forced aligner for word/segment timestamps. Shared by `qwen` and `anime-whisper`; `faster-whisper` never loads it |
| `TRANSCRIPT_SILENCE_FILTER` | `1` | Blank outputs that wholly match a known *silence* hallucination of the active backend/model/language. Set `0` to disable |
| `TRANSCRIPT_NOISE_FILTER` | `1` | Same, for the *noise/music* hallucination dataset (`server/data/noise_hallucinations.json`). Set `0` to disable |

Each backend's `<BACKEND>_MODEL_ID` identifies its model in two ways: it's the HuggingFace repo to load *and* the model name the server advertises on `/v1/models`. It is the *initial* model only — a transcription request naming a different model switches the server to it, as long as the active backend can load it (e.g. any CTranslate2 whisper repo on `faster-whisper`). The backend itself never changes per-transcription; swap it with `POST /v1/backend`, which reselects the model last used on the incoming backend this run, or — if it has not been used yet — that backend's `<BACKEND>_MODEL_ID`.

Backend-specific variables are prefixed with the backend name (`FASTER_WHISPER_`, `QWEN_`, `ANIME_WHISPER_`, `OPENROUTER_`), so one `env.sh` can carry the tuning for every backend side by side. Every value has a built-in constant in `server/backends/config.py`; an unset or empty variable keeps it.

### Faster Whisper backend (`TRANSCRIPT_BACKEND=faster-whisper`)

| Variable | Default | Purpose |
|---|---|---|
| `FASTER_WHISPER_MODEL_ID` | `Systran/faster-whisper-large-v3` | CTranslate2-converted HuggingFace repo (e.g. `Systran/faster-whisper-large-v3`, `…-medium`, `…-small`, `…-base`, `…-tiny`) |
| `FASTER_WHISPER_DEVICE` | *(empty / auto)* | `cuda` or `cpu`. Auto-detects when empty |
| `FASTER_WHISPER_COMPUTE_TYPE` | *(empty / auto)* | `float16`, `int8_float16`, `int8`, etc. Defaults to `float16` on CUDA, `int8` on CPU |
| `FASTER_WHISPER_BEAM_SIZE` | `5` | Decoder beam size |
| `FASTER_WHISPER_CPU_THREADS` | *(empty / auto)* | CPU-only intra-op threads. Empty = `os.cpu_count() // 2` (physical-core estimate; SMT siblings contend for int8 GEMM units). Ignored on CUDA |

Word and segment timestamps are produced directly by the Faster Whisper model — `POST /v1/audio/align` is not supported by this backend. The client always runs VAD, so this backend does not expose faster-whisper's internal VAD filter.

Segment timestamps are free (always emitted); word timestamps add a DTW alignment pass (~10–30% slower). Clients that only need SRT lines should request `timestamp_granularities=segment` to skip the alignment cost. The bundled SubsVibe client does this automatically when `TRANSCRIPT_BACKEND=faster-whisper`.

### Qwen3-ASR backend (`TRANSCRIPT_BACKEND=qwen`)

| Variable | Default | Purpose |
|---|---|---|
| `QWEN_MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace repo of the ASR model |
| `QWEN_MAX_NEW_TOKENS` | `512` | Generation cap per request |

Word/segment timestamps come from the shared forced aligner (`TRANSCRIPT_ALIGNER_ID`), loaded on the first request that asks for them.

### Anime Whisper backend (`TRANSCRIPT_BACKEND=anime-whisper`)

| Variable | Default | Purpose |
|---|---|---|
| `ANIME_WHISPER_MODEL_ID` | `litagin/anime-whisper` | HuggingFace repo of the ASR model |
| `ANIME_WHISPER_NO_REPEAT_NGRAM_SIZE` | `5` | Generation: suppress repetition hallucinations. Raise toward 10 if repetition still appears |
| `ANIME_WHISPER_REPETITION_PENALTY` | `1.0` | Generation: leave at 1.0 unless repetition persists |
| `ANIME_WHISPER_CHUNK_LENGTH_S` | `30.0` | Pipeline chunk length |
| `ANIME_WHISPER_BATCH_SIZE` | `16` | Pipeline batch size; lower if you hit OOM |

Runs via the `transformers` pipeline. This model is **Japanese-only**; the `language` form field is ignored on the wire and forced to Japanese. Per the model card, **`prompt` is dropped** — initial prompts cause this model to hallucinate and degrade severely. Word/segment timestamps come from the shared forced aligner (`TRANSCRIPT_ALIGNER_ID`); anime-whisper has none of its own.

### OpenRouter backend (`TRANSCRIPT_BACKEND=openrouter`)

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | *(none)* | Bearer key; the backend refuses to transcribe without one |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Endpoint base; `/audio/transcriptions` is appended |
| `OPENROUTER_MODEL_ID` | `microsoft/mai-transcribe-2` | Startup model. Any OpenRouter transcription id is accepted per request; only ids with a row in `MODEL_SPECS` (`server/backends/remote.py`) are tuned, the rest are fed like a Whisper endpoint (180 s requests, lossless audio) |
| `OPENROUTER_MAX_INPUT_SECONDS` | `3600` | Longest audio one `mai-transcribe-2` request may carry |
| `OPENROUTER_OPUS_BITRATE` | `80000` | Opus bitrate the audio is re-encoded at before forwarding |
| `OPENROUTER_TIMEOUT_SECONDS` | `600` | Outbound request timeout |
| `OPENROUTER_MAX_RETRIES` | `2` | Retries on timeouts, connection errors and 5xx. A 4xx, or a provider error wrapped in a 200, is final |

Nothing loads: `load`/`unload` only flip the loaded flag so the lifecycle endpoints keep their meaning. Audio arrives as WAV from the client, is re-encoded with PyAV's bundled FFmpeg (Opus in Ogg for `mai-transcribe-2`: the provider rejects every AAC form, and Opus at 80 kbps measured within noise of lossless) and leaves base64-encoded in a JSON body, whose 50 MB cap is twice the multipart one. `language` is forwarded when the client gives one and detected otherwise; `prompt` is dropped (OpenRouter only forwards it to Groq-hosted models). `mai-transcribe-2` returns every character as a word and each punctuation mark as its own word; the backend strips marks off tokens and folds mark-only tokens into the previous word's span, so the client's punctuation attachment works as it does on the aligner backends. Its `segments` carry no usable timing, so the response's `segments` is one span over all words and the client builds entries from `words`. A failed upstream call (after retries) is a 502 with the upstream detail. Each response's `usage.cost` is logged.

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

`GET /v1/models` — OpenAI-compatible list of one model (the currently active model id; initially the active backend's `<BACKEND>_MODEL_ID`).

### Manual model load / unload

Models lazy-load on first transcription, but you can warm them up or free VRAM explicitly:

- `POST /v1/model/load` — load the ASR model (no-op if already loaded)
- `POST /v1/aligner/load` — load only the forced aligner (no-op if already loaded)
- `POST /v1/model/unload` — unload both ASR and aligner

### Backend selection

`GET /v1/backend`

```json
{
  "backend": "faster-whisper",
  "model": "Systran/faster-whisper-medium",
  "supported": ["faster-whisper", "qwen", "anime-whisper", "openrouter"],
  "model_loaded": true,
  "aligner_loaded": false,
  "max_input_seconds": 180.0,
  "long_form": false,
  "entries_from": "segments"
}
```

The last three fields describe the active (backend, model) pair and the bundled client adopts them at session start instead of mirroring server variables: `max_input_seconds` is the longest audio one request may carry; `long_form` says the model takes minutes to an hour per request (the client splits the file evenly near its chunk maximum and scans only each part's tail for a seam) rather than Whisper's 30 s windows (latest speech onset inside the bounds); `entries_from` is `segments` when the backend's own segments are subtitle-shaped as returned and `words` when entries must be built from word timestamps.

`POST /v1/backend` — JSON body, swaps the active backend without restarting the server.

| Field | Type | Notes |
|---|---|---|
| `backend` | string | Required. One of the `supported` names; anything else is a 400 |
| `model` | string | Optional model id within the new backend. Omitted = the model last used on that backend this session, or the id it would have started with if it has not been used yet |
| `load` | bool | Default `false` (lazy — the new model loads on the next transcription). `true` loads eagerly, so an unloadable backend/model fails *this* request with a 400. The server stays on the requested backend, unloaded; there is no revert |

```json
{ "status": "switched", "backend": "qwen", "previous_backend": "faster-whisper",
  "model": "Qwen/Qwen3-ASR-1.7B", "model_loaded": false }
```

`status` is always `switched`. The switch is unconditional — re-selecting the active backend respawns its worker rather than short-circuiting, which costs one reload on a redundant call and keeps this a single step with a single outcome.

Switching **disposes the outgoing backend's worker child processes**, aligner first then ASR, exactly as `POST /v1/model/unload` does — the OS reclaims their VRAM before the incoming backend spawns its own. Changing backend or model is therefore a live operation: no server restart, no `scripts/env.sh` edit. Switching is meant to happen between runs; a transcription in flight when its worker is killed fails with a 500. The client follows that shape on its side — `--backend` is a separate call that switches and exits, and a session only ever reads the active backend.

### Transcribe

`POST /v1/audio/transcriptions` — multipart form.

| Field | Type | Notes |
|---|---|---|
| `file` | file | Any format; PyAV decodes to mono 16 kHz |
| `model` | string | Optional. Empty / omitted = use the active model. A different id unloads the active model and loads the requested one — it must be loadable by the active backend (400 with detail otherwise; the server reverts to the previous model) |
| `language` | string | ISO-639-1 (`en`, `zh`, `ja`, ...). Empty / `auto` / `detect` / `none` = auto-detect |
| `prompt` | string | Replaces the default ASR system context (transcription instructions / vocabulary hints) |
| `response_format` | string | `json` (default), `verbose_json`, `text` |
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

### Align

`POST /v1/audio/align` — multipart form: `file` (audio) + `text` (the transcript to align) + optional `language`. Returns word-level timestamps for externally-provided text via the forced aligner (`qwen` / `anime-whisper` backends; `faster-whisper` returns 501).

## Architecture Notes

- **Audio decoding**: PyAV decodes any input format to mono 16 kHz float32, normalising peaks to ±1.0.
- **Silence/noise hallucination filter**: on by default. Two datasets record the texts each backend/model/language emits for non-speech audio — `server/data/silence_hallucinations.json` for pure silence (built by `tests/test_silence_hallucinations.py`) and `server/data/noise_hallucinations.json` for background noise/music (BGM stings, channel-promo overlays). When a transcription's *whole* text matches one of those entries — compared with punctuation, symbols, whitespace and case stripped — the server returns empty `text` (and empty `segments`/`words`). The lookup is specific to the active backend, the active model and the requested/detected language; partial matches inside real speech are never touched. Disable each source independently with `TRANSCRIPT_SILENCE_FILTER=0` / `TRANSCRIPT_NOISE_FILTER=0`.
- **Inference threading**: ASR and alignment run via `asyncio.to_thread()` so the event loop stays responsive. A per-backend inference lock serialises calls to the same model instance.
- **Idle unload**: a background task unloads aligner first, then ASR, after `IDLE_UNLOAD_SECONDS` of no `/v1/audio/transcriptions` activity. Models reload on the next request.
- **Backends**: pluggable via `TRANSCRIPT_BACKEND` (`faster-whisper`, `qwen`, `anime-whisper`, `openrouter`) and swappable at runtime via `POST /v1/backend`. Each backend describes its current model through `profile()` (request cap, long-form or not, entry source), which `GET /v1/backend` reports. The active backend and model both live in the server process (`server/model.py`); the environment holds only startup values. The child entry point is chosen parent-side and receives its model id as a spawn argument, so nothing is written back to the environment. See `server/backends/`.
