# SubsVibe Transcription Server

FastAPI server exposing an OpenAI Whisper-compatible API. Every backend below is served at once; a request's `backend` field picks one and `TRANSCRIPT_BACKEND` is the default for requests naming none:

- `faster-whisper` (default) — Faster Whisper via CTranslate2; timestamps come from the model itself, no separate aligner.
- `qwen` — Qwen3-ASR plus an optional forced aligner for word/segment timestamps.
- `anime-whisper` — Japanese-only ASR fine-tuned on anime/galgame speech ([litagin/anime-whisper](https://huggingface.co/litagin/anime-whisper)); timestamps via the Qwen forced aligner.
- `openrouter` — remote: each request is re-encoded and forwarded to OpenRouter's transcription endpoint. `microsoft/mai-transcribe-2`, `x-ai/grok-stt-1.0` and `elevenlabs/scribe-v2` are long-form (up to an hour of audio per request) with their own word timestamps; nothing runs locally.

Per-backend details are in each backend's section below.

See the root [README.md](../README.md) for setup; start the server with `scripts/server.sh`.

## Environment Variables

Configured via `scripts/env.sh`.

### Server binding

| Variable | Default | Purpose |
|---|---|---|
| `TRANSCRIPT_HOST` | `0.0.0.0` | Bind address |
| `TRANSCRIPT_PORT` | `5183` | Bind port |

### Transcription backend selection

| Variable | Default | Purpose |
|---|---|---|
| `TRANSCRIPT_BACKEND` | `faster-whisper` | `faster-whisper`, `qwen`, `anime-whisper` or `openrouter`: the backend a request gets when it names none. Every backend is available at once; a request's `backend` field picks one |
| `TRANSCRIPT_MAX_INPUT_SECONDS` | `180` | Reject audio longer than this; client must split. Shared by the local backends (`openrouter` has `OPENROUTER_MAX_INPUT_SECONDS`); each backend's value is reported on `GET /v1/backend` |
| `TRANSCRIPT_MAX_CONCURRENT` | `2` | Transcriptions a local backend runs at once on its resident model; the rest queue in arrival order. Shared by the local backends (`openrouter` has `OPENROUTER_MAX_CONCURRENT`). faster-whisper runs both inferences together; qwen and anime-whisper serialise their GPU work behind their own lock, so the overlap there is the parent-side work |
| `TRANSCRIPT_ALIGNER_ID` | `Qwen/Qwen3-ForcedAligner-0.6B` | Forced aligner for word/segment timestamps. Shared by `qwen` and `anime-whisper`; `faster-whisper` never loads it |
| `TRANSCRIPT_SILENCE_FILTER` | `1` | Blank outputs that wholly match a known *silence* hallucination of the active backend/model/language. Set `0` to disable |
| `TRANSCRIPT_NOISE_FILTER` | `1` | Same, for the *noise/music* hallucination dataset (`server/data/noise_hallucinations.json`). Set `0` to disable |

Each backend's `<BACKEND>_MODEL_ID` identifies its model in two ways: it's the HuggingFace repo to load *and* the model name the server advertises on `/v1/models`. It is the model that backend uses for requests naming none. A request's `model` field picks another: on a local backend the server waits for that backend's in-flight requests to drain, unloads its resident model and loads the requested one, as long as the backend can load it (e.g. any CTranslate2 whisper repo on `faster-whisper`), and later requests naming none use it too; up to `TRANSCRIPT_MAX_CONCURRENT` requests run at once on the resident model. On `openrouter` nothing is resident, so nothing switches: each request is forwarded with the model it names, up to `OPENROUTER_MAX_CONCURRENT` at once, and sessions on different models run side by side. Backends never block each other: a `qwen` session and a `faster-whisper` session run together, each on its own worker. Two sessions alternating models on one local backend reload it on every request; run them on separate backends or the same model.

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
| `OPENROUTER_MAX_INPUT_SECONDS` | `3600` | Longest audio one long-form request (mai, Grok STT, Scribe) may carry |
| `OPENROUTER_OPUS_BITRATE` | `80000` | Opus bitrate the audio is re-encoded at before forwarding |
| `OPENROUTER_TIMEOUT_SECONDS` | `150` | Outbound request timeout; an hour chunk answers in about a minute |
| `OPENROUTER_MAX_RETRIES` | `2` | Retries on timeouts, connection errors and 5xx. A 4xx, or a provider error wrapped in a 200, is final |
| `OPENROUTER_MAX_CONCURRENT` | `8` | Requests forwarded at once; the rest queue in arrival order until a slot frees. Timeout and retries apply per forwarded call, after queueing |

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
{ "status": "ok", "model_loaded": true, "loaded": ["faster-whisper"], "default": "faster-whisper" }
```

`loaded` names the backends whose ASR model is in memory; `model_loaded` is whether any is.

### Models

`GET /v1/models` — OpenAI-compatible list with one entry per backend: the model it currently uses for requests naming none (initially its `<BACKEND>_MODEL_ID`), `owned_by` the backend name.

### Manual model load / unload

Models lazy-load on first transcription, but you can warm them up or free VRAM explicitly. Each takes an optional JSON body `{"backend": "<name>", "model": "<id>"}`; `backend` omitted (or no body) means the server default.

- `POST /v1/model/load` — load that backend's ASR model (no-op if already loaded). With `model`, select it first: on a local backend this waits for the backend's in-flight requests to drain, unloads the resident model and loads the new one, reverting the selection if it cannot load (400). Returns `{status, backend, model}`
- `POST /v1/aligner/load` — load only that backend's forced aligner (no-op if already loaded)
- `POST /v1/model/unload` — unload that backend's ASR and aligner, or every backend's when `backend` is omitted; waits for in-flight requests to drain. Returns `{status, backends, asr_unloaded, aligner_unloaded}` with the backend names each step touched

### Backend report

`GET /v1/backend?backend=<name>&model=<id>` — both optional. `backend` omitted means the server default; an unknown name is a 400.

```json
{
  "backend": "faster-whisper",
  "default": "faster-whisper",
  "model": "Systran/faster-whisper-medium",
  "supported": ["faster-whisper", "qwen", "anime-whisper", "openrouter"],
  "model_loaded": true,
  "aligner_loaded": false,
  "max_concurrent": 2,
  "max_input_seconds": 180.0,
  "long_form": false,
  "entries_from": "segments"
}
```

`model` is the id that backend uses for requests naming none. The last three fields describe the (backend, model) pair and the bundled client adopts them at session start instead of mirroring server variables: `max_input_seconds` is the longest audio one request may carry; `long_form` says the model takes minutes to an hour per request (the client splits the file evenly near its chunk maximum and scans only each part's tail for a seam) rather than Whisper's 30 s windows (latest speech onset inside the bounds); `entries_from` is `segments` when the backend's own segments are subtitle-shaped as returned and `words` when entries must be built from word timestamps. `?model=` asks for another model's profile, which matters on `openrouter` where every request carries its own model: the bundled client passes its `--model` here so a session chunks for the model it will send. On a local backend every model shares one profile.

There is no switch endpoint. Every supported backend is served at once, built on first use and kept for the run; a request picks one with its `backend` field and the bundled client names the backend it settled on (`--backend`, else `TRANSCRIPT_BACKEND`, else the server default) on every request. Each backend admits up to its concurrency (`max_concurrent`) and queues the rest in arrival order; backends never wait on each other.

### Transcribe

`POST /v1/audio/transcriptions` — multipart form.

| Field | Type | Notes |
|---|---|---|
| `file` | file | Any format; PyAV decodes to mono 16 kHz |
| `backend` | string | Optional. One of the `supported` names (400 otherwise); empty / omitted = the server default (`TRANSCRIPT_BACKEND`). Backends serve side by side, each admitting up to its own concurrency and queueing the rest |
| `model` | string | Optional. Empty / omitted = the backend's current model. On a local backend a different id waits for that backend's in-flight requests to drain, unloads its model and loads the requested one — it must be loadable there (400 with detail otherwise; the selection reverts). On `openrouter` the id simply goes out with the request |
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

`POST /v1/audio/align` — multipart form: `file` (audio) + `text` (the transcript to align) + optional `backend` (server default when omitted) and `language`. Returns word-level timestamps for externally-provided text via the forced aligner (`qwen` / `anime-whisper` backends; `faster-whisper` returns 501).

## Architecture Notes

- **Audio decoding**: PyAV decodes any input format to mono 16 kHz float32, normalising peaks to ±1.0.
- **Silence/noise hallucination filter**: on by default. Two datasets record the texts each backend/model/language emits for non-speech audio — `server/data/silence_hallucinations.json` for pure silence (built by `tests/test_silence_hallucinations.py`) and `server/data/noise_hallucinations.json` for background noise/music (BGM stings, channel-promo overlays). When a transcription's *whole* text matches one of those entries — compared with punctuation, symbols, whitespace and case stripped — the server returns empty `text` (and empty `segments`/`words`). The lookup is specific to the active backend, the active model and the requested/detected language; partial matches inside real speech are never touched. Disable each source independently with `TRANSCRIPT_SILENCE_FILTER=0` / `TRANSCRIPT_NOISE_FILTER=0`.
- **Inference threading and admission**: ASR and alignment run via `asyncio.to_thread()` so the event loop stays responsive. Each backend has an admission gate (`_Admission` in `server/server.py`) sized by its `max_concurrency()`: that many transcriptions run at once, the rest wait in arrival order without holding a thread. A model switch or unload on a local backend waits for the gate to drain and holds back new arrivals meanwhile, so a steady stream cannot starve it. Worker children (`server/worker.py`) multiplex calls by request id, one thread per call in the child; qwen and anime-whisper additionally serialise their GPU work behind a lock of their own, faster-whisper's CTranslate2 model runs calls together.
- **Idle unload**: a background task unloads each idle backend's aligner first, then ASR, after `IDLE_UNLOAD_SECONDS` of no `/v1/audio/transcriptions` activity, skipping a backend with requests in flight. Models reload on the next request.
- **Backends**: every supported backend (`faster-whisper`, `qwen`, `anime-whisper`, `openrouter`) is served at once; a request's `backend` field picks one and `TRANSCRIPT_BACKEND` is only the default for requests naming none. Each is built on first use and kept for the run (`server/model.py`); each describes its current model through `profile()` (request cap, long-form or not, entry source), which `GET /v1/backend` reports. The model a backend uses lives in the server process; the environment holds only startup values. The child entry point is chosen parent-side and receives its model id as a spawn argument, so nothing is written back to the environment. See `server/backends/`.
