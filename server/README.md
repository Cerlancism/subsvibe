# SubsVibe Transcription Server

FastAPI server exposing an OpenAI Whisper-compatible API. Backend is pluggable via `TRANSCRIPT_BACKEND`:

- `faster-whisper` (default) — Faster Whisper via CTranslate2; timestamps come from the model itself, no separate aligner.
- `qwen` — Qwen3-ASR plus an optional forced aligner for word/segment timestamps.
- `anime-whisper` — Japanese-only ASR fine-tuned on anime/galgame speech ([litagin/anime-whisper](https://huggingface.co/litagin/anime-whisper)); timestamps via the Qwen forced aligner.

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
| `TRANSCRIPT_BACKEND` | `faster-whisper` | `faster-whisper`, `qwen`, or `anime-whisper` |
| `TRANSCRIPT_MAX_INPUT_SECONDS` | `180` | Reject audio longer than this; client must split |
| `TRANSCRIPT_SILENCE_FILTER` | `1` | Blank outputs that wholly match a known *silence* hallucination of the active backend/model/language. Set `0` to disable |
| `TRANSCRIPT_NOISE_FILTER` | `1` | Same, for the *noise/music* hallucination dataset (`server/data/noise_hallucinations.json`). Set `0` to disable |

`TRANSCRIPT_MODEL_ID` identifies the model in two ways: it's the HuggingFace repo to load *and* the model name the server advertises on `/v1/models`. It is the *initial* model only — a transcription request naming a different model switches the server to it, as long as the configured backend can load it (e.g. any CTranslate2 whisper repo on `faster-whisper`). The backend itself never changes per-request.

### Qwen3-ASR backend (`TRANSCRIPT_BACKEND=qwen`)

| Variable | Default | Purpose |
|---|---|---|
| `TRANSCRIPT_MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace repo of the ASR model |
| `TRANSCRIPT_MODEL_PATH` | *(empty)* | Local path to cached model; empty = auto-download |
| `TRANSCRIPT_ALIGNER_ID` | `Qwen/Qwen3-ForcedAligner-0.6B` | HuggingFace repo of the forced aligner (used when timestamps are requested) |
| `TRANSCRIPT_ALIGNER_PATH` | *(empty)* | Local path to cached aligner; empty = auto-download |

### Faster Whisper backend (`TRANSCRIPT_BACKEND=faster-whisper`)

| Variable | Default | Purpose |
|---|---|---|
| `TRANSCRIPT_MODEL_ID` | `Systran/faster-whisper-large-v3` | CTranslate2-converted HuggingFace repo (e.g. `Systran/faster-whisper-large-v3`, `…-medium`, `…-small`, `…-base`, `…-tiny`) |
| `TRANSCRIPT_DEVICE` | *(empty / auto)* | `cuda` or `cpu`. Auto-detects when empty |
| `TRANSCRIPT_COMPUTE_TYPE` | *(empty / auto)* | `float16`, `int8_float16`, `int8`, etc. Defaults to `float16` on CUDA, `int8` on CPU |
| `TRANSCRIPT_BEAM_SIZE` | `5` | Decoder beam size |
| `TRANSCRIPT_CPU_THREADS` | *(empty / auto)* | CPU-only intra-op threads. Empty = `os.cpu_count() // 2` (physical-core estimate; SMT siblings contend for int8 GEMM units). Ignored on CUDA |

Word and segment timestamps are produced directly by the Faster Whisper model — `POST /v1/audio/align` is not supported by this backend. The client always runs VAD, so this backend does not expose faster-whisper's internal VAD filter.

Segment timestamps are free (always emitted); word timestamps add a DTW alignment pass (~10–30% slower). Clients that only need SRT lines should request `timestamp_granularities=segment` to skip the alignment cost. The bundled SubsVibe client does this automatically when `TRANSCRIPT_BACKEND=faster-whisper`.

### Anime Whisper backend (`TRANSCRIPT_BACKEND=anime-whisper`)

| Variable | Default | Purpose |
|---|---|---|
| `TRANSCRIPT_MODEL_ID` | `litagin/anime-whisper` | HuggingFace repo of the ASR model |
| `TRANSCRIPT_ALIGNER_ID` | `Qwen/Qwen3-ForcedAligner-0.6B` | Forced aligner used for word/segment timestamps (anime-whisper has none of its own) |
| `TRANSCRIPT_NO_REPEAT_NGRAM_SIZE` | `5` | Generation: suppress repetition hallucinations. Raise toward 10 if repetition still appears |
| `TRANSCRIPT_REPETITION_PENALTY` | `1.0` | Generation: leave at 1.0 unless repetition persists |
| `TRANSCRIPT_CHUNK_LENGTH_S` | `30.0` | Pipeline chunk length |
| `TRANSCRIPT_BATCH_SIZE` | `16` | Pipeline batch size; lower if you hit OOM |

Runs via the `transformers` pipeline. This model is **Japanese-only**; the `language` form field is ignored on the wire and forced to Japanese. Per the model card, **`prompt` is dropped** — initial prompts cause this model to hallucinate and degrade severely.

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

`GET /v1/models` — OpenAI-compatible list of one model (the currently active model id; initially `TRANSCRIPT_MODEL_ID`).

### Manual model load / unload

Models lazy-load on first transcription, but you can warm them up or free VRAM explicitly:

- `POST /v1/model/load` — load the ASR model (no-op if already loaded)
- `POST /v1/aligner/load` — load only the forced aligner (no-op if already loaded)
- `POST /v1/model/unload` — unload both ASR and aligner

### Transcribe

`POST /v1/audio/transcriptions` — multipart form.

| Field | Type | Notes |
|---|---|---|
| `file` | file | Any format; PyAV decodes to mono 16 kHz |
| `model` | string | Optional. Empty / omitted = use the active model. A different id unloads the active model and loads the requested one — it must be loadable by the configured backend (400 with detail otherwise; the server reverts to the previous model) |
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
- **Silence/noise hallucination filter**: on by default. Two datasets record the texts each backend/model/language emits for non-speech audio — `server/data/silence_hallucinations.json` for pure silence (built by `tests/test_silence_hallucinations.py`) and `server/data/noise_hallucinations.json` for background noise/music (BGM stings, channel-promo overlays). When a transcription's *whole* text matches one of those entries — compared with punctuation, symbols, whitespace and case stripped — the server returns empty `text` (and empty `segments`/`words`). The lookup is specific to the configured backend, the active model and the requested/detected language; partial matches inside real speech are never touched. Disable each source independently with `TRANSCRIPT_SILENCE_FILTER=0` / `TRANSCRIPT_NOISE_FILTER=0`.
- **Inference threading**: ASR and alignment run via `asyncio.to_thread()` so the event loop stays responsive. A per-backend inference lock serialises calls to the same model instance.
- **Idle unload**: a background task unloads aligner first, then ASR, after `IDLE_UNLOAD_SECONDS` of no `/v1/audio/transcriptions` activity. Models reload on the next request.
- **Backends**: pluggable via `TRANSCRIPT_BACKEND` (`faster-whisper`, `qwen`, `anime-whisper`). See `server/backends/`.
