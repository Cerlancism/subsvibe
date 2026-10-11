---
name: file-chunk-scan-window
description: "Plan for file-mode chunking on a long-form ASR backend (minutes-to-hours per request) — tail-only cut scan, no onset anchoring — the decision to host it as a server-side remote (OpenAI-compatible proxy) backend, and the client/server restrictions to lift for it. Measured facts about the current 30 s chunker that shape the plan."
metadata:
  node_type: memory
  type: project
  originSessionId: 1b8b56a3-fc2c-439d-ac89-498a1d5348f8
  modified: 2026-10-10T00:00:00.000Z
---

# File-mode chunking for a long-form backend

Builds on the `CoarseChunker` section of [[recovery-vad-webrtcvad]]. The
current 30 s chunker in `./client/vad.py` stays as it is; the plan below
applies only when a backend that takes minutes-to-hours per request lands.

**Why chunk at all on such a backend:** the point of the tail scan is a
*quality* cut — landing the chunk end on a speech onset rather than
chopping flat at the maximum length — which we believe gives the ASR
better input than an abrupt mid-word boundary. Cost is secondary; the
scan is kept small so a long window stays cheap, not to skip cutting.

## Facts that shape the plan (measured 2026-10-09 on a 15 min Japanese stream recording)

- The whole-window detector ladder in `next_chunk` costs **~0.19 s per 30 s
  chunk** (8.4 s per 15 min of audio, ~5 % of a 142 s large-v3 run). At 30 s
  the VAD is not worth optimising; it only matters once windows are minutes
  long.
- Silero's `get_speech_timestamps` is stateful: scanning a sub-band of a
  window with some lead-in context does **not** reproduce what a scan from
  the window start finds, and more context does not converge it. Any
  tail-only scan therefore cuts differently from today's chunker — fine for
  a new backend with its own chunk size, not a drop-in for the 30 s one.
- **SRT diffs cannot validate chunker changes**: two runs of the same code
  on the same file differ by ~1170 diff lines (324 vs 326 entries) —
  faster-whisper on GPU is not run-to-run deterministic, and the ASR-driven
  cursor cascades every variance into later cuts. Compare chunker decisions
  (end, method, `speech_start`) directly instead.

## Plan (user direction, 2026-10-09)

- Per-backend chunk bounds — **done 2026-10-10**: `ChunkProfile` table in
  `./client/chunking.py` keyed by server backend name, `<BACKEND>_CHUNK_MAX_SECONDS`
  / `<BACKEND>_CHUNK_MIN_SECONDS` overrides, `anchor_first_entry` flag per
  profile; `CoarseChunker` takes the bounds as constructor args. A long-form
  backend adds one table row. Keep the chunker rather than one request per file:
  it preserves progress/ETA, bounded retries and the `split_provisional`
  overlap for cue continuity.
- For the long-form backend, scan only the **last ~16 s before the window
  end** for the cut, and **skip `speech_start` / `anchor_first_entry`
  entirely** — no head scan, no onset carry. The anchor exists for Whisper
  pulling a chunk's first cue early; whether the new backend needs it is a
  separate question.
- Alternatives to the detector ladder are open for that backend (user:
  "reconsider other options").

## Where the long-form backend lives (decision 2026-10-10)

Implement it as **server backends that forward to third-party
OpenAI-compatible ASR endpoints**, not as the client connecting to an
endpoint directly. The server does not become a mere proxy: each remote
endpoint is one more row in the backend table and the server keeps
everything that makes backends uniform.

**One implementation, many backend names.** Third-party endpoints differ in
model ids, input caps, whether they return `segments`/`words`, timeouts and
hallucination behaviour, so each endpoint the project cares about is its own
backend *name* (e.g. `openai`, `groq`, …), all served by one class in
`./server/backends/remote.py`. Per-name config follows the existing prefix
convention — `<NAME>_BASE_URL`, `<NAME>_API_KEY`, `<NAME>_MODEL_ID`,
`<NAME>_MAX_INPUT_SECONDS`, `<NAME>_TIMEOUT_SECONDS`, plus a
returns-segments / returns-words flag — collected in a spec table in
`./server/backends/config.py` that `DEFAULT_MODEL_IDS` and the dispatch in
`_get_backend` (`./server/model.py`) read; the if/elif chain there becomes
table-driven so adding a provider is one config row, no code. Open question:
fixed rows in config vs. letting env name extra remotes (e.g. a
`TRANSCRIPT_REMOTE_BACKENDS` list) — start fixed.

**Aggregators (OpenRouter): one endpoint, many models.** A remote backend
name identifies an *endpoint*, and the model varies per request — already
supported, since the `model` form field switches the model within a backend
and `_last_model_ids` remembers it. Consequence: input cap, timestamps
returned, chunk bounds and hallucination profile are really per
**(backend, model)**, with the endpoint-level values only as defaults.

OpenRouter facts (verified 2026-10-10 against
`https://openrouter.ai/docs/guides/overview/multimodal/stt.md` and
`GET /api/v1/models?output_modalities=transcription`):

- `POST https://openrouter.ai/api/v1/audio/transcriptions`, **OpenAI
  multipart-compatible** — the OpenAI SDK works with a base-URL swap and a
  Bearer key. No chat-completions `input_audio` mode needed for this
  provider; keep that only as a caveat for others.
- Multipart fields honoured: `file`, `model`, `language` (ISO-639-1, auto
  when omitted), `temperature`, `response_format` (`json` | `verbose_json`;
  `text`/`srt`/`vtt` → 400), `timestamp_granularities[]` (`segment`,
  `word`; verbose only). **`prompt` is accepted but ignored** (Groq-only
  prompt exists solely in the JSON body as `provider.options.groq.prompt`) —
  treat like anime-whisper: drop it.
- `verbose_json` content is provider-dependent and some models **reject it
  with 400** (documented: `openai/gpt-4o-transcribe`,
  `microsoft/mai-transcribe-1.5`; `mai-transcribe-2` unknown). Detection
  must therefore handle both "fields missing" and "400 on verbose_json →
  retry as `json`", and **cache the outcome per model id** so the
  double request happens once per run. Response also carries
  `usage` and `X-Generation-Id`; no cost tracking planned.
- **Limits that bound "long-form"**: multipart body cap **25 MB** (16 kHz
  mono int16 WAV ≈ 13 min; FLAC roughly doubles it); **upstream 60 s
  timeout per request**. Base64 JSON `input_audio` past 25 MB is allowed
  only for OpenAI and Groq models. So on OpenRouter long-form means
  *minutes per request*, bounded by provider speed against the 60 s wall,
  not hours — the tail-only scan plan applies at minute windows; one
  request per file does not.
- Target models (ids exact; model pages checked 2026-10-10):
  - `openai/whisper-large-v3-turbo` — on OpenRouter since 2026-05-01;
    providers DeepInfra (`deepinfra/us`, P50 latency 1.06 s) and Groq
    (`groq`, P50 0.28 s). 99+ languages, ~12 % WER, accepts mp3/mp4/wav/webm/flac/ogg.
    OpenRouter routes between the two unless pinned with `provider.order` /
    `provider.only`; whisper family → expect native `segments`/`words`
    under `verbose_json`. Fast enough that the 25 MB cap, not the 60 s
    timeout, is the bound.
  - `microsoft/mai-transcribe-2` — released 2026-09-03, Azure only
    (`azure`), P50 latency 6.81 s. 60 languages, auto
    language id, code switching, faster than 1.5 on long-form. **Does
    support `verbose_json`** (segment timestamps) **and
    `timestamp_granularities: ["word"]`** per its page — unlike 1.5, so the
    detection path should find native timestamps; aligner only as fallback.
    Azure extras (diarization `provider.options.azure.diarization.enabled`,
    keyword biasing `provider.options.azure.phraseList.phrases`,
    verbatim/clean `provider.options.azure.enhancedMode.modelOptions.transcribeStyle`)
    live under the JSON body's `provider` field — **unverified whether the
    multipart path accepts them** (SDK `extra_body` would add a form field);
    not needed for subtitles, skip unless keyword biasing proves useful.
- Model discovery for new rows: the `output_modalities=transcription`
  models query; ids are opaque `author/slug` strings.

Config shape for this endpoint: backend name `openrouter`;
`OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`),
`OPENROUTER_API_KEY`, `OPENROUTER_MODEL_ID` (default
`openai/whisper-large-v3-turbo`), `OPENROUTER_TIMEOUT_SECONDS` (a little
over the upstream 60 s; `max_retries=0`), `OPENROUTER_MAX_INPUT_SECONDS`
derived from the 25 MB cap for the encoding used, with a per-model override
layer for slower models that hit the 60 s wall; client
`OPENROUTER_CHUNK_MAX_SECONDS` / `_MIN_SECONDS` row in `./client/chunking.py`.
Caps and chunk bounds cannot be detected, so a remote 4xx must surface as a
clear error rather than a retry.

Client side, unchanged by the aggregator case: `GET /v1/backend` reports
properties for the **active model**, so a session that passes `--model`
must select it first (the `--backend` step with `model`, or a `model` query
param on the GET) — otherwise the first transcription switches the model
after the client has fixed its chunk profile and cap. Routing for any
remote name defaults to the word-aligner path, which always works because
the server fills `words` natively or via the aligner fallback; the
segment-path shortcut is an optimisation for models whose native segments
prove good (whisper turbo is the first candidate).

What stays intact with server-side remote backends:

- `GET /v1/backend` still answers, so `sync_backend_with_server`, the
  `ChunkProfile` lookup and `max_input_seconds` adoption need no new client
  path — no client-side profile table, no "server unreachable" special case.
- The forced aligner stays available: a remote backend reports
  `has_secondary()` and reuses `_qwen_aligner`, so word timestamps,
  `/v1/audio/align` and `--llm-asr` keep working against a remote ASR that
  returns only text or segments. Client-direct cannot offer this.
- Hallucination filter applies, keyed by the remote backend name + model id,
  so each provider accumulates its own dataset.
- API keys / base URLs live server-side in one place; the in-process bench
  (`./tests/run_file_transcripts.py`) works unchanged via `switch_backend`.
- Lifecycle is trivial: no `ModelWorker`, `load`/`unload` near no-ops.

Costs to accept / handle:

- Audio crosses the wire twice and the server decodes then re-encodes it.
  Bounded because the client still chunks at minute scale; forward as FLAC
  rather than WAV to halve the hop.
- The server's outbound call needs its own explicit timeout and
  `max_retries=0`, on top of the client-side file-mode fix listed below —
  two timeouts to keep consistent.
- `MAX_INPUT_SECONDS` is global today; it must become per-backend (remote
  caps per name, configurable or unlimited) with `GET /v1/backend`
  reporting the active backend's value.

Shape of the work:

1. `./server/backends/remote.py`: one `Backend` class constructed from a
   spec + model id over the OpenAI SDK. `transcribe_result` forwards the
   chunk with `verbose_json`, maps `segments`/`words` through, falls back to
   the local aligner when `want_words` and the remote returned none.
2. Spec table + per-name env rows in `./server/backends/config.py`; the
   per-backend cap table; table-driven `_get_backend`.
3. Client: one `ChunkProfile` row per remote name in `./client/chunking.py`
   (long-form ones with large bounds and `anchor_first_entry=False`), and
   `backend_returns_segments` consulting the same per-name
   returns-segments knowledge — ideally reported on `GET /v1/backend` so
   the client table need not mirror it.

Client-direct only wins when there is no SubsVibe server at all; if wanted,
add it later as a thin fallback (client-side profile with its own prefixed
model id / bounds, skipping the server probe). Today, pointing
`TRANSCRIPT_BASE_URL` at a foreign endpoint half-works: `/v1/backend` 404s
→ `TRANSCRIPT_BACKEND` fallback, empty model id (no env var for it since
`TRANSCRIPT_MODEL_ID` was dropped; `--model` rescues), 30 s chunks,
180 s cap, and `--llm-asr` fails on the missing align endpoint.

## Restrictions to lift (verified by grep, 2026-10-08)

- `TRANSCRIPT_MAX_INPUT_SECONDS = 180` is `MAX_INPUT_SECONDS` in
  `./server/backends/config.py`, applied by every backend in
  `./server/backends/*.py`; a new backend simply does not apply it. The client
  no longer mirrors it: `GET /v1/backend` reports `max_input_seconds` and
  `max_input_seconds()` in `./client/transcribe.py` serves the live trim in
  `./client/pipeline.py` (done 2026-10-10).
- File-mode ASR calls (`_transcribe_segment_asr_*` in `./client/client.py`)
  use `transcribe_client` from `./client/transcribe.py` with **no timeout
  and SDK default retries** (600 s, 2 retries): a long request past that
  re-uploads twice. Needs an explicit per-request `timeout` and
  `max_retries=0` on the file-mode path (live already does
  `with_options(max_retries=0)` in `./client/pipeline.py`).
- `SERVER_BACKEND_TIMEOUT_SECONDS = 30` covers only lifecycle calls
  (`/backend`, `/model/load`), not the transcription POST — no change.
- Upload: 2 h of 16 kHz int16 WAV ≈ 230 MB, buffered in RAM on both sides;
  no size cap found in `./server/server.py`. FLAC (~2×) or minute-scale
  chunks avoid it.
- `anchor_first_entry` cps caps and `split_provisional` assume short
  chunks; still valid at minutes, meaningless for one-shot.
