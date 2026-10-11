---
name: file-chunk-scan-window
description: "File-mode chunking for a long-form ASR backend and the `openrouter` remote backend that will use it: decided settings (60 min chunks, Opus 80k in the JSON body, mai-transcribe-2 verbatim, auto language), the measured OpenRouter/Azure limits and codec acceptance behind them, what the response needs before our word path can use it, the server-side remote-backend design, and the client/server restrictions still to lift."
metadata:
  node_type: memory
  type: project
  originSessionId: 1b8b56a3-fc2c-439d-ac89-498a1d5348f8
  modified: 2026-10-11T17:10:32.124Z
---

# File-mode chunking for a long-form backend

Builds on the `CoarseChunker` section of [[recovery-vad-webrtcvad]]. The
30 s chunker in `./client/vad.py` stays as it is for local backends; what
follows applies to the `openrouter` backend, whose requests carry an hour.

**Why still chunk on such a backend:** the tail scan lands the chunk end on
a quiet seam instead of a flat cut at the maximum length, and chunking
keeps progress/ETA, bounded retries and the `split_provisional` overlap.
Cost of the scan is secondary.

## Facts that shape the chunker (measured 2026-10-09, 15 min Japanese clip)

- Silero's `get_speech_timestamps` is stateful: a scan of a sub-band with
  lead-in does not reproduce a scan from the window start. A tail-only scan
  therefore cuts differently from the 30 s chunker; fine for a backend with
  its own chunk size, not a drop-in.
- SRT diffs cannot validate chunker changes: two runs of the same code
  differ by ~1170 diff lines because faster-whisper on GPU is not
  deterministic and the ASR-driven cursor cascades the variance. Compare
  chunker decisions (end, method) directly. mai is not deterministic either:
  same FLAC twice agrees 99.4% of characters.

## Decided settings for the `openrouter` backend (user direction, 2026-10-12)

- **Model** `microsoft/mai-transcribe-2`, one request per chunk, verbatim.
- **Chunking**: `ChunkProfile` row `openrouter` in `./client/chunking.py`
  with `OPENROUTER_CHUNK_MAX_SECONDS = 3600`, a large min bound,
  `anchor_first_entry=False`, no head scan; cut at a quiet seam in the last
  ~16 s before the window end (`quiet_seam` in `./tests/openrouter_common.py`
  is the prototype). Cut a minute or two short of 60 so VBR Opus stays
  under the body cap. Per-backend `MAX_INPUT_SECONDS` = 3600 for this name.
- **Encode**: Opus 80 kbps VBR, `-application voip`, 16 kHz mono, Ogg
  container (`encode_ogg` in `./tests/test_openrouter_longform.py`, ffmpeg
  libopus). ~36.7 MB raw / 49 MB base64 per hour.
- **Transport**: JSON body `POST /api/v1/audio/transcriptions` with
  `input_audio: {data: <base64>, format: "ogg"}`, `response_format:
  verbose_json`, `timestamp_granularities: ["segment", "word"]`,
  `temperature: 0`, OpenRouter Bearer key. Prototype `post_json` in
  `./tests/test_openrouter_longform.py`. Not multipart (25 MB cap).
- **Language**: omit `language` for auto detect (returned `ja`); pass the
  response `language` through `transcribe_result`.
- **Style**: send no `provider` field (Azure default is verbatim). No
  `transcribeStyle`, no diarization, no `prompt` (accepted, ignored).
- **Response handling**: `normalize_words` (`./tests/openrouter_common.py`)
  before `attach_punctuation` + `entries_from_words`; trust `words`, ignore
  `segments` timing; a 200 whose body is `{"error": ...}` is a failure.
- **Timeouts / retries**: ~1 s wall per minute of audio (100 min in 92 s)
  with one 3x outlier seen; outbound timeout 600 s per hour chunk,
  `max_retries=2` on timeouts/5xx, none on 4xx. Skip chunks under 0.1 s
  (the chunker emits a 0.0 s file-tail chunk; the provider answers 400).
- **Cost**: `usage.cost` per response, ~$0.10 per hour of audio; log it
  per chunk.
- **Joiner fixes first**, in `write_srt` / `_join_word_tokens`
  (`./client/subtitle.py`), affecting every word backend: no space after
  CJK punctuation, glue ASCII digits to CJK, glue `々`. Without them mai
  output reads `3 チーム 11 人`, `甘 々`.
- **Not planned**: diarization, clean style, whisper on OpenRouter.

## OpenRouter / Azure facts the settings rest on (measured 2026-10-11/12)

Probes: `./tests/test_openrouter_longform.py` (mai, `--codec`, `--bitrate`,
`--transport`, `--style`, `--rebuild`), `./tests/test_openrouter_whisper.py`,
shared `./tests/openrouter_common.py`. Outputs, raw JSON and logs under
`tests/data/archive/2026-10-11-openrouter/` (15 min runs) and
`tests/data/archive/2026-10-12-openrouter-100min/` (100 min, codec and
bitrate runs). Env: `OPENROUTER_API_KEY` from `OPENROUTER_KEY_GENERIC` in
`./scripts/backends/env.openrouter.sh`.

- **Body caps**: multipart 25 MB (400 from OpenRouter); JSON body 50 MB
  (52428800 bytes, 413 at the edge). The documented 60 s upstream timeout
  is not enforced for mai.
- **Codecs accepted**: wav, flac, ogg/opus, mp3. Rejected with "Provider
  returned 400" in every shape: AAC (fdk HE-AAC, fdk LC, native; m4a, ADTS;
  any mime or JSON `format`) and webm/opus.
- **Sizes at 16 kHz mono**: WAV 1.92 MB/min, FLAC ~1.4 MB/min on this
  material (fits ~26 min per JSON request), Opus/MP3 at N kbps = N/133
  MB/min. 100 min one-shot works at Opus 32k (24 MB, 92 s, $0.17); mai's
  own 2 h limit comes before the cap at that rate.
- **Codec vs quality** (first 15 min, characters identical to FLAC, noise
  floor 99.4%): Opus 32k 95.6% and drops a clause; Opus/MP3 64k 97.5%;
  Opus 80k 98.6% with every known miss recovered; Opus 96k 98.0%, so the
  effect bottoms out at 80k. MP3 80k 98.1% but lost a clause again. Chunk
  length does not change the transcript (Ogg 15 min vs first 15 of 100 min
  99.3%). Hence 80k Opus at 60 min; a shorter cap buys nothing.
- **Formats**: only `json` / `verbose_json`; `srt`/`vtt`/`text` 400. SRT is
  always ours. mai returns `words` (every character a word, each mark its
  own word) and `segments`; segments were once all `start: 0, end: 0`.
- **Word normalisation is mandatory**: no mark reaches `trailing`
  otherwise, so `entries_from_words` never flushes on sentence ends.
  whisper providers embed marks in tokens (`さい。`); mai separates them.
  `normalize_words` handles both.
- **Diarization** (`provider.options.azure.diarization.enabled`, JSON body
  only) gives 2-3 alternating labels for 11 speakers: turn boundaries, not
  identity. `maxSpeakers` -> provider 400; snake-case is dropped silently.
- **Clean style** (`provider.options.azure.enhancedMode.modelOptions.transcribeStyle`)
  removes fillers and un-splices interjections but drops a few real words;
  same cost. Verbatim chosen.
- **Quality verdict vs local faster-whisper-large-v3** (read side by side,
  `tests/data/archive/2026-10-11/`): mai verbatim is the safe replacement.
  large-v3 fabricates lines, loops (`乾杯` x7) and turns crosstalk into word
  salad; mai never invents, keeps terms consistent, and misses only two
  short reactions and one polarity flip in 15 min. Cast-name readings are
  wrong on every system. Do not score with a text-distance metric against
  `sample_15mins_ja.control.srt`: it omits crosstalk and penalises the
  faithful transcript (`tests/srt_cer.py` was tried and deleted).
- **Speed / cost per 15 min**: mai one-shot 25-60 s, $0.026; mai through
  the 30 s chunker 714 s (~16 s fixed latency per request); whisper-large-v3
  chunked 369 s. Only mai is long-form.

## Where the backend lives (decision 2026-10-10)

Server-side: one class in `./server/backends/remote.py` that forwards to an
OpenAI-compatible endpoint, one backend *name* per endpoint (`openrouter`
first), per-name config by prefix (`OPENROUTER_BASE_URL`, `_API_KEY`,
`_MODEL_ID`, `_MAX_INPUT_SECONDS`, `_TIMEOUT_SECONDS`) in a spec table in
`./server/backends/config.py`, and a table-driven `_get_backend` in
`./server/model.py`. On an aggregator the model varies per request (the
`model` form field and `_last_model_ids` already support it), so caps and
chunk bounds are per (backend, model) with endpoint defaults.

What this keeps: `GET /v1/backend` (so `sync_backend_with_server`, the
`ChunkProfile` lookup and `max_input_seconds` adoption need no new client
path); the forced aligner as fallback via `has_secondary()`; the
hallucination filter keyed by backend + model; keys server-side; the
in-process bench `./tests/run_file_transcripts.py` unchanged; near no-op
`load`/`unload`.

Costs: audio crosses the wire twice (client WAV to server, server Opus to
OpenRouter); the server's outbound timeout and the client's file-mode
timeout must stay consistent; `MAX_INPUT_SECONDS` becomes per-backend and
`GET /v1/backend` reports it. A session that passes `--model` must select
it before the client fixes its chunk profile and cap.

Work: (1) `remote.py` with `transcribe_result` forwarding Opus in the JSON
body, mapping `words`/`segments`, aligner fallback when `want_words` and no
words came back; (2) spec table, per-name env rows, per-backend cap table,
table-driven dispatch; (3) client `ChunkProfile` row, and
`backend_returns_segments` fed from `GET /v1/backend` rather than a client
table. Client-direct to a foreign endpoint is not planned; today it
half-works (`/v1/backend` 404s, empty model id, 30 s chunks, 180 s cap).

## Restrictions to lift (verified by grep, 2026-10-08)

- `TRANSCRIPT_MAX_INPUT_SECONDS = 180` is `MAX_INPUT_SECONDS` in
  `./server/backends/config.py`, applied by every backend; make it
  per-backend. The client already adopts the server's value via
  `GET /v1/backend` (done 2026-10-10).
- File-mode ASR calls (`_transcribe_segment_asr_*` in `./client/client.py`)
  use `transcribe_client` with no timeout and SDK default retries (600 s,
  2 retries): a long request past that re-uploads twice. Give the file-mode
  path an explicit per-request `timeout` and bounded retries (live already
  uses `with_options(max_retries=0)` in `./client/pipeline.py`).
- Upload: an hour of 16 kHz int16 WAV is 115 MB, buffered in RAM on both
  sides; no size cap in `./server/server.py`. Acceptable at one hour; the
  server re-encodes to Opus before forwarding.
- `anchor_first_entry` cps caps and `split_provisional` assume short
  chunks; the `openrouter` profile disables the anchor.
