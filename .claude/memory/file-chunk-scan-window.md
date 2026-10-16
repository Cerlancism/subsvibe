---
name: file-chunk-scan-window
description: "The `openrouter` remote backend (implemented 2026-10-12) and the long-form even-split chunking it uses: the settings as built (58 min chunks, Opus 80k in the JSON body, mai-transcribe-2 verbatim, auto language), the measured OpenRouter/Azure limits and codec acceptance behind them, where each piece lives, and what is still open (--model ordering, hour-long WAV uploads)."
metadata:
  node_type: memory
  type: project
  originSessionId: 1b8b56a3-fc2c-439d-ac89-498a1d5348f8
  modified: 2026-10-16T14:42:52.234Z
---

# The `openrouter` backend and long-form chunking

Builds on the `CoarseChunker` section of [[recovery-vad-webrtcvad]]. The
short-form (30 s) strategy in `./client/vad.py` is unchanged for local
backends; `CoarseChunker._next_long_form` is the second strategy, used when
the server reports `long_form` for its active model.

**Why still chunk on such a backend:** the tail scan lands the chunk end on
a quiet seam instead of a flat cut at the maximum length, and chunking
keeps progress/ETA, bounded retries and the `split_provisional` overlap.
Cost of the scan is secondary.

**Even split (user direction, 2026-10-12):** never greedy. Each chunk
recomputes `parts = ceil(remaining / max)`, `target = remaining / parts`,
and scans only the last `LONG_FORM_SCAN_SECONDS` (60) before the target
with the detector ladder (latest Silero onset wins, energy seam last, flat
cut never seen so far). 61 min -> 2 x ~30.5, not 60 + 1. The reference SRT
is not consulted in this mode. Needs the container duration up front
(`_get_audio_duration`); at EOF the decoded length replaces it. Verified on
the 100 min clip: 2965 s + 3071 s.

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

## Settings as built (2026-10-12; constants in `./server/backends/config.py`)

- **Model** `microsoft/mai-transcribe-2`, one request per chunk, verbatim:
  no `provider` field, no `transcribeStyle`, no diarization. `prompt` is
  dropped (`send_prompt=False` in the spec). `language` is forwarded only
  when the client gives one; otherwise detected (returned `ja`).
- **Spec table** `MODEL_SPECS` in `./server/backends/remote.py`, keyed by
  model id: `long_form`, `max_input_seconds`, `codec`, `transport`,
  `trust_segments`, `send_prompt`, `send_language`. Unknown ids get
  `DEFAULT_SPEC` (short-form, 180 s, FLAC, entries from words). A whisper
  row later is one entry with `trust_segments=True`; the client then takes
  the short-form profile because the server reports `long_form: false`.
- **Chunking**: `openrouter` row in `./client/chunking.py` = the
  `LONG_FORM_PROFILE` (max 3480 s, min 600 s, `anchor_first_entry=False`,
  `long_form=True`), env `OPENROUTER_CHUNK_MAX_SECONDS` / `_MIN_SECONDS`.
  3480 keeps VBR Opus under the 50 MB body. `transcribe_file` clamps the
  maximum to the server's reported cap. Chunks under `MIN_REQUEST_SECONDS`
  (0.1 s) are not sent.
- **Encode**: `encode_opus` in `remote.py` through PyAV's bundled libopus
  (no ffmpeg binary): 80 kbps VBR, `application=voip`, 16 kHz mono, Ogg.
  ~0.6 MB/min on speech, 1.2 s per 7 min to encode.
- **Transport**: JSON body `POST <base>/audio/transcriptions` with
  `input_audio: {data, format: "ogg"}`, `verbose_json`, `["segment","word"]`,
  `temperature 0`, Bearer key. urllib, no SDK. Multipart exists in
  `_multipart` for a spec row that needs it.
- **Response**: `normalize_words` in `remote.py` = `fold_word_marks`
  (`./utils/text.py`, shared with the probes' `normalize_words`) plus
  rounding: letters in `word`, every mark in the previous word's
  `trailing`, mark-only tokens folded into the previous word. Words cross
  the wire with `trailing`; the client's `word_from_response`
  (`./client/transcribe.py`, both parse sites) keeps it and
  `attach_punctuation` then leaves such words untouched, so the response
  text is never searched. `segments` = one span from `segments_from_words`;
  `language` passed through; a 200 with an `error` body raises
  `RemoteError`. Server maps `RemoteError` to 502.
- **Timeouts / retries**: `OPENROUTER_TIMEOUT_SECONDS` 150 (was 600), `_MAX_RETRIES`
  2 with linear backoff on timeouts, connection errors, 5xx; 4xx and
  wrapped provider errors final. Client file mode: `with_options(timeout=
  FILE_ASR_TIMEOUT_SECONDS (900), max_retries=0)` in `transcribe_file`.
  Measured: 441 s chunk in 42 s, 480 s chunk in 16 s.
- **Cost**: `usage.cost` logged per request; $0.026 per 15 min.
- **Profile contract**: `Backend.profile()` -> `max_input_seconds`,
  `long_form`, `entries_from`; `GET /v1/backend` reports it;
  `adopt_backend_report` in `./client/transcribe.py` takes it (also used
  by `./tests/run_file_transcripts.py`). The server report wins over the
  client's `_BACKENDS_USE_*` tables.
- **Joiner fixes** landed first (commit f32dc2f): `_join_word_tokens` glues
  full-width closing marks, digits next to CJK, and `々〆ゝゞヽヾー`. Ported
  from `transit/feat/openrouter-backend` (2026-10-12): `entries_from_words`
  sees a sentence end through brackets and quotes (`。」`, `."`) and carries
  a trailing opener (`。「`) to the next cue; `attach_punctuation` bounds
  its search (`ATTACH_SEARCH_SLACK`) so a miss cannot cascade. That
  branch's own `remote.py` (whisper-large-v3, multipart WAV, local aligner
  fallback when no words come back) was not ported: `DEFAULT_SPEC` already
  feeds unknown ids Whisper-shaped, and the aligner fallback is not needed
  for mai.
- **Grok STT and Scribe (2026-10-12)**: `x-ai/grok-stt-1.0` and
  `elevenlabs/scribe-v2` share mai's row (`LONG_FORM_OPUS_SPEC`): 100 min
  as one 24 MB Opus 32k JSON request answered 200 (grok 77 s, $0.17;
  scribe 117 s, $0.18), per-character words with marks as words, one
  untimed segment. Scribe returns ISO-639-3 (`jpn`) and per-word
  `confidence`; `to_iso_code` maps the 3-letter codes (aliases in
  `./utils/language.py`) and the backend normalises the returned
  language. Both take `language`; mai runs are sent none (bench json
  per-run `"language": null`). Scribe/Grok leave 1-2 s holes inside a
  word (です -> で ... す), so `entries_from_words` needs
  `WORD_GAP_FLUSH_UNMARKED_SECONDS` (2 s) after an unmarked word; Scribe
  also emits `[音楽]` audio events, dropped in `normalize_words`. 5 min
  quality review 2026-10-13: mai > Scribe > local large-v3 > Grok >
  OpenRouter whisper (Scribe rewrites ハッシュタグ as `#` and flipped one
  polarity; Grok drops crosstalk). Whisper on OpenRouter: `DEFAULT_SPEC`
  (FLAC, JSON) verified on 16 requests, no row needed.
- **Not planned**: diarization, clean style.

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

## Where it lives

Server-side, as decided 2026-10-10: `RemoteBackend` in
`./server/backends/remote.py`, one backend *name* per endpoint, per-name
env by prefix (`OPENROUTER_*` in `config.py`, documented in
`./scripts/backends/env.openrouter.sh` and `./server/README.md`),
table-driven `_BACKEND_CLASSES` in `./server/model.py`. Kept: `GET
/v1/backend` as the only client sync path, hallucination filter keyed by
backend + model (no entries for `openrouter`), keys server-side, the
in-process bench. Audio crosses the wire twice (client WAV to server,
server Opus out); an hour of WAV is 115 MB in RAM on both sides, accepted.
Client-direct to a foreign endpoint is not supported.

## Concurrency (2026-10-16)

`openrouter` serves requests in parallel and per model: `RemoteBackend`
reports `max_concurrency()` = `OPENROUTER_MAX_CONCURRENT` (default 8) and
`transcribe_result` takes `model_id` per call (falling back to the id
`set_model_id` chose), reading only that local so a concurrent switch never
touches a call in flight. `transcribe` in `./server/server.py` skips
`switch_model` on such a backend and admits calls through an
`asyncio.Semaphore` (`_request_slots` / `_transcribe_in_slot`, rebuilt per
backend name and limit); the rest wait in arrival order holding no thread.
Remote timeout and retries are unchanged and run per call after the wait.
Local backends keep `max_concurrency()` = 1 and the switch-then-serialise
path, untouched by design (the switch-under-in-flight race there is still
open). `GET /v1/backend?model=` returns that model's profile and
`sync_backend_with_server(model=args.model)` passes it, which closed the
`--model` ordering item below. Verified with a fake parallel backend behind
uvicorn: 12 requests on 3 models, peak 8 in flight, no switch, each answer
from its own model.

## Open

- ~~`--model` on a session: the client fixes its profile from the server's
  *active* model at sync, before the first request switches it.~~ Closed
  2026-10-16: the sync asks `GET /v1/backend?model=` for the session's
  model. On local backends every model shares one profile anyway.
- Client file-mode timeout (`FILE_ASR_TIMEOUT_SECONDS`, 900 s) now also
  covers time spent queued behind `OPENROUTER_MAX_CONCURRENT` other
  requests; with 8 slots and hour chunks at about a minute each that is
  comfortable, but a much lower limit or many clients could reach it.
- `transcribe` in `./server/server.py` logs the whole text at INFO per
  request; an hour chunk makes that line enormous. Pre-existing.
- `write_srt`'s entry merge joins CJK entries with a space
  (`こんばんは こんばんは`); separate from the word joiner, untouched.
