# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SubsVibe captures system audio and produces live subtitles. The pipeline is:

```
[client]                                                [server]          [server]
SoundCard loopback -> PCM chunks -> Silero VAD -> speech segments -> Whisper API -> raw text -> LLM API -> subtitles
```

Each stage is decoupled via queues and runs in its own thread. There are also two non-live paths: `--input <file>` (file mode) transcribes a file to `.srt` next to the input, and `--llm-asr` which routes audio through a multimodal LLM (e.g. Gemma-4 via Ollama) instead of the FastAPI server.

## Common Commands

All scripts source `./scripts/core/venv.sh` to find the project venv and `./scripts/env.sh` for config — run from bash (Linux/macOS) or Git Bash (Windows).

```bash
scripts/setup.sh                            # one-shot: venv + PyTorch + locked deps + model download
scripts/server.sh                           # start the FastAPI transcription server
scripts/client.sh --live --translate        # capture loopback, produce live subtitles
scripts/client.sh --input audio.mp3         # file mode: transcribe -> audio.srt
scripts/dev/typecheck.sh                    # AST parse-check; swap to pyright per the script's comment
scripts/core/python.sh tests/run_file_transcripts.py  # batch file-mode bench: tests/data/*.json -> .srt per (backend, model), in-process
```

To update locked deps after editing `requirements.in`:

```bash
pip-compile requirements.in -o requirements.txt
```

Setup order matters: PyTorch installs *before* `pip-sync` so the platform-specific build (e.g. `torch==2.11.0+cu130`) satisfies the lockfile's plain `torch==2.11.0` pin without being replaced by a generic PyPI wheel.

## Environment Configuration

All env vars live in `./scripts/env.sh` (copy from `./scripts/env.example.sh`):

- **Transcription server** (`TRANSCRIPT_*`): `TRANSCRIPT_HOST`, `TRANSCRIPT_PORT`, `TRANSCRIPT_BACKEND`, `TRANSCRIPT_ALIGNER_ID`, `TRANSCRIPT_BASE_URL`, etc. — see `./server/README.md` for the full reference. Settings read by a single backend are prefixed with that backend's name (`FASTER_WHISPER_*`, `QWEN_*`, `ANIME_WHISPER_*`), including each backend's startup model `<BACKEND>_MODEL_ID` (there is no global model id var); every backend tunable has its constant default in `./server/backends/config.py`, with the env var as override (helpers in `./utils/env.py`). Client-side file-mode chunk bounds follow the same scheme per backend in `./client/chunking.py` (`<BACKEND>_CHUNK_MAX_SECONDS` / `_MIN_SECONDS`).
- **LLM backend** (`LLM_*`): `LLM_BASE_URL`, `LLM_MODEL_ID`, `LLM_API_KEY` — defaults to Ollama at `127.0.0.1:11434`. `LLM_ASR_MODEL_ID` selects the multimodal model used in `--llm-asr` mode.
- **Lifecycle**: `IDLE_UNLOAD_SECONDS` / `IDLE_CHECK_SECONDS` control automatic VRAM release.

## Path Conventions (agent docs)

In this file and anything under `./.claude/`:

- **Project-root paths**: write **relative to the repo root, prefixed with `./`**, e.g. `./client/render.py` — never bare `client/render.py`, never absolute (`c:\Users\...\client\render.py`), never deep parent-relative (`../../../client/render.py`).
- **In-skill / in-folder paths** (a skill file pointing at a sibling reference under its own directory): keep the sibling-relative form, e.g. `references/streaming.md`. No `./` needed.
- **Always wrap paths in backticks** — do **not** use markdown link syntax `[./path](./path)` in agent docs (including same-folder references). The backtick form is the convention; links add noise without value here.

## Project Structure

```
./client/        # Audio capture, VAD, transcription worker, LLM refinement, subtitle assembly
./server/        # FastAPI server + pluggable backends (server/backends/)
./utils/         # Shared helpers (language, logging_config, romanize, subtitle, text, time)
./tests/         # Mixed test bench: unit tests, manual model/integration tests, library try-outs, decision-driving analyses — check each file's header before assuming it's automatable
./scripts/       # Setup + run scripts; scripts/core/ has shared sourced helpers
./docs/plan.md   # Phased design / spec
./references/    # Reference implementations - DO NOT MODIFY
```

## PCM Format (fixed across all stages)

- 16000 Hz, mono, int16 little-endian
- ~512 frames per chunk (32 ms at 16 kHz) — VAD requires exactly this shape
- Transcriber encodes speech segments as WAV and POSTs to the Whisper API

## Key Architecture Decisions

- **Commit-on-silence live pipeline**: `./client/live_vad.py` wraps Silero VADIterator. Each PCM chunk yields zero or more `SegmentEvent`s — provisional (in-progress preview, refreshed ~1Hz) or final (immutable, on confirmed silence or force-flush after MAX_SEG_SECONDS). The pipeline transcribes (and optionally translates) each event once.
- **Queue-based stage decoupling**: each stage reads from an input queue and writes to an output queue; stages run in independent threads. Stale items older than `LIVE_LAG_TOLERANCE_SECONDS` are dropped, but **finals are sticky** — never dropped in favour of newer provisionals.
- **Transcription via API, not in-process**: `./client/transcribe.py` POSTs WAV segments to `/v1/audio/transcriptions` on a Whisper-compatible server. Configured via `TRANSCRIPT_BASE_URL`; the model id is the server's, read at session start.
- **LLM via OpenAI-compatible API**: `./client/llm.py` translates one utterance per call. History fed to the LLM contains *committed* (final) utterances only — provisional previews never enter context, so history never drifts on mid-sentence noise.
- **In-place rendering**: `./client/render.py` scrolls committed lines and overwrites the current provisional line via `\r`.
- **Pluggable ASR backends**: `./server/model.py` dispatches to `./server/backends/<name>.py` per `TRANSCRIPT_BACKEND` (`faster-whisper`, `qwen`, `anime-whisper`, and `openrouter` in `./server/backends/remote.py`, which forwards to OpenRouter with a per-model spec table). The `Backend` Protocol in `./server/backends/base.py` defines the contract; `transcribe_result` returns `{text, language, words, segments}` and `profile()` describes the current model (`max_input_seconds`, `long_form`, `entries_from`), which `GET /v1/backend` reports and the client adopts. Streaming is not supported — the server always returns one response per request.
- **Two file-mode chunking strategies** (`./client/vad.py`, profiles in `./client/chunking.py`): short form cuts at the latest speech onset inside the bounds (30 s Whisper windows); long form (`mai-transcribe-2`) splits the remaining audio into the fewest equal parts under the maximum and scans only each part's last minute for a seam, recomputed every chunk. The server's `long_form` report picks the strategy; the client clamps the maximum to the server's cap.
- **Runtime backend swap**: `TRANSCRIPT_BACKEND` is the startup selection only; the live one lives in `./server/model.py` (`active_backend` / `switch_backend`) and is swapped through `POST /v1/backend`. A switch disposes the outgoing backend's worker child processes exactly as an unload does, then selects the model last used on the incoming backend this run, falling back to the id that backend would have started with. The client never assumes its own `TRANSCRIPT_BACKEND` matches — `sync_backend_with_server` in `./client/transcribe.py` adopts the server's backend at session start, because entry post-processing differs per backend. Requesting a switch is a separate CLI step (`--backend`) run between sessions, never a session option: a switch tears down worker processes. The server-management flags are matched in a fixed order and each returns, so combining two of them, or pairing one with a session, runs the first and drops the rest — unsupported rather than rejected. `--load` is the exception: it is a modifier read by `--backend` and by a session, not a step. With `TRANSCRIPT_BACKEND` unset both sides fall back to `DEFAULT_BACKEND` in `./utils/backend.py` (`faster-whisper`) — the server used to default to `qwen` here, silently disagreeing with the client and with `./scripts/env.example.sh`.
- **Idle unload**: a background task in `./server/server.py` unloads aligner first, then ASR, after `IDLE_UNLOAD_SECONDS` of inactivity. Models lazy-reload on the next request.
- **Per-request models and parallel requests on the remote backend only**: `Backend.max_concurrency()` is 1 on the local backends (one resident model, calls serialised, a request naming another model switches the server first) and `OPENROUTER_MAX_CONCURRENT` (default 8) on `openrouter`, which takes the model id per `transcribe_result` call and never switches. `transcribe` in `./server/server.py` admits up to that many calls through an `asyncio.Semaphore` and queues the rest in arrival order; remote timeout and retries apply per forwarded call after the wait. `GET /v1/backend?model=` reports that model's profile, and the client passes its `--model` at sync so a session chunks for the model it sends.

## Server Endpoints (quick map)

- `POST /v1/audio/transcriptions` — OpenAI Whisper-compatible (multipart). Optional `timestamp_granularities=word|segment` triggers the forced aligner.
- `POST /v1/audio/align` — align externally-provided text against audio (returns word-level timestamps).
- `POST /v1/model/load` / `POST /v1/aligner/load` / `POST /v1/model/unload` — explicit lifecycle control; otherwise lazy.
- `GET /v1/backend` / `POST /v1/backend` — read or swap the active ASR backend at runtime.
- `GET /v1/health`, `GET /v1/models` — standard probes.

## References & Skills

- `./references/` holds upstream reference implementations used to guide server design — **not part of SubsVibe; do not modify**.
- Use the `/transcription-ref` skill for any work on the server, `./client/transcribe.py`, API design, or model backend behaviour.
- Use the `/openai-sdk-subsvibe` skill when touching OpenAI-SDK calls in the client (transcription or LLM).
- When writing/updating docs in `./docs/`, focus on functional spec and behaviour — no code examples.

## Project Memory

Local project memory lives under `./.claude/memory/`. Start at `./.claude/memory/MEMORY.md` — it indexes per-topic notes. Check items off in the topic file as they land, and add new index entries to `MEMORY.md` when starting a new topic.

Reference code in memory notes by **function or symbol name**, not line numbers (e.g. `_render` in `./client/render.py`, not `./client/render.py:84-120`). Line numbers rot on every edit; function names survive refactors. File-level path refs are fine.

## Platform Notes

- **macOS**: no native loopback — requires BlackHole or similar virtual audio device.
- **Linux**: PulseAudio must be running.
- **Windows**: native via WASAPI loopback. Use `127.0.0.1` (not `localhost`) in `TRANSCRIPT_BASE_URL` to avoid IPv6 loopback delay.
