# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SubsVibe captures system audio and produces live subtitles. The pipeline is:

```
[client]                                                [server]          [server]
SoundCard loopback -> PCM chunks -> Silero VAD -> speech segments -> Whisper API -> raw text -> LLM API -> subtitles
```

Each stage is decoupled via queues and runs in its own thread. There are also two non-live paths: `--input <file>` for batch transcription writing `.srt` next to the input, and `--llm-asr` which routes audio through a multimodal LLM (e.g. Gemma-4 via Ollama) instead of the FastAPI server.

## Common Commands

All scripts source `scripts/core/venv.sh` to find the project venv and `scripts/env.sh` for config — run from bash (Linux/macOS) or Git Bash (Windows).

```bash
scripts/setup.sh                            # one-shot: venv + PyTorch + locked deps + model download
scripts/server.sh                           # start the FastAPI transcription server
scripts/client.sh --live --translate        # capture loopback, produce live subtitles
scripts/client.sh --input audio.mp3         # batch transcribe -> audio.srt
scripts/dev/typecheck.sh                    # AST parse-check; swap to pyright per the script's comment
```

To update locked deps after editing `requirements.in`:

```bash
pip-compile requirements.in -o requirements.txt
```

Setup order matters: PyTorch installs *before* `pip-sync` so the platform-specific build (e.g. `torch==2.11.0+cu130`) satisfies the lockfile's plain `torch==2.11.0` pin without being replaced by a generic PyPI wheel.

## Environment Configuration

All env vars live in `scripts/env.sh` (copy from `scripts/env.example.sh`):

- **Transcription server** (`TRANSCRIPT_*`): `TRANSCRIPT_HOST`, `TRANSCRIPT_PORT`, `TRANSCRIPT_MODEL_NAME`, `TRANSCRIPT_MODEL_ID`, `TRANSCRIPT_ALIGNER_ID`, `TRANSCRIPT_BASE_URL`, etc. — see [server/README.md](server/README.md) for the full reference.
- **LLM backend** (`LLM_*`): `LLM_BASE_URL`, `LLM_MODEL_NAME`, `LLM_API_KEY` — defaults to Ollama at `127.0.0.1:11434`. `LLM_ASR_MODEL_NAME` selects the multimodal model used in `--llm-asr` mode.
- **Lifecycle**: `IDLE_UNLOAD_SECONDS` / `IDLE_CHECK_SECONDS` control automatic VRAM release.

## Project Structure

```
./client/        # Audio capture, VAD, transcription worker, LLM refinement, subtitle assembly
./server/        # FastAPI server + pluggable backends (server/backends/)
./utils/         # Shared helpers (logging_config, subtitle, text, time)
./tests/         # Manual integration tests against real models (not unit tests)
./scripts/       # Setup + run scripts; scripts/core/ has shared sourced helpers
./docs/plan.md   # Phased design / spec
./references/    # Reference implementations - DO NOT MODIFY
```

## PCM Format (fixed across all stages)

- 16000 Hz, mono, int16 little-endian
- ~512 frames per chunk (32 ms at 16 kHz) — VAD requires exactly this shape
- Transcriber encodes speech segments as WAV and POSTs to the Whisper API

## Key Architecture Decisions

- **Callback-based capture**: [client/capture.py](client/capture.py) emits PCM chunks to registered callbacks (e.g. `vad.on_chunk`), no central scheduler.
- **Queue-based stage decoupling**: each stage reads from an input queue and writes to an output queue; stages run in independent threads.
- **Transcription via API, not in-process**: [client/transcribe.py](client/transcribe.py) POSTs WAV segments to `/v1/audio/transcriptions` on a Whisper-compatible server. Configured via `TRANSCRIPT_BASE_URL` + `TRANSCRIPT_MODEL_NAME`.
- **LLM via OpenAI-compatible API**: [client/llm.py](client/llm.py) talks to any chat-completions endpoint (Ollama, vLLM, LM Studio, OpenAI). A sliding context window of recent subtitle history is sent alongside new segments so the LLM can correct cross-segment errors.
- **Provisional subtitles**: subtitle lines stay tentative until enough downstream context confirms them.
- **Pluggable ASR backends**: [server/model.py](server/model.py) dispatches to `server/backends/<name>.py` per `TRANSCRIPT_BACKEND` (currently only `qwen`). The `Backend` Protocol in [server/backends/base.py](server/backends/base.py) defines the contract; `transcribe_result` returns `{text, language, words, segments}`. Streaming is not supported — the server always returns one response per request.
- **Idle unload**: a background task in [server/server.py](server/server.py) unloads aligner first, then ASR, after `IDLE_UNLOAD_SECONDS` of inactivity. Models lazy-reload on the next request.

## Server Endpoints (quick map)

- `POST /v1/audio/transcriptions` — OpenAI Whisper-compatible (multipart). Optional `timestamp_granularities=word|segment` triggers the forced aligner.
- `POST /v1/audio/align` — align externally-provided text against audio (returns word-level timestamps).
- `POST /v1/model/load` / `POST /v1/aligner/load` / `POST /v1/model/unload` — explicit lifecycle control; otherwise lazy.
- `GET /v1/health`, `GET /v1/models` — standard probes.

## References & Skills

- [references/](references/) holds upstream reference implementations used to guide server design — **not part of SubsVibe; do not modify**.
- Use the `/transcription-ref` skill for any work on the server, [client/transcribe.py](client/transcribe.py), API design, or model backend behaviour.
- Use the `/openai-sdk-subsvibe` skill when touching OpenAI-SDK calls in the client (transcription or LLM).
- When writing/updating docs in [docs/](docs/), focus on functional spec and behaviour — no code examples.

## Platform Notes

- **macOS**: no native loopback — requires BlackHole or similar virtual audio device.
- **Linux**: PulseAudio must be running.
- **Windows**: native via WASAPI loopback. Use `127.0.0.1` (not `localhost`) in `TRANSCRIPT_BASE_URL` to avoid IPv6 loopback delay.
