# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SubsVibe captures system audio and produces live subtitles. The pipeline is:

```
[client]                                                [server]          [server]
SoundCard loopback -> PCM chunks -> Silero VAD -> speech segments -> Whisper API -> raw text -> LLM API -> subtitles
```

Each stage is decoupled via queues, running in its own thread.

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install pip-tools
pip-sync requirements/requirements.txt
```

To update locked deps after editing `requirements.in`:

```bash
pip-compile requirements/requirements.in -o requirements/requirements.txt
```

## Project Structure

```
./requirements/
  requirements.in      # abstract deps (client + server)
  requirements.txt     # locked deps (committed)
./client/
  capture.py           # Audio capture with callback-based PCM streaming (SoundCard loopback)
  vad.py               # Silero VAD speech filtering
  transcribe.py        # Transcription worker (Whisper API client)
  llm.py               # LLM context-aware subtitle refinement (OpenAI-compatible API)
  pipeline.py          # Wires all stages together
./server/
  server.py            # FastAPI transcription server (OpenAI Whisper-compatible)
  model.py             # Model backend abstraction (Faster Whisper / Qwen3-ASR)
```

## References

`./references/` contains reference implementations used to guide the transcription server design. These are not part of the SubsVibe codebase and should not be modified.

Use the `/transcription-ref` skill when working on anything transcription-related — the server (`./server/`), the client transcription worker (`./client/transcribe.py`), API design, or model backend behaviour.

## Design Plan

See `./docs/plan.md` for the full phased implementation plan.

When writing or updating plan docs (`./docs/`), focus on functional specification, expected capabilities, and behaviour. Do not include code examples.

## PCM Format (fixed across all stages)

- 16000 Hz, mono, int16 little-endian
- ~512 frames per chunk (32ms at 16kHz)
- VAD expects exactly 512-sample chunks at 16kHz
- Transcriber encodes speech segments as WAV and sends them to the Whisper API

## Key Architecture Decisions

- **Callback-based capture**: `./client/capture.py` emits PCM chunks to registered callbacks (e.g. `vad.on_chunk`)
- **Queue-based stage decoupling**: each processing stage reads from an input queue and writes to an output queue
- **Transcription via API**: `./client/transcribe.py` sends speech segments to `POST /v1/audio/transcriptions` on a Whisper-compatible server; configured via `transcription.base_url` + model name
- **LLM via API**: `./client/llm.py` sends chat completions requests to any OpenAI-compatible server (Ollama, vLLM, LM Studio, OpenAI, etc.); configured via `llm.base_url` + model name
- **Sliding context window** in `./client/llm.py`: recent subtitle history is sent alongside new transcription segments so the LLM can correct errors using cross-segment context
- **Provisional subtitles**: subtitles are not finalized until enough context confirms them

## Testing Audio Capture

```bash
python ./client/capture.py                        # Capture -> output.pcm, Ctrl+C to stop
python ./client/capture.py --seconds 10           # Capture 10 seconds
python ./client/capture.py --output test.pcm      # Custom output path
python ./client/capture.py --list                 # List loopback devices
ffplay -f s16le -ar 16000 -ac 1 output.pcm        # Playback captured audio
```

## Dependencies

- `SoundCard` - cross-platform loopback audio (WASAPI on Windows, PulseAudio on Linux)
- `numpy` - PCM format conversion (float32 <-> int16)
- `silero-vad[onnx-cpu]` - voice activity detection via ONNX Runtime (no PyTorch)
- `openai` - API client for both Whisper-compatible transcription and LLM chat completions

## Platform Notes

- **macOS**: No native loopback - requires BlackHole virtual audio device
- **Linux**: PulseAudio must be running
- **Windows**: Works natively via WASAPI loopback
