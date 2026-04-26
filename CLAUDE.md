# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SubsVibe captures system audio and produces live subtitles via local speech-to-text. The pipeline is:

```
SoundCard loopback -> PCM chunks -> Silero VAD -> speech segments -> Transcriber (Faster Whisper | Qwen3-ASR) -> raw text -> LLM -> subtitles
```

Each stage is decoupled via queues, running in its own thread.

## Development Setup

```bash
uv sync
```

## Project Structure

```
pyproject.toml       # Dependencies and project metadata (uv)
src/
  capture.py         # Audio capture with callback-based PCM streaming (SoundCard loopback)
  vad.py             # Silero VAD speech filtering
  transcribe.py      # Transcription worker (Faster Whisper or Qwen3-ASR)
  llm.py             # LLM context-aware subtitle refinement (OpenAI-compatible API)
  pipeline.py        # Wires all stages together
```

## Design Plan

See `docs/plan.md` for the full phased implementation plan.

When writing or updating plan docs (`docs/`), focus on functional specification, expected capabilities, and behaviour. Do not include code examples.

## PCM Format (fixed across all stages)

- 16000 Hz, mono, int16 little-endian
- ~512 frames per chunk (32ms at 16kHz)
- VAD expects exactly 512-sample chunks at 16kHz
- Transcriber expects numpy float32 arrays at 16kHz mono (both Faster Whisper and Qwen3-ASR accept this format)

## Key Architecture Decisions

- **Callback-based capture**: `capture.py` emits PCM chunks to registered callbacks (e.g. `vad.on_chunk`)
- **Queue-based stage decoupling**: each processing stage reads from an input queue and writes to an output queue
- **Pluggable transcription backends**: `transcribe.py` supports Faster Whisper (CPU-friendly, `base` model int8) or Qwen3-ASR-1.7B/0.6B (GPU, 52 languages, auto language detect) — selected via config
- **Sliding context window** in `llm.py`: recent subtitle history is sent alongside new transcription segments so the LLM can correct errors using cross-segment context
- **Provisional subtitles**: subtitles are not finalized until enough context confirms them
- **OpenAI SDK for LLM**: works with OpenAI API, Ollama, vLLM, LM Studio, or any OpenAI-compatible endpoint via `base_url`

## Testing Audio Capture

```bash
uv run python capture.py                        # Capture -> output.pcm, Ctrl+C to stop
uv run python capture.py --seconds 10           # Capture 10 seconds
uv run python capture.py --output test.pcm      # Custom output path
uv run python capture.py --list                 # List loopback devices
ffplay -f s16le -ar 16000 -ac 1 output.pcm      # Playback captured audio
```

## Dependencies

- `SoundCard` - cross-platform loopback audio (WASAPI on Windows, PulseAudio on Linux)
- `numpy` - PCM format conversion (float32 <-> int16)
- `silero-vad` + `torch` - voice activity detection
- `faster-whisper` - CTranslate2-based transcription (default backend: `base` model, `int8` on CPU)
- `qwen-asr` - Qwen3-ASR-1.7B/0.6B transcription (optional backend: 52 languages, GPU bfloat16, auto language detect)
- `openai` - LLM API client

## Platform Notes

- **macOS**: No native loopback - requires BlackHole virtual audio device
- **Linux**: PulseAudio must be running
- **Windows**: Works natively via WASAPI loopback
