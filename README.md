# SubsVibe

Real-time subtitles from system audio using local speech-to-text.

Captures your system's audio output (any app, any language), runs it through voice activity detection and speech recognition locally, and displays live subtitles. In live mode an LLM pass refines the output with context-aware correction and translation, and a subtitle stage emits timed SRT lines with line-wrap, CPS, and reading-time heuristics. 

A batch mode transcribes an existing audio or video file straight to an `.srt` (VAD segmentation, transcription, and SRT timing - no LLM refinement or translation).

## Demo
**(Turn up volume)**  

https://github.com/user-attachments/assets/b37d3e33-bccf-4005-acb2-2b9f02da6267

## Features

- **Two ways in.** Live mode captures system audio (any app, any language) and streams subtitles as you watch; batch mode (`--input <file>`) transcribes an existing audio or video file straight to an `.srt` alongside it.
- **Fully local.** Capture, VAD, and speech recognition all run on your machine - no audio leaves it. The LLM stage points at whatever you configure: local models (Ollama, LM Studio, vLLM) or a cloud endpoint.
- **Commit-on-silence live pipeline.** Each utterance is transcribed once when it ends, with mid-utterance previews shown in place - no fixed sliding window, no re-transcribing the same audio.
- **LLM refinement and translation (live mode).** A context-aware LLM pass corrects errors and translates, driven by committed history so context never drifts on mid-sentence noise.
- **Pluggable transcription backends.** A FastAPI server fronts Faster Whisper (default, CPU-friendly), Qwen3-ASR, or Anime Whisper behind one OpenAI Whisper-compatible API - switch via config, no pipeline changes.
- **Subtitle-aware output.** SRT lines come with line-wrap, CPS, and reading-time heuristics rather than raw transcript dumps.
- **Prompt context for batch.** Batch mode can feed prior transcripts or a reference `.srt` into each segment's ASR prompt via `--history` / `--context-src` (LLM refinement and `--translate` remain live-only).

Tuning of segment timing, subtitle wrapping, and translation-prompt quality is ongoing - see [docs/plan.md](docs/plan.md) for the full design and what's still planned.

## How it works

**Live mode** (`--live`) captures system audio and streams subtitles as you watch:

```
System Audio -> Voice Detection -> Speech-to-Text -> LLM Refinement -> Subtitles
```

**Batch mode** (`--input <file>`) turns an existing audio or video file into an `.srt`:

```
Audio/Video File -> Voice Detection -> Speech-to-Text -> Subtitles (.srt)
```

Batch mode stops at transcription and SRT timing - there is no LLM refinement or translation pass (those are live-only), though prior transcripts or a reference `.srt` can still be fed into each segment's ASR prompt via `--history` / `--context-src`.

All processing runs locally. No audio leaves your machine. The LLM stage works with local models (Ollama, LM Studio, vLLM) or cloud endpoints - your choice.


## Setup

Requires Python 3.14. Faster Whisper runs on GPU or CPU (int8); the Qwen3-ASR backend requires a GPU.

Run the scripts in `scripts/` from any POSIX shell — bash on Linux/macOS, or Git Bash on Windows.

```bash
cp scripts/env.example.sh scripts/env.sh    # first time only
# Edit scripts/env.sh and set PYTORCH_INSTALL_CMD for your platform.
# Get the right command from https://pytorch.org/get-started - pick your OS,
# package (Pip), and compute platform (CUDA 12.x / ROCm / CPU / etc.).
scripts/setup.sh                            # creates .venv, installs PyTorch + locked deps, downloads models
scripts/server.sh                           # start the transcription server
scripts/client.sh --live --translate        # capture loopback audio and produce live subtitles
```

The setup script installs PyTorch first (from the wheel index in `PYTORCH_INSTALL_CMD`), then `pip-sync` against `requirements.txt`. The platform-specific build's local version tag (e.g. `+cu130`, `+rocm6.2`, `+cpu`) satisfies the lockfile's plain torch pin, so your chosen wheel is preserved. To switch platforms, change `PYTORCH_INSTALL_CMD` in `scripts/env.sh` and re-run setup.

## Pipeline stages

| Stage | What it does | Live | Batch |
|-------|-------------|:----:|:-----:|
| **Input** | Live: records system audio via loopback (SoundCard). Batch: decodes an audio/video file to PCM (PyAV). | ✓ | ✓ |
| **VAD** | Silero VAD (with a WebRTC VAD second-chance pass) marks speech. Live drops silence/noise; batch keeps the whole timeline and only uses VAD to choose segment boundaries. | ✓ | ✓ |
| **Transcribe** | Converts speech to text (Faster Whisper, Qwen3-ASR, or Anime Whisper). | ✓ | ✓ |
| **LLM** | Corrects errors, adds context, translates (any OpenAI-compatible API). | ✓ | — |
| **Subtitle** | Times and wraps lines into SRT (line-wrap, CPS, reading-time heuristics). | ✓ | ✓ |

In live mode each stage runs in its own thread, connected by queues. Batch mode runs the same stages sequentially over a file (minus the LLM pass).

## Transcription backends

| Backend | Model size | Device | Strength |
|---------|-----------|--------|----------|
| **Faster Whisper** | tiny / base / small / medium / large-v3 | GPU or CPU (int8) | Fast, low memory, proven quality, ~100 languages |
| **Qwen3-ASR-1.7B** | 1.7B params | GPU (bfloat16) | 30 languages + 22 Chinese dialects, auto language detection, SOTA accuracy |
| **Qwen3-ASR-0.6B** | 0.6B params | GPU (bfloat16) | Lighter weight; ~2000× throughput at high concurrency on the vLLM backend |
| **Anime Whisper** | based on Whisper-large-v2 | GPU or CPU | Japanese-only, fine-tuned on Japanese media content |

All backends accept `(np.ndarray, sample_rate)` tuples, so the VAD stage feeds them identically. Switch via config - no pipeline changes needed. Qwen3-ASR streaming requires the vLLM backend (`qwen-asr[vllm]`).

## Platform support

| Platform | Status |
|----------|--------|
| Windows | Native (WASAPI loopback) |
| Linux | PulseAudio required |
| macOS | Requires [BlackHole](https://github.com/ExistentialAudio/BlackHole) or similar virtual audio device |
