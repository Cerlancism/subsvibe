# SubsVibe

Real-time subtitles from system audio using local speech-to-text.

Captures your system's audio output (any app, any language), runs it through voice activity detection and speech recognition locally, and displays live subtitles. An LLM pass refines the output with context-aware correction and translation, and a subtitle stage emits timed SRT lines with line-wrap, CPS, and reading-time heuristics.

## Demo
**(Turn up volume)**  

https://github.com/user-attachments/assets/b37d3e33-bccf-4005-acb2-2b9f02da6267

## Status

**Working end-to-end on Windows.** All five pipeline stages - capture, VAD, transcription, LLM refinement, and subtitle generation - are implemented and connected, producing live SRT output. A batch mode (`--input <audio>`) also transcribes any audio file directly to an `.srt` alongside it. The transcription server runs FastAPI with a Faster Whisper backend (Qwen3-ASR and Anime Whisper also supported). Live mode uses a commit-on-silence VAD pipeline: each utterance is transcribed once when it ends, with mid-utterance previews shown in place. Tuning of segment timing, subtitle wrapping, and translation-prompt quality is ongoing. See [docs/plan.md](docs/plan.md) for the full design and what's still planned.

## How it works

```
System Audio -> Voice Detection -> Speech-to-Text -> LLM Refinement -> Subtitles
```

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

| Stage | What it does |
|-------|-------------|
| **Capture** | Records system audio via loopback (SoundCard) |
| **VAD** | Filters silence/noise, emits only speech segments (Silero VAD) |
| **Transcribe** | Converts speech to text (Faster Whisper, Qwen3-ASR, or Anime Whisper) |
| **LLM** | Corrects errors, adds context, translates (any OpenAI-compatible API) |

Each stage runs in its own thread, connected by queues.

See [docs/plan.md](docs/plan.md) for detailed design and phase breakdown.

## Transcription backends

| Backend | Model size | Device | Strength |
|---------|-----------|--------|----------|
| **Faster Whisper** | tiny / base / small / medium / large-v3 | GPU or CPU (int8) | Fast, low memory, proven quality, ~100 languages |
| **Qwen3-ASR-1.7B** | 1.7B params | GPU (bfloat16) | 52 languages (incl. 22 Chinese dialects), auto language detection, SOTA accuracy |
| **Qwen3-ASR-0.6B** | 0.6B params | GPU (bfloat16) | Lighter weight; ~2000× throughput at high concurrency on the vLLM backend |
| **Anime Whisper** | based on Whisper-large-v2 | GPU or CPU | Japanese-only, fine-tuned on anime/galgame speech |

All backends accept `(np.ndarray, sample_rate)` tuples, so the VAD stage feeds them identically. Switch via config - no pipeline changes needed. Qwen3-ASR streaming requires the vLLM backend (`qwen-asr[vllm]`).

## Platform support

| Platform | Status |
|----------|--------|
| Windows | Native (WASAPI loopback) |
| Linux | PulseAudio required |
| macOS | Requires [BlackHole](https://github.com/ExistentialAudio/BlackHole) or similar virtual audio device |
