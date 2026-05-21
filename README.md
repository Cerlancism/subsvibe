# SubsVibe

Real-time subtitles from system audio using local speech-to-text.

Captures your system's audio output (any app, any language), runs it through voice activity detection and speech recognition locally, and displays live subtitles. An LLM pass refines the output with context-aware correction and translation, and a subtitle stage emits timed SRT lines with line-wrap, CPS, and reading-time heuristics.

## Status

**Working end-to-end on Windows.** All five pipeline stages - capture, VAD, transcription, LLM refinement, and subtitle generation - are implemented and connected, producing live SRT output. The transcription server runs FastAPI with a Qwen3-ASR backend (Faster Whisper backend planned). Tuning of segment timing, subtitle wrapping, and sliding-context refinement is ongoing. See [docs/plan.md](docs/plan.md) for the full design and what's still planned.

## How it works

```
System Audio -> Voice Detection -> Speech-to-Text -> LLM Refinement -> Subtitles
```

All processing runs locally. No audio leaves your machine. The LLM stage works with local models (Ollama, LM Studio, vLLM) or cloud endpoints - your choice.


## Setup

Requires Python 3.14. The Qwen3-ASR backend runs best on a GPU but will fall back to CPU.

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
| **Transcribe** | Converts speech to text (Faster Whisper or Qwen3-ASR) |
| **LLM** | Corrects errors, adds context, translates (any OpenAI-compatible API) |

Each stage runs in its own thread, connected by queues.

See [docs/plan.md](docs/plan.md) for detailed design and phase breakdown.

## Transcription backends

| Backend | Model size | Device | Strength |
|---------|-----------|--------|----------|
| **Faster Whisper** | base / small / medium | CPU (int8) or GPU | Fast, low memory, proven quality, ~100 languages |
| **Qwen3-ASR-1.7B** | 1.7B params | GPU (bfloat16) | 52 languages (incl. 22 Chinese dialects), auto language detection, SOTA accuracy |
| **Qwen3-ASR-0.6B** | 0.6B params | GPU (bfloat16) | Lighter weight; ~2000× throughput at high concurrency on the vLLM backend |

Both backends accept `(np.ndarray, sample_rate)` tuples, so the VAD stage feeds either one identically. Switch via config - no pipeline changes needed. Qwen3-ASR streaming requires the vLLM backend (`qwen-asr[vllm]`).

## Platform support

| Platform | Status |
|----------|--------|
| Windows | Native (WASAPI loopback) |
| Linux | PulseAudio required |
| macOS | Requires [BlackHole](https://github.com/ExistentialAudio/BlackHole) or similar virtual audio device |
