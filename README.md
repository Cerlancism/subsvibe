# SubsVibe

SubsVibe generates live subtitles from system audio and `.srt` subtitles from audio or video files.

Live mode captures audio playing on your machine and displays subtitles as speech is transcribed. An LLM can revise or translate the text using previous utterances as context. Audio capture and voice activity detection run locally; transcription and LLM processing can use local or remote services.

File mode transcribes an audio or video file and saves timed subtitles to an `.srt` alongside it. It does not include LLM refinement or translation. For those steps, the resulting file can be used with [chatgpt-subtitle-translator](https://github.com/Cerlancism/chatgpt-subtitle-translator).

## Demo
**(Turn up volume)**  

https://github.com/user-attachments/assets/b37d3e33-bccf-4005-acb2-2b9f02da6267

## Features

- **Live subtitles and file transcription.** Live mode (`--live`) captures system audio and displays subtitles as you watch. File mode (`--input <file>`) transcribes an audio or video file and saves an `.srt` alongside it.
- **Live previews.** Subtitles update in place while speech is ongoing and are finalized when a pause is detected or the segment reaches its length limit.
- **LLM refinement and translation.** In live mode, an LLM can revise transcriptions and translate them using previous finalized utterances as context.
- **Local or remote processing.** Audio capture and voice activity detection run locally. Transcription and LLM processing can each use a local server, another machine on your network, or a hosted API.
- **Configurable models and endpoints.** The bundled transcription server supports Faster Whisper (the default), Qwen3-ASR, and Anime Whisper. Set `TRANSCRIPT_BASE_URL` to the SubsVibe server's local or remote address. The LLM stage supports OpenAI-compatible endpoints, including Ollama, LM Studio, and vLLM.

## How it works

**Live mode** (`--live`) detects speech in system audio, transcribes it, and displays subtitles. An LLM can revise or translate the transcription:

```
System Audio -> Voice Detection -> Speech-to-Text -> LLM Refinement -> Subtitles
```

**File mode** (`--input <file>`) divides audio into segments for transcription and saves timed subtitles to an `.srt`:

```
Audio/Video File -> Voice Detection -> Speech-to-Text -> Subtitles (.srt)
```

File mode does not include LLM refinement or translation. The `--history` and `--context-src` options can provide previous transcriptions or a reference `.srt` as context for speech recognition. The saved subtitles can be translated or edited separately with [chatgpt-subtitle-translator](https://github.com/Cerlancism/chatgpt-subtitle-translator).

Audio capture and voice activity detection run on your machine. Transcription sends audio to the configured speech-to-text endpoint; the LLM stage sends text to its configured endpoint. Both services can run locally or remotely.


## Setup

Requires Python 3.14. Faster Whisper runs on GPU or CPU (int8); the Qwen3-ASR backend requires a GPU.

Run the scripts in `scripts/` from any POSIX shell - bash on Linux/macOS, or Git Bash on Windows.

```bash
cp scripts/env.example.sh scripts/env.sh    # first time only
# Edit scripts/env.sh and set PYTORCH_INSTALL_CMD for your platform.
# Get the right command from https://pytorch.org/get-started - pick your OS,
# package (Pip), and compute platform (CUDA 12.x / ROCm / CPU / etc.).
scripts/setup.sh                            # creates .venv, installs PyTorch + locked deps, downloads models
scripts/server.sh                           # start the transcription server
scripts/client.sh --live --translate        # live: capture loopback audio and produce live subtitles
scripts/client.sh --input video.mp4         # file: transcribe a file to video.srt alongside it
```

The setup script installs PyTorch first (from the wheel index in `PYTORCH_INSTALL_CMD`), then `pip-sync` against `requirements.txt`. The platform-specific build's local version tag (e.g. `+cu130`, `+rocm6.2`, `+cpu`) satisfies the lockfile's plain torch pin, so your chosen wheel is preserved. To switch platforms, change `PYTORCH_INSTALL_CMD` in `scripts/env.sh` and re-run setup.

Files SubsVibe writes (`.srt` subtitles, `--log-file` logs) and its console output use the platform line ending by default: CRLF on Windows, LF elsewhere. Set `SUBSVIBE_NEWLINE` in `scripts/env.sh` to `lf` or `crlf` to pin it, or `auto` for the default.

### Google Colab (remote GPU server)

To run the transcription server on a Colab GPU, see the [Colab setup helper](scripts/setup.colab.sh). The client runs on your local machine.

## Pipeline stages

| Stage | What it does | Live | File |
|-------|-------------|:----:|:----:|
| **Input** | Live: records system audio via loopback (SoundCard). File: decodes an audio/video file to PCM (PyAV). | ✓ | ✓ |
| **VAD** | Silero VAD (with a WebRTC VAD second-chance pass) marks speech. Live drops silence/noise; file mode keeps the whole timeline and only uses VAD to choose segment boundaries. | ✓ | ✓ |
| **Transcribe** | Converts speech to text (Faster Whisper, Qwen3-ASR, or Anime Whisper). | ✓ | ✓ |
| **LLM** | Corrects errors, adds context, translates (any OpenAI-compatible API). | ✓ | - |
| **Subtitle** | Times and wraps lines into SRT (line-wrap, CPS, reading-time heuristics). | ✓ | ✓ |

In live mode each stage runs in its own thread, connected by queues. File mode runs the same stages sequentially over a file (minus the LLM pass).

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
