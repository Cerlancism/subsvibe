# SubsVibe - Real-Time System Audio Subtitles

Capture system audio and produce live subtitles via local speech-to-text.

## Landscape

See [comparison.md](comparison.md) for a detailed comparison with existing open-source projects (Buzz, WhisperLive, LiveCaptions, Vibe, SubsAI, RealtimeSTT, whisper_streaming, whisper.cpp, etc.) and OS built-in solutions (Windows Live Captions, Google Live Caption). SubsVibe is the only open-source project combining native system audio loopback, neural VAD, pluggable transcription via an OpenAI-compatible Whisper server, and LLM-based sliding context refinement in a single pipeline.

## Phases

1. **Base - Audio Capture** *(done)* - SoundCard loopback -> PCM stream
2. **VAD - Voice Activity Detection** *(done)* - Silero VAD filters speech from silence (online for live, batch for file)
3. **Transcription** *(done)* - Send speech segments to an OpenAI Whisper-compatible server
4. **LLM Post-Processing** *(done; tuning ongoing)* - Context-aware subtitle refinement and translation
5. **Subtitle output** *(done; tuning ongoing)* - SRT line wrapping, timing, and live display

All pipeline stages run end-to-end on Windows. Active work is on segment merging/sub-slicing, subtitle wrap heuristics, and prompt quality - not on adding new stages.

## Live pipeline: commit-on-silence

Live mode does **not** use a fixed sliding window. Silero VAD runs in
streaming mode and the pipeline is driven by speech boundaries.

```
PCM (32 ms chunks)
  -> LiveVAD (silero VADIterator, online)
       emits SegmentEvent(pcm, start, end, final)
         - provisional: every ~1 s while a segment is open (preview only)
         - final: on confirmed silence, or force-flushed after MAX_SEG_SECONDS
  -> ASR worker (single thread)
       transcribes each event once
  -> Translate worker (optional, separate thread)
       translates each event; only finals append to history
  -> Renderer
       finals scroll up as committed lines (immutable)
       provisionals overwrite a single line in place via \r
```

Key properties:

- **No redundant ASR.** Each speech segment is transcribed once when it ends; the in-progress segment is re-transcribed only at the provisional cadence.
- **Stable history.** Committed lines never change. Translation context fed to the LLM contains finals only, so it cannot drift on mid-sentence noise.
- **Natural segment boundaries.** Whisper sees complete prosodic units.
- **Bounded latency.** End-of-speech latency = `MIN_SILENCE_MS` + ASR + (translate). A segment longer than `MAX_SEG_SECONDS` is force-finalised so monologues still produce output.
- **Stale-job drop.** Each stage drops a queued item older than `LIVE_LAG_TOLERANCE_SECONDS`, draining forward to the freshest item. Finals are sticky — they are never dropped in favour of a newer provisional.

## Client-Server Split

SubsVibe ships two components:

- **Client** (`client/`): audio capture, VAD, and pipeline. VAD runs locally - only completed speech segments are sent, not raw audio. Calls the transcription server and LLM server via HTTP using the `openai` SDK.
- **Transcription server** (`server/`, in scope): FastAPI server implementing `POST /v1/audio/transcriptions`. Pluggable model backend (Faster Whisper, Qwen3-ASR, or Anime Whisper); decodes audio via PyAV. The client is agnostic to which backend is running.
- **LLM server** (out of scope): any OpenAI-compatible chat server - Ollama, vLLM, LM Studio, OpenAI API, etc. Configured via `llm.base_url` + model name.

The client has no dependency on model-specific packages (`faster-whisper`, `qwen-asr`, `torch`, etc.).

## Why SoundCard

Single API for loopback recording across Windows (WASAPI) and Linux (PulseAudio). Loopback devices are discovered via `sc.all_microphones(include_loopback=True)`. macOS lacks native loopback - requires a virtual audio device like BlackHole.

## Architecture

```
[client]                                                [server]          [server]
SoundCard loopback -> PCM chunks -> Silero VAD -> speech segments -> Whisper API -> raw text -> LLM API -> subtitles
```

VAD runs on the client; only completed speech segments cross the network. Each client-side stage is decoupled via queues, running in its own thread.

### PCM format (fixed for all consumers)

- **Sample rate**: 16000 Hz (standard for speech transcription)
- **Channels**: 1 (mono)
- **Bit depth**: 16-bit signed integer (int16 little-endian)
- **Chunk size**: ~512 frames (32ms per chunk at 16kHz)

## Project Structure

```
subsvibe/
  client/
    capture.py       # SoundCard loopback, PCM chunking, shared live constants
    vad.py           # Silero VAD (batch, file mode): audio -> speech segments
    live_vad.py      # Silero VAD (online, live mode): chunks -> segment events
    transcribe.py    # Speech segment -> Whisper-compatible API call
    llm.py           # Committed-history translator via OpenAI-compatible API
    render.py        # Terminal renderer: scrolling commits + in-place provisional
    subtitle.py      # SRT line wrapping, timing, CPS heuristics
    pipeline.py      # Wires capture -> live_vad -> transcribe -> LLM -> render
    client.py        # CLI entry point
  server/
    server.py        # FastAPI transcription server (OpenAI Whisper-compatible)
    model.py         # Model backend abstraction
    backends/        # Pluggable backends (Qwen3-ASR; Faster Whisper planned)
    download_models.py
  requirements.in    # abstract deps (client + server combined)
  requirements.txt   # locked deps (pip-compile output)
```

## Setup

Dependencies are managed with `pip-tools`. Abstract deps live in `requirements.in`; the locked file is committed as `requirements.txt`.

One-shot setup via `scripts/setup.sh` (creates venv, installs PyTorch, syncs locked deps, downloads models):

Run from any POSIX shell (bash on Linux/macOS, Git Bash on Windows):

```
cp scripts/env.example.sh scripts/env.sh    # first time only - edit PYTORCH_INSTALL_CMD for your platform
scripts/setup.sh
```

`PYTORCH_INSTALL_CMD` in `scripts/env.sh` selects the PyTorch wheel index. Pick the right command from https://pytorch.org/get-started for your OS and compute platform (CUDA / ROCm / CPU / MPS). PyTorch is installed before `pip-sync` so the platform-specific wheel's local version tag (e.g. `2.11.0+cu130`, `+cpu`) satisfies the lockfile's plain torch pin (`2.11.0`) and isn't replaced.

To update locks after editing `requirements.in`:

```
pip-compile requirements.in -o requirements.txt
```

## Dependencies

All deps are in a single `requirements.in`:

```
soundcard>=0.4.5
numpy>=2.2.3
silero-vad[onnx-cpu]
openai
fastapi
uvicorn[standard]
av
qwen-asr
huggingface_hub[hf_xet]
# torch: install manually per pytorch.org/get-started (CUDA version-specific)
```

PyTorch is needed for the server backends (Faster Whisper and Qwen3-ASR) and must be installed separately before `pip-sync`.

## How It Works

1. Find loopback mic for default speaker via `sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)`
2. Open recorder at 16kHz - resamples manually via numpy if device doesn't support native resampling
3. Convert float32 -> mono int16, emit each chunk to registered callback(s)
4. Stop on Ctrl+C or after duration

## CLI (for testing)

```
python client/capture.py                        # Capture -> output.pcm, Ctrl+C to stop
python client/capture.py --seconds 10           # Capture 10 seconds
python client/capture.py --output test.pcm      # Custom output path
python client/capture.py --list                 # List loopback devices
ffplay -f s16le -ar 16000 -ac 1 output.pcm      # Playback test
```

## Troubleshooting

- **No loopback devices**: Ensure audio output is active
- **macOS**: Install [BlackHole](https://github.com/ExistentialAudio/BlackHole) - no native loopback
- **Linux**: PulseAudio must be running (`pulseaudio --check`)

---

## Phase 2 - Silero VAD

Filter PCM stream so only speech segments reach the transcriber. Silero VAD is a small (~2 MB) model that runs on CPU in real time via ONNX Runtime - no PyTorch required.

### Why VAD before Whisper

- Whisper hallucinates on silence/noise - VAD eliminates that
- Sending only speech segments cuts GPU/CPU work dramatically
- Gives clean segment boundaries (start/end timestamps) for subtitle timing

`VADIterator` expects 512-sample chunks at 16 kHz - matches our capture format exactly. Each chunk is converted from int16 numpy to float32. Returns `{start: float}` or `{end: float}` or `None` per chunk.

### Integration

- New file: `vad.py` - wraps VADIterator, accumulates speech chunks between start/end events, pushes complete segments to a `queue.Queue`
- `capture.py` registers `vad.on_chunk` as a callback

### Dependencies (added to `requirements.in`)

```
silero-vad[onnx-cpu]
```

Uses the ONNX CPU backend (`onnxruntime`) - no PyTorch dependency. Works on Python 3.14.

---

## Phase 3 - Transcription

### Client (`client/transcribe.py`)

Worker thread reads completed speech segments from the VAD queue, encodes each as a WAV buffer, and submits it to `POST /v1/audio/transcriptions`. Pushes returned text to the LLM queue. Configured via `transcription.base_url` and `transcription.model`.

### Server (`server/`)

A FastAPI server exposing an OpenAI Whisper-compatible API. The server is the only component that loads model weights.

**Endpoints**

- `GET /v1/models` - list available models
- `POST /v1/audio/transcriptions` - transcribe an uploaded audio file; returns JSON or `verbose_json` (with segments and optional word timestamps)

**Request parameters** (matching OpenAI Whisper API)

- `file` - audio file (any format; decoded server-side to mono 16kHz PCM via PyAV)
- `model` - model identifier
- `language` - optional ISO-639-1 language code; omit for auto-detection
- `prompt` - optional hint text to guide transcription style or vocabulary
- `response_format` - `json` (default) or `verbose_json`
- `timestamp_granularities` - `segment` (default) or `word` (where supported)

**Model backends** (selected via server config)

- **Faster Whisper** - CTranslate2-based, CPU-friendly, int8 quantization. Suitable for machines without a GPU.
- **Qwen3-ASR** - LLM-based ASR, GPU required (bfloat16). 52 languages with auto language detection; word-level timestamps via companion aligner model.

Audio is decoded on the server using PyAV to mono 16kHz PCM regardless of the input format, so the client can send standard WAV without pre-processing.

### Dependencies

All deps in a single `requirements.in`, one shared `.venv`.

```
soundcard>=0.4.5
numpy>=2.2.3
silero-vad[onnx-cpu]
openai
fastapi
uvicorn[standard]
av
torch
qwen-asr
```

#### Python and PyTorch

- **Python 3.14** supported - PyTorch 2.10+ includes full Python 3.14 support.
- PyTorch must be installed separately before `pip-sync`. Go to pytorch.org/get-started, select your OS and CUDA version, and run the generated command. The prebuilt wheel bundles the CUDA runtime - no separate CUDA Toolkit install needed.

---

## Phase 4 - LLM Post-Processing

Use an LLM to refine raw Whisper output and translate. Uses the OpenAI Python SDK (`openai` package) - works with OpenAI API, local servers (Ollama, vLLM, LM Studio), or any OpenAI-compatible endpoint via `base_url`.

### Why

- Whisper outputs segments in isolation - no cross-segment coherence
- Proper nouns, technical terms, acronyms get mangled without context
- Translation quality improves with surrounding context

### Committed history, not sliding window

The live pipeline calls the LLM **once per VAD segment**. Translation context is a short history of **committed** (i.e. final) utterances — not an overlapping audio window. Provisional outputs are previews shown to the user but never enter the LLM's history.

This avoids the failure mode where overlapping windows feed the same audio to the LLM multiple times and corrupt context with mid-sentence fragments.

### What the LLM handles

- **Translation**: full-utterance context driven by committed prior lines
- **Correction**: (file mode only, via `--history` / `--context-src`) optional prompt scaffolding when re-transcribing reference SRTs

### Integration

- `client/llm.py` - prompt formatting and structured-output parsing
- Consumes finalised segments from the ASR worker; emits to the renderer
- Configurable via env: `LLM_BASE_URL`, `LLM_MODEL_ID`, `LLM_API_KEY`
- Per-call timeout = `LIVE_LAG_TOLERANCE_SECONDS`

### Dependencies (added to `requirements.in`)

```
openai
```
