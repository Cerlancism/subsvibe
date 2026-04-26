# SubsVibe - Real-Time System Audio Subtitles

Capture system audio and produce live subtitles via local speech-to-text.

## Landscape

See [comparison.md](comparison.md) for a detailed comparison with existing open-source projects (Buzz, WhisperLive, LiveCaptions, Vibe, SubsAI, RealtimeSTT, whisper_streaming, whisper.cpp, etc.) and OS built-in solutions (Windows Live Captions, Google Live Caption). SubsVibe is the only open-source project combining native system audio loopback, neural VAD, pluggable transcription via an OpenAI-compatible Whisper server, and LLM-based sliding context refinement in a single pipeline.

## Phases

1. **Base - Audio Capture** *(current)* - SoundCard loopback -> PCM stream
2. **VAD - Voice Activity Detection** - Silero VAD filters speech from silence
3. **Transcription** - Send speech segments to an OpenAI Whisper-compatible server
4. **LLM Post-Processing** - Context-aware subtitle refinement and translation

## Client-Server Split

SubsVibe ships two components:

- **Client** (`client/`): audio capture, VAD, and pipeline. VAD runs locally — only completed speech segments are sent, not raw audio. Calls the transcription server and LLM server via HTTP using the `openai` SDK.
- **Transcription server** (`server/`, in scope): FastAPI server implementing `POST /v1/audio/transcriptions`. Pluggable model backend (Faster Whisper or Qwen3-ASR); decodes audio via PyAV. The client is agnostic to which backend is running.
- **LLM server** (out of scope): any OpenAI-compatible chat server — Ollama, vLLM, LM Studio, OpenAI API, etc. Configured via `llm.base_url` + model name.

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
    capture.py       # Audio capture with callback-based PCM streaming
    vad.py           # Silero VAD speech filtering
    transcribe.py    # Whisper API client, transcription worker
    llm.py           # LLM API client, context-aware subtitle refinement
    pipeline.py      # Wires all stages together
  server/
    server.py        # FastAPI transcription server (OpenAI Whisper-compatible)
    model.py         # Model backend abstraction (Faster Whisper / Qwen3-ASR)
  requirements/
    client.in        # abstract client deps
    client.txt       # locked client deps (pip-compile output)
    server.in        # abstract server deps
    server.txt       # locked server deps (pip-compile output)
```

## Setup

Dependencies are managed with `pip-tools`. Abstract deps live in `requirements/*.in`; locked versions are committed in `requirements/*.txt`.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install pip-tools

# install locked deps
pip-sync requirements/client.txt

# to update locks after editing a .in file
pip-compile requirements/client.in -o requirements/client.txt
```

## Dependencies

**Client** (`requirements/client.in`)

```
soundcard>=0.4.5
numpy>=2.2.3
```

The full client + server stack (silero-vad ONNX, faster-whisper, openai SDK) is PyTorch-free and supports Python 3.14. PyTorch is only needed if using the Qwen3-ASR server backend.

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

Filter PCM stream so only speech segments reach the transcriber. Silero VAD is a small (~2 MB) model that runs on CPU in real time via ONNX Runtime — no PyTorch required.

### Why VAD before Whisper

- Whisper hallucinates on silence/noise - VAD eliminates that
- Sending only speech segments cuts GPU/CPU work dramatically
- Gives clean segment boundaries (start/end timestamps) for subtitle timing

`VADIterator` expects 512-sample chunks at 16 kHz - matches our capture format exactly. Each chunk is converted from int16 numpy to float32. Returns `{start: float}` or `{end: float}` or `None` per chunk.

### Integration

- New file: `vad.py` - wraps VADIterator, accumulates speech chunks between start/end events, pushes complete segments to a `queue.Queue`
- `capture.py` registers `vad.on_chunk` as a callback

### Dependencies (added to `requirements/client.in`)

```
silero-vad[onnx-cpu]
```

Uses the ONNX CPU backend (`onnxruntime`) — no PyTorch dependency. Works on Python 3.14.

---

## Phase 3 - Transcription

### Client (`client/transcribe.py`)

Worker thread reads completed speech segments from the VAD queue, encodes each as a WAV buffer, and submits it to `POST /v1/audio/transcriptions`. Pushes returned text to the LLM queue. Configured via `transcription.base_url` and `transcription.model`.

### Server (`server/`)

A FastAPI server exposing an OpenAI Whisper-compatible API. The server is the only component that loads model weights.

**Endpoints**

- `GET /v1/models` — list available models
- `POST /v1/audio/transcriptions` — transcribe an uploaded audio file; returns JSON or `verbose_json` (with segments and optional word timestamps)

**Request parameters** (matching OpenAI Whisper API)

- `file` — audio file (any format; decoded server-side to mono 16kHz PCM via PyAV)
- `model` — model identifier
- `language` — optional ISO-639-1 language code; omit for auto-detection
- `prompt` — optional hint text to guide transcription style or vocabulary
- `response_format` — `json` (default) or `verbose_json`
- `timestamp_granularities` — `segment` (default) or `word` (where supported)

**Model backends** (selected via server config)

- **Faster Whisper** — CTranslate2-based, CPU-friendly, int8 quantization. Suitable for machines without a GPU.
- **Qwen3-ASR** — LLM-based ASR, GPU required (bfloat16). 52 languages with auto language detection; word-level timestamps via companion aligner model.

Audio is decoded on the server using PyAV to mono 16kHz PCM regardless of the input format, so the client can send standard WAV without pre-processing.

### Dependencies (added to `requirements/client.in` and `requirements/server.in`)

```
# client.in
openai

# server.in
fastapi
faster-whisper  # uses ctranslate2 + onnxruntime, no PyTorch
qwen-asr        # optional backend (GPU, PyTorch required)
```

---

## Phase 4 - LLM Post-Processing

Use an LLM to refine raw Whisper output into context-aware subtitles. Uses the OpenAI Python SDK (`openai` package) - works with OpenAI API, local servers (Ollama, vLLM, LM Studio), or any OpenAI-compatible endpoint via `base_url`.

### Why

- Whisper outputs segments in isolation - no cross-segment coherence
- Proper nouns, technical terms, acronyms get mangled without context
- Translation quality improves with surrounding context

### Sliding context window

Each new Whisper segment is sent to the LLM alongside a sliding window of recent subtitle history. The LLM can correct the new segment and revise recent lines if new context clarifies them (e.g. fix a mishearing, complete a split sentence, improve translation).

Subtitles are **provisional** until enough context confirms them - mimics how live captioners work.

### What the LLM handles

- **Correction**: fix Whisper errors using context (homophones, proper nouns, acronyms)
- **Translation**: full-sentence context, not word-by-word
- **Continuity**: handle sentences spanning multiple Whisper segments

### Integration

- New file: `llm.py` - sliding context window, prompt formatting, response parsing
- Consumes from transcription queue, produces final subtitle events
- Configurable: `base_url` / model name, target language, context window size
- Falls back to raw Whisper output if LLM is unavailable or too slow

### Dependencies (added to `requirements/client.in`)

```
openai
```
