# SubsVibe - Real-Time System Audio Subtitles

Capture system audio and produce live subtitles via local speech-to-text.

## Landscape

See [comparison.md](comparison.md) for a detailed comparison with existing open-source projects (Buzz, WhisperLive, LiveCaptions, Vibe, SubsAI, RealtimeSTT, whisper_streaming, whisper.cpp, etc.) and OS built-in solutions (Windows Live Captions, Google Live Caption). SubsVibe is the only open-source project combining native system audio loopback, neural VAD, Whisper transcription, and LLM-based sliding context refinement in a single pipeline.

## Phases

1. **Base - Audio Capture** *(current)* - SoundCard loopback -> PCM stream
2. **VAD - Voice Activity Detection** - Silero VAD filters speech from silence
3. **Transcription** - Faster Whisper transcribes speech segments
4. **LLM Post-Processing** - Context-aware subtitle refinement and translation

## Why SoundCard

Single API for loopback recording across Windows (WASAPI) and Linux (PulseAudio). Loopback devices are discovered via `sc.all_microphones(include_loopback=True)`. macOS lacks native loopback - requires a virtual audio device like BlackHole.

## Architecture

```
SoundCard loopback -> PCM chunks -> Silero VAD -> speech segments -> Faster Whisper -> raw text -> LLM -> subtitles
```

Each stage is decoupled via queues, running in its own thread.

### PCM format (fixed for all consumers)

- **Sample rate**: 16000 Hz (standard for speech transcription)
- **Channels**: 1 (mono)
- **Bit depth**: 16-bit signed integer (int16 little-endian)
- **Chunk size**: ~512 frames (32ms per chunk at 16kHz)

## Project Structure

```
subsvibe/
  src/
    capture.py       # Phase 1: Audio capture with callback-based PCM streaming
    vad.py           # Phase 2: Silero VAD speech filtering
    transcribe.py    # Phase 3: Faster Whisper transcription worker
    llm.py           # Phase 4: LLM context-aware subtitle refinement
    pipeline.py      # Wires all stages together
  pyproject.toml
```

## Setup

```
uv sync
```

## Dependencies (`pyproject.toml`)

```
soundcard>=0.4.5
numpy>=2.2.3
```

## How It Works

1. Find loopback mic for default speaker via `sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)`
2. Open recorder at 16kHz - resamples manually via numpy if device doesn't support native resampling
3. Convert float32 -> mono int16, emit each chunk to registered callback(s)
4. Stop on Ctrl+C or after duration

## CLI (for testing)

```
uv run python capture.py                        # Capture -> output.pcm, Ctrl+C to stop
uv run python capture.py --seconds 10           # Capture 10 seconds
uv run python capture.py --output test.pcm      # Custom output path
uv run python capture.py --list                 # List loopback devices
ffplay -f s16le -ar 16000 -ac 1 output.pcm  # Playback test
```

## Troubleshooting

- **No loopback devices**: Ensure audio output is active
- **macOS**: Install [BlackHole](https://github.com/ExistentialAudio/BlackHole) - no native loopback
- **Linux**: PulseAudio must be running (`pulseaudio --check`)

---

## Phase 2 - Silero VAD

Filter PCM stream so only speech segments reach the transcriber. Silero VAD is a small PyTorch model (~2 MB) that runs on CPU in real time.

### Why VAD before Whisper

- Whisper hallucinates on silence/noise - VAD eliminates that
- Sending only speech segments cuts GPU/CPU work dramatically
- Gives clean segment boundaries (start/end timestamps) for subtitle timing

`VADIterator` expects 512-sample chunks at 16 kHz - matches our capture format exactly. Each chunk is converted from int16 numpy to float32 torch tensor. Returns `{start: float}` or `{end: float}` or `None` per chunk.

### Integration

- New file: `vad.py` - wraps VADIterator, accumulates speech chunks between start/end events, pushes complete segments to a `queue.Queue`
- `capture.py` registers `vad.on_chunk` as a callback

### Dependencies (added)

```
silero-vad
torch
```

---

## Phase 3 - Faster Whisper Transcription

Transcribe speech segments from the VAD queue using Faster Whisper (CTranslate2 backend). ~4x faster than OpenAI Whisper, lower memory, supports int8/float16 quantization.

`WhisperModel("base", device="cpu", compute_type="int8")` - accepts numpy float32 arrays (16kHz mono). Call `model.transcribe(audio, vad_filter=False)` since we already ran Silero VAD. Returns segments with text, timestamps, and optional word-level timing.

Default to `base` for real-time on CPU. Upgrade to `small`/`medium` if GPU available.

### Integration

- New file: `transcribe.py` - worker thread reads speech segments from VAD queue, runs `model.transcribe()`, emits subtitle events
- Runs in its own thread to not block audio capture

### Dependencies (added)

```
faster-whisper
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

### Dependencies (added)

```
openai
```
