# SubsVibe

Real-time subtitles from system audio using local speech-to-text.

Captures your system's audio output (any app, any language), runs it through voice activity detection and speech recognition locally, and displays live subtitles. An LLM pass refines the output with context-aware correction and translation.

## How it works

```
System Audio -> Voice Detection -> Speech-to-Text -> LLM Refinement -> Subtitles
```

All processing runs locally. No audio leaves your machine. The LLM stage works with local models (Ollama, LM Studio, vLLM) or cloud endpoints - your choice.

## Status

**This project is currently in the planning stage.** See [docs/plan.md](docs/plan.md) for the full implementation plan.

## Pipeline stages

| Stage | What it does |
|-------|-------------|
| **Capture** | Records system audio via loopback (SoundCard) |
| **VAD** | Filters silence/noise, emits only speech segments (Silero VAD) |
| **Transcribe** | Converts speech to text (Faster Whisper) |
| **LLM** | Corrects errors, adds context, translates (any OpenAI-compatible API) |

Each stage runs in its own thread, connected by queues.

See [docs/plan.md](docs/plan.md) for detailed design and phase breakdown.

## Platform support

| Platform | Status |
|----------|--------|
| Windows | Native (WASAPI loopback) |
| Linux | PulseAudio required |
| macOS | Requires [BlackHole](https://github.com/ExistentialAudio/BlackHole) or similar virtual audio device |
