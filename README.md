# SubsVibe

Real-time subtitles from system audio using local speech-to-text.

Captures your system's audio output (any app, any language), runs it through voice activity detection and speech recognition locally, and displays live subtitles. An LLM pass refines the output with context-aware correction and translation.

## Status

**This project is currently in the planning stage.** See [docs/plan.md](docs/plan.md) for the full implementation plan.

## How it works

```
System Audio -> Voice Detection -> Speech-to-Text -> LLM Refinement -> Subtitles
```

All processing runs locally. No audio leaves your machine. The LLM stage works with local models (Ollama, LM Studio, vLLM) or cloud endpoints - your choice.


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

Both backends accept `(np.ndarray, sample_rate)` tuples, so the VAD stage feeds either one identically. Switch via config — no pipeline changes needed. Qwen3-ASR streaming requires the vLLM backend (`qwen-asr[vllm]`).

## Platform support

| Platform | Status |
|----------|--------|
| Windows | Native (WASAPI loopback) |
| Linux | PulseAudio required |
| macOS | Requires [BlackHole](https://github.com/ExistentialAudio/BlackHole) or similar virtual audio device |
