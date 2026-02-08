# Landscape Comparison: Open-Source Live Captioning & Transcription

How SubsVibe compares to existing open-source projects and OS built-in solutions.

---

## Quick Comparison Matrix

| Project | System Audio | Real-Time | VAD | LLM Refinement | Cross-Platform | Stars |
|---|---|---|---|---|---|---|
| **SubsVibe** | Yes (SoundCard loopback) | Yes (streaming pipeline) | Silero VAD | Yes (sliding context window) | Win/Linux/macOS* | -- |
| [Buzz](https://github.com/chidiwilliams/buzz) | No | Mic only | No dedicated | No | Win/Linux/macOS | ~17.7k |
| [WhisperLive](https://github.com/collabora/WhisperLive) | Browser tabs only | Yes (WebSocket) | Silero VAD | No | Linux (server) | ~3.8k |
| [LiveCaptions](https://github.com/abb128/LiveCaptions) | Yes (PulseAudio monitor) | Yes | Amplitude-based | No | Linux only | ~1.7k |
| [Vibe](https://github.com/thewh1teagle/vibe) | macOS only | Record-then-transcribe | No | Post-transcription only | Win/Linux/macOS | ~5.2k |
| [SubsAI](https://github.com/absadiki/subsai) | No | No (file-based) | Optional (via faster-whisper) | No | Win/Linux/macOS | ~1.6k |
| [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) | No (custom feed possible) | Yes (library) | Dual: WebRTC + Silero | No | Win/Linux/macOS | ~9.4k |
| [whisper_streaming](https://github.com/ufal/whisper_streaming) | No | Yes (~3.3s latency) | Silero VAD (optional) | No | Linux/macOS | ~3.5k |
| [whisper.cpp stream](https://github.com/ggerganov/whisper.cpp) | No | Yes (mic only) | Basic amplitude | No | All major | ~46.5k |
| [speech-to-text](https://github.com/reriiasu/speech-to-text) | No | Yes (mic via WebSocket) | Silero VAD | Yes (OpenAI API proofreading) | Win/Linux/macOS | ~612 |
| [LocalVocal](https://github.com/occ-ai/obs-localvocal) | No (OBS audio filter) | Yes | Silero VAD (ONNX) | No (LLM for translation only, not correction) | Win/Linux/macOS | ~1.4k |
| [FunASR](https://github.com/modelscope/FunASR) | No (server/toolkit) | Yes (streaming + 2pass) | FSMN-VAD (neural) | No (2pass self-correction) | Linux primarily | ~13.9k |
| [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) | No (mic via browser) | Yes (AlignAtt streaming) | Silero VAD | No | Win/Linux/macOS | ~9.4k |
| Windows Live Captions | Yes (OS-level) | Yes | Proprietary | Unknown | Windows 11 only | Closed |
| Google Live Caption | Yes (OS-level) | Yes | Proprietary | Unknown | Android / Chrome | Closed |
| macOS Live Captions | Yes (OS-level) | Yes | Neural (on-device) | Unknown | macOS only (Apple Silicon) | Closed |

\* macOS requires BlackHole virtual audio device for loopback.

---

## Detailed Project Profiles

### Buzz

**GitHub**: [chidiwilliams/buzz](https://github.com/chidiwilliams/buzz) | **License**: MIT | **Language**: Python (PyQt GUI)

The most popular open-source Whisper desktop app. Supports multiple STT backends (OpenAI Whisper, faster-whisper, whisper.cpp), file transcription, YouTube URL import, and live mic transcription with a presentation overlay window. Offers speaker diarization, noise reduction, and export to SRT/VTT/TXT.

**Key difference from SubsVibe**: No system audio loopback -- mic only for live use. No LLM post-processing. Targets a different use case: transcribing your own voice or pre-recorded files, not captioning audio already playing on your system.

---

### WhisperLive

**GitHub**: [collabora/WhisperLive](https://github.com/collabora/WhisperLive) | **License**: MIT | **Language**: Python + JS + Swift

Client-server architecture with faster-whisper and Silero VAD on the backend. Chrome/Firefox extensions route browser tab audio to the server via WebSocket. Supports RTSP/HLS streams. Docker-first deployment.

**Key difference from SubsVibe**: System audio capture is limited to browser tabs via extensions -- desktop apps, games, and video players are not captured. Client-server model adds deployment complexity and network latency. No LLM refinement.

---

### LiveCaptions

**GitHub**: [abb128/LiveCaptions](https://github.com/abb128/LiveCaptions) | **License**: GPL-3.0 | **Language**: C + GTK

Linux desktop app that captions system audio via PulseAudio monitor sources. Uses AprilASR (LSTM transducer, ONNX) instead of Whisper. Includes token-level confidence fading, profanity filter, and D-Bus text stream for external apps.

**Key difference from SubsVibe**: Linux-only (PulseAudio). Uses AprilASR not Whisper -- English-only in practice with lower accuracy. Simple amplitude-based silence detection instead of neural VAD. No LLM refinement. Tightly coupled C architecture vs. SubsVibe's decoupled queue-based pipeline.

---

### Vibe

**GitHub**: [thewh1teagle/vibe](https://github.com/thewh1teagle/vibe) | **License**: MIT | **Language**: TypeScript + Rust + Go (Tauri v2)

Cross-platform desktop app using whisper.cpp via a Go sidecar (Sona). Polished GUI with batch file processing, multiple export formats (SRT, VTT, PDF, DOCX), GPU acceleration (CoreML/Metal/Vulkan), speaker diarization, and web content extraction. Optional Claude/Ollama integration for post-transcription summarization.

**Key difference from SubsVibe**: Fundamentally file-based (record-then-transcribe). No true real-time streaming pipeline. No VAD. System audio only works on macOS via ScreenCaptureKit. LLM is used only for post-transcription summary, not real-time correction.

---

### SubsAI

**GitHub**: [absadiki/subsai](https://github.com/absadiki/subsai) | **License**: GPL-3.0 | **Language**: Python (Streamlit UI)

Multi-engine subtitle generator supporting 8 Whisper backends through a unified interface. Offers Web UI, CLI, and Python API. Includes subtitle translation (NLLB-200, M2M100, mBART), auto-sync via ffsubsync, interactive subtitle editor, and video embedding.

**Key difference from SubsVibe**: Entirely file-based -- no real-time capability, no audio capture. VAD is optional and delegated to faster-whisper. No LLM refinement. Synchronous single-pass architecture. Heavy dependency footprint (PyTorch, torchaudio, transformers, multiple engines).

---

### RealtimeSTT

**GitHub**: [KoljaB/RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) | **License**: MIT | **Language**: Python

A library (not application) for real-time speech-to-text with faster-whisper. Notable for its **dual VAD system**: WebRTCVAD for fast initial speech detection, then SileroVAD for verification -- reduces false positives. Supports wake word activation (Porcupine, OpenWakeWord) and callback-based recording events. Accepts custom audio chunk feeding.

**Key difference from SubsVibe**: A library, not a complete pipeline. No system audio capture (though custom chunk feeding makes integration possible). No LLM refinement. No subtitle output or display. Maintainer has stepped back from active development.

---

### whisper_streaming

**GitHub**: [ufal/whisper_streaming](https://github.com/ufal/whisper_streaming) | **License**: MIT | **Language**: Python

Research-grade streaming transcription with a "**local agreement**" policy: output is only emitted when consecutive Whisper runs agree on the same text, providing stability without an LLM. Supports faster-whisper, whisper-timestamped, OpenAI API, and Whisper MLX backends. Optional Silero VAD. ~3.3 second latency.

**Key difference from SubsVibe**: No system audio capture (mic via ALSA only). The local agreement approach solves output instability differently from SubsVibe's LLM context window -- lighter weight but less powerful (can't fix semantic errors or proper nouns). Being superseded by authors' newer SimulStreaming project.

---

### whisper.cpp (stream example)

**GitHub**: [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp) | **License**: MIT | **Language**: C/C++

The gold standard for raw Whisper inference performance. Zero-dependency C/C++ implementation with massive hardware acceleration support (NEON, AVX, Metal, CoreML, CUDA, Vulkan, OpenVINO). The `stream` example provides basic continuous mic transcription at ~500ms intervals.

**Key difference from SubsVibe**: The stream example is self-described as "naive" -- a demo, not a production solution. No system audio. Extremely basic amplitude-based VAD. No LLM, no subtitle formatting. C/C++ makes integration with Python LLM pipelines difficult.

---

### speech-to-text (reriiasu)

**GitHub**: [reriiasu/speech-to-text](https://github.com/reriiasu/speech-to-text) | **License**: MIT | **Language**: Python + HTML/JS

Web-based GUI with real-time mic transcription via WebSocket. Uses Silero VAD + faster-whisper -- the same core stack as SubsVibe. **Includes OpenAI API integration for text proofreading** -- the only other open-source project found with LLM-based correction.

**Key difference from SubsVibe**: No system audio capture (mic only). LLM proofreading is simple per-segment correction via cloud OpenAI API, not SubsVibe's sliding context window with provisional subtitles. No local LLM support (Ollama, vLLM, etc.). Web UI only, no subtitle overlay.

---

### LocalVocal

**GitHub**: [occ-ai/obs-localvocal](https://github.com/occ-ai/obs-localvocal) | **License**: GPL-2.0 | **Language**: C++ | **Stars**: ~1.4k

OBS Studio plug-in that adds real-time transcription, translation, and captioning as an audio filter. Uses whisper.cpp for inference with GGML-format models. Supports 100+ languages, captions displayed via OBS text sources, file output (.txt/.srt), RTMP stream caption embedding, and synchronized recording timestamps. The most feature-rich translation stack of any open-source captioning tool: Whisper's built-in translator, local NMT via CTranslate2 (M2M100, NLLB, T5), 7 cloud providers (AWS, Azure, Claude, DeepL, Google Cloud, OpenAI, Papago), and a custom API option with configurable endpoint URL (can point to Ollama or other local servers with manual setup). Extensive hardware acceleration: CUDA, Vulkan, hipBLAS/ROCm, Metal, CoreML, OpenCL, OpenBLAS, AVX/SSE/AVX2/AVX512. Extremely practical for the large OBS user base -- streamers get production-ready captioning with zero additional tooling.

**Key difference from SubsVibe**: Deeply integrated with OBS -- transcribes audio sources attached as filters, making it the most turnkey solution for streamers. Has Silero VAD via ONNX Runtime (active/hybrid/disabled modes). LLM APIs (OpenAI, Claude, custom endpoints) are used as translation backends but not for cross-segment context-aware correction -- each segment is translated independently. SubsVibe's differentiator is system-wide audio capture outside OBS and LLM-based sliding context refinement across segments. SubsVibe's decoupled pipeline could also serve as a foundation for an OBS plugin in the future, bringing context-aware LLM refinement into the OBS ecosystem.

---

### FunASR

**GitHub**: [modelscope/FunASR](https://github.com/modelscope/FunASR) | **License**: MIT | **Language**: Python/C++ | **Stars**: ~13.9k

Comprehensive end-to-end speech recognition toolkit from Alibaba DAMO Academy. Flagship model is **Paraformer** (non-autoregressive ASR). Provides ASR, VAD (FSMN-VAD), punctuation restoration (CT-Transformer), speaker diarization, and multi-talker ASR. Notable **2pass mode**: real-time streaming via Paraformer-streaming for immediate results, then offline Paraformer-large corrects errors at sentence boundaries. Docker-deployable server with WebSocket support.

**Key difference from SubsVibe**: A server/toolkit, not a desktop application -- no system audio capture. The 2pass architecture achieves a similar goal to SubsVibe's LLM refinement (correcting first-pass errors with broader context) but uses a dedicated ASR model rather than a general-purpose LLM, so it cannot correct domain-specific proper nouns, acronyms, or technical terminology. Strongest for Chinese/Mandarin; SubsVibe with Whisper may be better for English and European languages.

---

### WhisperLiveKit

**GitHub**: [QuentinFuxa/WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) | **License**: MIT | **Language**: Python + JS | **Stars**: ~9.4k

Local, low-latency real-time speech transcription with speaker diarization and web UI. Built on whisper_streaming (SimulStreaming). Uses the **AlignAtt** streaming policy for ultra-low latency simultaneous transcription. Supports faster-whisper and Whisper MLX backends. Multi-user WebSocket server with browser frontend. Automatic silence chunking via Silero VAD. One-command deployment.

**Key difference from SubsVibe**: No system audio capture -- microphone only via browser WebRTC. No LLM refinement. The AlignAtt streaming policy is a more sophisticated approach to streaming than simple chunking, but lacks cross-segment context-aware correction. Client-server architecture (WebSocket) vs. SubsVibe's local desktop pipeline.

---

### OS Built-in Solutions (Closed Source)

**Windows Live Captions** (Windows 11 22H2+), **Google Live Caption** (Android 10+ / Chrome), and **macOS Live Captions** (macOS Sonoma 14+ / Apple Silicon) provide the "north star" user experience: seamless OS-level system audio capture, highly optimized on-device models, polished caption overlay, and zero configuration. Apple's implementation runs on the Neural Engine and exposes a `SpeechAnalyzer` API (WWDC 2025) with `SpeechTranscriber` (STT) and `SpeechDetector` (VAD) modules, supporting provisional "volatile results" that refine over time -- conceptually similar to SubsVibe's provisional subtitles.

**SubsVibe's advantages over OS built-ins**:
- Open-source and auditable
- Cross-platform (not locked to one OS)
- User choice of STT model (swap between tiny/base/small/medium/large)
- LLM-based contextual refinement for proper nouns, technical terms, acronyms
- Configurable target language and translation via any OpenAI-compatible LLM endpoint
- Works with local LLMs (Ollama, vLLM, LM Studio) for full privacy

---

## What Makes SubsVibe Unique

No existing open-source project combines all four of these capabilities:

1. **Native cross-platform system audio loopback** -- SoundCard provides WASAPI (Windows) and PulseAudio (Linux) loopback capture. Only LiveCaptions (Linux-only) and the closed-source OS features offer comparable system audio capture.

2. **LLM sliding context window with provisional subtitles** -- Recent subtitle history is sent alongside new Whisper segments so the LLM can correct errors across segment boundaries. Subtitles are held as provisional until enough context confirms them. Only reriiasu/speech-to-text has any LLM integration, and it's simple per-segment proofreading.

3. **Complete four-stage decoupled pipeline** -- Capture -> VAD -> Whisper -> LLM, each stage running in its own thread with queue-based communication. Most projects implement one or two stages.

4. **Local-first LLM flexibility** -- OpenAI SDK with configurable `base_url` works with Ollama, vLLM, LM Studio, or any OpenAI-compatible endpoint. Full privacy without cloud dependency.

---

## Ideas from the Landscape

Patterns from other projects worth considering:

| Idea | Source | Description |
|---|---|---|
| Dual VAD | RealtimeSTT | WebRTCVAD for fast initial detection + SileroVAD for confirmation reduces false positives |
| Local agreement | whisper_streaming | Emit output only when consecutive Whisper runs agree -- complementary to LLM refinement |
| Multiple STT backends | Buzz, SubsAI | Supporting whisper.cpp alongside faster-whisper gives users more performance options |
| SRT/VTT export | Buzz, Vibe | Save caption history to subtitle file formats |
| Browser extension | WhisperLive | Alternative audio source for browser-only use cases |
| Speaker diarization | Vibe, Buzz, WhisperX | Identify different speakers in the audio |
| 2pass correction | FunASR | Streaming model for instant results + offline model for sentence-boundary error correction |
| FSMN-VAD | FunASR | Industrial-scale neural VAD; potentially superior to Silero for Chinese/Asian languages |
| AlignAtt streaming | WhisperLiveKit | State-of-the-art streaming policy for lower-latency simultaneous transcription |
| Variable-length ASR | Moonshine | Compute scales with input length (unlike Whisper's fixed 30s chunks) -- ideal for VAD-segmented pipelines |
| Audio event detection | SenseVoice | Detect [laughter], [applause], speaker emotion alongside transcription |
| Cloud translation providers | LocalVocal | DeepL, Google Cloud, Azure, AWS, Papago as translation backends beyond LLM |
| ONNX inference runtime | Sherpa-ONNX | Unified runtime supporting Whisper, Paraformer, SenseVoice, transducer models via ONNX |
