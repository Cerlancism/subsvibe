---
name: webrtcvad
description: >-
  Reference for py-webrtcvad, the Python wrapper around Google's WebRTC Voice
  Activity Detector (`import webrtcvad`, the `Vad` class, `is_speech`). Use this
  whenever working with webrtcvad: splitting PCM audio into speech vs. silence,
  segmenting recordings, gating a transcription/ASR pipeline on speech, building
  a frame_generator / vad_collector, or choosing an aggressiveness mode. Trigger
  it for the strict audio-format constraints (16-bit mono PCM at 8/16/32/48 kHz,
  frames of exactly 10/20/30 ms), for `is_speech` raising errors or returning
  garbage, for the sliding-window "triggered" segmenter pattern, and any time
  someone mentions webrtcvad, WebRTC VAD, or wiseman/py-webrtcvad — even if they
  don't name the exact API. Note this is the classic energy/GMM WebRTC VAD, NOT a
  neural model like Silero; pick the right tool when both are in play.
---

# py-webrtcvad

py-webrtcvad is a thin Python binding over the voice-activity detector shipped
in Google's WebRTC stack. It answers one narrow question per audio frame: *does
this frame contain speech?* It is fast, dependency-free at runtime, and runs on
CPU with no model file — but it is a classic signal-processing VAD (energy + a
Gaussian mixture over frequency bands), not a neural network. That trade-off
drives almost everything below: it is cheap and instant, but pickier about input
format and noisier in its judgments than a model like Silero.

## The one rule that breaks everyone: input format is strict

WebRTC VAD rejects anything that isn't shaped exactly right, and the failure
modes are unhelpful — either a raised exception or a silently wrong answer. Get
these four things right and the rest is easy:

- **16-bit signed PCM** — `int16` samples, little-endian, passed as raw `bytes`.
  Not float, not numpy arrays, not 8-bit, not 24-bit.
- **Mono** — one channel. Downmix stereo before you start.
- **Sample rate ∈ {8000, 16000, 32000, 48000} Hz** — no other rate is accepted.
- **Frame duration ∈ {10, 20, 30} ms** — and *only* those. The frame's byte
  length must equal `sample_rate * (duration_ms / 1000) * 2` (the `* 2` is the
  two bytes per int16 sample). A frame of any other length is the single most
  common cause of `is_speech` raising an error.

> If you have audio in any other form (mp3, 44.1 kHz, float32, stereo), resample
> and convert *first* — e.g. with `ffmpeg`, `soundfile` + `librosa`, or
> `pydub`. The VAD will not do it for you.

## Core API

```python
import webrtcvad

vad = webrtcvad.Vad()       # default aggressiveness (mode 0)
vad = webrtcvad.Vad(2)      # or construct with a mode 0-3
vad.set_mode(3)             # change mode later

# frame must be exactly 10/20/30 ms of 16-bit mono PCM bytes at sample_rate
speech: bool = vad.is_speech(frame_bytes, sample_rate)
```

That is the entire surface area: construct, optionally set a mode, and call
`is_speech(frame, sample_rate) -> bool` per frame. There is no streaming state,
no internal buffer, no segmentation — `is_speech` is a pure per-frame classifier.
All the useful behaviour (turning a stream of yes/no flags into clean speech
segments) is logic *you* build on top, shown below.

### Aggressiveness modes (0–3)

The mode controls how aggressively the detector filters out non-speech. Higher
is stricter about calling something speech:

- **0** — least aggressive. Lets the most audio through as "speech"; good when
  missing speech is worse than including some noise (high recall).
- **1, 2** — middle ground. **Mode 2 or 3 is the usual choice for noisy,
  real-world audio.**
- **3** — most aggressive. Filters hardest; best in noisy environments where you
  want only confident speech, at the cost of clipping quiet or short utterances.

There is no universally correct mode — it depends on your noise floor and
whether false positives or false negatives hurt more. Start at 2 or 3 for
captured system/desktop audio, drop toward 0–1 for clean close-mic recordings
where you don't want to lose soft speech.

## The real pattern: per-frame flags → clean segments

A raw stream of `is_speech` results flickers — a single noisy frame mid-speech
reads as silence, a lip-smack mid-silence reads as speech. Chopping on every flip
gives you shredded, useless segments. The canonical fix from the project's
`example.py` is a **sliding-window state machine with hysteresis**: only *start*
a segment once a window is mostly-voiced, and only *end* it once a window is
mostly-unvoiced. This rides through brief glitches in both directions.

Two pieces make it work — a frame generator and the collector. Read
`references/example.py.md` for the full, runnable reference implementation
(reading/writing WAV, the `Frame` class, both functions, and a `main`). Adapt it
rather than retyping from memory; the padding/ratio bookkeeping is easy to get
subtly wrong.

The shape of it:

```python
# 1. Slice contiguous PCM into fixed-duration Frames (carrying a timestamp).
for frame in frame_generator(frame_duration_ms=30, audio=pcm, sample_rate=sr):
    ...

# 2. Feed frames through the collector; it yields one bytes blob per detected
#    utterance, already trimmed to start/end on confident speech transitions.
segments = vad_collector(sr, frame_duration_ms=30,
                         padding_duration_ms=300, vad=vad, frames=frames)
for segment in segments:
    # `segment` is concatenated PCM bytes for one speech run — transcribe,
    # write to a WAV, etc.
    ...
```

How the collector decides (the part worth internalising):

- It keeps a **ring buffer** of the last `padding_duration_ms / frame_duration_ms`
  frames (e.g. 300 ms / 30 ms = 10 frames).
- **Not triggered (in silence):** when **>90%** of the ring buffer is voiced, it
  *triggers* — emits the buffered padding frames as the segment's lead-in (so you
  don't clip the word's onset) and starts collecting.
- **Triggered (in speech):** when **>90%** of the ring buffer is unvoiced, it
  *de-triggers* — yields the collected segment and resets.
- Trailing audio still buffered when the stream ends is flushed as a final
  segment.

The `padding_duration_ms` (typically ~300 ms) is the smoothing window: longer
padding tolerates longer gaps within one utterance but merges nearby utterances;
shorter padding splits more eagerly. The 0.9 ratios are the hysteresis
thresholds — they're what stop a single odd frame from starting or ending a
segment.

## Common pitfalls

- **Frame length mismatch.** Computing frame size in *samples* but passing the
  wrong byte count (forgetting `* 2` for int16) — or feeding a leftover partial
  frame at the end of the buffer. `frame_generator` deliberately stops before any
  short tail (`while offset + n < len(audio)`); honour that.
- **Passing numpy or float audio.** `is_speech` wants raw `bytes` of int16. From
  numpy: `arr.astype(np.int16).tobytes()`. From float in [-1, 1]:
  `(arr * 32767).astype(np.int16).tobytes()`.
- **Unsupported sample rate.** 44100 Hz (CD audio) and 22050 Hz are *not*
  supported — resample to 16000 or 48000 first.
- **Expecting word-level precision.** This is frame-level (10–30 ms) and tuned
  for telephony-style speech presence, not phoneme boundaries. For tight word
  timings, follow VAD with a forced aligner.
- **Treating it as a recognizer.** It detects *speech vs. not*, not *who* or
  *what*. Music, laughter, and some noises can read as speech, especially at low
  aggressiveness.

## Installation

```bash
pip install webrtcvad
```

If you hit a C-compiler error building the extension (common on Windows or in
minimal CI images), use the prebuilt-wheel fork instead — same `import
webrtcvad`, same API:

```bash
pip install webrtcvad-wheels
```

## Picking webrtcvad vs. a neural VAD (e.g. Silero)

Reach for **webrtcvad** when you want a tiny, instant, model-free CPU detector
and can tolerate some noise sensitivity — quick segmentation, pre-filtering, or
environments where pulling in PyTorch is overkill. Reach for a **neural VAD like
Silero** when accuracy in noise matters more than footprint, or when you're
already in a torch pipeline. They have different input contracts (Silero takes
float tensors and its own chunk sizes), so don't assume frames are
interchangeable — if a project already standardises on one, match it rather than
mixing the two.
