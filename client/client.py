from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
import wave
from collections import deque
from pathlib import Path

import av
import numpy as np
from openai import APIConnectionError, APIStatusError, OpenAI

from utils.logging_config import setup_logging

setup_logging()
log = logging.getLogger("subsvibe.client")

TRANSCRIPT_HOST = os.environ.get("TRANSCRIPT_HOST", "127.0.0.1")
TRANSCRIPT_PORT = os.environ.get("TRANSCRIPT_PORT", "8000")
TRANSCRIPT_MODEL_NAME = os.environ.get("TRANSCRIPT_MODEL_NAME", "qwen3-asr")
TRANSCRIPT_BASE_URL = os.environ.get("TRANSCRIPT_BASE_URL", f"http://{TRANSCRIPT_HOST}:{TRANSCRIPT_PORT}")
TRANSCRIPT_API_KEY = os.environ.get("TRANSCRIPT_API_KEY", "not-needed-locally")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "qwen3.5-instruct:4b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

_client = OpenAI(api_key=TRANSCRIPT_API_KEY, base_url=TRANSCRIPT_BASE_URL)
_llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def _get_audio_duration(path: Path) -> float:
    try:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            return float(stream.duration * stream.time_base)
    except Exception as e:
        log.warning("could not get audio duration: %s", e)
        return 0.0


LIVE_SAMPLE_RATE = 16000
LIVE_WINDOW_SECONDS = 5
LIVE_TICK_SECONDS = 1

_TRANSLATE_SYSTEM = (
    "You are a real-time subtitle translator working with a sliding window ASR system. "
    f"Every {LIVE_TICK_SECONDS} second(s) you receive a new {LIVE_WINDOW_SECONDS}-second transcript window. "
    "Each window heavily overlaps with the previous one — only the last second or so is genuinely new. "
    "The transcript is raw ASR output and may contain mid-sentence fragments, repeated phrases, "
    "or mis-heard words that get corrected in later windows. "
    "Your job: translate the complete thought visible in the current window into natural English. "
    "Use the history to understand context and spot ASR corrections "
    "(e.g. a word that was wrong before now appears correctly — prefer the corrected form). "
    "Focus on what is new or corrected compared to the previous window. "
    "Output only the English translation of the current window, no explanations."
)

TRANSLATE_HISTORY_LEN = 10


def _translate(text: str, history: list[tuple[str, str]]) -> str:
    messages: list[dict] = [{"role": "system", "content": _TRANSLATE_SYSTEM}]
    if history:
        context_lines = "\n".join(
            f"transcript: {raw}\ntranslation: {tr}" for raw, tr in history
        )
        messages.append({
            "role": "user",
            "content": f"Recent context (oldest to newest):\n{context_lines}",
        })
        messages.append({"role": "assistant", "content": "Understood."})
    messages.append({"role": "user", "content": f"Current window transcript: {text}"})
    resp = _llm_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=messages,
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


def _fmt_ts_filename(seconds: float) -> str:
    return _fmt_ts(seconds).replace(":", "-").replace(".", "-")


def _print_timestamps(words: list, segments: list, granularity: str, *, translate: bool = False) -> float:
    """Print timestamps and return total translation time in seconds."""
    t_translate = 0.0
    if granularity == "word":
        for w in words:
            start = _fmt_ts(w.get("start", 0))
            end = _fmt_ts(w.get("end", 0))
            print(f"  [{start} -> {end}]  {w.get('word', w.get('text', ''))}")
    elif granularity == "segment":
        for seg in segments:
            start = _fmt_ts(seg.get("start", 0))
            end = _fmt_ts(seg.get("end", 0))
            text = seg.get("text", "")
            prefix = f"  [{start} -> {end}]  "
            print(f"{prefix}{text}")
            if translate and text:
                t0 = time.monotonic()
                print(f"{' ' * (len(prefix) - 3)}-> {_translate(text, [])}")
                t_translate += time.monotonic() - t0
    return t_translate


def transcribe_file(
    path: Path,
    *,
    model: str,
    language: str | None,
    timestamps: str,
    translate: bool,
) -> None:
    audio_duration = _get_audio_duration(path)
    want_timestamps = timestamps != "none"
    fmt = "verbose_json" if want_timestamps else "json"

    with path.open("rb") as f:
        size = path.stat().st_size
        log.info("sending %s (%dB) -> %s", path.name, size, TRANSCRIPT_BASE_URL)
        kwargs: dict = dict(
            model=model,
            file=(path.name, f),
            response_format=fmt,
        )
        if language:
            kwargs["language"] = language
        t0 = time.monotonic()
        result = _client.audio.transcriptions.create(**kwargs)
        elapsed = time.monotonic() - t0

    text = result if isinstance(result, str) else result.text
    log.info("received in %.2fs — %r", elapsed, text[:80])

    t_translate = 0.0
    if want_timestamps:
        words = list(getattr(result, "words", None) or [])
        segments = list(getattr(result, "segments", None) or [])
        t_translate = _print_timestamps(
            [w if isinstance(w, dict) else w.__dict__ for w in words],
            [s if isinstance(s, dict) else s.__dict__ for s in segments],
            timestamps,
            translate=translate,
        )
    else:
        print(text)
        if translate:
            t0 = time.monotonic()
            print(f"  -> {_translate(text, [])}")
            t_translate = time.monotonic() - t0

    if translate:
        total_time = elapsed + t_translate
        buffer = audio_duration - total_time
        real_time_pct = 100.0 * audio_duration / total_time if total_time > 0 else 0
        log.info(
            "audio=%.2fs, transcript=%.2fs, translate=%.2fs, total=%.2fs, buffer=%.2fs (%.1f%% real-time)",
            audio_duration, elapsed, t_translate, total_time, buffer, real_time_pct,
        )




def _encode_wav(pcm_float32: np.ndarray, sample_rate: int = LIVE_SAMPLE_RATE) -> bytes:
    """Encode a float32 mono PCM array as WAV bytes (int16)."""
    pcm_int16 = (np.clip(pcm_float32, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def live_capture(
    *,
    model: str,
    language: str | None,
    translate: bool,
    window: int = LIVE_WINDOW_SECONDS,
    tick: int = LIVE_TICK_SECONDS,
) -> None:
    import warnings

    import soundcard as sc
    from soundcard import SoundcardRuntimeWarning

    warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

    mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
    log.info("capturing loopback from: %s", mic.name)

    samples_per_tick = LIVE_SAMPLE_RATE * tick
    samples_per_window = LIVE_SAMPLE_RATE * window
    ring: deque[np.ndarray] = deque()
    ring_len = 0  # total samples currently buffered
    ticks_elapsed = 0
    history: list[tuple[str, str]] = []

    log.info("starting live capture — window=%ds tick=%ds (Ctrl+C to stop)", window, tick)

    with mic.recorder(samplerate=LIVE_SAMPLE_RATE, channels=1) as recorder:
        while True:
            chunk = recorder.record(numframes=samples_per_tick).reshape(-1).astype(np.float32)
            ring.append(chunk)
            ring_len += len(chunk)
            ticks_elapsed += 1

            if ring_len < samples_per_window:
                continue

            # Trim ring to window length
            while ring_len > samples_per_window:
                oldest = ring[0]
                excess = ring_len - samples_per_window
                if excess >= len(oldest):
                    ring.popleft()
                    ring_len -= len(oldest)
                else:
                    ring[0] = oldest[excess:]
                    ring_len -= excess

            window_audio = np.concatenate(list(ring))
            wav_bytes = _encode_wav(window_audio)

            win_end = ticks_elapsed * tick
            win_start = max(0, win_end - window)
            filename = f"{_fmt_ts_filename(win_start)}-{_fmt_ts_filename(win_end)}.wav"

            t0 = time.monotonic()
            try:
                result = _client.audio.transcriptions.create(
                    model=model,
                    file=(filename, wav_bytes, "audio/wav"),
                    response_format="json",
                    **({"language": language} if language else {}),
                )
            except APIConnectionError:
                log.error("could not connect to transcription server at %s", TRANSCRIPT_BASE_URL)
                continue
            except APIStatusError as exc:
                log.error("server error %s: %s", exc.status_code, exc.message)
                continue
            elapsed = time.monotonic() - t0

            text = result if isinstance(result, str) else result.text
            if not text:
                continue

            if translate:
                t_tx0 = time.monotonic()
                translation = _translate(text, history)
                t_translate = time.monotonic() - t_tx0
                history.append((text, translation))
                if len(history) > TRANSLATE_HISTORY_LEN:
                    history.pop(0)
                print(f"{text}")
                print(f"  -> {translation}")
                log.info("transcript=%.2fs translate=%.2fs", elapsed, t_translate)
            else:
                print(text)
                log.info("transcript=%.2fs", elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SubsVibe client - real time transcription and translation.",
    )
    parser.add_argument("-i", "--input", type=Path, default=None, help="Audio file to transcribe (mp3, wav, …)")
    parser.add_argument("--live", action="store_true", help="Live capture from default system audio output (loopback)")
    parser.add_argument("--model", default=TRANSCRIPT_MODEL_NAME, help="Model name")
    parser.add_argument("--language", default=None, help="ISO-639-1 language code (default: auto-detect)")
    parser.add_argument(
        "--timestamps",
        default="none",
        choices=["none", "segment", "word"],
        help="Include timestamps in output (default: none)",
    )
    parser.add_argument("--translate", action="store_true", help="Translate each segment to English via LLM")

    args = parser.parse_args()

    if args.live:
        try:
            live_capture(
                model=args.model,
                language=args.language,
                translate=args.translate,
            )
        except KeyboardInterrupt:
            log.info("stopped")
        except APIConnectionError:
            sys.exit(f"error: could not connect to transcription server at {TRANSCRIPT_BASE_URL}")
    elif args.input is not None:
        if not args.input.exists():
            parser.error(f"File not found: {args.input}")
        try:
            transcribe_file(
                args.input,
                model=args.model,
                language=args.language,
                timestamps=args.timestamps,
                translate=args.translate,
            )
        except APIConnectionError:
            sys.exit(f"error: could not connect to transcription server at {TRANSCRIPT_BASE_URL}")
        except APIStatusError as exc:
            sys.exit(f"error: server returned {exc.status_code}: {exc.message}")
    else:
        parser.error("provide --input or --live")


if __name__ == "__main__":
    main()
