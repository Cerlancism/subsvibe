from __future__ import annotations

import logging
import sys
from pathlib import Path

import av
import numpy as np

from subtitle import entries_from_words, write_srt
from transcribe import TRANSCRIPT_BASE_URL, TRANSCRIPT_MODEL_NAME, client as transcribe_client, normalize_language
from utils.logging_config import setup_logging
from utils.subtitle import overlapping_text, read_srt
from utils.text import attach_punctuation
from utils.time import format_timestamp

setup_logging()
log = logging.getLogger("subsvibe.client")

REFERENCE_CONTEXT_PAD_SECONDS = 3.0


def _get_audio_duration(path: Path) -> float:
    try:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            return float(stream.duration * stream.time_base)
    except Exception as e:
        log.warning("could not get audio duration: %s", e)
        return 0.0


def peak_normalize(pcm: np.ndarray, *, start: float, end: float) -> np.ndarray:
    """Scale samples so the peak hits ~99% of int16 max. Skip near-silent
    segments so we don't amplify pure noise."""
    if not pcm.size:
        return pcm
    peak = int(np.abs(pcm).max())
    if peak < 500:
        return pcm
    gain = (32767 * 0.99) / peak
    out = np.clip(pcm.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
    log.info("peak-normalized [%.2f–%.2f] gain=%+.1fdB (peak %d → %d)",
             start, end, 20 * np.log10(gain), peak, int(peak * gain))
    return out


def _extract_wav_segment(path: Path, start: float, end: float) -> bytes:
    import io
    import wave

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16p", layout="mono", rate=16000)
        seek_ts = int(start / float(stream.time_base))
        container.seek(seek_ts, stream=stream)
        for packet in container.demux(stream):
            for frame in packet.decode():
                pts_sec = float(frame.pts * stream.time_base)
                if pts_sec > end + 0.1:
                    break
                for resampled in resampler.resample(frame):
                    frames.append(resampled.to_ndarray()[0])
            else:
                continue
            break
        for resampled in resampler.resample(None):
            frames.append(resampled.to_ndarray()[0])

    pcm = np.concatenate(frames).astype(np.int16) if frames else np.zeros(0, dtype=np.int16)
    # pcm = peak_normalize(pcm, start=start, end=end)  # disabled

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _transcribe_segment(
    path: Path,
    seg: dict,
    *,
    model: str,
    language: str | None,
    prompt: str | None,
) -> list[dict]:
    start, end = seg["start"], seg["end"]
    filename = f"seg_{start:.3f}-{end:.3f}.wav"
    wav = _extract_wav_segment(path, start, end)

    kwargs: dict = dict(
        model=model,
        file=(filename, wav, "audio/wav"),
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt

    result = transcribe_client.audio.transcriptions.create(**kwargs)

    log.debug(f"segment result: ", result)

    full_text = (result if isinstance(result, str) else (result.text or "")).strip()
    raw_words = list(getattr(result, "words", None) or [])

    def _field(w: object, name: str, default: object = "") -> object:
        if isinstance(w, dict):
            return w.get(name, default)
        return getattr(w, name, default)

    bare_words: list[dict] = []
    for w in raw_words:
        token = _field(w, "word") or _field(w, "text") or ""
        bare_words.append({
            "text": str(token),
            "start": start + float(_field(w, "start", 0)),
            "end": start + float(_field(w, "end", 0)),
        })

    words = attach_punctuation(bare_words, full_text)
    entries = entries_from_words(words)
    if entries:
        return entries

    # fallback when the server returns no words (e.g. timestamps disabled)
    if full_text:
        return [{"start": start, "end": end, "text": full_text}]
    return []


def _build_segment_prompt(
    base_prompt: str | None,
    reference_entries: list[dict] | None,
    seg: dict,
) -> tuple[str | None, dict | None]:
    """Return (prompt, reference_match). reference_match is the {start, end, text}
    span from the reference SRT (or None if nothing overlapped). The lookup
    window is padded by REFERENCE_CONTEXT_PAD_SECONDS on each side."""
    if not reference_entries:
        return base_prompt, None
    pad_start = seg["start"] - REFERENCE_CONTEXT_PAD_SECONDS
    pad_end = seg["end"] + REFERENCE_CONTEXT_PAD_SECONDS
    match = overlapping_text(reference_entries, pad_start, pad_end)
    if match is None:
        return base_prompt, None
    reference_block = f"Reference: {match['text']}"
    if base_prompt:
        return f"{base_prompt}\n{reference_block}", match
    return reference_block, match


def transcribe_file(
    path: Path,
    *,
    model: str,
    language: str | None,
    prompt: str | None,
    output: Path | None = None,
    reference_srt: Path | None = None,
) -> None:
    from vad import get_speech_segments

    audio_duration = _get_audio_duration(path)
    log.info("audio duration: %.1fs", audio_duration)

    reference_entries: list[dict] | None = None
    if reference_srt is not None:
        reference_entries = read_srt(reference_srt)
        log.info("loaded %d reference subtitle(s) from %s", len(reference_entries), reference_srt.name)

    segments = get_speech_segments(path, reference_entries=reference_entries)
    if not segments:
        log.warning("no speech detected in %s", path.name)
        return

    log.info("transcribing %d VAD segment(s) (audio=%.1fs)", len(segments), audio_duration)

    all_entries: list[dict] = []

    for i, seg in enumerate(segments, 1):
        log.info("segment %d/%d  [%s-%s]  %.1fs", i, len(segments), format_timestamp(seg["start"]), format_timestamp(seg["end"]), seg["end"] - seg["start"])
        seg_prompt, ref_match = _build_segment_prompt(prompt, reference_entries, seg)
        if ref_match is not None:
            log.info("segment %d reference context %d chars", i, len(ref_match["text"]))
        all_entries.extend(_transcribe_segment(path, seg, model=model, language=language, prompt=seg_prompt))

    all_entries.sort(key=lambda e: e["start"])

    out_path = output if output is not None else path.with_suffix(".srt")
    write_srt(all_entries, out_path)
    print(f"subtitles written to: {out_path}")


SERVER_REQUEST_TIMEOUT_SECONDS = 60


def _server_request(method: str, path: str) -> dict:
    import urllib.request
    url = f"{TRANSCRIPT_BASE_URL}{path}"
    req = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
    try:
        with urllib.request.urlopen(req, timeout=SERVER_REQUEST_TIMEOUT_SECONDS) as resp:
            import json
            return json.loads(resp.read())
    except Exception as exc:
        sys.exit(f"error: {exc}")


def main() -> None:
    import argparse

    from openai import APIConnectionError, APIStatusError

    from pipeline import live_capture

    parser = argparse.ArgumentParser(
        description="SubsVibe client - transcription and live subtitles.",
    )
    parser.add_argument("-i", "--input", type=Path, default=None, help="Audio/video file to subtitle (mp3, wav, mp4, …)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .srt path (default: alongside --input with .srt suffix)")
    parser.add_argument("--live", action="store_true", help="Live capture from default system audio output (loopback)")
    parser.add_argument("--model", default=TRANSCRIPT_MODEL_NAME, help="Model name")
    parser.add_argument("--language", default=None, help="Language hint: ISO-639-1 code (e.g. ja, zh) or canonical name (e.g. Japanese). Default: auto-detect")
    parser.add_argument("--prompt", default=None, help="Optional context appended to the ASR system prompt to bias vocabulary or style (e.g. proper nouns, jargon)")
    parser.add_argument("--context-src", default=None, help="Context source (file mode only). Path to an .srt file whose entries overlapping each VAD segment are appended to --prompt. Other formats reserved for future use.")
    parser.add_argument("--translate", action="store_true", help="Translate live subtitles to English via LLM (--live only)")

    server_group = parser.add_argument_group("server management")
    server_group.add_argument("--health", action="store_true", help="Check server health and model load state")
    server_group.add_argument("--load", action="store_true", help="Ask the server to load the ASR model")
    server_group.add_argument("--unload", action="store_true", help="Ask the server to unload the ASR model")

    args = parser.parse_args()

    try:
        args.language = normalize_language(args.language)
    except ValueError as exc:
        parser.error(str(exc))

    if args.health:
        result = _server_request("GET", "/health")
        loaded = result.get("model_loaded", "unknown")
        print(f"status: {result.get('status', '?')}  model_loaded: {loaded}")
        return

    if args.load:
        result = _server_request("POST", "/model/load")
        print(f"{result.get('status', '?')}: {result.get('model', '')}")
        return

    if args.unload:
        result = _server_request("POST", "/model/unload")
        status = result.get("status", "?")
        model = result.get("model", "")
        parts = []
        if result.get("asr_unloaded"):
            parts.append("asr")
        if result.get("aligner_unloaded"):
            parts.append("aligner")
        detail = f" ({', '.join(parts)})" if parts else ""
        print(f"{status}: {model}{detail}")
        return

    if args.live:
        if args.context_src is not None:
            parser.error("--context-src is only supported with --input")
        try:
            live_capture(
                model=args.model,
                language=args.language,
                prompt=args.prompt,
                do_translate=args.translate,
            )
        except KeyboardInterrupt:
            log.info("stopped")
        except APIConnectionError:
            sys.exit(f"error: could not connect to transcription server at {TRANSCRIPT_BASE_URL}")
    elif args.input is not None:
        if not args.input.exists():
            parser.error(f"File not found: {args.input}")
        if args.translate:
            parser.error("--translate is only supported with --live")
        reference_srt: Path | None = None
        if args.context_src is not None:
            ctx_path = Path(args.context_src)
            if not ctx_path.exists():
                parser.error(f"Context source not found: {ctx_path}")
            if ctx_path.suffix.lower() == ".srt":
                reference_srt = ctx_path
            else:
                parser.error(f"--context-src only supports .srt files for now (got {ctx_path.suffix})")
        try:
            transcribe_file(args.input, model=args.model, language=args.language, prompt=args.prompt, output=args.output, reference_srt=reference_srt)
        except APIConnectionError:
            sys.exit(f"error: could not connect to transcription server at {TRANSCRIPT_BASE_URL}")
        except APIStatusError as exc:
            sys.exit(f"error: server returned {exc.status_code}: {exc.message}")
    else:
        parser.error("provide --input, --live, or a server management flag (--health, --load, --unload)")


if __name__ == "__main__":
    main()
