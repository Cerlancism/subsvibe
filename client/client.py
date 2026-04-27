from __future__ import annotations

import logging
import sys
from pathlib import Path

import av
import numpy as np

from transcribe import TRANSCRIPT_BASE_URL, TRANSCRIPT_MODEL_NAME, client as transcribe_client, normalize_language
from utils.logging_config import setup_logging

setup_logging()
log = logging.getLogger("subsvibe.client")


def _get_audio_duration(path: Path) -> float:
    try:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            return float(stream.duration * stream.time_base)
    except Exception as e:
        log.warning("could not get audio duration: %s", e)
        return 0.0


def _srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


SRT_MIN_DURATION_SECONDS = 0.5
SRT_READING_BUFFER_SECONDS = 1.0
SRT_NEXT_GAP_SECONDS = 0.001


def _normalize_durations(entries: list[dict]) -> list[dict]:
    """For each entry, extend its end by up to SRT_READING_BUFFER_SECONDS
    (giving the reader extra time), capped 1 ms before the next entry's start.
    If an entry still cannot reach SRT_MIN_DURATION_SECONDS, merge it forward."""
    out: list[dict] = [dict(e) for e in entries]
    i = 0
    while i < len(out):
        e = out[i]

        target_end = e["end"] + SRT_READING_BUFFER_SECONDS
        if i + 1 < len(out):
            target_end = min(target_end, out[i + 1]["start"] - SRT_NEXT_GAP_SECONDS)
        new_end = max(e["end"], target_end)

        if new_end - e["start"] >= SRT_MIN_DURATION_SECONDS or i + 1 >= len(out):
            e["end"] = new_end
            i += 1
            continue

        nxt = out[i + 1]
        merged_text = f"{e['text'].strip()} {nxt['text'].strip()}".strip()
        out[i + 1] = {"start": e["start"], "end": nxt["end"], "text": merged_text}
        del out[i]
        # re-check the merged entry on the next loop iteration
    return out


def _write_srt(entries: list[dict], out_path: Path) -> None:
    entries = _normalize_durations(entries)
    with out_path.open("w", encoding="utf-8") as f:
        for i, e in enumerate(entries, 1):
            f.write(f"{i}\n")
            f.write(f"{_srt_timestamp(e['start'])} --> {_srt_timestamp(e['end'])}\n")
            f.write(f"{e['text'].strip()}\n\n")
    log.info("wrote %d subtitle(s) to %s", len(entries), out_path)


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
) -> list[dict]:
    start, end = seg["start"], seg["end"]
    filename = f"seg_{start:.3f}-{end:.3f}.wav"
    wav = _extract_wav_segment(path, start, end)

    kwargs: dict = dict(
        model=model,
        file=(filename, wav, "audio/wav"),
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )
    if language:
        kwargs["language"] = language

    result = transcribe_client.audio.transcriptions.create(**kwargs)

    raw_segments = list(getattr(result, "segments", None) or [])
    entries = []
    for s in raw_segments:
        sd = s if isinstance(s, dict) else s.__dict__
        seg_start = start + float(sd.get("start", 0))
        seg_end = start + float(sd.get("end", 0))
        text = sd.get("text", "").strip()
        if text:
            entries.append({"start": seg_start, "end": seg_end, "text": text})

    if not entries:
        text = (result if isinstance(result, str) else result.text).strip()
        if text:
            entries.append({"start": start, "end": end, "text": text})

    return entries


def transcribe_file(path: Path, *, model: str, language: str | None) -> None:
    from vad import get_speech_segments

    audio_duration = _get_audio_duration(path)
    log.info("audio duration: %.1fs", audio_duration)

    segments = get_speech_segments(path)
    if not segments:
        log.warning("no speech detected in %s", path.name)
        return

    log.info("transcribing %d VAD segment(s) (audio=%.1fs)", len(segments), audio_duration)

    all_entries: list[dict] = []

    for i, seg in enumerate(segments, 1):
        log.info("segment %d/%d  [%.2f–%.2f]", i, len(segments), seg["start"], seg["end"])
        all_entries.extend(_transcribe_segment(path, seg, model=model, language=language))

    all_entries.sort(key=lambda e: e["start"])

    out_path = path.with_suffix(".srt")
    _write_srt(all_entries, out_path)
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
        description="SubsVibe client — transcription and live subtitles.",
    )
    parser.add_argument("-i", "--input", type=Path, default=None, help="Audio/video file to subtitle (mp3, wav, mp4, …)")
    parser.add_argument("--live", action="store_true", help="Live capture from default system audio output (loopback)")
    parser.add_argument("--model", default=TRANSCRIPT_MODEL_NAME, help="Model name")
    parser.add_argument("--language", default=None, help="Language hint: ISO-639-1 code (e.g. ja, zh) or canonical name (e.g. Japanese). Default: auto-detect")
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
        print(f"{result.get('status', '?')}: {result.get('model', '')}")
        return

    if args.live:
        try:
            live_capture(
                model=args.model,
                language=args.language,
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
        try:
            transcribe_file(args.input, model=args.model, language=args.language)
        except APIConnectionError:
            sys.exit(f"error: could not connect to transcription server at {TRANSCRIPT_BASE_URL}")
        except APIStatusError as exc:
            sys.exit(f"error: server returned {exc.status_code}: {exc.message}")
    else:
        parser.error("provide --input, --live, or a server management flag (--health, --load, --unload)")


if __name__ == "__main__":
    main()
