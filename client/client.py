from __future__ import annotations

import logging
import sys
from pathlib import Path

import av
import numpy as np
from openai import OpenAI

from history import compose_prompt, select_history
from subtitle import entries_from_words, write_srt
from transcribe import (
    LLM_ASR_MODEL_ID,
    TRANSCRIPT_BACKEND,
    TRANSCRIPT_BASE_URL,
    TRANSCRIPT_MODEL_ID,
    build_llm_asr_system_prompt,
    align_words,
    get_asr_client,
    llm_asr_chat_transcribe,
    normalize_language,
)
from utils.language import is_cjk
from utils.logging_config import setup_logging
from utils.subtitle import overlapping_text, read_srt
from utils.text import attach_punctuation
from utils.time import format_timestamp

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


def _extract_wav_segment(path: Path, start: float, end: float) -> tuple[bytes, float]:
    """Extract [start, end] from `path`, peak-normalise, and encode as WAV.

    Returns (wav_bytes, gain_db). The whole file was already normalised once
    before VAD; this second pass scales each segment to its own peak so a
    quiet segment in an otherwise loud file still hits the ASR at full level.
    """
    from capture import encode_wav, peak_normalize

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        # Decode to float32 mono so peak_normalize works in the same units
        # as the live path. encode_wav handles the final int16 conversion.
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
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

    pcm = np.concatenate(frames).astype(np.float32) if frames else np.zeros(0, dtype=np.float32)
    pcm, gain_db = peak_normalize(pcm)
    return encode_wav(pcm), gain_db


def _words_to_entries(
    bare_words: list[dict],
    full_text: str,
    *,
    fallback_start: float,
    fallback_end: float,
) -> list[dict]:
    """Common post-processing: punctuation attachment + word→entry segmentation.
    Falls back to a single full-segment entry if word timing is unavailable."""
    if bare_words:
        words = attach_punctuation(bare_words, full_text)
        entries = entries_from_words(words)
        if entries:
            return entries
    if full_text:
        return [{"start": fallback_start, "end": fallback_end, "text": full_text}]
    return []


def _transcribe_segment_asr(
    seg: dict,
    wav: bytes,
    *,
    asr_client: OpenAI,
    model: str,
    language: str | None,
    prompt: str | None,
) -> list[dict]:
    """OpenAI-compatible audio.transcriptions.create() path.
    Returns SRT entries for the segment.

    Routing by TRANSCRIPT_BACKEND:
      - "faster-whisper": request segment-level timestamps and map the
        model's own segments directly to SRT entries (no word aligner).
      - "qwen" / "anime-whisper" (default): request word-level timestamps
        (produced server-side by the Qwen forced aligner) and run
        attach_punctuation + entries_from_words to build SRT entries from
        the aligned word stream."""
    if TRANSCRIPT_BACKEND == "faster-whisper":
        return _transcribe_segment_asr_segments(
            seg, wav, asr_client=asr_client, model=model, language=language, prompt=prompt,
        )
    if TRANSCRIPT_BACKEND not in {"qwen", "anime-whisper"}:
        log.warning(
            "unknown TRANSCRIPT_BACKEND=%r; using word-aligner SRT path",
            TRANSCRIPT_BACKEND,
        )
    return _transcribe_segment_asr_words(
        seg, wav, asr_client=asr_client, model=model, language=language, prompt=prompt,
    )


def _asr_kwargs(
    seg: dict,
    wav: bytes,
    *,
    model: str,
    language: str | None,
    prompt: str | None,
    granularity: str,
) -> dict:
    start, end = seg["start"], seg["end"]
    filename = f"seg_{start:.3f}-{end:.3f}.wav"
    kwargs: dict = dict(
        model=model,
        file=(filename, wav, "audio/wav"),
        response_format="verbose_json",
        timestamp_granularities=[granularity],
    )
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt
    return kwargs


def _transcribe_segment_asr_words(
    seg: dict,
    wav: bytes,
    *,
    asr_client: OpenAI,
    model: str,
    language: str | None,
    prompt: str | None,
) -> list[dict]:
    """Word-aligner path (Qwen3-ASR style)."""
    start, end = seg["start"], seg["end"]
    kwargs = _asr_kwargs(seg, wav, model=model, language=language, prompt=prompt, granularity="word")
    result = asr_client.audio.transcriptions.create(**kwargs)
    log.debug("segment result: %s", result)

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

    return _words_to_entries(bare_words, full_text, fallback_start=start, fallback_end=end)


def _transcribe_segment_asr_segments(
    seg: dict,
    wav: bytes,
    *,
    asr_client: OpenAI,
    model: str,
    language: str | None,
    prompt: str | None,
) -> list[dict]:
    """Segment-trust path (faster-whisper style). Maps Whisper segments
    directly to SRT entries without running the word aligner."""
    start, end = seg["start"], seg["end"]
    kwargs = _asr_kwargs(seg, wav, model=model, language=language, prompt=prompt, granularity="segment")
    result = asr_client.audio.transcriptions.create(**kwargs)
    log.debug("segment result: %s", result)

    full_text = (result if isinstance(result, str) else (result.text or "")).strip()
    raw_segments = list(getattr(result, "segments", None) or [])

    def _field(s: object, name: str, default: object = "") -> object:
        if isinstance(s, dict):
            return s.get(name, default)
        return getattr(s, name, default)

    entries: list[dict] = []
    for s in raw_segments:
        text = str(_field(s, "text", "") or "").strip()
        if not text:
            continue
        entries.append({
            "start": round(start + float(_field(s, "start", 0)), 3),
            "end": round(start + float(_field(s, "end", 0)), 3),
            "text": text,
        })

    if entries:
        return entries
    if full_text:
        return [{"start": start, "end": end, "text": full_text}]
    return []


def _transcribe_segment_llm(
    seg: dict,
    wav: bytes,
    *,
    asr_client: OpenAI,
    model: str,
    language: str | None,
    base_prompt: str | None,
    history_text: str | None,
    reference_text: str | None,
    align_base_url: str,
) -> list[dict]:
    """Chat-completions path for multimodal LLMs (e.g. gemma4:e4b on Ollama).
    Calls /v1/audio/align on the transcription server for word timestamps."""
    start, end = seg["start"], seg["end"]
    system_prompt = build_llm_asr_system_prompt(
        language=language,
        base_prompt=base_prompt,
        history=history_text,
        reference=reference_text,
    )
    full_text = llm_asr_chat_transcribe(asr_client, model, wav, system_prompt=system_prompt)
    if not full_text:
        return []

    try:
        raw_words = align_words(align_base_url, wav, full_text, language)
    except Exception as exc:
        log.warning("alignment failed for segment [%.2f-%.2f]: %s", start, end, exc)
        raw_words = []

    bare_words = [
        {
            "text": str(w.get("text", "")),
            "start": start + float(w.get("start", 0.0)),
            "end": start + float(w.get("end", 0.0)),
        }
        for w in raw_words
    ]
    return _words_to_entries(bare_words, full_text, fallback_start=start, fallback_end=end)


def _build_segment_context(
    reference_entries: list[dict] | None,
    history_texts: list[str] | None,
    seg: dict,
) -> tuple[str | None, str | None, dict | None]:
    """Resolve per-segment context. Returns
    (history_text, reference_text, reference_match). The lookup window for
    reference is padded by REFERENCE_CONTEXT_PAD_SECONDS on each side."""
    history_text = "\n".join(history_texts) if history_texts else None

    match: dict | None = None
    if reference_entries:
        pad_start = seg["start"] - REFERENCE_CONTEXT_PAD_SECONDS
        pad_end = seg["end"] + REFERENCE_CONTEXT_PAD_SECONDS
        match = overlapping_text(reference_entries, pad_start, pad_end)
    reference_text = match["text"] if match else None

    return history_text, reference_text, match


def transcribe_file(
    path: Path,
    *,
    asr_client: OpenAI,
    model: str,
    language: str | None,
    prompt: str | None,
    output: Path | None = None,
    reference_srt: Path | None = None,
    history: int = 0,
    history_seconds: float = 0.0,
    use_llm_asr: bool = False,
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
    history_enabled = history > 0 or history_seconds > 0
    history_buf: list[tuple[float, str]] = []  # (end_time, text)

    for i, seg in enumerate(segments, 1):
        wav, gain_db = _extract_wav_segment(path, seg["start"], seg["end"])
        log.info(
            "segment %d/%d  [%s-%s]  %.1fs  %+.1fdB",
            i, len(segments),
            format_timestamp(seg["start"]), format_timestamp(seg["end"]),
            seg["end"] - seg["start"], gain_db,
        )
        history_texts = select_history(history_buf, count=history, seconds=history_seconds, now=seg["start"]) or None
        history_text, reference_text, ref_match = _build_segment_context(reference_entries, history_texts, seg)
        if ref_match is not None:
            log.info("segment %d reference context %d chars", i, len(ref_match["text"]))
        if history_texts:
            log.info("segment %d history context %d entries", i, len(history_texts))

        if use_llm_asr:
            seg_entries = _transcribe_segment_llm(
                seg, wav,
                asr_client=asr_client, model=model, language=language,
                base_prompt=prompt, history_text=history_text, reference_text=reference_text,
                align_base_url=TRANSCRIPT_BASE_URL,
            )
        else:
            seg_prompt = compose_prompt(prompt, history_text, reference_text)
            seg_entries = _transcribe_segment_asr(
                seg, wav,
                asr_client=asr_client, model=model, language=language, prompt=seg_prompt,
            )
        all_entries.extend(seg_entries)
        if history_enabled:
            for e in seg_entries:
                txt = (e.get("text") or "").strip()
                if txt:
                    history_buf.append((float(e["end"]), txt))

    all_entries.sort(key=lambda e: e["start"])

    out_path = output if output is not None else path.with_suffix(".srt")
    write_srt(all_entries, out_path)
    print(f"subtitles written to: {out_path}")


SERVER_REQUEST_TIMEOUT_SECONDS = 60


def _server_request(method: str, path: str) -> dict:
    import urllib.request
    url = f"{TRANSCRIPT_BASE_URL.rstrip('/')}{path}"
    log.debug("server request: %s %s", method, url)
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
    parser.add_argument("--model", default=None, help=f"Model name (default: server's configured model, or {LLM_ASR_MODEL_ID} with --llm-asr; override with TRANSCRIPT_MODEL_ID={TRANSCRIPT_MODEL_ID or '<unset>'})")
    parser.add_argument("--llm-asr", action="store_true", help="Route audio to the LLM backend (LLM_BASE_URL) instead of the FastAPI transcription server. Use with multimodal LLMs that accept audio (e.g. gemma4:e4b on Ollama)")
    parser.add_argument("--language", default=None, help="Language hint: ISO-639-1 code (e.g. ja, zh) or canonical name (e.g. Japanese). Default: auto-detect")
    parser.add_argument("--prompt", default=None, help="Optional context appended to the ASR system prompt to bias vocabulary or style (e.g. proper nouns, jargon)")
    parser.add_argument("--context-src", default=None, help="Context source (file mode only). Path to an .srt file whose entries overlapping each VAD segment are appended to --prompt. Other formats reserved for future use.")
    parser.add_argument("--history", type=int, default=0, metavar="N", help="Append up to the last N committed transcripts to each segment's prompt. In live mode only finalised segments count (provisionals never enter history). Default: 0 (disabled). Combine with --history-seconds to cap both ways.")
    parser.add_argument("--history-seconds", type=float, default=0.0, metavar="T", help="Time-bounded history window: include prior segments whose end falls within the last T seconds before the current segment's start. Combine with --history to additionally cap by count. Default: 0 (disabled).")
    parser.add_argument("--translate", nargs="?", const="English", default=None, metavar="TARGET", help="Translate live subtitles via LLM (--live only). Optional value is free-text target language passed to the LLM (e.g. 'English', 'simplified Chinese', 'casual Japanese'). Default when bare: English.")
    parser.add_argument("--translate-prompt", default=None, metavar="TEXT", help="Extra context appended to the translator's system prompt (--live + --translate only). Use for proper-noun glossaries, tone hints, or domain vocabulary (e.g. 'Speakers: Ana, Koko. Render Koko-chan with the suffix.').")
    parser.add_argument("--translate-system-prompt", default=None, metavar="TEXT", help="EXPERIMENTAL: fully replace the translator's built-in system prompt (--live + --translate only). The replacement must itself specify target language and behaviour — the --translate TARGET and --translate-prompt values are ignored when this is set. Mutually exclusive with --translate-prompt.")
    parser.add_argument("--translate-history-seconds", type=float, default=None, metavar="T", help="EXPERIMENTAL: override the translator's history time window independently of --history-seconds (--live + --translate only). Default: inherit --history-seconds. Pass 0 to disable the time window for the translator only.")
    parser.add_argument("--translate-temperature", type=float, default=0.0, metavar="T", help="Sampling temperature for the LLM translator (--live + --translate only). Default: 0 (deterministic). Raise (e.g. 0.7-1.0) for creative / persona-style outputs via --translate-system-prompt or --translate-prompt.")
    parser.add_argument("--romanize", action=argparse.BooleanOptionalAction, default=None, help="Show a romanization line (romaji/pinyin/transliteration) between the transcript and translation (--live only). Backend is picked from --language: ja->pykakasi, zh->pypinyin, anything else (incl. auto-detect)->generic (anyascii). Default: on for CJK source languages (ja/zh/ko), off otherwise. Force with --romanize / disable with --no-romanize.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Client log verbosity (default: INFO)")
    parser.add_argument("--log-file", default=None, metavar="PATH", help="Also write logs to this file (plain text, no ANSI colours)")
    parser.add_argument("--log-file-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"], metavar="LEVEL", help="Log level for --log-file (default: same as --log-level). Useful for capturing DEBUG to file while keeping the console at INFO.")

    server_group = parser.add_argument_group("server management")
    server_group.add_argument("--health", action="store_true", help="Check server health and model load state")
    server_group.add_argument("--load", action="store_true", help="Ask the server to load the ASR model")
    server_group.add_argument("--load-aligner", action="store_true", help="Ask the server to load only the forced aligner (used by --llm-asr for word timestamps)")
    server_group.add_argument("--unload", action="store_true", help="Ask the server to unload all loaded models (ASR + aligner)")

    args = parser.parse_args()

    setup_logging(
        level=getattr(logging, args.log_level),
        log_file=args.log_file,
        log_file_level=getattr(logging, args.log_file_level) if args.log_file_level else None,
    )

    asr_client, asr_model, asr_base_url = get_asr_client(args.llm_asr, args.model)

    if args.llm_asr:
        # gemma4 / multimodal LLMs don't take Qwen3-ASR's canonical language names.
        # Pass the user's hint through unchanged; the model will most likely ignore it.
        pass
    else:
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

    if args.load_aligner:
        result = _server_request("POST", "/aligner/load")
        print(f"aligner {result.get('status', '?')}")
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

    if args.history < 0:
        parser.error("--history must be >= 0")
    if args.history_seconds < 0:
        parser.error("--history-seconds must be >= 0")

    if args.live:
        if args.context_src is not None:
            parser.error("--context-src is only supported with --input")
        if args.translate_prompt is not None and args.translate is None:
            parser.error("--translate-prompt requires --translate")
        if args.translate_system_prompt is not None and args.translate is None:
            parser.error("--translate-system-prompt requires --translate")
        if args.translate_system_prompt is not None and args.translate_prompt is not None:
            parser.error("--translate-system-prompt and --translate-prompt are mutually exclusive")
        if args.translate_history_seconds is not None:
            if args.translate is None:
                parser.error("--translate-history-seconds requires --translate")
            if args.translate_history_seconds < 0:
                parser.error("--translate-history-seconds must be >= 0")
        if args.translate_temperature != 0.0 and args.translate is None:
            parser.error("--translate-temperature requires --translate")
        if args.translate_temperature < 0:
            parser.error("--translate-temperature must be >= 0")
        # Tri-state: explicit --romanize/--no-romanize wins; otherwise default
        # on only for CJK source languages (anyascii on Latin-script sources is
        # rarely useful, and auto-detect can't be resolved up front).
        romanize = is_cjk(args.language) if args.romanize is None else args.romanize
        try:
            live_capture(
                asr_client=asr_client,
                asr_base_url=asr_base_url,
                model=asr_model,
                language=args.language,
                prompt=args.prompt,
                history=args.history,
                history_seconds=args.history_seconds,
                translate_target=args.translate,
                translate_prompt=args.translate_prompt,
                translate_system=args.translate_system_prompt,
                translate_history_seconds=args.translate_history_seconds,
                translate_temperature=args.translate_temperature,
                romanize=romanize,
            )
        except KeyboardInterrupt:
            log.info("stopped")
        except APIConnectionError:
            sys.exit(f"error: could not connect to transcription backend at {asr_base_url}")
    elif args.input is not None:
        if not args.input.exists():
            parser.error(f"File not found: {args.input}")
        if args.translate is not None:
            parser.error("--translate is only supported with --live")
        if args.translate_prompt is not None:
            parser.error("--translate-prompt is only supported with --live")
        if args.translate_system_prompt is not None:
            parser.error("--translate-system-prompt is only supported with --live")
        if args.translate_history_seconds is not None:
            parser.error("--translate-history-seconds is only supported with --live")
        if args.translate_temperature != 0.0:
            parser.error("--translate-temperature is only supported with --live")
        if args.romanize is not None:
            parser.error("--romanize/--no-romanize is only supported with --live")
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
            transcribe_file(args.input, asr_client=asr_client, model=asr_model, language=args.language, prompt=args.prompt, output=args.output, reference_srt=reference_srt, history=args.history, history_seconds=args.history_seconds, use_llm_asr=args.llm_asr)
        except APIConnectionError:
            sys.exit(f"error: could not connect to transcription backend at {asr_base_url}")
        except APIStatusError as exc:
            sys.exit(f"error: server returned {exc.status_code}: {exc.message}")
    else:
        parser.error("provide --input, --live, or a server management flag (--health, --load, --unload)")


if __name__ == "__main__":
    main()
