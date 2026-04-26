from __future__ import annotations

import gc
import os
import threading

import numpy as np

TRANSCRIPT_MODEL_ID = os.environ.get("TRANSCRIPT_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
TRANSCRIPT_ALIGNER_ID = os.environ.get("TRANSCRIPT_ALIGNER_ID", "Qwen/Qwen3-ForcedAligner-0.6B")
SAMPLE_RATE = 16000

_model: object | None = None
_timestamp_model: object | None = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()

SENTENCE_END_MARKERS = (".", "!", "?", "。", "！", "？")
CLOSING_PUNCTUATION = set(".,!?;:)]}、。，！？；：」』）》〉】")
OPENING_PUNCTUATION = set("([{'\"「『《〈【")


def _contains_cjk(value: str) -> bool:
    for ch in value:
        code = ord(ch)
        if (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0x3040 <= code <= 0x30FF
            or 0xF900 <= code <= 0xFAFF
        ):
            return True
    return False


def _join_tokens(tokens: list[str]) -> str:
    text = ""
    for token in tokens:
        piece = token.strip()
        if not piece:
            continue
        if not text:
            text = piece
            continue
        prev, nxt = text[-1], piece[0]
        if (
            nxt in CLOSING_PUNCTUATION
            or prev in OPENING_PUNCTUATION
            or (_contains_cjk(prev) and _contains_cjk(nxt))
        ):
            text += piece
        else:
            text += f" {piece}"
    return text.strip()


def _build_segments(words: list[dict]) -> list[dict]:
    segments: list[dict] = []
    current: list[dict] = []

    def flush():
        if not current:
            return
        text = _join_tokens([str(w.get("text", "") or "") for w in current])
        if text:
            segments.append({
                "start": round(float(current[0]["start"]), 3),
                "end": round(float(current[-1]["end"]), 3),
                "text": text,
            })
        current.clear()

    for word in words:
        token = str(word.get("text", "") or "")
        if current:
            gap = float(word["start"]) - float(current[-1]["end"])
            span = float(word["end"]) - float(current[0]["start"])
            if gap >= 1.0 or span >= 12.0:
                flush()
        current.append(word)
        if token.endswith(SENTENCE_END_MARKERS):
            flush()

    flush()
    return segments


def _load_asr() -> object:
    import torch
    from qwen_asr import Qwen3ASRModel

    kwargs: dict = {"max_new_tokens": 512}
    if torch.cuda.is_available():
        kwargs.update(device_map="cuda:0", dtype="bfloat16", attn_implementation="sdpa")
    else:
        kwargs.update(device_map="cpu", dtype="float32", attn_implementation="eager")
    return Qwen3ASRModel.from_pretrained(TRANSCRIPT_MODEL_ID, **kwargs)


def _load_timestamp() -> object:
    import torch
    from qwen_asr import Qwen3ASRModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if torch.cuda.is_available() else "float32"
    return Qwen3ASRModel.from_pretrained(
        TRANSCRIPT_MODEL_ID,
        forced_aligner=TRANSCRIPT_ALIGNER_ID,
        forced_aligner_kwargs={"device_map": device, "dtype": dtype},
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
        max_new_tokens=512,
    )


def get_model() -> object:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            _model = _load_asr()
    return _model


def get_timestamp_model() -> object:
    global _timestamp_model
    if _timestamp_model is not None:
        return _timestamp_model
    with _model_lock:
        if _timestamp_model is None:
            _timestamp_model = _load_timestamp()
    return _timestamp_model


def unload_model() -> None:
    global _model, _timestamp_model
    with _model_lock:
        _model = None
        _timestamp_model = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def transcribe(audio: np.ndarray, language: str | None = None, prompt: str | None = None) -> str:
    result = transcribe_result(audio, language=language, prompt=prompt, return_timestamps=False)
    return result["text"]


def transcribe_stream(
    audio: np.ndarray,
    language: str | None = None,
    prompt: str | None = None,
    return_timestamps: bool = False,
):
    """Yields (chunk_text, chunk_audio, chunk_offset, chunk_language) per ASR chunk.

    After the last chunk, yields a final (None, aligned_words, aligned_segments, full_text)
    tuple when return_timestamps=True, or (None, [], [], full_text) otherwise.
    """
    from qwen_asr.inference.utils import MAX_ASR_INPUT_SECONDS, split_audio_into_chunks

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        yield (None, [], [], "")
        return

    context = (
        "Transcribe the speech faithfully and conservatively. "
        "Preserve the language that is actually spoken. "
        "Do not translate, summarize, or rewrite."
    )
    if prompt:
        context += f" {prompt}"

    parts = split_audio_into_chunks(wav=audio, sr=SAMPLE_RATE, max_chunk_sec=MAX_ASR_INPUT_SECONDS)
    chunks = [(chunk.astype(np.float32, copy=False), SAMPLE_RATE) for chunk, _ in parts] or [(audio, SAMPLE_RATE)]
    offsets = [float(off) for _, off in parts] if parts else [0.0]

    model = get_timestamp_model() if return_timestamps else get_model()

    chunk_texts: list[str] = []
    chunk_langs: list[str] = []
    chunk_audios: list[np.ndarray] = []

    for (chunk_audio, _), offset in zip(chunks, offsets):
        with _infer_lock:
            results = model.transcribe([(chunk_audio, SAMPLE_RATE)], context=context, language=language or None)
        text = (getattr(results[0], "text", "") or "").strip() if results else ""
        lang = (getattr(results[0], "language", "") or "").strip() if results else ""
        chunk_texts.append(text)
        chunk_langs.append(lang)
        chunk_audios.append(chunk_audio)
        yield (text, chunk_audio, offset, lang or language)

    full_text = "".join(chunk_texts)

    if not return_timestamps:
        yield (None, [], [], full_text)
        return

    aligner = getattr(model, "forced_aligner", None)
    if aligner is None:
        yield (None, [], [], full_text)
        return

    audios_for_align = [(a, SAMPLE_RATE) for a in chunk_audios]
    with _infer_lock:
        aligned = aligner.align(
            audio=audios_for_align,
            text=chunk_texts,
            language=[l or "" for l in chunk_langs],
        )

    words: list[dict] = []
    for offset, result in zip(offsets, aligned):
        for item in getattr(result, "items", []):
            text = str(getattr(item, "text", "") or "").strip()
            start = round(float(getattr(item, "start_time", 0.0) or 0.0) + offset, 3)
            end = round(float(getattr(item, "end_time", 0.0) or 0.0) + offset, 3)
            if text or end > start:
                words.append({"text": text, "start": start, "end": end})

    yield (None, words, _build_segments(words), full_text)


def transcribe_result(
    audio: np.ndarray,
    language: str | None = None,
    prompt: str | None = None,
    return_timestamps: bool = False,
) -> dict:
    from qwen_asr.inference.utils import MAX_ASR_INPUT_SECONDS, split_audio_into_chunks

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return {"text": "", "language": None, "words": [], "segments": []}

    context = (
        "Transcribe the speech faithfully and conservatively. "
        "Preserve the language that is actually spoken. "
        "Do not translate, summarize, or rewrite."
    )
    if prompt:
        context += f" {prompt}"

    parts = split_audio_into_chunks(wav=audio, sr=SAMPLE_RATE, max_chunk_sec=MAX_ASR_INPUT_SECONDS)
    chunks = [(chunk.astype(np.float32, copy=False), SAMPLE_RATE) for chunk, _ in parts] or [(audio, SAMPLE_RATE)]
    offsets = [float(off) for _, off in parts] if parts else [0.0]

    if return_timestamps:
        model = get_timestamp_model()
        aligner = getattr(model, "forced_aligner", None)
        if aligner is None:
            raise ValueError(
                "timestamps require a forced aligner — set TRANSCRIPT_ALIGNER_ID or cache "
                "Qwen/Qwen3-ForcedAligner-0.6B locally"
            )
        with _infer_lock:
            asr_results = model.transcribe(chunks, context=context, language=language or None)

        texts = [(getattr(r, "text", "") or "").strip() for r in asr_results]
        langs = [(getattr(r, "language", "") or "").strip() for r in asr_results]
        full_text = "".join(texts)

        audios_for_align = [(chunk, SAMPLE_RATE) for chunk, _ in chunks]
        with _infer_lock:
            aligned = aligner.align(
                audio=audios_for_align,
                text=texts,
                language=[l or "" for l in langs],
            )

        words: list[dict] = []
        for offset_sec, result in zip(offsets, aligned):
            for item in getattr(result, "items", []):
                text = str(getattr(item, "text", "") or "").strip()
                start = round(float(getattr(item, "start_time", 0.0) or 0.0) + offset_sec, 3)
                end = round(float(getattr(item, "end_time", 0.0) or 0.0) + offset_sec, 3)
                if text or end > start:
                    words.append({"text": text, "start": start, "end": end})

        detected_language = next((l for l in langs if l), None)
        return {
            "text": full_text,
            "language": detected_language,
            "words": words,
            "segments": _build_segments(words),
        }

    with _infer_lock:
        results = get_model().transcribe(chunks, context=context, language=language or None)

    if not results:
        return {"text": "", "language": None, "words": [], "segments": []}

    texts = [(getattr(r, "text", "") or "").strip() for r in results]
    langs = [(getattr(r, "language", "") or "").strip() for r in results]
    return {
        "text": "".join(texts),
        "language": next((l for l in langs if l), None),
        "words": [],
        "segments": [],
    }
