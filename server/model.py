from __future__ import annotations

import gc
import os
import threading

import numpy as np

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
SAMPLE_RATE = 16000

_model = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()


def _load() -> object:
    import torch
    from qwen_asr import Qwen3ASRModel

    kwargs: dict = {"max_new_tokens": 512}
    if torch.cuda.is_available():
        kwargs.update(device_map="cuda:0", dtype="bfloat16", attn_implementation="sdpa")
    else:
        kwargs.update(device_map="cpu", dtype="float32", attn_implementation="eager")

    return Qwen3ASRModel.from_pretrained(MODEL_ID, **kwargs)


def get_model() -> object:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            _model = _load()
    return _model


def unload_model() -> None:
    global _model
    with _model_lock:
        _model = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def transcribe(audio: np.ndarray, language: str | None = None, prompt: str | None = None) -> str:
    from qwen_asr.inference.utils import MAX_ASR_INPUT_SECONDS, split_audio_into_chunks

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return ""

    context = (
        "Transcribe the speech faithfully and conservatively. "
        "Preserve the language that is actually spoken. "
        "Do not translate, summarize, or rewrite."
    )
    if prompt:
        context += f" {prompt}"

    parts = split_audio_into_chunks(wav=audio, sr=SAMPLE_RATE, max_chunk_sec=MAX_ASR_INPUT_SECONDS)
    chunks = [(chunk.astype(np.float32, copy=False), SAMPLE_RATE) for chunk, _ in parts] or [(audio, SAMPLE_RATE)]

    with _infer_lock:
        results = get_model().transcribe(chunks, context=context, language=language or None)

    if not results:
        return ""
    return "".join((getattr(r, "text", "") or "").strip() for r in results)
