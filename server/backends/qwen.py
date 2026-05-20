from __future__ import annotations

import gc
import logging
import os
import threading

import numpy as np

from backends._qwen_aligner import (
    QwenAligner,
    segments_from_words,
    strip_word_trailing,
)
from backends.base import Backend
from utils.language import to_canonical_name
from utils.text import strip_hallucinations

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

log = logging.getLogger("subsvibe.qwen")


def _log_gpu_mem(tag: str) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            log.info("gpu mem %s: allocated=%.1fMB reserved=%.1fMB", tag, allocated, reserved)
    except ImportError:
        pass


def _release_gpu(model: object | None) -> None:
    if model is not None and hasattr(model, "to"):
        try:
            model.to("cpu")
        except Exception as exc:
            log.warning("failed to move model to CPU before unload: %s", exc)
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass


TRANSCRIPT_MODEL_ID = os.environ.get("TRANSCRIPT_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
SAMPLE_RATE = 16000
MAX_INPUT_SECONDS = float(os.environ.get("TRANSCRIPT_MAX_INPUT_SECONDS", "180"))


class QwenBackend(Backend):
    def __init__(self) -> None:
        self._model: object | None = None
        self._model_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._aligner = QwenAligner(self._infer_lock)

    def _load_asr(self) -> object:
        import torch
        from qwen_asr import Qwen3ASRModel

        kwargs: dict = {"max_new_tokens": 512}
        if torch.cuda.is_available():
            kwargs.update(device_map="cuda:0", dtype="bfloat16", attn_implementation="sdpa")
        else:
            kwargs.update(device_map="cpu", dtype="float32", attn_implementation="eager")
        return Qwen3ASRModel.from_pretrained(TRANSCRIPT_MODEL_ID, **kwargs)

    def load(self) -> None:
        with self._model_lock:
            if self._model is None:
                self._model = self._load_asr()

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        _log_gpu_mem("before ASR unload")
        with self._model_lock:
            model, self._model = self._model, None
        _release_gpu(model)
        _log_gpu_mem("after ASR unload")

    def load_aligner(self) -> None:
        self._aligner.load()

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                self._model = self._load_asr()
        return self._model

    def has_secondary(self) -> bool:
        return self._aligner.is_loaded()

    def unload_secondary(self) -> None:
        self._aligner.unload()

    def align(
        self,
        audio: np.ndarray,
        text: str,
        language: str | None,
    ) -> list[dict]:
        return self._aligner.align_one(audio, text, language, MAX_INPUT_SECONDS)

    def transcribe_result(
        self,
        audio: np.ndarray,
        language: str | None,
        prompt: str | None,
        want_words: bool,
    ) -> dict:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return {"text": "", "language": None, "words": [], "segments": []}

        duration = audio.size / SAMPLE_RATE
        if duration > MAX_INPUT_SECONDS:
            raise ValueError(
                f"audio is {duration:.1f}s, exceeds server max {MAX_INPUT_SECONDS:.0f}s - split on the client"
            )

        chunks = [(audio, SAMPLE_RATE)]
        canonical_language = to_canonical_name(language)

        with self._infer_lock:
            results = self._get_model().transcribe(chunks, context=prompt, language=canonical_language)

        if not results:
            return {"text": "", "language": None, "words": [], "segments": []}

        texts = [strip_hallucinations((getattr(r, "text", "") or "").strip()) for r in results]
        langs = [(getattr(r, "language", "") or "").strip() for r in results]
        full_text = "".join(texts)

        if not want_words:
            return {
                "text": full_text,
                "language": next((l for l in langs if l), None),
                "words": [],
                "segments": [],
            }

        align_langs = [l or "" for l in langs]
        words = self._aligner.align_chunks(chunks, texts, align_langs)
        segments = segments_from_words(words, full_text)
        return {
            "text": full_text,
            "language": next((l for l in langs if l), None),
            "words": strip_word_trailing(words),
            "segments": segments,
        }
