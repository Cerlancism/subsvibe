"""Qwen3-ASR backend, isolated in a child process.

ASR runs in its own worker; the forced aligner runs in a separate worker
owned by this backend. `_infer_lock` is held parent-side to serialise GPU
work across the two so they don't fight for VRAM."""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

from backends._qwen_aligner import (
    QwenAligner,
    segments_from_words,
    strip_word_trailing,
)
from backends.base import Backend
from utils.language import to_canonical_name
from utils.text import strip_hallucinations
from worker import ModelWorker

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

log = logging.getLogger("subsvibe.qwen")

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------


class _QwenASRChild:
    def __init__(self) -> None:
        from utils.logging_config import setup_logging
        setup_logging()
        self._log = logging.getLogger("subsvibe.qwen.child")
        self._model_id = os.environ.get("TRANSCRIPT_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
        self._max_input_seconds = float(os.environ.get("TRANSCRIPT_MAX_INPUT_SECONDS", "180"))

        import torch
        from qwen_asr import Qwen3ASRModel

        kwargs: dict[str, Any] = {"max_new_tokens": 512}
        if torch.cuda.is_available():
            kwargs.update(device_map="cuda:0", dtype="bfloat16", attn_implementation="sdpa")
        else:
            kwargs.update(device_map="cpu", dtype="float32", attn_implementation="eager")
        self._log.info("loading qwen3-asr %s", self._model_id)
        self._model = Qwen3ASRModel.from_pretrained(self._model_id, **kwargs)
        self._log.info("qwen3-asr ready")

    def max_input_seconds(self) -> float:
        return self._max_input_seconds

    def transcribe(
        self,
        audio: np.ndarray,
        canonical_language: str | None,
        prompt: str | None,
    ) -> dict[str, Any]:
        """Returns {"texts": [...], "languages": [...]} - one entry per
        chunk. Chunking is currently always [audio], but the shape matches
        the model API for future expansion."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return {"texts": [], "languages": []}

        duration = audio.size / SAMPLE_RATE
        if duration > self._max_input_seconds:
            raise ValueError(
                f"audio is {duration:.1f}s, exceeds server max {self._max_input_seconds:.0f}s - split on the client"
            )

        chunks = [(audio, SAMPLE_RATE)]
        results = self._model.transcribe(chunks, context=prompt, language=canonical_language)
        texts = [(getattr(r, "text", "") or "").strip() for r in results]
        languages = [(getattr(r, "language", "") or "").strip() for r in results]
        return {"texts": texts, "languages": languages}


def _qwen_child_entry() -> _QwenASRChild:
    return _QwenASRChild()


# ---------------------------------------------------------------------------
# Parent process
# ---------------------------------------------------------------------------


class QwenBackend(Backend):
    def __init__(self) -> None:
        # Shared lock serialises ASR + aligner GPU work in the parent.
        self._infer_lock = threading.Lock()
        self._worker = ModelWorker(_qwen_child_entry, name="qwen-asr")
        self._aligner = QwenAligner(self._infer_lock)

    def load(self) -> None:
        self._worker.start()

    def is_loaded(self) -> bool:
        return self._worker.is_alive()

    def unload(self) -> None:
        self._worker.stop()

    def load_aligner(self) -> None:
        self._aligner.load()

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
        # MAX_INPUT_SECONDS is enforced in the child for transcription; the
        # aligner reuses the same env-driven bound.
        max_seconds = float(os.environ.get("TRANSCRIPT_MAX_INPUT_SECONDS", "180"))
        return self._aligner.align_one(audio, text, language, max_seconds)

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

        canonical_language = to_canonical_name(language)
        if not self._worker.is_alive():
            self._worker.start()

        with self._infer_lock:
            result = self._worker.call("transcribe", audio, canonical_language, prompt)

        texts_raw = result.get("texts", [])
        langs = result.get("languages", [])
        if not texts_raw:
            return {"text": "", "language": None, "words": [], "segments": []}

        texts = [strip_hallucinations(t) for t in texts_raw]
        full_text = "".join(texts)
        detected_lang = next((l for l in langs if l), None)

        if not want_words:
            return {
                "text": full_text,
                "language": detected_lang,
                "words": [],
                "segments": [],
            }

        # Word alignment needs the same chunked audio the ASR saw.
        chunks = [(audio, SAMPLE_RATE)]
        align_langs = [l or "" for l in langs]
        words = self._aligner.align_chunks(chunks, texts, align_langs)
        segments = segments_from_words(words, full_text)
        return {
            "text": full_text,
            "language": detected_lang,
            "words": strip_word_trailing(words),
            "segments": segments,
        }
