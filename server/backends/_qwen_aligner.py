"""Qwen3-ForcedAligner, isolated in a child process.

Same rationale as faster_whisper.py: deleting a GPU-bound HuggingFace
model from a worker thread leaves dangling CUDA references. We host it
in a spawn'd subprocess and reclaim VRAM by killing the child.

Parent side: QwenAligner — dispatcher used by qwen / anime-whisper.
Child side : _QwenAlignerChild — loads Qwen3ForcedAligner and serves
align requests.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

from utils.language import to_canonical_name
from worker import ModelWorker

log = logging.getLogger("subsvibe.qwen_aligner")

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------


class _QwenAlignerChild:
    def __init__(self) -> None:
        from utils.logging_config import setup_logging
        setup_logging()
        self._log = logging.getLogger("subsvibe.qwen_aligner.child")
        self._model_id = os.environ.get("TRANSCRIPT_ALIGNER_ID", "Qwen/Qwen3-ForcedAligner-0.6B")

        import torch
        from qwen_asr import Qwen3ForcedAligner

        kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            kwargs.update(device_map="cuda:0", dtype="bfloat16", attn_implementation="sdpa")
        else:
            kwargs.update(device_map="cpu", dtype="float32", attn_implementation="eager")
        self._log.info("loading qwen aligner %s", self._model_id)
        self._model = Qwen3ForcedAligner.from_pretrained(self._model_id, **kwargs)
        self._log.info("qwen aligner ready")

    def align(
        self,
        chunks: list[tuple[np.ndarray, int]],
        texts: list[str],
        languages: list[str],
    ) -> list[dict]:
        aligned = self._model.align(audio=chunks, text=texts, language=languages)
        words: list[dict] = []
        for result in aligned:
            for item in getattr(result, "items", []):
                text = str(getattr(item, "text", "") or "").strip()
                start = round(float(getattr(item, "start_time", 0.0) or 0.0), 3)
                end = round(float(getattr(item, "end_time", 0.0) or 0.0), 3)
                if text or end > start:
                    words.append({"text": text, "start": start, "end": end})
        return words


def _aligner_child_entry() -> _QwenAlignerChild:
    return _QwenAlignerChild()


# ---------------------------------------------------------------------------
# Parent process
# ---------------------------------------------------------------------------


class QwenAligner:
    """Parent-side dispatcher for the forced aligner.

    Inference is serialised through an externally supplied lock so the
    aligner shares a GPU queue with the ASR backend that owns it."""

    def __init__(self, infer_lock: threading.Lock) -> None:
        self._infer_lock = infer_lock
        self._worker = ModelWorker(_aligner_child_entry, name="qwen-aligner")

    def load(self) -> None:
        self._worker.start()

    def is_loaded(self) -> bool:
        return self._worker.is_alive()

    def unload(self) -> None:
        self._worker.stop()

    def align_chunks(
        self,
        chunks: list[tuple[np.ndarray, int]],
        texts: list[str],
        languages: list[str],
    ) -> list[dict]:
        """Run the forced aligner on already-decoded audio chunks.
        Returns flat [{"text", "start", "end"}, ...] across all results."""
        if not self._worker.is_alive():
            self._worker.start()
        with self._infer_lock:
            return self._worker.call("align", chunks, texts, languages)

    def align_one(
        self,
        audio: np.ndarray,
        text: str,
        language: str | None,
        max_input_seconds: float,
    ) -> list[dict]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not text.strip():
            return []

        duration = audio.size / SAMPLE_RATE
        if duration > max_input_seconds:
            raise ValueError(
                f"audio is {duration:.1f}s, exceeds server max {max_input_seconds:.0f}s - split on the client"
            )

        chunks = [(audio, SAMPLE_RATE)]
        return self.align_chunks(chunks, [text], [to_canonical_name(language) or ""])


def strip_word_trailing(words: list[dict]) -> list[dict]:
    """Drop the SubsVibe-specific `trailing` field; rename `text`->`word`
    to match the OpenAI `TranscriptionWord` schema."""
    return [
        {"word": w.get("text", ""), "start": w["start"], "end": w["end"]}
        for w in words
    ]


def segments_from_words(words: list[dict], full_text: str) -> list[dict]:
    """Default segment from aligner output: one segment spanning the
    aligner's first->last item, carrying the full ASR text."""
    if not words:
        return []
    text = (full_text or "").strip()
    if not text:
        return []
    return [{
        "start": round(float(words[0]["start"]), 3),
        "end": round(float(words[-1]["end"]), 3),
        "text": text,
    }]
