from __future__ import annotations

import gc
import logging
import os
import threading

import numpy as np

from utils.language import to_canonical_name

log = logging.getLogger("subsvibe.qwen_aligner")

TRANSCRIPT_ALIGNER_ID = os.environ.get("TRANSCRIPT_ALIGNER_ID", "Qwen/Qwen3-ForcedAligner-0.6B")
SAMPLE_RATE = 16000


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
            log.warning("failed to move aligner to CPU before unload: %s", exc)
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass


class QwenAligner:
    """Forced aligner wrapper around Qwen3-ForcedAligner.

    Reusable across backends that lack a native aligner. Owns its own
    load/unload lifecycle. Inference is serialised through an externally
    supplied lock so it can share a GPU queue with an ASR model."""

    def __init__(self, infer_lock: threading.Lock) -> None:
        self._model: object | None = None
        self._lock = threading.Lock()
        self._infer_lock = infer_lock

    def _load(self) -> object:
        import torch
        from qwen_asr import Qwen3ForcedAligner

        kwargs: dict = {}
        if torch.cuda.is_available():
            kwargs.update(device_map="cuda:0", dtype="bfloat16", attn_implementation="sdpa")
        else:
            kwargs.update(device_map="cpu", dtype="float32", attn_implementation="eager")
        return Qwen3ForcedAligner.from_pretrained(TRANSCRIPT_ALIGNER_ID, **kwargs)

    def load(self) -> None:
        self._get()

    def _get(self) -> object:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                self._model = self._load()
        return self._model

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        _log_gpu_mem("before aligner unload")
        with self._lock:
            model, self._model = self._model, None
        _release_gpu(model)
        _log_gpu_mem("after aligner unload")

    def align_chunks(
        self,
        chunks: list[tuple[np.ndarray, int]],
        texts: list[str],
        languages: list[str],
    ) -> list[dict]:
        """Run the forced aligner on already-decoded audio chunks.
        Returns flat [{"text", "start", "end"}, ...] across all results."""
        with self._infer_lock:
            aligned = self._get().align(audio=chunks, text=texts, language=languages)

        words: list[dict] = []
        for result in aligned:
            for item in getattr(result, "items", []):
                text = str(getattr(item, "text", "") or "").strip()
                start = round(float(getattr(item, "start_time", 0.0) or 0.0), 3)
                end = round(float(getattr(item, "end_time", 0.0) or 0.0), 3)
                if text or end > start:
                    words.append({"text": text, "start": start, "end": end})
        return words

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
