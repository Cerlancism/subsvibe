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
from utils.text import strip_hallucinations

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

log = logging.getLogger("subsvibe.anime_whisper")

TRANSCRIPT_MODEL_ID = os.environ.get("TRANSCRIPT_MODEL_ID", "litagin/anime-whisper")
SAMPLE_RATE = 16000
MAX_INPUT_SECONDS = float(os.environ.get("TRANSCRIPT_MAX_INPUT_SECONDS", "180"))

# README: no_repeat_ngram_size=5 is the recommended value to suppress
# whisper-family repetition hallucinations. repetition_penalty stays at 1.0.
NO_REPEAT_NGRAM_SIZE = int(os.environ.get("TRANSCRIPT_NO_REPEAT_NGRAM_SIZE", "5"))
REPETITION_PENALTY = float(os.environ.get("TRANSCRIPT_REPETITION_PENALTY", "1.0"))
CHUNK_LENGTH_S = float(os.environ.get("TRANSCRIPT_CHUNK_LENGTH_S", "30.0"))
BATCH_SIZE = int(os.environ.get("TRANSCRIPT_BATCH_SIZE", "16"))


def _log_gpu_mem(tag: str) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            log.info("gpu mem %s: allocated=%.1fMB reserved=%.1fMB", tag, allocated, reserved)
    except ImportError:
        pass


def _release() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass


class AnimeWhisperBackend(Backend):
    """Japanese anime-domain ASR (litagin/anime-whisper).

    Built on a fine-tuned kotoba-whisper-v2.0 with no native word-level
    aligner, so word/segment timestamps are produced by composing the
    Qwen3-ForcedAligner. The model is Japanese-only - the `language`
    argument is ignored on the wire and forced to Japanese."""

    def __init__(self) -> None:
        self._pipe: object | None = None
        self._model_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._aligner = QwenAligner(self._infer_lock)

    def _load(self) -> object:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        log.info(
            "loading anime-whisper %s (device=%s dtype=%s chunk=%.1fs batch=%d)",
            TRANSCRIPT_MODEL_ID, device, dtype, CHUNK_LENGTH_S, BATCH_SIZE,
        )
        return pipeline(
            "automatic-speech-recognition",
            model=TRANSCRIPT_MODEL_ID,
            device=device,
            torch_dtype=dtype,
            chunk_length_s=CHUNK_LENGTH_S,
            batch_size=BATCH_SIZE,
        )

    def load(self) -> None:
        with self._model_lock:
            if self._pipe is None:
                self._pipe = self._load()

    def is_loaded(self) -> bool:
        return self._pipe is not None

    def unload(self) -> None:
        _log_gpu_mem("before ASR unload")
        with self._model_lock:
            self._pipe = None
        _release()
        _log_gpu_mem("after ASR unload")

    def _get_pipe(self) -> object:
        if self._pipe is not None:
            return self._pipe
        with self._model_lock:
            if self._pipe is None:
                self._pipe = self._load()
        return self._pipe

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
        # Anime-whisper is Japanese-only; force the aligner language to match.
        del language
        return self._aligner.align_one(audio, text, "ja", MAX_INPUT_SECONDS)

    def transcribe_result(
        self,
        audio: np.ndarray,
        language: str | None,
        prompt: str | None,
        want_words: bool,
    ) -> dict:
        # README explicitly warns: initial prompt causes hallucinations and
        # degrades quality on this model. Drop it.
        del language, prompt

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return {"text": "", "language": None, "words": [], "segments": []}

        duration = audio.size / SAMPLE_RATE
        if duration > MAX_INPUT_SECONDS:
            raise ValueError(
                f"audio is {duration:.1f}s, exceeds server max {MAX_INPUT_SECONDS:.0f}s - split on the client"
            )

        generate_kwargs = {
            "language": "Japanese",
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            "repetition_penalty": REPETITION_PENALTY,
        }

        pipe = self._get_pipe()
        with self._infer_lock:
            result = pipe(audio, generate_kwargs=generate_kwargs)

        raw_text = (result.get("text") if isinstance(result, dict) else "") or ""
        full_text = strip_hallucinations(raw_text.strip())

        if not want_words:
            return {
                "text": full_text,
                "language": "ja",
                "words": [],
                "segments": [],
            }

        chunks = [(audio, SAMPLE_RATE)]
        words = self._aligner.align_chunks(chunks, [full_text], ["Japanese"])
        segments = segments_from_words(words, full_text)
        return {
            "text": full_text,
            "language": "ja",
            "words": strip_word_trailing(words),
            "segments": segments,
        }
