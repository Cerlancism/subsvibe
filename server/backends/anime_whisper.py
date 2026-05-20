"""Anime-whisper backend, isolated in a child process.

Japanese-only ASR fine-tune on top of kotoba-whisper-v2.0. No native
word-level aligner, so word/segment timestamps come from the shared
Qwen3-ForcedAligner (`QwenAligner`)."""
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
from utils.text import strip_hallucinations
from worker import ModelWorker

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

log = logging.getLogger("subsvibe.anime_whisper")

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------


class _AnimeWhisperChild:
    def __init__(self) -> None:
        from utils.logging_config import setup_logging
        setup_logging()
        self._log = logging.getLogger("subsvibe.anime_whisper.child")
        self._model_id = os.environ.get("TRANSCRIPT_MODEL_ID", "litagin/anime-whisper")
        self._max_input_seconds = float(os.environ.get("TRANSCRIPT_MAX_INPUT_SECONDS", "180"))
        self._no_repeat_ngram_size = int(os.environ.get("TRANSCRIPT_NO_REPEAT_NGRAM_SIZE", "5"))
        self._repetition_penalty = float(os.environ.get("TRANSCRIPT_REPETITION_PENALTY", "1.0"))
        self._chunk_length_s = float(os.environ.get("TRANSCRIPT_CHUNK_LENGTH_S", "30.0"))
        self._batch_size = int(os.environ.get("TRANSCRIPT_BATCH_SIZE", "16"))

        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._log.info(
            "loading anime-whisper %s (device=%s dtype=%s chunk=%.1fs batch=%d)",
            self._model_id, device, dtype, self._chunk_length_s, self._batch_size,
        )
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self._model_id,
            device=device,
            torch_dtype=dtype,
            chunk_length_s=self._chunk_length_s,
            batch_size=self._batch_size,
        )
        self._log.info("anime-whisper ready")

    def transcribe(self, audio: np.ndarray) -> dict[str, Any]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return {"text": ""}

        duration = audio.size / SAMPLE_RATE
        if duration > self._max_input_seconds:
            raise ValueError(
                f"audio is {duration:.1f}s, exceeds server max {self._max_input_seconds:.0f}s - split on the client"
            )

        generate_kwargs = {
            "language": "Japanese",
            "no_repeat_ngram_size": self._no_repeat_ngram_size,
            "repetition_penalty": self._repetition_penalty,
        }
        result = self._pipe(audio, generate_kwargs=generate_kwargs)
        text = (result.get("text") if isinstance(result, dict) else "") or ""
        return {"text": text}


def _anime_child_entry() -> _AnimeWhisperChild:
    return _AnimeWhisperChild()


# ---------------------------------------------------------------------------
# Parent process
# ---------------------------------------------------------------------------


class AnimeWhisperBackend(Backend):
    """Japanese anime-domain ASR. Language argument is ignored on the wire
    and forced to Japanese throughout."""

    def __init__(self) -> None:
        self._infer_lock = threading.Lock()
        self._worker = ModelWorker(_anime_child_entry, name="anime-whisper")
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
        del language  # Japanese-only
        max_seconds = float(os.environ.get("TRANSCRIPT_MAX_INPUT_SECONDS", "180"))
        return self._aligner.align_one(audio, text, "ja", max_seconds)

    def transcribe_result(
        self,
        audio: np.ndarray,
        language: str | None,
        prompt: str | None,
        want_words: bool,
    ) -> dict:
        # README warns: initial prompts cause hallucinations on this model.
        del language, prompt

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return {"text": "", "language": None, "words": [], "segments": []}

        if not self._worker.is_alive():
            self._worker.start()

        with self._infer_lock:
            result = self._worker.call("transcribe", audio)

        raw_text = result.get("text", "") or ""
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
