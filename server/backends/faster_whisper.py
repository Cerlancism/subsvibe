"""Faster-whisper backend, isolated in a child process.

The CTranslate2 WhisperModel cannot be safely destructed in-process (the
destructor on a worker thread SIGSEGVs on Windows). To make `--unload`
actually work, the model lives in a spawn'd subprocess and `unload` kills
it; the OS reclaims GPU memory unconditionally.

This file has two halves:
  - parent side: FasterWhisperBackend, a thin dispatcher that forwards
    transcribe/load/unload through a ModelWorker.
  - child side: FasterWhisperChild, the actual WhisperModel host. Imports
    faster-whisper / torch only when constructed (i.e. in the child).
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from backends.base import Backend
from utils.language import to_iso_code
from utils.text import strip_hallucinations
from worker import ModelWorker

log = logging.getLogger("subsvibe.faster_whisper")

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------

def _resolve_device_compute() -> tuple[str, str]:
    device_env = os.environ.get("TRANSCRIPT_DEVICE", "")
    compute_env = os.environ.get("TRANSCRIPT_COMPUTE_TYPE", "")
    if device_env:
        device = device_env
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    compute_type = compute_env or ("float16" if device == "cuda" else "int8")
    return device, compute_type


class _FasterWhisperChild:
    """In-subprocess WhisperModel host. One instance per worker."""

    def __init__(self) -> None:
        from utils.logging_config import setup_logging
        setup_logging()
        self._log = logging.getLogger("subsvibe.faster_whisper.child")
        self._model_id = os.environ.get("TRANSCRIPT_MODEL_ID", "Systran/faster-whisper-large-v3")
        self._beam_size = int(os.environ.get("TRANSCRIPT_BEAM_SIZE", "5"))
        self._max_input_seconds = float(os.environ.get("TRANSCRIPT_MAX_INPUT_SECONDS", "180"))

        from faster_whisper import WhisperModel

        device, compute_type = _resolve_device_compute()
        self._log.info(
            "loading faster-whisper model %s (device=%s compute_type=%s)",
            self._model_id, device, compute_type,
        )
        self._model = WhisperModel(self._model_id, device=device, compute_type=compute_type)
        self._log.info("faster-whisper model ready")

    def transcribe_result(
        self,
        audio: np.ndarray,
        language: str | None,
        prompt: str | None,
        want_words: bool,
    ) -> dict[str, Any]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return {"text": "", "language": None, "words": [], "segments": []}

        duration = audio.size / SAMPLE_RATE
        if duration > self._max_input_seconds:
            raise ValueError(
                f"audio is {duration:.1f}s, exceeds server max {self._max_input_seconds:.0f}s "
                "- split on the client"
            )

        iso_language = to_iso_code(language)
        cleaned_prompt = strip_hallucinations(prompt) if prompt else prompt
        if cleaned_prompt != prompt:
            self._log.warning(
                "prompt contained hallucination patterns; cleaned %d -> %d chars",
                len(prompt or ""), len(cleaned_prompt or ""),
            )
        self._log.info("transcribe language=%s prompt=%r", iso_language or "auto", cleaned_prompt or None)

        segments_iter, info = self._model.transcribe(
            audio,
            language=iso_language,
            initial_prompt=cleaned_prompt or None,
            condition_on_previous_text=True,
            beam_size=self._beam_size,
            word_timestamps=want_words,
        )

        text_parts: list[str] = []
        out_segments: list[dict] = []
        out_words: list[dict] = []
        for seg in segments_iter:
            seg_text = (seg.text or "").strip()
            text_parts.append(seg.text or "")
            out_segments.append({
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": seg_text,
            })
            if want_words and getattr(seg, "words", None):
                for w in seg.words:
                    out_words.append({
                        "word": (w.word or "").strip(),
                        "start": round(float(w.start), 3),
                        "end": round(float(w.end), 3),
                    })

        full_text = "".join(text_parts).strip()
        detected_lang = getattr(info, "language", None) if info else None
        return {
            "text": full_text,
            "language": detected_lang,
            "words": out_words,
            "segments": out_segments,
        }


def _child_entry() -> _FasterWhisperChild:
    """Top-level callable passed to ModelWorker. Must be picklable, which
    is why it's a module-level function (not a bound method or lambda)."""
    return _FasterWhisperChild()


# ---------------------------------------------------------------------------
# Parent process
# ---------------------------------------------------------------------------


class FasterWhisperBackend(Backend):
    def __init__(self) -> None:
        self._worker = ModelWorker(_child_entry, name="faster-whisper")

    def load(self) -> None:
        self._worker.start()

    def is_loaded(self) -> bool:
        return self._worker.is_alive()

    def unload(self) -> None:
        self._worker.stop()

    def load_aligner(self) -> None:
        return None

    def has_secondary(self) -> bool:
        return False

    def unload_secondary(self) -> None:
        return None

    def align(
        self,
        audio: np.ndarray,
        text: str,
        language: str | None,
    ) -> list[dict]:
        raise NotImplementedError(
            "faster-whisper backend does not support standalone alignment; "
            "request word/segment timestamps via /v1/audio/transcriptions instead"
        )

    def transcribe_result(
        self,
        audio: np.ndarray,
        language: str | None,
        prompt: str | None,
        want_words: bool,
    ) -> dict:
        if not self._worker.is_alive():
            self._worker.start()
        return self._worker.call(
            "transcribe_result", audio, language, prompt, want_words,
        )
