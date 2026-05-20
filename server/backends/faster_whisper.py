from __future__ import annotations

import gc
import logging
import os
import threading

import numpy as np

from backends.base import Backend
from utils.language import to_iso_code
from utils.text import strip_hallucinations

log = logging.getLogger("subsvibe.faster_whisper")

TRANSCRIPT_MODEL_ID = os.environ.get("TRANSCRIPT_MODEL_ID", "Systran/faster-whisper-large-v3")
TRANSCRIPT_COMPUTE_TYPE = os.environ.get("TRANSCRIPT_COMPUTE_TYPE", "")
TRANSCRIPT_DEVICE = os.environ.get("TRANSCRIPT_DEVICE", "")
TRANSCRIPT_BEAM_SIZE = int(os.environ.get("TRANSCRIPT_BEAM_SIZE", "5"))
SAMPLE_RATE = 16000
MAX_INPUT_SECONDS = float(os.environ.get("TRANSCRIPT_MAX_INPUT_SECONDS", "180"))


def _resolve_device_compute() -> tuple[str, str]:
    if TRANSCRIPT_DEVICE:
        device = TRANSCRIPT_DEVICE
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if TRANSCRIPT_COMPUTE_TYPE:
        compute_type = TRANSCRIPT_COMPUTE_TYPE
    else:
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


class FasterWhisperBackend(Backend):
    def __init__(self) -> None:
        self._model: object | None = None
        self._model_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _load(self) -> object:
        from faster_whisper import WhisperModel

        device, compute_type = _resolve_device_compute()
        log.info("loading faster-whisper model %s (device=%s compute_type=%s)",
                 TRANSCRIPT_MODEL_ID, device, compute_type)
        return WhisperModel(TRANSCRIPT_MODEL_ID, device=device, compute_type=compute_type)

    def load(self) -> None:
        with self._model_lock:
            if self._model is None:
                self._model = self._load()

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        # CTranslate2 has its own CUDA allocator; torch.cuda.* calls don't free
        # its memory and can crash the process when invoked from a worker
        # thread that doesn't own the CUDA context. Hold _infer_lock so the
        # destructor can't run while a transcription is still using the model.
        with self._model_lock, self._infer_lock:
            model, self._model = self._model, None
        del model
        gc.collect()

    def load_aligner(self) -> None:
        return None

    def has_secondary(self) -> bool:
        return False

    def unload_secondary(self) -> None:
        return None

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                self._model = self._load()
        return self._model

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
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return {"text": "", "language": None, "words": [], "segments": []}

        duration = audio.size / SAMPLE_RATE
        if duration > MAX_INPUT_SECONDS:
            raise ValueError(
                f"audio is {duration:.1f}s, exceeds server max {MAX_INPUT_SECONDS:.0f}s - split on the client"
            )

        iso_language = to_iso_code(language)
        cleaned_prompt = strip_hallucinations(prompt) if prompt else prompt
        if cleaned_prompt != prompt:
            log.warning("prompt contained hallucination patterns; cleaned %d -> %d chars", len(prompt or ""), len(cleaned_prompt or ""))
        log.info("transcribe language=%s prompt=%r", iso_language or "auto", cleaned_prompt or None)
        model = self._get_model()
        with self._infer_lock:
            segments_iter, info = model.transcribe(
                audio,
                language=iso_language,
                initial_prompt=cleaned_prompt or None,
                condition_on_previous_text=True,
                beam_size=TRANSCRIPT_BEAM_SIZE,
                word_timestamps=want_words,
            )
            segments_list = list(segments_iter)

        # Segment timestamps are free with faster-whisper - always emit them.
        # word_timestamps=True adds a DTW alignment pass (~10-30% slower).
        text_parts: list[str] = []
        out_segments: list[dict] = []
        out_words: list[dict] = []
        for seg in segments_list:
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
