from __future__ import annotations

import gc
import logging
import os
import threading
from typing import Iterator

import numpy as np

from backends.base import Backend
# from utils.text import (
#     CLOSING_PUNCTUATION,
#     OPENING_PUNCTUATION,
#     SENTENCE_END_MARKERS,
#     SOFT_BREAK_MARKERS,
#     attach_punctuation,
#     contains_cjk,
#     is_overlong,
# )

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
TRANSCRIPT_ALIGNER_ID = os.environ.get("TRANSCRIPT_ALIGNER_ID", "Qwen/Qwen3-ForcedAligner-0.6B")
SAMPLE_RATE = 16000
MAX_INPUT_SECONDS = float(os.environ.get("TRANSCRIPT_MAX_INPUT_SECONDS", "180"))

HALLUCINATION_REPEAT_THRESHOLD = 20
HALLUCINATION_PATTERN_MAX_LEN = 20

ASR_SYSTEM_CONTEXT = (
    "Transcribe all dialogue. "
    "Do not translate, summarize, or rewrite."
)


def _build_asr_context(prompt: str | None) -> str:
    return prompt if prompt else ASR_SYSTEM_CONTEXT


def _strip_hallucinations(text: str, threshold: int = HALLUCINATION_REPEAT_THRESHOLD) -> str:
    if not text:
        return text

    def fix_char_repeats(s: str, thresh: int) -> str:
        out: list[str] = []
        i, n = 0, len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                out.append(s[i])
            else:
                out.append(s[i:i + count])
            i += count
        return "".join(out)

    def fix_pattern_repeats(s: str, thresh: int, max_len: int = HALLUCINATION_PATTERN_MAX_LEN) -> str:
        n = len(s)
        if n < thresh * 2:
            return s
        i = 0
        out: list[str] = []
        while i <= n - thresh * 2:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i:i + k]
                if not all(s[i + r * k:i + r * k + k] == pattern for r in range(1, thresh)):
                    continue
                end = i + thresh * k
                while end + k <= n and s[end:end + k] == pattern:
                    end += k
                out.append(pattern)
                out.append(fix_pattern_repeats(s[end:], thresh, max_len))
                i = n
                found = True
                break
            if not found:
                out.append(s[i])
                i += 1
        if i < n:
            out.append(s[i:])
        return "".join(out)

    return fix_pattern_repeats(fix_char_repeats(text, threshold), threshold)


# def _join_tokens(tokens: list[str]) -> str:
#     text = ""
#     for token in tokens:
#         piece = token.strip()
#         if not piece:
#             continue
#         if not text:
#             text = piece
#             continue
#         prev, nxt = text[-1], piece[0]
#         if (
#             nxt in CLOSING_PUNCTUATION
#             or prev in OPENING_PUNCTUATION
#             or (contains_cjk(prev) and contains_cjk(nxt))
#         ):
#             text += piece
#         else:
#             text += f" {piece}"
#     return text.strip()
#
#
# def _accumulated_text(current: list[dict]) -> str:
#     parts: list[str] = []
#     for w in current:
#         parts.append(str(w.get("text", "") or ""))
#         trailing = str(w.get("trailing", "") or "")
#         if trailing:
#             parts.append(trailing)
#     return _join_tokens(parts).strip()
#
#
# def _endswith_any(s: str, markers: frozenset[str]) -> bool:
#     return bool(s) and s[-1] in markers


def _strip_trailing(words: list[dict]) -> list[dict]:
    """Drop the SubsVibe-specific `trailing` field; rename `text`→`word` to
    match the OpenAI `TranscriptionWord` schema. The client re-attaches
    punctuation from the response `text` field."""
    return [
        {"word": w.get("text", ""), "start": w["start"], "end": w["end"]}
        for w in words
    ]


# def _build_segments(enriched: list[dict]) -> list[dict]:
#     segments: list[dict] = []
#     current: list[dict] = []
#
#     def flush():
#         if not current:
#             return
#         text = _accumulated_text(current)
#         if text:
#             segments.append({
#                 "start": round(float(current[0]["start"]), 3),
#                 "end": round(float(current[-1]["end"]), 3),
#                 "text": text,
#             })
#         current.clear()
#
#     for word in enriched:
#         if current:
#             gap = float(word["start"]) - float(current[-1]["end"])
#             span = float(word["end"]) - float(current[0]["start"])
#             if gap >= 1.0 or span >= 12.0:
#                 flush()
#         current.append(word)
#         trailing = str(word.get("trailing", "") or "").rstrip()
#         if _endswith_any(trailing, SENTENCE_END_MARKERS):
#             flush()
#         elif _endswith_any(trailing, SOFT_BREAK_MARKERS) and is_overlong(_accumulated_text(current)):
#             flush()
#
#     flush()
#     return segments


def _segments_from_words(words: list[dict], full_text: str) -> list[dict]:
    """Default segment from aligner output: one segment spanning the
    aligner's first→last item, carrying the full ASR text."""
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


class QwenBackend(Backend):
    def __init__(self) -> None:
        self._model: object | None = None
        self._aligner_model: object | None = None
        self._model_lock = threading.Lock()
        self._aligner_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _load_asr(self) -> object:
        import torch
        from qwen_asr import Qwen3ASRModel

        kwargs: dict = {"max_new_tokens": 512}
        if torch.cuda.is_available():
            kwargs.update(device_map="cuda:0", dtype="bfloat16", attn_implementation="sdpa")
        else:
            kwargs.update(device_map="cpu", dtype="float32", attn_implementation="eager")
        return Qwen3ASRModel.from_pretrained(TRANSCRIPT_MODEL_ID, **kwargs)

    def _load_aligner(self) -> object:
        import torch
        from qwen_asr import Qwen3ForcedAligner

        kwargs: dict = {}
        if torch.cuda.is_available():
            kwargs.update(device_map="cuda:0", dtype="bfloat16", attn_implementation="sdpa")
        else:
            kwargs.update(device_map="cpu", dtype="float32", attn_implementation="eager")
        return Qwen3ForcedAligner.from_pretrained(TRANSCRIPT_ALIGNER_ID, **kwargs)

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
        self._get_aligner()

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                self._model = self._load_asr()
        return self._model

    def _get_aligner(self) -> object:
        if self._aligner_model is not None:
            return self._aligner_model
        with self._aligner_lock:
            if self._aligner_model is None:
                self._aligner_model = self._load_aligner()
        return self._aligner_model

    def has_secondary(self) -> bool:
        return self._aligner_model is not None

    def unload_secondary(self) -> None:
        _log_gpu_mem("before aligner unload")
        with self._aligner_lock:
            model, self._aligner_model = self._aligner_model, None
        _release_gpu(model)
        _log_gpu_mem("after aligner unload")

    def _align_chunks(
        self,
        chunks: list[tuple[np.ndarray, int]],
        texts: list[str],
        languages: list[str],
    ) -> list[dict]:
        """Run the forced aligner on already-decoded audio chunks.
        Returns flat [{"text", "start", "end"}, ...] across all results."""
        with self._infer_lock:
            aligned = self._get_aligner().align(
                audio=chunks,
                text=texts,
                language=languages,
            )

        words: list[dict] = []
        for result in aligned:
            for item in getattr(result, "items", []):
                text = str(getattr(item, "text", "") or "").strip()
                start = round(float(getattr(item, "start_time", 0.0) or 0.0), 3)
                end = round(float(getattr(item, "end_time", 0.0) or 0.0), 3)
                if text or end > start:
                    words.append({"text": text, "start": start, "end": end})
        return words

    def align(
        self,
        audio: np.ndarray,
        text: str,
        language: str | None,
    ) -> list[dict]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not text.strip():
            return []

        duration = audio.size / SAMPLE_RATE
        if duration > MAX_INPUT_SECONDS:
            raise ValueError(
                f"audio is {duration:.1f}s, exceeds server max {MAX_INPUT_SECONDS:.0f}s - split on the client"
            )

        chunks = [(audio, SAMPLE_RATE)]
        return self._align_chunks(chunks, [text], [language or ""])

    def transcribe_result(
        self,
        audio: np.ndarray,
        language: str | None,
        prompt: str | None,
        return_timestamps: bool,
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

        with self._infer_lock:
            results = self._get_model().transcribe(chunks, context=_build_asr_context(prompt), language=language or None)

        if not results:
            return {"text": "", "language": None, "words": [], "segments": []}

        texts = [_strip_hallucinations((getattr(r, "text", "") or "").strip()) for r in results]
        langs = [(getattr(r, "language", "") or "").strip() for r in results]
        full_text = "".join(texts)

        if not return_timestamps:
            return {
                "text": full_text,
                "language": next((l for l in langs if l), None),
                "words": [],
                "segments": [],
            }

        align_langs = [l or "" for l in langs]
        # log.info("aligner input: audio_seconds=%.2f text_lens=%s langs=%s",
        #          audio.size / SAMPLE_RATE, [len(t) for t in texts], align_langs)

        words = self._align_chunks(chunks, texts, align_langs)

        # nonzero = sum(1 for w in words if w["end"] > w["start"])
        # log.info("aligner output: %d words, %d with nonzero span", len(words), nonzero)
        # if nonzero < len(words):
        #     zero_words = [w for w in words if w["end"] <= w["start"]]
        #     log.info("aligner zero-span words (%d): %s", len(zero_words),
        #              [(w["text"], w["start"], w["end"]) for w in zero_words[:30]])
        #     log.info("aligner full word sequence: %s",
        #              [(w["text"], w["start"], w["end"]) for w in words])

        # enriched = attach_punctuation(words, full_text)
        # segments = _build_segments(enriched)
        segments = _segments_from_words(words, full_text)
        # log.info("segments (%d): %s", len(segments),
        #          [(s["start"], s["end"], s["text"]) for s in segments])
        return {
            "text": full_text,
            "language": next((l for l in langs if l), None),
            "words": _strip_trailing(words),
            "segments": segments,
        }

    def transcribe_stream(
        self,
        audio: np.ndarray,
        language: str | None,
        prompt: str | None,
        return_timestamps: bool,
    ) -> Iterator[tuple]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            yield (None, [], [], "")
            return

        duration = audio.size / SAMPLE_RATE
        if duration > MAX_INPUT_SECONDS:
            raise ValueError(
                f"audio is {duration:.1f}s, exceeds server max {MAX_INPUT_SECONDS:.0f}s - split on the client"
            )

        chunks = [(audio, SAMPLE_RATE)]

        with self._infer_lock:
            results = self._get_model().transcribe(chunks, context=_build_asr_context(prompt), language=language or None)

        text = _strip_hallucinations((getattr(results[0], "text", "") or "").strip()) if results else ""
        lang = (getattr(results[0], "language", "") or "").strip() if results else ""

        yield (text, audio, 0.0, lang or language)

        if not return_timestamps:
            yield (None, [], [], text)
            return

        words = self._align_chunks(chunks, [text], [lang or ""])

        # enriched = attach_punctuation(words, text)
        # yield (None, _strip_trailing(enriched), _build_segments(enriched), text)
        yield (None, _strip_trailing(words), _segments_from_words(words, text), text)
