from __future__ import annotations

import gc
import os
import threading
from typing import Iterator

import numpy as np

from backends.base import Backend

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

TRANSCRIPT_MODEL_ID = os.environ.get("TRANSCRIPT_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
TRANSCRIPT_ALIGNER_ID = os.environ.get("TRANSCRIPT_ALIGNER_ID", "Qwen/Qwen3-ForcedAligner-0.6B")
SAMPLE_RATE = 16000
MAX_INPUT_SECONDS = float(os.environ.get("TRANSCRIPT_MAX_INPUT_SECONDS", "120"))

SENTENCE_END_MARKERS = (".", "!", "?", "。", "！", "？")
CLOSING_PUNCTUATION = set(".,!?;:)]}、。，！？；：」』）》〉】")
OPENING_PUNCTUATION = set("([{'\"「『《〈【")

HALLUCINATION_REPEAT_THRESHOLD = 20
HALLUCINATION_PATTERN_MAX_LEN = 20

ASR_SYSTEM_CONTEXT = (
    "Transcribe the speech faithfully and conservatively. "
    "Preserve the language that is actually spoken. "
    "Do not translate, summarize, or rewrite."
)


def _build_asr_context(prompt: str | None) -> str:
    return f"{ASR_SYSTEM_CONTEXT} {prompt}" if prompt else ASR_SYSTEM_CONTEXT


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


def _contains_cjk(value: str) -> bool:
    for ch in value:
        code = ord(ch)
        if (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0x3040 <= code <= 0x30FF
            or 0xF900 <= code <= 0xFAFF
        ):
            return True
    return False


def _join_tokens(tokens: list[str]) -> str:
    text = ""
    for token in tokens:
        piece = token.strip()
        if not piece:
            continue
        if not text:
            text = piece
            continue
        prev, nxt = text[-1], piece[0]
        if (
            nxt in CLOSING_PUNCTUATION
            or prev in OPENING_PUNCTUATION
            or (_contains_cjk(prev) and _contains_cjk(nxt))
        ):
            text += piece
        else:
            text += f" {piece}"
    return text.strip()


def _build_segments(words: list[dict]) -> list[dict]:
    segments: list[dict] = []
    current: list[dict] = []

    def flush():
        if not current:
            return
        text = _join_tokens([str(w.get("text", "") or "") for w in current])
        if text:
            segments.append({
                "start": round(float(current[0]["start"]), 3),
                "end": round(float(current[-1]["end"]), 3),
                "text": text,
            })
        current.clear()

    for word in words:
        token = str(word.get("text", "") or "")
        if current:
            gap = float(word["start"]) - float(current[-1]["end"])
            span = float(word["end"]) - float(current[0]["start"])
            if gap >= 1.0 or span >= 12.0:
                flush()
        current.append(word)
        if token.endswith(SENTENCE_END_MARKERS):
            flush()

    flush()
    return segments


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
        with self._model_lock:
            self._model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

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
        with self._aligner_lock:
            self._aligner_model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

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
                f"audio is {duration:.1f}s, exceeds server max {MAX_INPUT_SECONDS:.0f}s — split on the client"
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

        with self._infer_lock:
            aligned = self._get_aligner().align(
                audio=chunks,
                text=texts,
                language=[l or "" for l in langs],
            )

        words: list[dict] = []
        for result in aligned:
            for item in getattr(result, "items", []):
                text = str(getattr(item, "text", "") or "").strip()
                start = round(float(getattr(item, "start_time", 0.0) or 0.0), 3)
                end = round(float(getattr(item, "end_time", 0.0) or 0.0), 3)
                if text or end > start:
                    words.append({"text": text, "start": start, "end": end})

        return {
            "text": full_text,
            "language": next((l for l in langs if l), None),
            "words": words,
            "segments": _build_segments(words),
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
                f"audio is {duration:.1f}s, exceeds server max {MAX_INPUT_SECONDS:.0f}s — split on the client"
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

        with self._infer_lock:
            aligned = self._get_aligner().align(
                audio=chunks,
                text=[text],
                language=[lang or ""],
            )

        words: list[dict] = []
        for result in aligned:
            for item in getattr(result, "items", []):
                item_text = str(getattr(item, "text", "") or "").strip()
                start = round(float(getattr(item, "start_time", 0.0) or 0.0), 3)
                end = round(float(getattr(item, "end_time", 0.0) or 0.0), 3)
                if item_text or end > start:
                    words.append({"text": item_text, "start": start, "end": end})

        yield (None, words, _build_segments(words), text)
