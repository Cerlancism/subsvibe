# Project memory index

- `./.claude/memory/known-issues/live-render.md` — live render known-issues
  backlog (open + closed), covering `./client/render.py` + `./client/pipeline.py`
  emit logic.

- `./.claude/memory/recovery-vad-webrtcvad.md` — webrtcvad in SubsVibe: the
  live recovery VAD in `./client/live_vad.py` (Silero stays primary; its
  recovery pass was removed) — design, tuning knob, amplified-noise-floor
  risk — plus file-input chunking in `./client/vad.py` (`CoarseChunker` +
  `split_provisional`: on-the-fly cursor chunking whose next cut comes from
  the previous chunk's ASR output) and the live early-split fallback
  (`_scan_split_point`: webrtcvad → energy quiet-window seam once a primary
  segment passes `LIVE_SPLIT_TARGET_SECONDS` without Silero finalising).

- `./.claude/memory/silence-hallucinations.md` — silence + noise hallucination
  datasets (`./server/data/silence_hallucinations.json` built by
  `./tests/test_silence_hallucinations.py`, `./server/data/noise_hallucinations.json`
  hand-curated) and the default-on server filter (`./server/hallucination_filter.py`):
  per backend/model/language exact-match blocklist, punctuation/case-insensitive,
  `TRANSCRIPT_SILENCE_FILTER=0` / `TRANSCRIPT_NOISE_FILTER=0` to disable each source.

- `./.claude/memory/backend-investigations.md` — open investigations on the ASR
  backends (`./server/backends/`). Current item: whether `faster-whisper` can
  return transcription confidence (avg_logprob / no_speech_prob / word
  probability) and how it'd flow through `transcribe_result`.

- `./.claude/memory/ja-romanization.md` — Japanese romaji gauge: cutlet config &
  why, its failure classes (which are detectable), the rejected alternatives
  (pykakasi, full unidic, from-scratch LLM prompts), the chosen hybrid corrector
  (`romanize_ja_fix` in `./client/llm.py`), and the gate-vs-harness test split.
