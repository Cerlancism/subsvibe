# Project memory index

- `./.claude/memory/known-issues/live-render.md` — live render known-issues
  backlog (open + closed), covering `./client/render.py` + `./client/pipeline.py`
  emit logic.

- `./.claude/memory/recovery-vad-webrtc.md` — webrtcvad as the recovery VAD
  in `./client/live_vad.py` (Silero stays primary; its recovery pass was
  removed): design, tuning knob, and the amplified-noise-floor risk to
  watch in live use.

- `./.claude/memory/ja-romanization.md` — Japanese romaji gauge: cutlet config &
  why, its failure classes (which are detectable), the rejected alternatives
  (pykakasi, full unidic, from-scratch LLM prompts), the chosen hybrid corrector
  (`romanize_ja_fix` in `./client/llm.py`), and the gate-vs-harness test split.
