# Live render polish — fix punch list

Branch: `fix/live-render-polish`. Tracks the issues raised in the live
rendering audit (render.py + pipeline.py emit logic). Updated as items land.

## Done

- **#1 Sliced-final ↔ tail key collision** — commit `e1b1c53`.
  Tail prov keyed as `(ev.start, "tail")`. Renderer gains
  `discard_provisional(key)`. Pipeline calls it at utterance-close paths
  (slicing branch with no holds, whole-path skip on `ev.final`).

- **#3 `history_buf` unbounded growth**. Trim policy: once buffer span
  exceeds 2h, drop everything older than 1h before the newest entry.
  `_trim_history()` runs after each append in pipeline.py.

- **#2 Carried translation can mismatch**. `pending_final` now only
  carries the prior `_prov_translation` / `_pending_final_translation`
  when the source transcript is similar enough (SequenceMatcher ratio
  ≥ 0.6). Wholesale rewrites blank the slot until the real translation
  arrives via `commit()`. Only affects the transient pending preview;
  scrollback always uses the fresh translate-worker result.

## Pending — data accuracy

- **#4 `tail_text = " ".join(...)`** (pipeline.py:500). Wrong joiner
  for CJK. Use `""` when language is `ja`/`zh`, or use the backend's
  separator. Affects displayed tail prov text only.

- **#5 `_reanchor_if_prompt_trimmed` slack** (pipeline.py:74-108).
  `_ANCHOR_SLACK = 0.5` is wide; a legitimately absolute entry starting
  <0.5s before `committed_until` would get shifted into the wrong
  frame. Add a log line when the shift fires so misclassifications are
  observable in traces.

## Pending — dynamism

- **#6 Header `ts` vs `lag` semantic mismatch** (render.py:122-128).
  `ts` anchors to pending-final (older content) while `lag` reads from
  prov (newer). Header reads as "this old timestamp, this fresh lag".
  Either anchor both to the same slot or split into two header lines.

- **#7 Refresh cadence** (render.py:77). `refresh_per_second=12` can
  feel jittery on slow remote SSH. Drop to 8Hz for a calmer feel.

## Pending — visual

- **#8 Separator readability** (render.py:33). `SEPARATOR = "  "` lets
  pending + next-prov read as one continuous sentence, especially after
  `soft_wrap` wraps. Use a visible glyph (` │ ` in dim grey).

- **#9 Block height jumps 3↔6** across phases. Held+next renders six
  lines while prov-only renders three. Pad the empty phase so the
  in-place region keeps a constant height — calmer scroll.

- **#10 Tag chip in header** (render.py:176-179). `tail`/`sliced`
  labels are useful for debugging, noise for end users. Gate behind log
  level or a `--debug-render` flag.

- **#11 `_install_log_handler` is destructive** (render.py:426-440).
  Replaces all root handlers; wipes whatever `setup_logging` configured
  (file handler, JSON formatter). Restored at exit, but during the
  session file logging is lost. Wrap instead: keep existing handlers,
  remove only the stderr stream handler, add `RichHandler`.

- **#12 Light-theme colors**. `grey80` washes out on light terminals.
  If light themes are in scope, switch to a `rich.theme`-aware palette.

## Out of scope (file-mode audit)

- File-mode (`--input` → `.srt`) was checked against the recent live
  work and shows no regression. The only shared-module behavioural
  change was `utils/subtitle.overlapping_text` joiner `" "` → `"\n"`
  (`ebfddee`), affecting `--context-src` prompt formatting only.
