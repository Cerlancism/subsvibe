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

- **#4 Tail-text joiner CJK-aware**. New `utils.language.is_spaceless`
  (ja/zh/yue/th/lo/my/km). Pipeline uses `""` joiner for spaceless
  languages, `" "` otherwise. Auto-detect (`language=None`) falls back
  to space.

- **#5 `_reanchor_if_prompt_trimmed` observability**. Now logs a
  warning each time the shift fires, with `min_start`, `max_end`,
  `committed_until`, and `tail_span`. Reviewers can spot borderline
  misclassifications when `min_start` sits close to `committed_until`
  (i.e. the entries might be legitimately absolute with aligner drift).

- **#6 Per-slot mini-headers**. When pending and prov are both shown,
  the header line now joins two pre-styled mini-headers via
  `_compose_headers` (same SEPARATOR as the content rows). Each side's
  ts/lag/entries/tag reflect its own slot. Single-slot cases unchanged.

- **#15 Sliced-utterance residue lost on VAD-final** (final fix:
  cursor-trim audio). Initial fix (text-prefix strip via
  `_strip_committed_prefix` + `committed_text_by_utt`) recovered residue
  but introduced a duplication class: prefix-mismatch fallback emitted
  the full cheap text, and `_PREFIX_NOISE`'s narrow punct set let
  routine cycle-to-cycle ASR variation drift into the warn-and-emit-full
  branch. Replaced with a deterministic split: each cycle slices
  `ev.pcm` at `int(committed_until * LIVE_SAMPLE_RATE)` and sends only
  the residue audio to ASR. Entries returned are 0-based on the tail
  and get shifted back by `tail_start_abs` when building sub_ev.
  Side effects: `_split_entries` is now positional (commit
  `entries[:-1]`, hold last; finals commit all) — no `silence_tail_s`
  rule, no text comparison. `_reanchor_if_prompt_trimmed`,
  `_PREFIX_NOISE`, `_normalise_for_prefix`, `_strip_committed_prefix`,
  `committed_text_by_utt` all deleted. New `_MIN_TAIL_SECONDS = 0.1`
  guards against sub-100ms ASR calls when cursor sits at end of audio.
  Cheap-path fall-through with `committed_until > 0` emits the text as
  a `sliced` commit (final) or `tail` prov (provisional) covering
  `[tail_start_abs, ev.end]`. Acoustic-context loss across the cursor
  is mitigated by `seg_prompt` carrying prior committed text via
  `--history` / `--history-seconds`. Recommended command shape stays
  `--live --language <L> --translate <T> --history-seconds 5`.

## Pending — dynamism

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

- **#14 `--live --llm-asr` is broken** (orthogonal to render polish, but
  blocks the multimodal-LLM live path). `live_capture` /
  `live_transcribe` always call `asr_client.audio.transcriptions.create`
  (Whisper-compat `/v1/audio/transcriptions`). With `--llm-asr` the
  `asr_client` is the Ollama HTTP client which only exposes
  `/v1/chat/completions` → first segment 404s. File-mode (`--input
  --llm-asr`) routes correctly via `_transcribe_segment_llm` →
  `llm_asr_chat_transcribe`; live-mode never wires that branch in.
  Affects both with and without `--translate` (failure is at the ASR
  boundary, before translation runs). `scripts/client.sh:33` documents
  `--live --llm-asr` as supported, so this is doc-drift / rotted intent.
  Fix sketch: plumb `use_llm_asr` into `live_capture` →
  `live_transcribe`; branch to `llm_asr_chat_transcribe` (chat-completions
  multimodal path) for the cheap `with_entries=False` route. The
  `with_entries=True` path (slicing / tail prov / reanchor) can't be
  supported — multimodal chat returns no word/segment timestamps, so
  long utterances would only force-flush via VAD `MAX_SEG_SECONDS`.
  Latency caveat: chat completions are slower than Whisper-server, so
  provisional refresh cadence will be sluggish.

- **#13 Held + non-matching pending/prov causes 3↔6 jump**. Normal flow:
  `pending_final(A)` → `prov(B)` promotes A to pending → `commit(A)`
  clears pending A, sets held A while prov B remains → 6-line region.
  3s timer later held A flushes → back to 3 lines (prov B alone). The
  intentional in-place recolor (held alone after commit, no overlap) is
  preserved only when no other utterance is racing. Proposed fix: in
  `commit()`, detect non-matching pending/prov still occupying the
  region and bypass the held slot — print the committed lines straight
  to scrollback (committed colors) and leave the live region at 3 lines
  for the ongoing prov. Tradeoff: viewer loses the 3s committed-glow on
  A in the overlap case, but B's prov already signals A finished.
  Single-utterance flow (no overlap) keeps the existing held-with-timer
  path. Supersedes #9 (which was misdiagnosed as a pad-empty-slots
  problem).

## Out of scope (file-mode audit)

- File-mode (`--input` → `.srt`) was checked against the recent live
  work and shows no regression. The only shared-module behavioural
  change was `utils/subtitle.overlapping_text` joiner `" "` → `"\n"`
  (`ebfddee`), affecting `--context-src` prompt formatting only.
