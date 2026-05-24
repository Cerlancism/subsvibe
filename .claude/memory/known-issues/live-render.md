# Live render polish — known issues & backlog

Tracks issues raised in the live rendering audit (`./client/render.py`
+ `./client/pipeline.py` emit logic). Check items off as they land.

## Correctness

- [x] **#1 Sliced-final ↔ tail key collision** — commit `e1b1c53`. Tail prov
  keyed as `(ev.start, "tail")`. Renderer gains `discard_provisional(key)`.
  Pipeline calls it at utterance-close paths (slicing branch with no holds,
  whole-path skip on `ev.final`).

- [x] **#2 Carried translation can mismatch** (superseded by #16). Originally
  added a SequenceMatcher ratio ≥ 0.6 gate to blank the pending slot's
  translation on wholesale rewrites. Replaced in #16 by unconditional
  carry across all prov/pending slot transitions — the slicer's boundaries
  are trustworthy enough that residue/refinement text always shares
  semantic content with the prior translation. Scrollback was never
  affected (always uses the fresh translate-worker result).

- [x] **#3 `history_buf` unbounded growth**. Trim policy: once buffer span
  exceeds 2h, drop everything older than 1h before the newest entry.
  `_trim_history()` runs after each append in `./client/pipeline.py`.

- [x] **#4 Tail-text joiner CJK-aware**. New `utils.language.is_spaceless`
  (ja/zh/yue/th/lo/my/km). Pipeline uses `""` joiner for spaceless
  languages, `" "` otherwise. Auto-detect (`language=None`) falls back to
  space.

- [x] **#5 `_reanchor_if_prompt_trimmed` observability** (obsolete after
  #15's final fix). Originally added a warning when the shift fired
  (`min_start`, `max_end`, `committed_until`, `tail_span`) to spot
  borderline misclassifications. The function itself was deleted in #15's
  audio-cursor rewrite — the cheap-path entries are now 0-based on the
  trimmed tail and shifted back by `tail_start_abs`, so prompt-trim
  reanchoring no longer applies. Kept here for audit trail only.

- [x] **#6 Per-slot mini-headers**. When pending and prov are both shown,
  the header line now joins two pre-styled mini-headers via
  `_compose_headers` (same SEPARATOR as the content rows). Each side's
  ts/lag/entries/tag reflects its own slot. Single-slot cases unchanged.

- [x] **#15 Sliced-utterance residue lost on VAD-final** (final fix:
  cursor-trim audio). Initial fix (text-prefix strip via
  `_strip_committed_prefix` + `committed_text_by_utt`) recovered residue
  but introduced a duplication class: prefix-mismatch fallback emitted
  the full cheap text, and `_PREFIX_NOISE`'s narrow punct set let
  routine cycle-to-cycle ASR variation drift into the warn-and-emit-full
  branch. Replaced with a deterministic split: each cycle slices
  `ev.pcm` at `int(round(committed_until * LIVE_SAMPLE_RATE))` and sends only
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

- [x] **#17 Pending-stage flicker eliminated by queue-first commit**. The
  pending slot (transcript shown immediately, translation fills in later)
  was the source of every remaining flicker on the live region: sliced
  sub-commits landed pending with `_pending_final_translation = None`,
  the suppress-prov-while-pending-untranslated rule hid the prov, and the
  composite header packed pending+prov onto one wrapped line. Fix is
  architectural: **finals are queue-first now**. ASR-final and sliced
  sub-finals go straight to `translate_q` without touching the renderer;
  the translate worker commits transcript+translation together via
  `renderer.commit()`. Provisionals still emit `_emit_transcript` for
  the tail preview (throwaway, brief translation-less window is fine).
  Renderer cleanup: deleted `pending_final()`, `_pending_final_*` state,
  `_compose_headers`, `_compose`, `STYLE_NEXT_*`, `STYLE_SEPARATOR`,
  `SEPARATOR`, and the suppress-prov rule. `_render()` is now held +
  prov only. `provisional_translation` no longer dispatches between
  slots — single-slot. Net: render.py shrinks ~110 lines.
  Side effects:
  - Latency: viewer sees the new commit one LLM round-trip later than
    before, but during that round-trip the prior committed line stays
    held (no blank region).
  - Issues #6 (composite header), #9/#13 (3↔6 height jump) become
    moot — there's no two-slot path anymore.
  - Translate failure path unchanged: on timeout for a final, the
    translate worker still falls back to `_emit(job, translation=None)`
    so the transcript reaches scrollback (just slightly later).

- [x] **#16 Prov translation blanks when slicer rotates prov key**. The
  cheap-path prov uses key `ev.start`; once slicing engages with a held
  tail, the new prov key becomes `(ev.start, "tail")`. The renderer's
  `provisional_transcript` saw `new_utt=True` on the rotation and
  unconditionally blanked `_prov_translation`. The slicing tail's
  transcript covers only the residue past `committed_until`, so a
  similarity gate (initial attempt) failed to fire and the tail still
  rendered translation-blank for one LLM round-trip. Real-session
  screenshots confirmed visible flicker every slicing cycle.

  Final fix: drop the similarity gate entirely. `_transcripts_similar`,
  `CARRY_TRANSLATION_SIMILARITY`, and the `SequenceMatcher` import are
  all deleted. Both `pending_final` and `provisional_transcript` now
  carry unconditionally — whenever a candidate slot's key matches, its
  translation is reused as a placeholder. Trust the slicer's boundaries
  (faster-whisper segments / `entries_from_words` carve at aligner-chosen
  breaks; residue and refinement text share semantic content with the
  prior translation). Pipeline still passes `inherit_from=ev.start` from
  `_emit_transcript` when the new key is namespaced
  (`isinstance(key, tuple)`); no-op for same-key refinements and for
  whole-utterance cheap-path emits. Commit/scrollback path untouched —
  it never used the carry mechanism, always writes the fresh
  translate-worker result verbatim.

- [x] **#21 Stale tail prov visible after sliced sub-final commits**.
  With queue-first provs (#20), a tail prov can race a sliced sub-final
  to the translate worker: cycle K's ASR returns 1 entry → queued as
  tail prov for `(parent, "tail")`. Cycle K+1's ASR returns 2 entries →
  cycle K's lone entry is now `entries[:-1]` so it gets promoted as a
  sliced sub-final. Translate worker FIFO processes tail K first (renders
  the entry's text as a dim prov), then the sub-final (commits the SAME
  entry as bold). Both keys differ, so `commit()` doesn't clear the prov
  slot — viewer sees the same line twice (bold above, dim below) until
  the next cycle's tail repopulates the slot.

  Fix in `./client/pipeline.py`:
  1. Sub-final's meta carries `parent_start = ev.start` so downstream can
     identify the parent utterance's tail key.
  2. `_enqueue_translate_with_backoff` extended: when the new job is a
     sliced sub-final (final + `parent_start` set), it also drops any
     pending tail prov with `item.event.start == parent_start` from
     translate_q. Covers the case where tail hasn't been processed yet.
  3. Translate worker's final branch: after `_emit()` for a sliced
     sub-final, calls `renderer.discard_provisional((parent_start, "tail"))`.
     Covers the case where tail already rendered before this sub-final
     was processed.

  Verified: tail prov from cycle K (overlapping content) drops either
  in-queue or post-render. Tail prov from cycle K+1 (fresh post-cursor
  content) queues after sub-final and processes normally.

- [x] **#22 #21's post-commit discard creates blank-tail gap**. Live-run
  debug.log showed `discard_provisional FIRED` on every sliced sub-final
  commit. Each fire blanked the slot until the next cycle's tail finished
  its LLM round-trip (~500-600ms) — visible as a recurring gap after
  every sub-final commit. Sequence:
  - cycle K tail rendered → on screen as T_K
  - cycle K+1 ASR returns N entries → enqueues sub-final + new tail
    T_K+1 to translate_q (FIFO)
  - translate worker pulls sub-final → LLM → commit → #21 discard fires
    → tail slot blank ← THE GAP
  - translate worker pulls T_K+1 → LLM → renders → tail slot filled

  Text-comparison gating (prefix match, similarity ratio) was considered
  and rejected: cross-cycle ASR variance returns similar-but-not-identical
  text (e.g. `だろここ` vs `だろうこ`), so no reliable string-based
  predicate distinguishes the duplicate-race case from the fresh-tail
  case.

  Fix: queue-state gate. Translate worker's sub-final branch peeks
  translate_q (via new `_has_pending_tail(parent_start)` helper which
  drains + restores) before firing the discard. If a successor tail for
  the same parent is already queued, skip the discard — the queued tail
  will overwrite the slot naturally when it renders, no gap. If no
  successor is queued (e.g., VAD-final closed the utterance and the
  final ASR call produced no new tail), discard fires so the stale tail
  doesn't leak.

  Safety: `_has_pending_tail` is called from the translate worker, the
  sole consumer of translate_q, so the drain+restore can't race a
  concurrent `.get()`. ASR worker may push during the peek window
  (false-negative → discard fires when it shouldn't); benign — tail
  blanks for one LLM round-trip, same as before the fix.

  Tradeoff: when the discard is skipped, the on-screen tail (showing
  cycle K's text for the just-committed audio range) overlaps the bold
  sub-final commit for one LLM round-trip — viewer sees a "duplicate"
  briefly until T_K+1 renders. Considered better than the blank gap.
  No effect on committed scrollback (`_emit()` runs unconditionally),
  no effect on which segments are committed (still `entries[:-1]` only).

  Follow-up — second discard source found in live debug.log: the ASR
  worker had TWO synchronous `discard_provisional` calls for the same
  scenario (line 630 in the slicing-path no-holds branch and line 658
  in the cheap-path final branch). Both fired BEFORE the queued
  sub-final committed, so the post-commit gate couldn't help — the
  tail was already blank by the time the gate ran. Fix: removed both
  eager discards. The cheap-path final's meta now carries `parent_start`
  so the translate-worker post-commit gate handles cleanup uniformly
  across both paths (sliced sub-final from entry-driven promotion AND
  cheap-path final from cursor-advance promotion). Other early-exit
  discards (lines 474, 519) kept — those return without queuing any
  sub-final, so the gate would never fire and the eager discard is
  the only cleanup path.

- [x] **#20 Queue-first provs (extends #17 to provisionals)**. #19's
  held-linger only masked the prior utterance's frame; the new utterance's
  own prov still rendered transcript-only for a full LLM round-trip
  (often many seconds when prov translate jobs queue behind a slow final
  commit-translate). Per user direction, the right model is: a prov
  never appears on screen without its own translation attached.

  Fix in `./client/pipeline.py`: provs now go through `translate_q` instead
  of rendering immediately via `_emit_transcript`. The translate worker
  calls `_emit_transcript` and `_emit_translation` back-to-back so the
  prov appears already paired. Empty / refused translations skip the
  render entirely (next prov cycle retries). Sliced sub-jobs (always
  finals) and the slicing tail prov route through the same path.

  Backpressure: new `_enqueue_translate_with_backoff` mirrors
  `_enqueue_with_backoff` for translate_q. Same-utterance prov jobs older
  than `LIVE_PROVISIONAL_BACKOFF_SECONDS` are collapsed before push so
  the queue can't accrue nested provs when the LLM trails ASR cadence.
  Finals are always preserved (helper short-circuits).

  Effects:
  - New utterance's first prov visible latency: +1 LLM round-trip (was
    +ASR only). Same-utterance refinements: +1 LLM round-trip each, but
    backoff collapses the queue so only the freshest gets through.
  - "Don't show new prov before previous final commits" satisfied for
    free by FIFO order on translate_q — K+1's first prov queues after
    K's final and waits its turn behind it.
  - #19's deferred-held-flush gate retained as defense-in-depth: with
    queue-first provs, `_emit_transcript`+`_emit_translation` fire under
    microseconds of each other so the held lingers only between the two
    calls (invisible at the 12Hz refresh rate). The gate still protects
    against any future path that emits a prov without a paired
    translation.

- [x] **#19 Defer held flush until new prov has its translation**. The
  "translation-blank under new prov" gap that #18 left as an accepted cost
  was still visible in real-session recordings: when a new utterance's
  first provisional landed, `provisional_transcript` blanked
  `_prov_translation` (per #18's "no cross-utterance carry" rule) AND
  flushed the held line to scrollback in the same call — leaving a 3-line
  region with transcript + blank translation for one LLM round-trip.

  Fix in `./client/render.py` only: `provisional_transcript` now flushes
  held only when the new prov already has a translation (same-key
  refinement, or `inherit_from` carry from a slicing-tail rotation).
  Otherwise held stays on screen and the prov renders below it as a
  second 3-line block. `provisional_translation` gains a held-flush at
  the end so that when the new prov's own LLM call returns, held scrolls
  to scrollback and the live region collapses back to 3 lines with the
  fully-paired prov. Other flush paths (`commit`, `__exit__`) untouched.

  Cases verified: same-utterance refinement (held flushes as before),
  first prov of new utterance with held on screen (held lingers, no
  blank flicker, flushes on translation arrival), slicing-tail rotation
  (translation carried via `inherit_from`, held flushes immediately),
  translation timeout for prov (next commit flushes held), Ctrl+C
  (`__exit__` flushes held). #18's invariant (never pair the wrong
  translation under a transcript) preserved — the prior committed line
  stays as its own bold block, not folded under the new transcript.

  Tradeoff: live region grows 3->6 lines during the overlap window
  (cursor jumps down briefly). Considered worse than the blank-flicker
  alternative.

- [x] **#18 Cross-utterance translation carry experiment, reverted**.
  After #17's queue-first refactor, the viewer sees a one-LLM-round-trip
  gap between a fresh utterance's prov transcript appearing and its
  translation landing. Tried filling the gap by carrying the held
  (just-committed) line's translation as a placeholder under the new
  prov. Two failure modes surfaced in real-session screenshots:
  1. **Held already flushed**: held flushes on the *first* prov refresh
     of the next utterance (no time-based timer with `COMMIT_HOLD_SECONDS`
     removed). Subsequent utterances arrive after the in-progress prov
     has already eaten the held — fallback finds nothing.
  2. **Stale text under new transcript**: in the race where a final's
     translate is still in flight when the next utterance starts, the
     held-fallback would pair the prior utterance's English under the
     new utterance's Japanese.
  Both rejected: carrying a prior utterance's translation under new
  transcript is misleading regardless of whether held is populated.
  Final decision: **accept the gap**. No cross-utterance carry; new
  provs are translation-blank until their own LLM call returns. The
  same-utterance inherit_from carry (slicing-tail key rotation, where
  the new prov is literally a substring of the same utterance) stays —
  that's truthful.

  Concurrency-based fixes (separate prov/final translate workers, or
  priority queue) were discussed and deferred. Local Ollama would
  handle 2 concurrent LLM calls fine; remote rate-limited APIs would
  not. Re-visit if the gap becomes intolerable in practice.

## Dynamism

- [ ] **#7 Refresh cadence** (`./client/render.py:66`). `refresh_per_second=12`
  still set; can feel jittery on slow remote SSH. Drop to 8Hz for a calmer
  feel.

## Visual

- [x] **#8 Separator readability** (obsolete after #17). The `SEPARATOR`
  constant and the pending+prov composite header were deleted when finals
  went queue-first — there's no two-slot composite line to separate
  anymore. Kept for audit trail.

- [x] **#9 Block height jumps 3↔6** (obsolete after #17, superseded by
  #19's accepted tradeoff). The old pending+prov+held 6-line phase is gone
  with the two-slot path. The 3↔6 jump that remains today is the
  intentional held-linger window introduced in #19 (held stays under the
  new prov while the prov's translation is in flight), explicitly chosen
  over the blank-flicker alternative.

- [ ] **#10 Tag chip in header** (`./client/render.py:151-154`). `tail`/`sliced`
  labels are useful for debugging, noise for end users. Gate behind log
  level or a `--debug-render` flag. Scope grew: the header now also carries
  `dur=`, `lag=`, `n=`, `prov=` fields — the whole row reads as debug.
  Consider gating all of these together.

- [x] **#11 `_install_log_handler` is destructive** — file-handler half
  fixed in `1152c0b`. `_install_log_handler` now preserves
  `logging.FileHandler` instances when swapping in `RichHandler` (see
  `./client/render.py:404-406`, restore loop also retained), so file logs
  survive a live session. Non-file stream handlers are still dropped,
  but that's the intended swap (stderr StreamHandler → RichHandler on the
  same console). Closing as resolved — re-open if the broader "wrap, don't
  replace" treatment is wanted.

- [ ] **#12 Light-theme colors**. `grey80` / `grey42` / `cyan` still
  hard-coded in `./client/render.py:25-29`. Washes out on light terminals.
  If light themes are in scope, switch to a `rich.theme`-aware palette.

- [x] **#13 Held + non-matching pending/prov causes 3↔6 jump** (obsolete
  after #17). The whole scenario was framed around the pending slot
  (`pending_final(A)` → `prov(B)` → `commit(A)` overlapping pending+prov+held).
  #17's queue-first refactor deleted `pending_final()` and the pending
  slot entirely; there's no longer a path that holds an un-translated
  final in the live region while a different prov is open. The remaining
  3↔6 jump is the #19 held-linger window (same key family, single open
  utterance), accepted as a tradeoff.

- [ ] **#14 `--live --llm-asr` is broken** (orthogonal to render polish,
  but blocks the multimodal-LLM live path). `live_capture` /
  `live_transcribe` always call `asr_client.audio.transcriptions.create`
  (Whisper-compat `/v1/audio/transcriptions`). With `--llm-asr` the
  `asr_client` is the Ollama HTTP client which only exposes
  `/v1/chat/completions` → first segment 404s. File-mode (`--input
  --llm-asr`) routes correctly via `_transcribe_segment_llm` →
  `llm_asr_chat_transcribe`; live-mode never wires that branch in. Affects
  both with and without `--translate` (failure is at the ASR boundary,
  before translation runs). `./scripts/client.sh:33` documents `--live
  --llm-asr` as supported, so this is doc-drift / rotted intent. Fix
  sketch: plumb `use_llm_asr` into `live_capture` → `live_transcribe`;
  branch to `llm_asr_chat_transcribe` (chat-completions multimodal path)
  for the cheap `with_entries=False` route. The `with_entries=True` path
  (slicing / tail prov / reanchor) can't be supported — multimodal chat
  returns no word/segment timestamps, so long utterances would only
  force-flush via VAD `MAX_SEG_SECONDS`. Latency caveat: chat completions
  are slower than Whisper-server, so provisional refresh cadence will be
  sluggish.

## Out of scope (file-mode audit)

- File-mode (`--input` → `.srt`) was checked against the recent live work
  and shows no regression. The only shared-module behavioural change was
  `./utils/subtitle.overlapping_text` joiner `" "` → `"\n"` (`ebfddee`),
  affecting `--context-src` prompt formatting only.
