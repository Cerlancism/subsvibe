# Live render polish — known issues & backlog

Issues from the live-rendering audit (`./client/render.py` +
`./client/pipeline.py` emit logic). Open items carry full context; closed
items are one-line fix summaries (git history holds the detail). Both
sections sorted by issue number.

## Open

- [ ] **#14 `--live --llm-asr` is broken**. `live_capture`/`live_transcribe`
  always call `asr_client.audio.transcriptions.create` (Whisper-compat). Under
  `--llm-asr` the client only exposes `/v1/chat/completions` → first segment
  404s. File mode routes correctly via `_transcribe_segment_llm` →
  `llm_asr_chat_transcribe`; live never wires it. `./scripts/client.sh`
  documents this as supported (doc-drift). Fix: plumb `use_llm_asr` into
  `live_capture` → `live_transcribe`, branch to `llm_asr_chat_transcribe` for
  the cheap `with_entries=False` route. The `with_entries=True` path (slicing/
  tail/reanchor) can't be supported — multimodal chat returns no timestamps,
  so long utterances only force-flush via VAD `MAX_SEG_SECONDS`. Latency:
  chat completions are slower, so prov refresh will be sluggish.

- [ ] **#36 LOW — two cosmetic/pathological nits** (neither user-visible in practice):
  - `_ensure_entries` (`./client/transcribe.py`) synthesises
    `{"start": 0.0, "end": round(segment_duration, 3), ...}`; when
    `segment_duration ≈ 0` this is a zero-span entry. Pathological (sub-ms
    audio), and file mode repairs zero-spans downstream anyway. Could clamp
    `end` to a small floor if it ever surfaces.
  - `self._last_provisional_sample = 0` in `LiveVAD._flush`
    (`./client/live_vad.py`) is a dead write — after a flush no segment is open,
    and every re-open path (`start` branch, force-flush re-open) reassigns the
    marker before it's next read. Harmless; remove for clarity.

## Closed

- [x] **#1 Sliced-final ↔ tail key collision** (`e1b1c53`). Tail prov keyed
  `(ev.start, "tail")`; renderer gains `discard_provisional(key)`, called at
  utterance-close paths.
- [x] **#2 Carried translation mismatch** (superseded by #16). Replaced a
  SequenceMatcher gate with unconditional carry across prov/pending transitions.
- [x] **#3 `history_buf` unbounded growth**. `_trim_history()` in
  `./client/pipeline.py` (`HISTORY_TRIM_AFTER_SECONDS` / `HISTORY_KEEP_SECONDS`): once
  span > 2h, drop everything older than 1h before newest.
- [x] **#4 Tail-text joiner CJK-aware**. New `utils.language.is_spaceless`
  (ja/zh/yue/th/lo/my/km); `""` joiner for spaceless langs, `" "` otherwise,
  space on auto-detect.
- [x] **#5 `_reanchor_if_prompt_trimmed` observability** (obsolete after #15).
  Function deleted in #15's audio-cursor rewrite.
- [x] **#6 Per-slot mini-headers** (moot after #17). Composite header deleted
  with the two-slot path.
- [x] **#7 Refresh cadence** (won't fix). 12Hz stays; SSH jitter not worth
  tuning for.
- [x] **#8 Separator readability** (obsolete after #17). `SEPARATOR` + composite
  header deleted.
- [x] **#9 Block height 3↔6 jump** (obsolete after #17). Two-slot phase gone;
  remaining 3↔6 is the #19 held-linger window, accepted.
- [x] **#10 Debug fields in header** (won't fix). The `tail`/`sliced`/`dur=`/
  `lag=` row stays as-is; gating behind a flag not needed.
- [x] **#11 `_install_log_handler` destructive** (`1152c0b`). Now preserves
  `FileHandler` instances when swapping in `RichHandler`, so file logs survive.
- [x] **#12 Light-theme colors**. `STYLE_*` constants are now semantic
  `rich.theme` keys; `THEMES` dark/light palettes in `./client/render.py`,
  picked by new `--theme {dark,light}` flag (live-only, default dark) plumbed
  through `live_capture` → `LiveRenderer`. Bonus find: the old `grey80` was an
  invalid rich color silently rendering UNSTYLED — dark palette now uses
  `grey82`, realising the original dim-prov intent.
- [x] **#13 Held + non-matching prov 3↔6 jump** (obsolete after #17). Pending
  slot deleted; remaining 3↔6 is #19's held-linger, accepted.
- [x] **#15 Sliced-utterance residue lost on VAD-final** (cursor-trim audio).
  Each cycle slices `ev.pcm` at `committed_until * LIVE_SAMPLE_RATE`, sends only
  residue to ASR; entries 0-based on the tail, shifted by `tail_start_abs`.
  `_split_entries` is positional (commit `entries[:-1]`, hold last). Deleted the
  prefix-strip machinery (`_reanchor_if_prompt_trimmed`, `_PREFIX_NOISE`,
  `_strip_committed_prefix`, etc.). Acoustic-context loss mitigated by
  `seg_prompt` via `--history`/`--history-seconds`.
- [x] **#16 Prov translation blanks on slicer key rotation**. Dropped the
  similarity gate entirely (`_transcripts_similar`, `CARRY_TRANSLATION_SIMILARITY`
  gone); `pending_final` + `provisional_transcript` carry translation
  unconditionally when keys match. Trust the slicer's boundaries. Commit/
  scrollback path untouched (always uses fresh translate result).
- [x] **#17 Pending-stage flicker → queue-first finals**. ASR-finals and sliced
  sub-finals go straight to `translate_q`, never touch the renderer; translate
  worker commits transcript+translation together via `renderer.commit()`.
  Deleted `pending_final()`, `_pending_final_*`, `_compose_headers`, `_compose`,
  `STYLE_NEXT_*`, `SEPARATOR`, suppress-prov rule. `_render()` is held + prov
  only. Cost: new commit visible one LLM round-trip later, but prior held stays
  during the window (no blank). Makes #6/#9/#13 moot.
- [x] **#18 Cross-utterance translation carry, reverted**. Tried carrying the
  held line's translation under a new prov; failed (held often already flushed;
  pairs wrong translation under new transcript in a race). Decision: accept the
  one-round-trip gap, no cross-utterance carry. Same-utterance `inherit_from`
  carry stays (truthful — new prov is a substring of the same utterance).
- [x] **#19 Defer held flush until new prov has its translation**
  (`./client/render.py`). `provisional_transcript` flushes held only when the
  new prov already has a translation (same-key refinement or `inherit_from`
  carry); otherwise held lingers and prov renders below. `provisional_translation`
  flushes held when the prov's own translation arrives. Tradeoff: live region
  grows 3→6 lines during overlap — chosen over blank-flicker.
- [x] **#20 Queue-first provs** (extends #17). Provs go through `translate_q`
  instead of immediate `_emit_transcript`; translate worker calls
  `_emit_transcript` + `_emit_translation` back-to-back so a prov never appears
  without its translation. New `_enqueue_translate_with_backoff` collapses
  same-utterance prov jobs older than `LIVE_PROVISIONAL_BACKOFF_SECONDS`; finals
  always preserved. "Don't show new prov before previous final commits" falls
  out of FIFO order for free.
- [x] **#21 Stale tail prov after sliced sub-final commits** (`./client/pipeline.py`).
  Tail prov can race a sub-final (same entry rendered dim then bold). Fix:
  sub-final meta carries `parent_start`; `_enqueue_translate_with_backoff` drops
  any queued tail prov for that parent; translate worker calls
  `discard_provisional((parent_start, "tail"))` post-commit.
- [x] **#22 #21's discard creates a blank-tail gap**. Post-commit discard blanked
  the slot for one LLM round-trip every sub-final. Fix: queue-state gate —
  translate worker peeks `translate_q` (`_has_pending_tail(parent_start)`,
  drain+restore) and skips the discard if a successor tail is already queued
  (it'll overwrite naturally). Text-comparison gating rejected (cross-cycle ASR
  variance gives similar-not-identical text). Also removed two eager
  synchronous discards in the ASR worker that fired before the queued sub-final
  committed; cheap-path final's meta now carries `parent_start` so the
  post-commit gate handles both paths uniformly.
- [x] **#23 Live-region long lines now wrap in place** (`./client/render.py`).
  Root cause: `Console(soft_wrap=True)` makes every `print()` render with
  `no_wrap=True`/`overflow="ignore"` — including Live's refresh, where
  `LiveRender` CROPS over-long rows to console width (history only looked
  right because the terminal wrapped the uncropped flush output). Fix: new
  `_row` helper builds every subtitle row as
  `Text(..., no_wrap=False, overflow="fold")` (per-Text settings beat the
  print-level options); same kwargs on the `_header` assemble. Applied in both
  `_render` and `_flush_held_locked`, so live region and scrollback wrap
  identically.
- [x] **#24 Stale tail prov leaks in no-translate live mode** (`./client/pipeline.py`).
  No-translate path commits sub-finals directly with no #21/#22-style cleanup.
  Fix in the two `not translate_target` branches: sliced path discards
  `(ev.start, "tail")` when `commits and (not holds or ev.final)`; cheap-path
  final discards it when `ev.final and committed_until > 0`. `_emit` is
  synchronous so no blank-gap concern.
- [x] **#25 No-translate cheap-path commits every provisional** (`./client/pipeline.py`).
  The `not translate_target: _emit(...)` branch ran for provs too, giving every
  prov refinement a permanent scrollback entry. Fix: split by `ev.final` —
  finals call `_emit` (+ #24 discard), provs call `_emit_transcript` only.
- [x] **#26 Live timestamps drift from real wall time on long sessions**.
  Model: displayed `ts` = **segment START** anchored to real wall time; segment
  END stays sample/VAD-derived (`duration = ev.end - ev.start` at the emit
  sites), never wall-anchored. Displayed `ts` is a monotonic-anchored wall clock
  via `_mono_wall(mono)` = `capture_start_wall + (mono - capture_start)`.
  Re-anchor model (replaced the original per-commit re-snapshot): the
  `(capture_start, capture_start_wall)` pair is re-anchored to *now* at **every
  VAD speech-start** (capture worker, first event for an `ev.start`, under
  `_anchor_lock`), so NTP slew / suspend skew / crystal drift is re-absorbed
  once per utterance. The per-utterance start is then latched
  `utt_mono = now_mono - (ev.end - ev.start)`, back-calculating the true open:
  at first sighting `ev.end ≈ now` and `ev.start` is the open position, so this
  resolves to the real onset — correct for the primary VAD AND for recovery
  (whose ~1s `PRESPEECH_PAD_SECONDS` onset backtrack is already inside
  `ev.start`). `utt_start_mono` is reused for later events of the same utterance
  (popped on VAD final → bit-stable `ts` across provisional refreshes). Older
  in-flight utterances' `ts` stay correct after a re-anchor because the
  `(wall, mono)` pair is always an internally-consistent linear map (both
  sampled at the same instant). `_emit` does NOT re-snapshot (the per-VAD-start
  re-anchor covers slew); the pair's sole writer is the capture worker
  (first-chunk init + per-start re-anchor). `_audio_wall` keeps a separate
  session-pinned anchor (`_audio_anchor_wall`) for infrequent, non-re-rendered
  log lines. Lag math: `time.monotonic() - job.enqueued_at`; `datetime.now()` is
  display-only.
- [x] **#27 Recovery-opened segments run past `LIVE_MAX_SEGMENT_SECONDS`**.
  Part 1 (cap doesn't bite) was fixed by the recovery `watch_end` rework: in
  `LiveVAD.feed` (`./client/live_vad.py`) every open-segment path now falls
  through to the `open_samples >= self._max_samples` check unless it actually
  flushed (primary `start` can't early-return while a segment is open; primary
  `end` is ignored for recovery-owned segments without returning; the sidecar's
  `watch_end` advisory closes recovery segments on trailing silence), and a
  recovery-driven force-flush does NOT auto re-open. Part 2 (defence-in-depth):
  client mirrors the server cap via `TRANSCRIPT_MAX_INPUT_SECONDS` in
  `./client/transcribe.py`; `_transcribe_one` (`./client/pipeline.py`) trims
  `pcm_to_send` to the cap's head with a warning before `encode_wav`/POST, so a
  future VAD regression degrades to truncated text instead of a server 500.
- [x] **#29 Force-flush `n=1` splice, re-enabled with sample-accurate clamp**
  (`_transcribe_worker`, `./client/pipeline.py`). `can_splice` = `n>=2 OR (n==1
  AND committed_until>0)`. Splice start clamped to `max(held_start_abs_samples,
  prior_audio_end)` where `prior_audio_end = tail_start_abs + commits[-1].end`
  (cycle-end floor catches overlap with both prior and this-cycle commits).
  Clamp recomputed from locally-captured `committed_until` (independent of pop
  ordering). Removed `_MIN_TAIL_SECONDS` and `_TIME_EPS` (residue rolls forward
  anyway; both unused).
- [x] **#30 Pair-path correctness (held-line refinement actually lands)**
  (`_translate_worker`, `./client/pipeline.py` + `revise_held_translation` in
  `./client/render.py`). The tail-prov array path intended to keep the last
  committed line's translation revisable but had three bugs: (1) `last_committed`
  was never cleared when its held line flushed, so it fed an off-screen line into
  `translate_pair` and the revise silently no-op'd forever; (2) the `buf[-1]`
  history rewrite matched on transcript TEXT, so two identical short lines could
  cross-rewrite; (3) the array path could emit an empty `paired[1]`, the
  blank-translation flicker the queue-first design avoids. Also: the history fed
  to `translate_pair` INCLUDED the held line being re-translated, so the model
  saw it in the immutable "do not re-translate" block and echoed the stale copy
  instead of refining. Fixes: `revise_held_translation` returns bool → caller
  nulls `last_committed` on no-op; `buf` gains a 4th `_utt_key` field and rewrites
  match by key; `if paired[1]:` guards both array emits; `_hist_pairs` gains
  `exclude_last` to drop the re-translated line from the history block. #31
  builds on this.
- [x] **#32 Squash consecutive utterances, count-as-boundary** (`translate_pair`
  in `./client/llm.py`, `replace_held` in `./client/render.py`, cross-VAD branch
  in `_translate_worker`). #31's `translate_pair([A,B])` often returned 1
  translation instead of 2 — the model merging A+B because they ARE one
  continuous utterance (JA `男の人は`+`資料を…`). #31 treated count==1 as failure
  (→ per-line, A kept its wrong standalone translation). Now the COUNT is the
  boundary signal: `translate_pair` accepts length 1-or-2 (None only on
  refusal/parse-error/other counts); **len==1 → SQUASH** (merge A+B into one
  held line via the new `replace_held`, which rewrites the held transcript +
  translation + duration IN PLACE without flushing — A keeps its slot, start-ts
  and key so the next utterance squashes again; B gets no separate line and no
  buf entry; `buf[-1]` and `last_committed` become the merged line). **len==2 →
  SEPARATE** (the #31 refine-A-then-commit-B path, unchanged). **None →**
  per-line commit for B. The tail-prov array path is gated to `len==2` (a
  length-1 there isn't an actionable positional pair for a prov → per-line). CJK
  joiner via `is_spaceless`. A run of continuous speech merges into one growing
  line until the model returns 2 (a new thought) — natural boundary detection,
  no punctuation/length heuristic.
- [x] **#31 Cross-VAD pair translation** (`_translate_worker`,
  `./client/pipeline.py`). A short utterance A that's the grammatical subject of
  the next utterance B got a wrong
  standalone translation that never improved: the `translate_pair` refinement
  only fired for tail provs of the SAME growing utterance, and a separate fast B
  (no prov stage — ASR busy or speech < `LIVE_PROVISIONAL_MIN_INTERVAL_SECONDS`)
  committed A standalone before B existed. Fix: a new branch on the FINAL path
  re-translates `[A, B]` via `translate_pair` when B is a NEW utterance and A is
  still held, lands A's refinement via `revise_held_translation` BEFORE
  `_emit(B)` flushes A to scrollback (so the refined text is what scrolls —
  user chose "A scrolls up refined, live region shows just B", NOT a persistent
  2-slot region), then commits B with its paired translation. `last_committed`
  became a 3-tuple `(transcript, key, parent_start)`; the **parent-start guard**
  (`b_parent == last_committed[2]`) excludes sub-finals of the same long
  utterance (tail path's job) so only a genuinely new utterance pairs.
  No-double-emit: the branch `continue`s on pair success or falls through (no
  emit) to per-line on `paired is None` / timeout. Reuses the existing array-path
  machinery: `_hist_pairs(exclude_last=1)` to drop A from the immutable history
  block, key-matched (`buf[-1][3] == a_key`) buf rewrite, empty-`paired[*]`
  guards. Supersedes #18's "accept the one-round-trip gap, no cross-utterance
  carry" decision — #18 failed because it was a display-only STRING CARRY (paint
  A's old translation under B's transcript) that raced; this is a real
  key-guarded LLM re-translation where `revise_held_translation` no-ops if A's
  slot is gone, so a refinement can never land on the wrong line. Renderer
  unchanged (single held slot): the refine-before-flush ordering needs no 2-slot
  region. Cost: `translate_pair` (2 lines) replaces `translate` (1 line) on every
  new-utterance final while A is held — accepted.
- [x] **#28 `live_transcribe` always returns entries; whole-utterance path
  deleted**. Invariant: non-empty text ⇒ ≥1 entry. New `_ensure_entries` in
  `./client/transcribe.py` synthesises `[0, segment_duration]` when the
  aligner/cheap-JSON yields none (mirrors file mode's `_words_to_entries`);
  applied on BOTH the `with_entries=False` and `=True` paths. `live_transcribe`
  gained a `segment_duration` param (caller passes `len(pcm_to_send)/
  LIVE_SAMPLE_RATE` — 0-based on the trimmed tail). Pipeline `_transcribe_worker`
  then collapsed: the separate whole-utterance branch (cheap-path event-shift +
  duplicate `_emit`/`_emit_transcript` + `committed_until` bespoke handling) is
  gone — every non-empty text flows the unified sliced loop (n==1 final commits
  the lone entry, n==1 prov holds it as the tail). Removed the `sliced` gate
  (always true now) and the n==0 special-casing in `can_splice`. `with_entries`
  KEPT but RENAMED `with_entries` → `want_segments` (param in `live_transcribe`
  + local in `_transcribe_worker`). It is the pipeline's *intent* — "carve this
  open utterance into multiple entries so completed pieces promote to the live
  display before VAD closes it" — gated on `duration >= LIVE_ENTRIES_MIN_DURATION
  (= MAX_SEGMENT_SECONDS/2) or committed_until > 0`. That threshold is a pipeline
  PROMOTION-TIMING choice, NOT a backend-cost gate: the pipeline carries zero
  `TRANSCRIPT_BACKEND` refs. transcribe.py alone owns the backend cost of
  honouring the intent (qwen/anime-whisper word timestamps = a second
  forced-aligner pass under `_infer_lock`; faster-whisper segments are free).
  transcribe.py local `use_segments` → `backend_returns_segments` to avoid
  confusion with `want_segments`. Backend asymmetry: qwen
  returns nothing for `granularity=segment` (server sets `want_words = "word" in
  granularities`), so qwen has only two states (cheap text-only / full
  word-align) — no free middle tier; faster-whisper gives segments from the same
  decode regardless. `_ensure_entries` DEBUG-logs when the synthetic fallback
  fires.
- [x] **#33 HIGH — silent worker death stalls pipeline** (`./client/pipeline.py`).
  Per-iteration `except Exception: log.exception(...)` in `_transcribe_worker`
  (around `_transcribe_one`, inside the existing `asr_idle` try/finally) and in
  the translate loop — the translate body was extracted to a nested
  `_translate_one(job)` (`nonlocal last_committed`; loop-level `continue`s
  became `return`s) so the catch-all wraps one job per iteration.
  `_capture_worker` gained `except Exception: log.exception` and its `finally`
  now sets `stop_event` before `vad.close()`, waking the main thread's
  `stop_event.wait()`, whose teardown pushes the `None` sentinel through
  `asr_q` → `translate_q` so workers drain and the UI tears down cleanly.
- [x] **#34 `translate_q` concurrent drain race** (`./client/pipeline.py`). The
  ASR thread (`_enqueue_translate_with_backoff`) and the translate thread
  (`_drain_stale` + `_has_pending_tail`) both did non-atomic drain-all-then-reput
  on `translate_q` with no lock, so a final could be reordered behind a prov or a
  job dropped mid-interleave. Fix: new `translate_lock` guards all three
  drain/refill regions (backoff enqueue incl. its short-circuit final `put`,
  stale-drain via a new optional `lock=` param on `_drain_stale`, pending-tail
  peek). The lock is NEVER held across the worker's blocking `translate_q.get()`
  (only the bounded `get_nowait` drains), so it can't deadlock the producer.
  `asr_q` needs no lock (single producer/consumer). `_has_pending_tail` docstring
  corrected (it is not the only consumer).
- [x] **#35 Pair/squash translation made OPT-IN (`--translate-pair`), default
  off** (`_translate_worker` in `./client/pipeline.py`, flag in
  `./client/client.py`). Real-use verdict on #30-#32: the LLM was NOT reliable
  at deciding merge-vs-separate (squashed lines that were distinct, kept
  fragments apart), and in-place revision of an already-read held line is
  distracting — the viewer re-reads text they already consumed. Both pair paths
  (tail-prov held-line refinement AND the cross-VAD pair/squash branch) are now
  gated on a `translate_pairing` param plumbed from a new `--translate-pair`
  flag (requires `--translate`; live-only). DEFAULT: every line translates
  independently via the per-line `translate()` path — one call per line, no
  look-back, committed lines never revised or merged on screen. The #30-#32
  machinery (`translate_pair` count-as-signal, `replace_held`,
  `revise_held_translation`, `last_committed` bookkeeping, `_hist_pairs
  exclude_last`) is unchanged and still maintained — `last_committed` is still
  tracked when the flag is off (cheap, never read). If pairing quality matters
  again, improving the model/prompt beats re-enabling by default.
- [x] **#37 HIGH — recovery-driven force-flush lost the held trailing entry**
  (diagnosed from a real session log: ~4.5s of committed output vanished).
  The pipeline's `can_splice` assumed every force-flush final had splice
  carryover, but `LiveVAD.feed`'s cap branch skips the stash for
  recovery-driven flushes — so the held trailing entry's `request_splice()`
  was silently dropped at the `_flush_stash_pcm is None` guard and the tail
  prov was overwritten by the next utterance: text AND audio lost, guaranteed
  (not the "degraded path" the docstring claimed). Fix: `SegmentEvent` gains
  `spliceable: bool = False`, set True by the cap branch iff the stash was
  actually populated; `can_splice` in `_transcribe_one` now requires
  `ev.spliceable`, so non-spliceable force-flushes fall into the existing
  commit-at-chop-boundary path (warn line gained `spliceable=%s`). Also
  covers the latent razor-edge where a silence-end/watch-end final lands at
  dur >= MAX with no stash. Verified: primary-driven cap flush still
  `spliceable=True` + stash populated (splice path unchanged);
  recovery-driven cap flush `spliceable=False`, no stash, no auto re-open.
