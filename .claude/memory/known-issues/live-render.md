# Live render polish — known issues & backlog

Issues from the live-rendering audit (`./client/render.py` +
`./client/pipeline.py` emit logic). Open items carry full context; closed
items are one-line fix summaries (git history holds the detail). Both
sections sorted by issue number.

## Open

- [ ] **#7 Refresh cadence**. `refresh_per_second=12` in `LiveRenderer.__init__`
  (`./client/render.py`) can feel jittery on slow remote SSH. Drop to 8Hz.

- [ ] **#10 Debug fields in header**. `_header` in `./client/render.py` shows
  `tail`/`sliced` tags plus `dur=`/`lag=`/`n=`/`prov=` — useful for debugging,
  noise for end users. Gate the whole row behind a log level or `--debug-render`
  flag.

- [ ] **#12 Light-theme colors**. `STYLE_*` constants (`grey80`/`grey42`/`cyan`)
  hard-coded at module top of `./client/render.py`, wash out on light
  terminals. Switch to a `rich.theme`-aware palette if light themes are in scope.

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

- [ ] **#23 Live-region long lines don't wrap until they scroll**. Affects the
  held-commit row and the prov row — anything rendered in-place via `Live`
  from `_render()`. Longer-than-terminal lines render on one visual row and
  clip; once the row scrolls to history (`_flush_held_locked` →
  `console.print(Group(...))`) it wraps. `Console(soft_wrap=True)` is set in
  `__init__` but Live's in-place renderable doesn't honour it for `Group`
  children the way `console.print` does on flush. Fix sketch: wrap each row in
  a width-aware container (`overflow="fold"` / `no_wrap=False` on each `Text`,
  or a `Padding`/`Layout` that respects console width). One fix covers both rows.

- [ ] **#27 Recovery-opened segments run past `LIVE_MAX_SEGMENT_SECONDS` →
  server 500**. Real capture: recovery opened a segment at 00:45:34, next
  finalisation 337s later, server rejected the 335.9s WAV (max 180s) →
  `server error 500`. Two parts:
  1. **10s force-flush cap doesn't bite.** `open_samples >= self._max_samples`
     in `LiveVAD.process_chunk` (`./client/live_vad.py`) only fires in the
     "no boundary this chunk" tail. If Silero re-emits `start`/`end` every
     chunk, or the recovery-end branch keeps short-circuiting, the cap is
     never reached. Audit the recovery-owned long-segment path (is Silero
     bouncing through the line ~224 early-return? is the recovery sidecar's
     `watch_end` stalling?), then add a hard ceiling at the top of
     `process_chunk` that fires regardless of branch.
  2. **Defence-in-depth.** `./client/transcribe.py` should split/refuse
     segments past a configurable max BEFORE the WAV POST (read
     `TRANSCRIPT_MAX_INPUT_SECONDS` or probe `/v1/health`), surfacing a
     warning instead of a 500. Fixing (1) restores the invariant; (2) keeps
     the client safe if a future cap is mis-tuned.

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
- [x] **#8 Separator readability** (obsolete after #17). `SEPARATOR` + composite
  header deleted.
- [x] **#9 Block height 3↔6 jump** (obsolete after #17). Two-slot phase gone;
  remaining 3↔6 is the #19 held-linger window, accepted.
- [x] **#11 `_install_log_handler` destructive** (`1152c0b`). Now preserves
  `FileHandler` instances when swapping in `RichHandler`, so file logs survive.
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
- [x] **#29 Force-flush `n=1` splice, re-enabled with sample-accurate clamp**
  (`_transcribe_worker`, `./client/pipeline.py`). `can_splice` = `n>=2 OR (n==1
  AND committed_until>0)`. Splice start clamped to `max(held_start_abs_samples,
  prior_audio_end)` where `prior_audio_end = tail_start_abs + commits[-1].end`
  (cycle-end floor catches overlap with both prior and this-cycle commits).
  Clamp recomputed from locally-captured `committed_until` (independent of pop
  ordering). Removed `_MIN_TAIL_SECONDS` and `_TIME_EPS` (residue rolls forward
  anyway; both unused).
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

## Out of scope (file-mode audit)

File mode (`--input` → `.srt`) checked against recent live work, no regression.
Only shared-module change was `./utils/subtitle.overlapping_text` joiner
`" "` → `"\n"` (`ebfddee`), affecting `--context-src` prompt formatting only.

# Unsorted Reviews

## Real Bugs

### HIGH — Silent worker death stalls pipeline (`pipeline.py`)

`_transcribe_worker`, `_translate_worker`, and `_capture_worker` catch only API errors.

Any other exception, such as `KeyError` on an entry dict, NumPy errors, `encode_wav` failures, or device glitches, can kill the daemon thread silently.

Result:

* No `None` is pushed.
* `stop_event` is never set.
* The UI can freeze forever.

**Fix:** Add a per-iteration catch-all:

```python
try:
    ...
except Exception:
    log.exception(...)
```

Also set `stop_event` when capture dies.

---

### HIGH/MED — `translate_q` concurrent drain race (`pipeline.py`)

`_enqueue_translate_with_backoff` runs on the ASR thread, while `_drain_stale` and `_has_pending_tail` run on the translate thread.

Both perform non-atomic drain-all-then-reput operations without a lock.

The `_has_pending_tail` docstring claim that the “translate worker is only consumer” is wrong.

Worst case:

* A final job is reordered behind a provisional job.
* A job is dropped.

**Fix:** Use one lock around all `translate_q` drain/refill operations.

---

### MED — `last_committed` stale key causes wrong-context translation

`pipeline.py`, `_translate_worker`

`last_committed` is set on every final, but it is never cleared when the held slot flushes.

The next tail feeds stale `pair_with[0]` into `translate_pair`.

`revise_held_translation` correctly no-ops, but the tail translation was built against an off-screen line, and `buf[-1]` may get rewritten.

**Fix:** Null `last_committed` when the held slot is discarded.

---

### MED — `buf[-1]` rewrite by text equality (`pipeline.py`)

`buf[-1][1] == pair_with[0]` matches on transcript text, not key.

Two identical short lines, such as:

```text
Okay.
Yeah.
```

can cause the wrong history entry to be rewritten.

**Fix:** Key by `_utt_key`.

---

## Low / Noise

* `_emit_translation` can emit an empty `paired[1]` under `keep_held`, causing the blank-translation flicker that the design exists to avoid. The array path lacks the `elif translation:` guard that the per-line path has.

* `_ensure_entries` can produce a zero-span entry if `segment_duration ≈ 0`. This is pathological, and file mode repairs it anyway.

* `_last_provisional_sample = 0` in `_flush` is a dead write. Harmless.
