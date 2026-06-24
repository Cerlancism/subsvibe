# Recovery VAD: webrtcvad (replaces the Silero recovery pass)

The *recovery* VAD in `./client/live_vad.py` — the second-chance,
recall-over-precision pass over the peak-normalised silence accumulator
after the primary VADIterator missed speech — is webrtcvad (energy/GMM,
per-frame, model-free). The Silero stateless recovery path
(`get_speech_timestamps` at threshold 0.1) was removed; Silero remains the
**primary** live VAD (`VADIterator`) and the file-input seed pass in
`./client/vad.py` untouched. (The file-input *subslice* chain later adopted
webrtcvad too — see the last section.)

## Status

- [x] webrtcvad installed and locked (`webrtcvad==2.0.10` in
  `requirements.txt`; built from source on this machine — needs MSVC on a
  fresh Windows box, `webrtcvad-wheels` is the prebuilt fallback).
- [x] `webrtcvad_speech_timestamps` in `./client/live_vad.py`: the canonical
  py-webrtcvad vad_collector hysteresis (300ms ring, >90% trigger/de-trigger)
  adapted to return `[{start, end}]` sample spans. Span end is the last
  *voiced* frame, not the de-trigger frame, so the recovery loop's
  trailing-silence math isn't skewed by the ring padding.
- [x] `_recovery_loop` runs webrtcvad only; tuning knob is the module
  constant `RECOVERY_WEBRTCVAD_AGGRESSIVENESS` (set to 3 — see primary-reset
  section below; recovery is a quiet backstop, not the lead detector).
- [x] Recovery-owned segments run on the recovery VAD start-to-end: primary
  start/end events are ignored while open (no takeover — user decision);
  `_flush` resets the primary's states at the recovery close.
- [x] Idle refresh of the primary (`PRIMARY_IDLE_RESET_SECONDS = 1.5`):
  reset_states whenever no segment is open and no VAD finding / reset for
  1.5s. See rationale below.
- [ ] Live validation — target scenario is a concert recording (see below).

## Primary habituation & the reset policy (measured)

Silero VADIterator is a stateful RNN; sustained audio conditions its hidden
state and suppresses speech probabilities (speech scoring 0.74 fresh scored
0.02 after 20s of noise; even 3s of *faint* noise halved detection). User's
real-world symptom: at concerts, singing→speech transitions were missed
until SubsVibe was restarted — restart = reset_states with extra steps.

Findings that shaped the design:
- Reset *mid-utterance* is useless (0.054): Silero needs to see the onset
  transient on fresh state. So no reset at recovery *hit*.
- Reset at a *silence boundary before the next onset* is what works → reset
  at recovery-segment close + idle refresh while closed.
- Ungated periodic resets can wipe an in-progress pre-threshold ramp
  (measured kills at +192ms/+768ms after onset), but the stateless recovery
  backstop catches anything dropped ~1.5s later — worst case is latency,
  not loss. Hence idle reset is ungated (a webrtcvad-confirmed-quiet gate
  was considered and rejected: at a concert the tail is never quiet, the
  gate would hold forever).
- False positives from resets on noise: none observed (max 0.153).

Concert division of labour: fresh Silero is the only cheering-robust
discriminator (crowd roar scores low, speech-over-crowd high) — idle
refresh keeps it ready. webrtcvad cannot reject music/crowd at any
aggressiveness, so recovery runs at 3 to minimise junk segments over
singing/cheering; it exists to catch quiet speech the primary misses, not
to lead. Neither VAD reliably ignores *singing* — lyric segments during
songs are accepted (user: fine).

## Why it might work

The recovery pass peak-normalises its window first (`peak_normalize` in
`./client/capture.py`, cap +20dB), which cancels webrtcvad's core weakness
(quiet speech is invisible to an energy-based detector). Per-frame
classification is orders of magnitude cheaper than `get_speech_timestamps`
over a 30s window (~150ms on CPU), so the sidecar budget stops mattering.

## Known risk (observed in synthetic smoke test)

Normalisation amplifies the noise floor too. Broadband noise at +20dB read
as voiced at *every* aggressiveness 0–3 in a synthetic test — white noise is
webrtcvad's worst case, real desktop silence is cleaner, but two failure
modes to watch live:

1. `watch_onset`: spurious recovery hits on amplified noise floor → junk
   segments fed to ASR.
2. `watch_end`: noise floor keeps reading as speech → recovery-opened
   segments never see trailing silence → only close on force-flush at
   LIVE_MAX_SEGMENT_SECONDS (the exact loop the recovery-end logic exists
   to prevent).

Aggressiveness is already at the strictest (3). If (2) still shows up,
webrtcvad is wrong for `watch_end` and a split setup (webrtcvad onset /
silero end) or restoring the silero recovery pass is the answer (the
removed path was a stateless `get_speech_timestamps` call at threshold
0.1 over the same normalised window — trivial to reinstate). During
sustained cheering/music, (2) is *expected* — recovery segments close via
force-flush at LIVE_MAX_SEGMENT_SECONDS, which is tolerated: each close
resets the primary, and the primary (not recovery) is the lead detector.

## File-input subslice pass (added later, same detector)

`SUBSLICE_PASSES` in `./client/vad.py` is now a chain of `(engine, params)`
tuples consumed one per recursion level of `_split_oversized`:
silero (threshold 0.8, min_silence 50ms) → webrtcvad (aggressiveness 3) →
quiet-split (energy argmin, unchanged final fallback) → even-split.

- Same role as live recovery: a second-chance pass that only runs when the
  pass before it failed (here: a piece still > MAX_SEGMENT_SECONDS).
- `_webrtcvad_boundaries` reuses `webrtcvad_speech_timestamps` from
  `./client/live_vad.py` verbatim (same 16kHz, sample-span contract) and
  per-piece peak-normalises first — file-level normalisation targets the
  global peak, so a locally quiet span is otherwise invisible to an
  energy-based detector.
- Boundary contract matches `_vad_boundaries`: speech-*start* times, so
  leading silence attaches to the preceding piece; duplicate/zero-length
  pieces are dropped by the recursion's `e > s` guard and slivers are
  merged by `_bundle_to_target`.
- The ">=300ms unvoiced gap makes cuts conservative" assumption was wrong on
  real (concert) audio: webrtcvad de-triggered on every brief pause and
  shattered spans into ~0.5s pieces, and `_bundle_to_target`'s old
  either-short merge rule then glued sliver runs toward MAX (30s) with tiny
  orphans at bundle caps. Fixed with one knob, `TARGET_SEGMENT_SECONDS`
  (5.0), driving all three shaping steps: `_thin_boundaries` drops subslice
  cut points closer than it to the previously kept one (applies to *both*
  subslice passes — silero at min_silence 50ms is equally chatty), bundling
  merges only while the current bundle is below it, and a final file tail
  shorter than it folds back into the previous segment. A short mid-file
  piece needs no special case: it starts the next bundle and grows forward.
  (A separate MIN_SEGMENT_SECONDS was considered and rejected — user: reuse
  the existing post-processing knob instead.)
- Live side checked and deliberately untouched: recovery consumes only the
  first span's start (onset) and last span's end (trailing-silence gate =
  LIVE_MIN_SILENCE_MS), so de-trigger chatter creates no extra live
  segments, and live segments are utterance-shaped by design — the 3-5s
  aim is a batch-chunking concern only. `webrtcvad_speech_timestamps`
  itself is shared and unchanged.
- Validated with synthetic smoke test (noise bursts + 1s gaps: boundaries
  found at the gaps; unbroken noise: fell through to quiet-split) plus
  pure-function tests of thinning/bundling (sliver runs now bundle to ~5s).
- Validated end-to-end on real-speech test files (gradio sample clips +
  music bed): chatty speech with 0.2-1.2s pauses gave seed pieces down to
  0.40s but final segments 5.9-10.7s; 90s speech-over-music gave finals
  5.4-10.3s; zero segments under 3s in either. Direct exercise of the
  webrtcvad pass on the 90s busy span: raw boundary gaps as small as
  0.03s (the shatter source, confirmed), all removed by thinning.
  Aggressiveness is NOT the spacing knob — lower modes mark more frames
  voiced (fewer de-triggers, fewer cuts) but spacing is guaranteed only by
  `_thin_boundaries`/`TARGET_SEGMENT_SECONDS`. On sustained busy audio
  webrtcvad never de-triggers at all (one 90s span) and the chain falls
  through to quiet-split as designed.
- [ ] Validate on a real long-monologue file where the silero 0.8 subslice
  used to fall through to quiet-split.

## Live early-split (added later — brings the file-side chain to live)

The live path now has an early-split fallback (`_scan_split_point` in
`./client/live_vad.py`), porting the file-side webrtcvad→quiet-split chain to
the streaming pipeline. Motivation: a continuous talker who never pauses
`LIVE_MIN_SILENCE_MS` (400ms) used to be held until the hard
`LIVE_MAX_SEGMENT_SECONDS` (16s) force-flush, which chops mid-word. Now, once a
*primary*-owned segment grows past `LIVE_SPLIT_TARGET_SECONDS` (5.0,
`./client/capture.py`) **and** Silero has not finalised it on silence, a
fallback scan looks for a seam and finalises there.

- Strictly a fallback, in this priority order: Silero silence 'end' (always
  wins when it fires) → webrtcvad span gap → energy quiet-window → 16s hard
  cap. Mirrors `SUBSLICE_PASSES` + `_quiet_split` ordering on the file side.
- Runs **inline on the capture thread** (not the recovery sidecar — user
  decision): the audio is already in `_open_pcm`, and threading a third
  sidecar mode through the generation machinery was higher risk. Paced to one
  pass per `SPLIT_SCAN_INTERVAL_SECONDS` (0.5) so webrtcvad never runs every
  32ms chunk. Gated on `not _open_via_recovery` (recovery segments close via
  the sidecar's watch_end, untouched).
- On a hit: `_split_open_pcm_at` partitions `_open_pcm` head/tail at the seam
  (lossless, verified), flushes the head as a final, re-opens at the seam
  keeping the tail. No force-flush stash / `request_splice` here — nothing is
  dropped (tail preserved directly), and the finalised segment is < 16s so the
  pipeline's force-flush detector (`duration >= LIVE_MAX_SEGMENT_SECONDS`,
  `./client/pipeline.py`) won't request a splice for it anyway.
- webrtcvad's 300ms hysteresis padding means `webrtcvad_speech_timestamps`
  only resolves gaps from ~500ms up (measured: 400ms gap → 1 span, 500ms →
  detected 360ms). So `LIVE_SPLIT_MIN_GAP_MS` (200) is effectively reached
  only by clear pauses; sub-500ms dips fall to the energy pass. This is *why*
  the two-tier chain exists live (a single webrtcvad pass couldn't hit the
  sub-400ms intent).
- Energy pass (`LIVE_SPLIT_QUIET_*`) is the live analogue of `_quiet_split`
  but with one extra gate the file side lacks: `LIVE_SPLIT_QUIET_MAX_RATIO`
  (0.5) — the quietest window must be ≤ half the band median or it returns
  None. File-mode `_quiet_split` always cuts because it only runs on
  already-oversized pieces; the live pass runs on *every* over-target scan, so
  without the ratio gate seamless continuous speech would be cut at an
  arbitrary ~5s point every scan. With it, gapless audio falls through to the
  16s cap.
- **Minimum-head / earliest-seam selection** (corrects the original
  widest-gap / global-argmin port): both passes now require the finalised head
  `[_open_start_sample, seam]` to be ≥ `LIVE_SPLIT_TARGET_SECONDS` and take the
  *earliest* qualifying seam, not the global widest gap / quietest window. This
  is the live mirror of `_bundle_to_target` ("don't break a bundle until it
  reaches TARGET"). Two reasons it matters: (1) the original picked the widest
  gap *anywhere* in the buffer, so a clean pause early in the segment produced
  a sub-target head — the exact short-clip ASR starvation TARGET exists to
  prevent; (2) global-widest/quietest drifted segments toward the 16s cap
  rather than landing just above TARGET. A clean pause *before* TARGET is now
  ignored as a split point (head would starve ASR); the scan retries next tick
  until a seam past TARGET appears, else the hard cap fires. The file side
  doesn't need this because it runs offline on complete segments (global
  optimum is correct there); the live path runs on a growing prefix, where
  "earliest acceptable past TARGET" is the right rule.
- This supersedes the earlier "Live side checked and deliberately untouched"
  note in the file-input section above (that was true when only the *recovery*
  consumer existed; the early-split is a new, separate live consumer of
  `webrtcvad_speech_timestamps`, which itself remains shared and unchanged).
- [x] Validated with synthetic smoke test: clear 600ms gap → webrtcvad cut at
  the gap; 150ms trough → energy fallback cut at the trough; flat continuous
  speech → None (falls to hard cap); head/tail partition lossless.
- [x] Validated min-head / earliest-seam selection (both passes): seam before
  TARGET ignored → None or skipped to next; earliest qualifying seam past
  TARGET chosen (not widest/global); head fed to ASR always ≥ TARGET.
- [ ] Live validation on continuous-monologue audio (no 400ms pauses): confirm
  segments land in (5s, 16s) instead of all at the 16s chop, and seams don't
  chop mid-word audibly.
