# Recovery VAD: webrtcvad (replaces the Silero recovery pass)

The *recovery* VAD in `./client/live_vad.py` — the second-chance,
recall-over-precision pass over the peak-normalised silence accumulator
after the primary VADIterator missed speech — is webrtcvad (energy/GMM,
per-frame, model-free). The Silero stateless recovery path
(`get_speech_timestamps` at threshold 0.1) was removed; Silero remains the
**primary** live VAD (`VADIterator`) and the file-input pass in
`./client/vad.py` untouched.

## Status

- [x] webrtcvad installed and locked (`webrtcvad==2.0.10` in
  `requirements.txt`; built from source on this machine — needs MSVC on a
  fresh Windows box, `webrtcvad-wheels` is the prebuilt fallback).
- [x] `webrtc_speech_timestamps` in `./client/live_vad.py`: the canonical
  py-webrtcvad vad_collector hysteresis (300ms ring, >90% trigger/de-trigger)
  adapted to return `[{start, end}]` sample spans. Span end is the last
  *voiced* frame, not the de-trigger frame, so the recovery loop's
  trailing-silence math isn't skewed by the ring padding.
- [x] `_recovery_loop` runs webrtcvad only; tuning knob is the module
  constant `RECOVERY_WEBRTC_AGGRESSIVENESS` (set to 3 — see primary-reset
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
webrtcvad is wrong for `watch_end` and a split setup (webrtc onset /
silero end) or restoring the silero recovery pass is the answer (the
removed path was a stateless `get_speech_timestamps` call at threshold
0.1 over the same normalised window — trivial to reinstate). During
sustained cheering/music, (2) is *expected* — recovery segments close via
force-flush at LIVE_MAX_SEGMENT_SECONDS, which is tolerated: each close
resets the primary, and the primary (not recovery) is the lead detector.
