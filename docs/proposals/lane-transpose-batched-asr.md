# Proposal: Lane-Transpose Batched Decode for faster-whisper

**Status:** proposal (not implemented)
**Scope:** file/batch mode only (`--input <file>`). Live mode is explicitly out of scope.
**Affects:** `./server/backends/faster_whisper.py` (child process), a new
extension class subclassing `faster_whisper.BatchedInferencePipeline`.

## Problem

faster-whisper exposes two decode paths off one loaded `WhisperModel`:

- **Sequential** (`WhisperModel.transcribe`) — one 30 s window at a time.
  Preserves `condition_on_previous_text` (each window is prompted with the
  prior window's output) and the temperature fallback ladder
  (re-decode a failed window at a higher temperature). This is what the
  current `_FasterWhisperChild` uses, with `condition_on_previous_text=True`
  and `beam_size=5`.
- **Batched** (`BatchedInferencePipeline.transcribe`) — VAD-splits the audio,
  stacks N chunks into one tensor, decodes the batch in parallel (~3-4x on
  long audio). To make parallelism legal it **hard-codes
  `condition_on_previous_text=False`** and **uses only the first temperature**
  (no fallback ladder). It also runs its *own* Silero VAD.

For SubsVibe's file mode we want the batched speedup **without** discarding
cross-window conditioning, the temperature ladder, or our own
(already-computed) VAD boundaries.

## Why the stock batched path is the wrong shape

1. **Redundant VAD.** SubsVibe already VAD-splits client-side. The stock
   batched path re-runs Silero on audio we've already segmented.
2. **No conditioning.** `condition_on_previous_text=False` is forced — every
   chunk decodes blind, weakening terminology/continuity across boundaries.
3. **No fallback.** A single scalar temperature per batch means a chunk that
   trips `compression_ratio_threshold` / `log_prob_threshold` has no recovery
   path and can emit a hallucination loop.

(1) is already solvable with the public API: passing `clip_timestamps` bypasses
`vad_filter` entirely. (2) and (3) are what this proposal forks.

## Core idea: contiguous lanes, column-wise (transposed) batching

Sequential conditioning and batch parallelism conflict only because the naive
batch groups *temporally adjacent* chunks — chunk N+1 depends on chunk N's
output, so they cannot decode in the same batch. The fix is to choose batch
membership so that no two chunks in a batch are adjacent, while still feeding
each chunk its true predecessor's text.

Split the chunk sequence into **L contiguous lanes** (`L = batch_size`), each a
contiguous stretch of the audio:

```
lane 0:  c0  c1  c2  ... c(k-1)
lane 1:  ck  ...
...
lane L-1: ...                  c(N-1)
```

Then decode **column by column** (the transpose): batch step *j* contains the
*j*-th chunk of every lane.

```
step 0 = [ lane0[0], lane1[0], ..., lane(L-1)[0] ]   # far apart in audio -> independent
step 1 = [ lane0[1], lane1[1], ..., lane(L-1)[1] ]
...
```

Two properties hold simultaneously:

- **Parallel-safe:** the L entries in any step come from L different lanes
  (audio regions `~total/L` apart), so none is the predecessor of another ->
  legal to decode in one batch.
- **Correctly conditioned:** within a lane, step *j* is prompted with the
  decoded text of step *j-1* — its **true temporal predecessor**. This is the
  same adjacency the sequential path preserves.

### Cost: lane seams

The only conditioning that cannot be preserved is across **lane boundaries**:
`lane1[0]` (first chunk of lane 1) cannot be prompted by `lane0[last]`, because
the last chunk of lane 0 is decoded in the *final* step while `lane1[0]` is
decoded in the *first* step — they are computed at opposite ends of the
schedule.

Number of conditioning breaks = number of lanes = `batch_size`.

Compare the three regimes over N chunks:

| Path                  | Conditioning breaks | Parallelism      |
|-----------------------|---------------------|------------------|
| Sequential            | 0                   | none (1 chunk)   |
| Stock batched         | N (none preserved)  | full (batch_size)|
| **Lane-transpose**    | **batch_size**      | full (batch_size)|

So `batch_size` becomes a quality/speed dial, but in a far better place than
stock: at `batch_size=8` over 64 chunks we go from 63 missing conditioning
links down to 7, at full parallelism. Lane seams land at predictable positions
(every `ceil(N/L)` chunks) and can optionally be softened by seeding the first
chunk of each lane with the **static** `initial_prompt` (glossary/history),
which is always batch-safe.

## Temperature fallback inside the lane schedule

The ladder is inherently per-chunk and sequential: decode at 0.0, evaluate
thresholds, re-decode only the failures at the next temperature, repeat. It
layers on top of the lane schedule as follows:

- Decode batch step *j* at temperature[0].
- Evaluate each entry against `compression_ratio_threshold` /
  `log_prob_threshold` / `no_speech_threshold`.
- Collect failed entries; re-batch **just those** at temperature[1]; repeat up
  the ladder until pass or list exhausted.
- A chunk's accepted text is only available to condition its lane-successor
  **after** it passes, so a mid-lane failure stalls that one lane's next step
  (but not the other lanes). Partial batches shrink as the ladder climbs.

This keeps recovery behaviour identical to sequential; it only costs some batch
fullness on hard audio — i.e. speedup degrades gracefully toward sequential
exactly when the audio is hard, which is acceptable.

## Implementation approach

**Do not vendor the module.** Subclass `BatchedInferencePipeline` in
`./server/backends/faster_whisper.py` (child side) and override the minimum.
Reuse upstream `encode`, `_split_segments_by_timestamps`, `add_word_timestamps`,
`get_prompt`, and `generate_with_fallback` unchanged — they are stable building
blocks.

Methods to override / add:

- **Lane planner** (new): given `clip_timestamps` (our VAD boundaries) and
  `batch_size`, build L contiguous lanes and the column schedule. Pure
  index math; no model calls. Handles ragged tails (last column is a partial
  batch when `N % L != 0`).
- **`generate_segment_batched`** (override): accept a **per-entry prompt list**
  instead of one shared prompt, so each lane carries its own running
  conditioning. Today it builds `prompts = [prompt.copy()] * batch_size`
  (one static prompt); we instead pass each entry the encoded prior-output of
  its lane. `model.generate` already accepts a list of prompts (one per batch
  entry), so the CT2 call is unchanged.
- **Segment generator** (override `_batched_segments_generator`): drive the
  column schedule instead of the consecutive `range(0, len, batch_size)` slice;
  after each step, decode text per entry and stash it as the next prompt for
  that lane; run the temperature-ladder re-batching loop; restore absolute
  timestamps from chunk offsets (upstream already does this via
  `restore_speech_timestamps` / `chunk_metadata["offset"]`).

Everything stays inside the spawned child process, behind the existing
`ModelWorker` boundary. The parent `FasterWhisperBackend` and the
`transcribe_result` contract (`{text, language, words, segments}`) are
unchanged, so the server and `./client/transcribe.py` need no edits.

## Configuration

- `TRANSCRIPT_BATCH_SIZE` (new) — `0`/`1` = current sequential path (default,
  zero behaviour change); `>1` = lane-transpose batched path with that many
  lanes. File mode reads it; live mode ignores it.
- Existing `TRANSCRIPT_BEAM_SIZE`, `TRANSCRIPT_MAX_INPUT_SECONDS`,
  device/compute resolution all carry over.

## Scope guards

- **Live mode unchanged and unaffected.** Each live request is one short
  VAD-closed segment = one chunk = a batch of one; batching cannot help and the
  sequential path is kept. The new code path is only reached when file mode
  passes a multi-chunk clip list with `TRANSCRIPT_BATCH_SIZE > 1`.
- **Default off.** With `TRANSCRIPT_BATCH_SIZE` unset, `_FasterWhisperChild`
  behaves exactly as today.
- **No public-API change.** Subclass + env knob only; no fork of the installed
  package, so faster-whisper can still be upgraded normally (the subclass
  depends only on the stable helper methods listed above).

## Open questions / risks

- **Lane length skew.** Our VAD chunks vary in duration; equal *chunk counts*
  per lane != equal *audio* per lane. For speedup this is fine (cost tracks
  chunk count, not duration). Worth confirming no lane ends up pathologically
  long.
- **Seam placement vs. content.** Lane seams fall at fixed index positions,
  which may land mid-sentence. Optionally snap lane boundaries to the longest
  silences among the VAD gaps so seams coincide with natural breaks — cheap,
  uses data we already have.
- **Word-timestamp path.** `add_word_timestamps` consumes `encoder_output` and
  per-segment sizing; confirm it composes with the transposed schedule (it
  operates per batch step, so it should, but needs a check).
- **Benchmark first.** Before building the full ladder+conditioning override,
  measure stock `BatchedInferencePipeline` fed our `clip_timestamps` +
  static `initial_prompt` on a representative file, to quantify the real
  speedup and the quality delta the conditioning is meant to recover. If the
  delta is small, the simpler path may be enough.
