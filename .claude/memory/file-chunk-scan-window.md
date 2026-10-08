---
name: file-chunk-scan-window
description: "Plan for file-mode chunking on a long-form ASR backend (minutes-to-hours per request) — tail-only cut scan, no onset anchoring — and the client/server restrictions to lift for it. Measured facts about the current 30 s chunker that shape the plan."
metadata:
  node_type: memory
  type: project
  originSessionId: 1b8b56a3-fc2c-439d-ac89-498a1d5348f8
  modified: 2026-10-08T16:15:12.268Z
---

# File-mode chunking for a long-form backend

Builds on the `CoarseChunker` section of [[recovery-vad-webrtcvad]]. The
current 30 s chunker in `./client/vad.py` stays as it is; the plan below
applies only when a backend that takes minutes-to-hours per request lands.

## Facts that shape the plan (measured 2026-10-09 on a 15 min Japanese stream recording)

- The whole-window detector ladder in `next_chunk` costs **~0.19 s per 30 s
  chunk** (8.4 s per 15 min of audio, ~5 % of a 142 s large-v3 run). At 30 s
  the VAD is not worth optimising; it only matters once windows are minutes
  long.
- Silero's `get_speech_timestamps` is stateful: scanning a sub-band of a
  window with some lead-in context does **not** reproduce what a scan from
  the window start finds, and more context does not converge it. Any
  tail-only scan therefore cuts differently from today's chunker — fine for
  a new backend with its own chunk size, not a drop-in for the 30 s one.
- **SRT diffs cannot validate chunker changes**: two runs of the same code
  on the same file differ by ~1170 diff lines (324 vs 326 entries) —
  faster-whisper on GPU is not run-to-run deterministic, and the ASR-driven
  cursor cascades every variance into later cuts. Compare chunker decisions
  (end, method, `speech_start`) directly instead.

## Plan (user direction, 2026-10-09)

- Per-backend `CHUNK_MAX_SECONDS` (env-only today; faster-whisper keeps 30 s,
  its training window). Keep the chunker rather than one request per file:
  it preserves progress/ETA, bounded retries and the `split_provisional`
  overlap for cue continuity.
- For the long-form backend, scan only the **last ~16 s before the window
  end** for the cut, and **skip `speech_start` / `anchor_first_entry`
  entirely** — no head scan, no onset carry. The anchor exists for Whisper
  pulling a chunk's first cue early; whether the new backend needs it is a
  separate question.
- Alternatives to the detector ladder are open for that backend (user:
  "reconsider other options").

## Restrictions to lift (verified by grep, 2026-10-08)

- `TRANSCRIPT_MAX_INPUT_SECONDS = 180` is read per backend in
  `./server/backends/*.py` (and trims live segments in
  `./client/pipeline.py`); a new backend simply does not apply it.
- File-mode ASR calls (`_transcribe_segment_asr_*` in `./client/client.py`)
  use `transcribe_client` from `./client/transcribe.py` with **no timeout
  and SDK default retries** (600 s, 2 retries): a long request past that
  re-uploads twice. Needs an explicit per-request `timeout` and
  `max_retries=0` on the file-mode path (live already does
  `with_options(max_retries=0)` in `./client/pipeline.py`).
- `SERVER_BACKEND_TIMEOUT_SECONDS = 30` covers only lifecycle calls
  (`/backend`, `/model/load`), not the transcription POST — no change.
- Upload: 2 h of 16 kHz int16 WAV ≈ 230 MB, buffered in RAM on both sides;
  no size cap found in `./server/server.py`. FLAC (~2×) or minute-scale
  chunks avoid it.
- `anchor_first_entry` cps caps and `split_provisional` assume short
  chunks; still valid at minutes, meaningless for one-shot.
