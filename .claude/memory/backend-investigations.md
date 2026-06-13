# Backend investigations — backlog

Open questions / investigations on the ASR backends under `./server/backends/`.
Closed items are one-line outcome summaries (git history holds detail).

## Open

- [ ] **Does `faster-whisper` expose transcription confidence?** Check whether
  the faster-whisper backend (`./server/backends/faster-whisper.py`) can return
  a per-segment / per-word confidence score. faster-whisper's `Segment` /
  `Word` objects carry `avg_logprob`, `no_speech_prob`, and word-level
  `probability` — confirm which are populated for our config, how they map to a
  usable [0,1] confidence, and whether they survive into `transcribe_result`'s
  `{text, language, words, segments}` shape (`Backend` Protocol in
  `./server/backends/base.py`). Motivation: a confidence signal could feed the
  hallucination filter (`./server/hallucination_filter.py`) or client-side
  provisional/final gating instead of the exact-match blocklist.

## Closed
