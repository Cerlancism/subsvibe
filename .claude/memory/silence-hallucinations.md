# Silence hallucination dataset + server filter

## Dataset

`./server/data/silence_hallucinations.json` — texts each ASR backend/model/
language emits for pure silence. Built by `./tests/test_silence_hallucinations.py`
(merge-only; re-runs extend, never overwrite). `_runs` holds per-sample-length
run/hallucinated counts; `_meta` lists the silence samples used.

- Samples are gitignored (`tests/samples/*.mp3`); regenerate with
  `bash scripts/dev/gen-silence-samples.sh` (ffmpeg, 1/2/3/5/10/15/30 s).
- Sweeps ran 10–13 repeats per cell on temp servers (port 8001 — never the
  user's persistent 8000 server).

Key findings:
- faster-whisper large-v3, qwen, anime-whisper: hallucinate on ~100% of
  silence runs at every duration; medium varies (0–100% per language/length).
- Hallucinations are stable/deterministic — same text 10/10 in most cells —
  so an exact-match blocklist works; duration is not a defense.
- anime-whisper always emits exactly `…`.
- small/base/tiny only hallucinate "You" on en.

## Server filter

`is_silence_hallucination` in `./server/silence_filter.py`, called from the
transcribe endpoint in `./server/server.py`. On by default;
`TRANSCRIPT_SILENCE_FILTER=0` disables.

- Lookup is backend → active model id → language specific (user requirement:
  "read by backend kind, model and language specific"). Language resolves via
  `to_iso_code` in `./utils/language.py` (handles qwen's canonical-name
  detected language); unknown language falls back to the union of that
  model's languages.
- Match = whole output equals a recorded text after stripping whitespace +
  Unicode P/S categories + casefold (`_normalize`). Partial matches inside
  real speech are never touched. Pure-punctuation outputs blank only when the
  model has a pure-punctuation entry (anime-whisper `…` normalizes to "").
- On match the server blanks `text` and empties `segments`/`words`.
- Blocklist loads once per process (`lru_cache`); a model id absent from the
  dataset gets no filtering.

## Gotchas

- [ ] Dataset sweeps must run against a server started with
  `TRANSCRIPT_SILENCE_FILTER=0`, or known hallucinations return blanked and
  `_runs` under-reports (noted in the test docstring).
- Blocklisting is inherently lossy: a genuine lone "Thank you." utterance in
  en on large-v3 gets blanked too. Accepted trade-off.
