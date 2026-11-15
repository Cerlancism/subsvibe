---
name: mai-phrase-list
description: Keyword-biasing phrase lists for mai-transcribe-2 (built 2026-11-15) - how --phrases reaches Azure through OpenRouter, the measured caps (1000 phrases, 50 chars), and the dilution curve that says keep lists short and targeted.
metadata:
  type: project
---

# Phrase lists on mai-transcribe-2

Builds on [[file-chunk-scan-window]] (the `openrouter` backend).

## How it flows (as built 2026-11-15)

- Client: `--phrases FILE` (repeatable, concatenated in order) and
  `--phrase-groups a,b` (picks named groups of grouped files, in the order
  named). `load_phrases` / `set_session_phrases` in `./client/transcribe.py`;
  `request_backend_fields` adds a `phrases` form field (JSON array) to every
  request, including the hybrid commit route. Warns past
  `PHRASE_LIST_SOFT_MAX` (800, 80% of the cap; user's choice 2026-11-15 -
  the warning is a cap guard, not a quality gate). Not supported with `--llm-asr`.
- Server: `_parse_phrases` in `./server/server.py` -> `transcribe_result(...,
  phrases=)` (keyword arg on the `Backend` Protocol; local backends ignore).
- Remote: spec field `phrase_list_provider` ("azure" on `MAI_SPEC` only) ->
  body `provider.options.azure.phraseList.phrases`. `_fit_phrase_list` drops
  phrases over `AZURE_PHRASE_MAX_CHARS` (50) and cuts to
  `AZURE_PHRASE_LIST_MAX` (1000). OpenRouter silently ignores unknown option
  keys, so a wrong shape fails silently - verify by output, not status.
- Bench: run keys `phrases`, `phrase_groups`, `tag` in
  `./tests/run_file_transcripts.py`. Unit test `./tests/test_phrases.py`.
- Lists are per job and live outside the repo; a grouped file (named
  groups of phrases) is the format for one broad list to pick subsets from.

## Measured (one 5 min Japanese sample, text-only API calls, vs control SRT)

Score = target names spelled right /32, role lines with both character and
cast right /15 (the coherence check). Same config reruns reproduce within 1.

- none 3/0; 11 hand-picked 9/1; target series 22 18/6; target + guests 30
  **24/10**; + core terms 82 19/6; core + target 74 18/4; broad 234 14/2;
  500 6/1; 1000 3/0.
- Dilution is by count: junk padding hurts as much as competing names.
- Order matters but not monotonically (target group first beat last at 234,
  lost at 74) - do not claim "first wins".
- Hard caps: 1001 phrases -> 400; a 51-char phrase -> 400 (50 ok).
- Danger: 200 OK but garbage. 1000 x 14+ char phrases (or 250 x 20 with
  repeated ー) made the transcript degenerate into loops; 500 natural 19-char
  phrases (~10k chars) silently dropped content. No false positives seen in
  any normal-size run.
- Ceiling: the list carries spelling only, no reading. Names whose reading is
  far from the spelling stay wrong (unusual kanji name readings, foreign
  names written in kanji, a name heard as a commoner homophone).

**How to apply:** give each job a focused list (its own names + guests +
core terms), well under ~100; a broad generalised list is a fallback, not
the default.
