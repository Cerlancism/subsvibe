# Japanese romanization — findings & decisions

The romaji line is a **display gauge** for transcription quality, not a
standards-compliant transliteration. The bar is "sounds correct read aloud" so a
viewer never mistakes a romaji quirk for an ASR error. Everything below is judged
by that bar.

## Engine: cutlet (default on main)

`_make_cutlet` in `./utils/romanize.py` uses cutlet (fugashi/MeCab + bundled
unidic-lite). Key config, all decided empirically:

- `use_he=False`, `use_wo=False` — **inverted from intuition**: True *keeps* the
  literal he/wo. False gives the spoken particle values へ=e, を=o. `use_wa`
  already defaults True (は=wa). Getting this backwards is the bug that motivated
  the whole audit.
- `use_foreign_spelling=False` — keeps katakana phonetic (コーヒー=koohii) instead
  of guessing English spellings.

**pykakasi was evaluated and rejected** as an alternative engine: it emits literal
particles everywhere (これは=koreha, を=wo, へ=he) — strictly worse for the gauge.
cutlet reads particles correctly. Keep cutlet.

## cutlet's failure classes (from token-level pos1/kana/pron inspection)

- **A — lexicalized particle**: single token where `kana != pron`. こんにちは has
  `kana=コンニチハ` but `pron=コンニチワ`; cutlet romanizes from kana → "konnichiha".
  This is the **only detectable class** (kana≠pron on a non-particle token — note
  real particles は also have kana≠pron but `pos1=助詞` triggers cutlet's correct
  wa-handling). こんばんは is the same.
- **B — context kanji reading**: 兄 alone reads *ani* (right in 私の兄, wrong in
  お兄ちゃん where it should be *nii*). Token is fully self-consistent
  (kana==pron) — cutlet has no idea it's wrong. **Undetectable per-token.**
- **C — segmentation glitch**: unidic-lite splits 月曜日 into 月曜+日 → "getsuyou
  hi". Each token is individually correct; only the boundary is wrong.
  **Undetectable from the token stream.**

The exceptions-based fix (overriding class-A surfaces) was explored and parked on
the `experimental/japanese-romanisation` branch — it can only safely cover the
~2 class-A entries; B/C need context and corrupt standalone readings if forced.
Full unidic (vs unidic-lite) was also tried and reverted — ~1GB download, marginal
gain, orphaned-dir cleanup pain.

## LLM enhancement — the corrector decision

Goal: use the LLM to fix cutlet's class-A/B/C mis-soundings. Evaluated three
shapes against cutlet (harness: `./tests/test_ja_romaji_llm.py`):

1. **From-scratch, generic prompt** — rejected. At 4b it HALLUCINATES phantom
   words (名前=nama, 兄=ane, invented particles). For a gauge, random fabrication
   is *worse* than cutlet's predictable errors — it masquerades as an ASR fault.
2. **From-scratch, JA-specialized prompt** — better, but still fabricates at 4b
   (美味しい=hitoame). Only clean at 9b. Rejected for the 4b target.
3. **Hybrid corrector** (`romanize_ja_fix` in `./client/llm.py`) — **chosen.**
   Feeds the LLM cutlet's draft + the source and asks it to EDIT only what
   mis-sounds. The draft anchors the model: the job shrinks to a few tokens, so
   even at 4b there's no room to fabricate. Won at 4b: **5/6 on known-bad, ~zero
   regressions** on cases cutlet already gets right (returns correct drafts
   verbatim, including the `oishii` the from-scratch prompt mangled).

Remaining 4b corrector blemishes (both *plausible-Japanese*, not phantom words):
私の兄は医者です over-corrects 兄→oniichan; one stray inserted `n` (mi ni→min ni).

**Intended placement (not yet wired):** committed (final) lines only. Provisional
previews stay pure cutlet so the ~1Hz refresh path takes no LLM call. Optional
gate: run the corrector only when tokens show the class-A signal (kana≠pron), so
the clean common path costs zero calls.

## Tests — gate vs. harness (important for future maintenance)

- `./tests/test_ja_romaji.py` — **deterministic regression gate.** Pure cutlet,
  no network. PASS corpus fails loudly on real regressions (e.g. particle flags
  flipping back); KNOWN_BAD corpus does NOT fail on drift — prints a
  "promote to PASS" notice if cutlet ever improves. Safe to automate.
- `./tests/test_ja_romaji_llm.py` — **evaluation harness, NOT a test.** Calls a
  live LLM → non-deterministic, needs a running server + model pulled, no
  assertions, always exits 0. Run by hand. If CI/pytest is ever added, EXCLUDE
  this (like `./tests/test_qwen_aligner.py`, which also hits real models).
