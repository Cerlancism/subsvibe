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
- **Arabic digits**: cutlet leaves them unread (2番 -> "2 ban"). Left as-is —
  both a rule-based digit->kanji pre-converter and an LLM-prompt number rule were
  prototyped and dropped (the pre-converter can't tell a decimal from a
  version/IP, and the prompt rule over-applied counters). Numbers stay as cutlet
  renders them; not worth the complexity for a pronunciation gauge.

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

**Prompt examples removed (2026-06):** the original corrector prompt embedded
concrete examples (こんにちは, お兄ちゃん, 月曜日, "leave koohii") — three of which
were literally the eval's KNOWN_BAD cases, so the 5/6 baseline was partly
answer-leaked. Re-evaluated with a category-only prompt (no word examples):
still fixes all four headline cases (genuine generalization), no longer
over-corrects 兄→oniichan in 私の兄は医者です, but scores 4/6 (watakushi→watashi
and nan'you hi→nan youbi unfixed) and gained one regression: コーヒー restyled
koohii→Kōhī/kohii. Four generic phrasings tried (style rule, copy
character-for-character, ASCII-only output, doubled-vowels-by-design) — none
suppress it; the explicit "leave koohii" example was load-bearing at 4b. Accepted
as the cost of an unleaked prompt.

Sharpening the category bullets to chase the two remaining misses was also tried
and reverted: an "over-formal reading" elaboration on the kanji bullet broke the
お兄ちゃん fix, and a "rejoin with its natural compound reading" elaboration made
the model falsely join mi ni ikimashita→minikimashita in the GOOD block, while
neither miss budged. The conservatism rule ("change ONLY if it genuinely
mis-sounds") is what keeps 4b safe — watakushi and nan'you hi sit on its
protected side, so they are not reachable by generic prompt guidance without
breaking better-established fixes.

**Targeted examples re-added (final state).** Two minimal examples went back into
the prompt for exactly the unreachable cases: 私→'watashi' (word example) and
曜日 words read '...youbi' (pattern statement). Measured example mechanics at 4b:

- A word-example fixes exactly the word it names and does NOT generalize even to
  the nearest neighbor — an お姉ちゃん→oneechan example did not fix お兄ちゃん.
- Examples compete for attention: adding the お姉ちゃん example broke the working
  私→watashi fix. Keep the example count minimal; every addition needs a re-eval.
- A PATTERN statement does generalize: the single 曜日 rule fixes 月曜日, 何曜日,
  and 日曜日 alike. Prefer pattern statements over word lists when one exists.
- Side effect: with this prompt the koohii→kohii restyle blemish disappeared
  (stable across two runs).

Eval hygiene: `./tests/test_ja_romaji_llm.py` KNOWN_BAD now carries per-row leak
annotations (prompt-leaked rows confirm copying, not generalization) plus fresh
unleaked probes found by probing cutlet directly: お母さん (Ohahasan), 一日中
(Ichi nichichuu), 日曜日. Keep annotations in sync when editing the prompt.

Final score 6/9: all leaked + pattern rows fixed, plus konnichiwa/konbanwa from
category description alone. Unfixed (all cutlet-level errors passed through, no
fabrication): お兄ちゃん, お母さん (partial: Ohasan), 一日中.

Remaining 4b corrector blemishes (all *plausible-Japanese*, not phantom words):
koohii→kohii restyle (above); one stray inserted `n` (mi ni→min ni).

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
