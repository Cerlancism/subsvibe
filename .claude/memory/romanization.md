# Romanization (ja/zh/ko display gauge)

Romaji/pinyin/RR shown on the subtitle display is a **read-aloud gauge of ASR
quality** — a viewer sounds it out to sanity-check what Whisper heard. It never
feeds the LLM translator or the ASR prompt. So the bar is "sounds correct when
read aloud", NOT romanization-standard compliance.

Code: `make_romanizer` in `./utils/romanize.py`. JA overrides:
`JA_ROMAJI_EXCEPTIONS` in `./utils/data/ja_romaji_exceptions.py`.

## Engine choice: cutlet, not pykakasi

- [x] Evaluated pykakasi (the pre-`d8d47c5` engine) head-to-head against cutlet
  + full UniDic. **cutlet wins decisively** on the thing we care about: it
  reads grammatical particles correctly (は→wa, へ→e, を→o) via morphology,
  while pykakasi emits literal kana everywhere (これは→"koreha", を→"wo",
  こんばんは→"konbanha"). Do NOT switch back to pykakasi.
- cutlet flags are inverted from intuition: `use_he`/`use_wo` must be **False**
  for へ→e / を→o (True *keeps* literal he/wo). `use_wa` defaults True (hepburn).

## Why the JA exceptions list is exactly 2 entries (and can't safely grow)

cutlet's `exceptions` hook is **token-keyed and context-free** — looked up per
tokenized word surface, not on the whole input string. That constraint sorts
every mis-sound into three classes:

- **Class A — single-token lexicalized particle.** `こんにちは` is ONE token
  (感動詞) with kana=コンニチハ but pron=コンニチワ; cutlet reads kana → "ha".
  Safe to override: the hook fires (single token) and there are no context side
  effects. A wide scan (~60 greetings/interjections/fillers) flags **exactly**
  `こんにちは` and `こんちは`. Everything else (こんばんは→"Konban wa", では→"De
  wa") tokenizes the particle separately and already sounds right. → **these 2
  are the whole list.**
- **Class B — kanji-reading ambiguity.** お母さん→"Ohahasan", 兄さん→"Anisan",
  お父さん→"Ochichisan". The analyzer picks a valid-but-contextually-wrong
  reading (母=ハハ, 兄=アニ — correct standalone, wrong in the お~さん honorific).
  **Cannot fix via exceptions:** a token key like 兄→nii fixes お兄ちゃん but
  corrupts the *common* standalone reading (私の兄→"nii", should be "ani"). Net
  harm, because standalone 兄/母/父 are far more frequent than the honorific.
  Would need context-aware substitution the hook can't do. → **excluded.**
- **Class C — segmentation glitch.** しょうり (hiragana)→"Sho uri". Input-
  specific; the kanji form 勝利→"Shouri" is correct and is what ASR actually
  emits. → **excluded.**

Upstream cutlet issues confirm these are known and won't-fix (#33 こんにちは, #62
しょうり, #55/#68 family terms, #52 exceptions-are-per-token). subsvibe repo has
no open issues of its own.

## Re-verifying after a cutlet / UniDic bump

Run the Class-A scan: push a broad greeting/interjection list through a stock
`cutlet.Cutlet()` and flag any **single-token** word whose `kana` ends in ハ/ヘ/ヲ
while `pron` ends in ワ/エ/オ. Only single-token flags belong in the list
(multi-token particles already sound right; per-token fixes for Class B cause
regressions). The harness logic lives in this note's history — recreate from the
description; the scratch scripts were not committed.
