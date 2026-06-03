"""Hand-curated romaji overrides for cutlet's Japanese romanizer.

Romaji in SubsVibe is a *read-aloud gauge of ASR quality* shown on the subtitle
display — a viewer sounds it out to sanity-check what Whisper heard. So an entry
belongs here only when cutlet+UniDic produces romaji that **sounds wrong** when
read aloud, not merely off-standard.

The one failure mode that needs overriding: a greeting/interjection that UniDic
lexicalizes as a single token, so an internal は/へ/を never tokenizes as a
grammatical particle. cutlet then reads the literal kana (ha/he/wo) even though
the word is *pronounced* wa/e/o. `こんにちは` is the canonical case — UniDic tags
it one 感動詞 (interjection), and cutlet emits "konnichiha".

`utils.romanize` feeds this dict to `cutlet.Cutlet().exceptions.update(...)`,
cutlet's supported override hook. Keys match on the token *surface* form.

Keep this set minimal — a wrong/over-eager entry would mangle audio that was
transcribed correctly, defeating the gauge. Most greetings (こんばんは, 今日は,
今晩は, では …) do NOT belong here: UniDic splits their particle into its own
token, so cutlet's built-in particle rule already sounds them correctly.

Why other known cutlet mis-sounds are deliberately excluded
-----------------------------------------------------------
cutlet's exception hook is **token-keyed and context-free** — it matches the
surface of each tokenized word, not the whole input string. That rules out the
other two families of mis-sound:

- Kanji-reading ambiguity (お母さん -> "Ohahasan", 兄さん -> "Anisan"): the
  analyzer picks a valid-but-wrong reading (母=ハハ, 兄=アニ — correct standalone,
  wrong in the お~さん honorific). A token key like 兄->nii would fix お兄ちゃん
  but corrupt the *common* standalone reading (私の兄 -> "nii", should be "ani").
  Net harm, so excluded.
- Segmentation glitches (hiragana しょうり -> "Sho uri"): input-specific; the
  kanji form 勝利 -> "Shouri" is correct, and kanji is what ASR emits. Excluded.

Both need context-aware substitution the hook can't express. Only single-token
lexicalized particles (the こんにちは class) are safe to override.

Re-verifying after a cutlet / UniDic upgrade
--------------------------------------------
The membership of this set is empirical, not theoretical. To re-derive it, run a
broad list of common greetings/interjections/set-phrases through a stock
``cutlet.Cutlet`` and flag any **single-token** word whose ``kana`` ends in ハ/ヘ/ヲ
while its ``pron`` ends in ワ/エ/オ (i.e. cutlet will read the wrong sound). Only
single-token flags belong here — multi-token particles already sound right, and
per-token fixes for the kanji-reading family regress standalone readings. As of
cutlet 0.5 + full UniDic 3.1, a ~60-phrase scan flags exactly the two below.
"""
from __future__ import annotations

# surface form -> romaji that matches how the word is actually pronounced
JA_ROMAJI_EXCEPTIONS: dict[str, str] = {
    "こんにちは": "konnichi wa",
    "こんちは": "konchi wa",  # casual contraction, same lexicalization as above
}
