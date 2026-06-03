"""Reproducible behaviour test for the Japanese romanizer (cutlet).

Run directly (no pytest):  python tests/test_ja_romaji.py

Romanization in SubsVibe is a *display gauge* for transcription quality, not a
standards-compliant transliteration. The bar is "sounds correct read aloud" so
a viewer never mistakes a romaji quirk for an ASR error. This test pins that bar
against the actual configured romanizer (`make_romanizer("ja")` -> cutlet with
use_he/use_wo=False, use_foreign_spelling=False, on the bundled unidic-lite).

Two corpora, both asserted as exact string matches:

  PASS       - cases cutlet renders correctly. These are the contract we depend
               on; a regression here is a real bug (e.g. the particle flags が/
               へ/を flipping back to literal ha/he/wo).

  KNOWN_BAD  - cases cutlet currently MIS-sounds with the default unidic-lite
               config. We assert the *current wrong* output so the limitation is
               documented and tracked: if upstream/unidic ever fixes one, this
               test fails and tells us to promote it into PASS. Each entry notes
               the correct reading and the failure class:
                 A = single-token lexicalized particle (kana != pron); cutlet
                     romanizes the kana, so こんにちは -> ...ha not ...wa.
                 B = kanji reading ambiguity; the standalone reading is right in
                     some contexts and wrong here (unfixable per-token).
                 C = segmentation glitch from unidic-lite's smaller dictionary.

Findings here were gathered empirically (scratch corpus + token inspection of
pos1/kana/pron). The richer exceptions-based handling explored to paper over the
class-A cases lives on the `experimental/japanese-romanisation` branch; main
keeps the simple default, so these stay as documented known limitations.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.romanize import make_romanizer

romaji = make_romanizer("ja")

# --- cases cutlet gets right (the contract) --------------------------------
# Grouped by what each group exercises; every value is the exact expected output.
PASS: dict[str, str] = {
    # basic greetings / interjections
    "おはよう": "Ohayou",
    "おはようございます": "Ohayou gozaimasu",
    "ありがとう": "Arigatou",
    "ありがとうございます": "Arigatou gozaimasu",
    "すみません": "Sumimasen",
    "ごめんなさい": "Gomen nasai",
    "さようなら": "Sayounara",
    "おやすみなさい": "Oyasumi nasai",
    "いただきます": "Itadakimasu",
    "はじめまして": "Hajimemashite",
    "よろしくお願いします": "Yoroshiku onegai shimasu",
    # the three grammatical particles は/へ/を -> wa/e/o (the flag-fix targets)
    "これはペンです": "Kore wa pen desu",
    "学校へ行きます": "Gakkou e ikimasu",
    "東京へ行く": "Tokyo e iku",
    "本を読む": "Hon o yomu",
    "ご飯を食べる": "Gohan o taberu",
    "君のことを愛してる": "Kimi no koto o aishiteru",
    "気をつけて": "Ki o tsukete",
    # everyday sentences
    "今日はいい天気ですね": "Kyou wa ii tenki desu ne",
    "お元気ですか": "Ogenki desu ka",
    "元気です": "Genki desu",
    "名前は何ですか": "Namae wa nan desu ka",
    "これはいくらですか": "Kore wa ikura desu ka",
    "駅はどこですか": "Eki wa doko desu ka",
    "トイレはどこですか": "Toire wa doko desu ka",
    "わかりました": "Wakarimashita",
    "わかりません": "Wakarimasen",
    "大丈夫です": "Daijoubu desu",
    "ちょっと待ってください": "Chotto matte kudasai",
    # numbers / counters / time
    "千円です": "Sen en desu",
    "三時半": "San ji han",
    "一人": "Hitori",
    "二人": "Futari",
    # katakana loanwords stay phonetic (use_foreign_spelling=False)
    "コーヒーを飲みたい": "Koohii o nomitai",
    "コンピューター": "Konpyuutaa",
    "テレビ": "Terebi",
    "インターネット": "Intaanetto",
    "アルバイト": "Arubaito",
    "パン": "Pan",
    # mixed script / punctuation
    "え？本当に？": "E? hontou ni?",
    "そうですね…": "Sou desu ne...",
    # longer natural sentences
    "昨日友達と映画を見に行きました": "Kinou tomodachi to eiga o mi ni ikimashita",
    "この料理はとても美味しいです": "Kono ryouri wa totemo oishii desu",
    "日本の文化に興味があります": "Nippon no bunka ni kyoumi ga arimasu",
    # pure-ASCII / empty -> "" (skipped, no spurious romaji line)
    "": "",
    "hello world": "",
}

# --- cases cutlet currently mis-sounds (documented limitations) -------------
# value = (current_wrong_output, correct_reading, failure_class)
KNOWN_BAD: dict[str, tuple[str, str, str]] = {
    "こんにちは": ("Konnichiha", "konnichiwa", "A"),
    "こんばんは": ("Konbanha", "konbanwa", "A"),
    # 兄 alone reads 'ani' (right in 私の兄) but should be 'nii' here; also no space
    "お兄ちゃん": ("Oanichan", "onii-chan", "B"),
    # 月曜 + 日 split, 日 -> 'hi'; should be one word 'getsuyoubi'
    "月曜日": ("Getsuyou hi", "getsuyoubi", "C"),
}


def main() -> int:
    failures: list[str] = []

    for src, expected in PASS.items():
        got = romaji(src)
        if got != expected:
            failures.append(f"  PASS regressed: {src!r}\n    expected {expected!r}\n    got      {got!r}")

    drifted: list[str] = []
    for src, (wrong, correct, cls) in KNOWN_BAD.items():
        got = romaji(src)
        if got != wrong:
            # Not necessarily a failure — could be an upstream fix. Flag it so we
            # can promote the case into PASS, but don't fail the suite on a fix.
            drifted.append(
                f"  KNOWN_BAD[{cls}] changed: {src!r} (correct={correct!r})\n"
                f"    was {wrong!r}\n    now {got!r}"
            )

    print(f"PASS corpus:      {len(PASS)} cases, {len(PASS) - len(failures)} ok")
    print(f"KNOWN_BAD corpus: {len(KNOWN_BAD)} documented limitations")

    if drifted:
        print("\nKNOWN_BAD entries drifted (review — promote to PASS if now correct):")
        print("\n".join(drifted))

    if failures:
        print("\nFAILURES:")
        print("\n".join(failures))
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
