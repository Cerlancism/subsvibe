"""Evaluate the hybrid Japanese romaji corrector against rule-based cutlet.

Run directly (needs a live LLM at LLM_BASE_URL):
    python tests/test_ja_romaji_llm.py

This is an EVALUATION harness, not a pass/fail gate (it calls a live LLM, so it
is non-deterministic — never assert on it). It compares two engines on Japanese:

  cutlet   - utils.romanize.make_romanizer("ja"), the rule-based default on main.
  hybrid   - client.llm.romanize_ja_fix(source, cutlet_draft): cutlet produces a
             draft, the LLM edits only what mis-sounds. The candidate for wiring
             into committed (final) lines.

The bet behind the hybrid: anchoring a small (4b) model on cutlet's mostly-correct
draft fixes cutlet's predictable errors (こんにちは->...ha, お兄ちゃん->oanichan,
月曜日->getsuyou hi) WITHOUT the phantom-word hallucinations a small model produces
when romanizing from scratch. Earlier from-scratch prompts (generic + JA-specific)
were evaluated and rejected; see git history for that comparison.

Read the GOOD block for regressions (does the corrector corrupt lines cutlet
already gets right?) and the KNOWN_BAD block for fixes. No assertions — judge by
ear; the loose matcher is only a hint. Not wired into SubsVibe.

Each corrector call is also TIMED. The per-call latency is the cost of running
the corrector on one committed line; compare it against a normal translation
call to decide whether to reuse the translation model (default 4b) for the
corrector or spawn a smaller dedicated model alongside it. Sweep models via
LLM_MODEL_ID, e.g.:
    LLM_MODEL_ID=qwen3.5:0.8b python tests/test_ja_romaji_llm.py
    LLM_MODEL_ID=qwen3.5:2b   python tests/test_ja_romaji_llm.py
    LLM_MODEL_ID=qwen3.5:4b   python tests/test_ja_romaji_llm.py
"""
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client.llm import romanize_ja_fix
from utils.romanize import make_romanizer

cutlet_romanize = make_romanizer("ja")


def _hybrid(text: str) -> str:
    """The candidate: cutlet draft -> LLM corrector."""
    return romanize_ja_fix(text, cutlet_romanize(text))


def _timed(fn, text: str) -> tuple[str, float]:
    """Run `fn(text)`, returning (output, elapsed_ms). Errors are captured as the
    output string so one bad call doesn't abort the sweep."""
    t0 = time.perf_counter()
    try:
        out = fn(text)
    except Exception as e:
        out = f"<error: {type(e).__name__}>"
    return out, (time.perf_counter() - t0) * 1000.0


def _stats(samples: list[float]) -> str:
    """mean / median / max, in ms — tail (max) matters most for a live pipeline."""
    if not samples:
        return "n/a"
    s = sorted(samples)
    mean = sum(s) / len(s)
    median = s[len(s) // 2]
    return f"mean {mean:6.0f}  median {median:6.0f}  max {s[-1]:6.0f}  (ms, n={len(s)})"


# Cases cutlet handles correctly — the corrector must not regress these.
GOOD = [
    "おはようございます",
    "ありがとうございます",
    "これはペンです",
    "学校へ行きます",
    "本を読む",
    "今日はいい天気ですね",
    "名前は何ですか",
    "コーヒーを飲みたい",
    "昨日友達と映画を見に行きました",
    "この料理はとても美味しいです",
]

# (source, correct spoken reading) — the cases cutlet mis-sounds. The corrector
# is judged on whether it repairs the reading here.
KNOWN_BAD = [
    ("こんにちは", "konnichiwa"),
    ("こんばんは", "konbanwa"),
    ("お兄ちゃん", "oniichan"),
    ("月曜日", "getsuyoubi"),
    # a few more lexicalized / context cases worth probing
    ("私の兄は医者です", "watashi no ani wa isha desu"),
    ("今日は何曜日ですか", "kyou wa nan youbi desu ka"),
]


def _norm(s: str) -> str:
    """Loose match for tallying: lowercase, drop spaces/hyphens/apostrophes."""
    return s.lower().replace(" ", "").replace("-", "").replace("'", "")


def _hit(got: str, correct: str) -> bool:
    g, c = _norm(got), _norm(correct)
    return c in g or g in c


def main() -> int:
    model = os.environ.get("LLM_MODEL_ID", "(default)")
    w = 30
    print(f"model: {model}\n")

    # Warm up: the first call to a freshly-loaded Ollama model pays a one-time
    # load cost (weights into VRAM). Time it separately so it doesn't skew the
    # per-call latency we use for the wiring decision.
    _, warmup_ms = _timed(_hybrid, "テスト")
    print(f"warmup (first call, includes model load): {warmup_ms:.0f} ms\n")

    latencies: list[float] = []  # steady-state corrector call times, ms

    print("=== cases cutlet gets right (does the hybrid corrector regress them?) ===\n")
    print(f"{'source':<{w}}{'cutlet':<{w}}{'hybrid':<{w}}ms")
    print("-" * (w * 3 + 8))
    for src in GOOD:
        h, ms = _timed(_hybrid, src)
        latencies.append(ms)
        print(f"{src:<{w}}{cutlet_romanize(src):<{w}}{h:<{w}}{ms:6.0f}")

    print("\n=== cases cutlet mis-sounds (does the hybrid corrector fix them?) ===\n")
    print(f"{'source':<{w}}{'correct':<{w}}{'cutlet':<{w}}{'hybrid':<{w}}ms")
    print("-" * (w * 4 + 8))
    hyb_hits = 0
    for src, correct in KNOWN_BAD:
        h, ms = _timed(_hybrid, src)
        latencies.append(ms)
        hit = _hit(h, correct)
        hyb_hits += hit
        h_disp = ("OK " if hit else "   ") + h
        print(f"{src:<{w}}{correct:<{w}}{cutlet_romanize(src):<{w}}{h_disp:<{w}}{ms:6.0f}")

    n = len(KNOWN_BAD)
    print(f"\nQuality:  hybrid corrector matched correct reading on {hyb_hits}/{n} known-bad cases.")
    print(f"Latency:  {_stats(latencies)}")
    print("(Evaluation only — read the table and judge by ear. Latency is the per-")
    print(" committed-line cost of the corrector; compare against the translation")
    print(" call to decide reuse-same-model vs spawn-a-smaller-one.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
