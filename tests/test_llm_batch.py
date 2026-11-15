"""Unit test (offline): streamed timed-entry translation in ./client/llm.py.

`translate_entries` sends committed lines as timed JSON and streams back
{start, end, source, translation} entries, handing each over as it closes;
the model may merge or split lines, and times are clamped into the input span
with strictly increasing starts. `_request` is stubbed with a fake chunk
stream, so no LLM is contacted.

Run: scripts/core/python.sh tests/test_llm_batch.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "client"))
sys.path.insert(0, str(ROOT))

import llm  # noqa: E402


def chunks(text: str, size: int = 3):
    for i in range(0, len(text), size):
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text[i:i + size]))])


def main() -> None:
    # Parser: objects come out as they close, braces/quotes inside strings ignored.
    parser = llm._EntryStreamParser()
    reply = '```json\n{"entries": [{"source": "a } \\" {", "n": 1}, {"source": "b"}]}\n```'
    got = []
    for ch in reply:
        got.extend(parser.feed(ch))
    assert got == [{"source": 'a } " {', "n": 1}, {"source": "b"}], got

    sent: list[dict] = []
    replies: list[str] = []

    def fake_request(messages, **kwargs):
        sent.append({"messages": messages, **kwargs})
        return chunks(replies.pop(0))

    llm._request = fake_request
    hist = [(1.0, 2.0, "東京", "Tokyo")]
    lines = [(10.0, 12.5, "一"), (12.5, 15.0, "二")]

    # One entry per line; history and lines go in as timed JSON; streamed.
    replies[:] = [json.dumps({"entries": [
        {"start": 10.0, "end": 12.5, "source": "一!", "translation": "one"},
        {"start": 12.5, "end": 15.0, "source": "二", "translation": "two"},
    ]}, ensure_ascii=False)]
    out: list[dict] = []
    assert llm.translate_entries(lines, hist, on_entry=out.append) == 2
    assert [e["source"] for e in out] == ["一!", "二"] and out[1]["start"] == 12.5
    msgs = sent[-1]["messages"]
    assert sent[-1]["stream"] is True and sent[-1]["response_format"]["type"] == "json_schema"
    assert '"start": 1.0, "end": 2.0, "source": "東京", "translation": "Tokyo"' in msgs[1]["content"]
    assert '{"start": 10.0, "end": 12.5, "text": "一"}' in msgs[-1]["content"]

    # Model merges both lines; bad / out-of-span / repeated times are fixed up.
    replies[:] = ['{"entries": [{"start": 9, "end": 99, "source": "一二", "translation": "x"},'
                  ' {"start": "?", "source": "", "translation": ""},'
                  ' {"start": 9, "end": 1, "source": "三", "translation": "y"}]}']
    out.clear()
    assert llm.translate_entries(lines, hist, on_entry=out.append) == 2
    assert (out[0]["start"], out[0]["end"]) == (10.0, 15.0), out[0]
    assert out[1]["start"] > out[0]["start"] and out[1]["end"] >= out[1]["start"], out[1]

    # Single and batched calls share the system + history prefix (prompt cache).
    captured = {}

    def fake_complete(messages, **kwargs):
        captured["m"] = messages
        return "x"

    llm._complete = fake_complete
    llm.translate("一", hist)
    assert captured["m"][:-1] == msgs[:-1]

    # Unparseable reply: nothing handed over, 0 returned (caller falls back).
    replies[:] = ["sorry"]
    out.clear()
    assert llm.translate_entries(lines, hist, on_entry=out.append) == 0 and not out
    print("ok")


if __name__ == "__main__":
    main()
