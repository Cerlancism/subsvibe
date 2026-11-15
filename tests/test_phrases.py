"""Unit test for the keyword-biasing phrase list (no server, no network).

Run directly (no pytest):  scripts/core/python.sh tests/test_phrases.py

Covers the plumbing only: the client's --phrases file shapes and the form
field it adds to every request, the server's parse of that field, and the
remote backend's request body (Azure `phraseList` under `provider.options`
for mai-transcribe-2, nothing for a model without a phrase-list input). What
the list does to a transcript is measured against the real API, not here.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "client", ROOT / "server"):
    sys.path.insert(0, str(_p))
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

import numpy as np  # noqa: E402

import transcribe  # noqa: E402
from backends import remote  # noqa: E402


def _write(data: object) -> str:
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(data, fh, ensure_ascii=False)
    fh.close()
    return fh.name


def check_loader() -> None:
    load = transcribe.load_phrases
    assert load(_write(["東京", " Alpha! ", "", "東京"])) == ["東京", "Alpha!"]
    assert load(_write({"phrases": ["a", "b"]})) == ["a", "b"]
    grouped = _write({"description": "x", "phrases": {"places": ["東京", "Kyoto"], "people": ["山田太郎", "Kyoto"]}})
    assert load(grouped) == ["東京", "Kyoto", "山田太郎"]
    # Groups in the order named; an ungrouped file is taken whole; files concatenate.
    assert load([grouped], ["people", "places"]) == ["山田太郎", "Kyoto", "東京"]
    assert load([grouped, _write(["大阪", "山田太郎"])], ["places"]) == ["東京", "Kyoto", "大阪", "山田太郎"]
    for bad in ({"phrases": "a"}, ["a", 1], {"other": []}):
        try:
            load(_write(bad))
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
    try:
        load([grouped], ["places", "nope"])
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("accepted an unknown group")


def check_request_fields() -> None:
    transcribe._session_backend = "openrouter"
    transcribe.set_session_phrases(None)
    assert transcribe.request_backend_fields() == {"extra_body": {"backend": "openrouter"}}
    transcribe.set_session_phrases(["山田太郎", "Alpha"])
    extra = transcribe.request_backend_fields("faster-whisper")["extra_body"]
    assert extra["backend"] == "faster-whisper"
    assert json.loads(extra["phrases"]) == ["山田太郎", "Alpha"]
    transcribe._session_backend = ""
    assert set(transcribe.request_backend_fields()["extra_body"]) == {"phrases"}
    transcribe.set_session_phrases(None)
    assert transcribe.request_backend_fields() == {}


def check_server_parse() -> None:
    from fastapi import HTTPException

    from server import _parse_phrases

    assert _parse_phrases(None) is None
    assert _parse_phrases(" ") is None
    assert _parse_phrases('["a", " ", " b "]') == ["a", "b"]
    assert _parse_phrases('[" "]') is None
    for bad in ("not json", '{"a": 1}', '["a", 2]'):
        try:
            _parse_phrases(bad)
        except HTTPException as exc:
            assert exc.status_code == 400
            continue
        raise AssertionError(f"accepted {bad!r}")


def check_remote_body() -> None:
    sent: list[dict] = []

    def fake_post(self, model_id: str, body: bytes, content_type: str) -> dict:
        sent.append(json.loads(body))
        return {"text": "", "words": []}

    remote.RemoteBackend._post = fake_post  # type: ignore[method-assign]
    backend = remote.RemoteBackend("microsoft/mai-transcribe-2")
    audio = np.zeros(16000, dtype=np.float32)

    backend.transcribe_result(audio, None, None, True, phrases=["山田太郎", "Alpha"])
    assert sent[-1]["provider"] == {"options": {"azure": {"phraseList": {"phrases": ["山田太郎", "Alpha"]}}}}
    backend.transcribe_result(audio, None, None, True)
    assert "provider" not in sent[-1]
    backend.transcribe_result(audio, None, None, True, model_id="x-ai/grok-stt-2.0", phrases=["山田太郎"])
    assert "provider" not in sent[-1]
    # Azure caps: over-long phrases dropped, the rest cut to the count cap.
    long_list = ["あ" * 51] + [f"p{i}" for i in range(remote.AZURE_PHRASE_LIST_MAX + 5)]
    backend.transcribe_result(audio, None, None, True, phrases=long_list)
    fitted = sent[-1]["provider"]["options"]["azure"]["phraseList"]["phrases"]
    assert len(fitted) == remote.AZURE_PHRASE_LIST_MAX and fitted[0] == "p0"


def main() -> None:
    check_loader()
    check_request_fields()
    check_server_parse()
    check_remote_body()
    print("ok")


if __name__ == "__main__":
    main()
