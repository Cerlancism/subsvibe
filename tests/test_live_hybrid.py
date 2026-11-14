"""Unit test for the hybrid live configuration (no server, no network).

Run directly (no pytest):  scripts/core/python.sh tests/test_live_hybrid.py

Covers the config surface only: the commit LLM is off unless a model is named,
is refused on a remote endpoint without its own key, routes only `final` calls
when on, and the commit ASR route is absent without a commit backend. The
pipeline behaviour (preview-only provisionals, the session-backend retry) needs
live audio and is not covered here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "client"))

for _name in ("LIVE_COMMIT_LLM_MODEL_ID", "LIVE_COMMIT_LLM_BASE_URL", "LIVE_COMMIT_LLM_API_KEY"):
    os.environ.pop(_name, None)

import llm  # noqa: E402
from transcribe import resolve_commit_route  # noqa: E402


def main() -> None:
    # Nothing named: no commit LLM, finals stay on LLM_*.
    assert llm.configure_commit_llm(None) is False
    assert llm._llm_for(True) == (llm.llm_client, llm.LLM_MODEL_ID)

    # A remote endpoint with no key of its own is refused, not sent LLM_API_KEY.
    os.environ["LIVE_COMMIT_LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
    assert llm.configure_commit_llm("some/model") is False
    assert llm.commit_llm_client is None

    # With a key: finals go to the commit LLM, previews do not.
    os.environ["LIVE_COMMIT_LLM_API_KEY"] = "test-key"
    assert llm.configure_commit_llm("some/model") is True
    client, model = llm._llm_for(True)
    assert client is llm.commit_llm_client and model == "some/model"
    assert llm._llm_for(False) == (llm.llm_client, llm.LLM_MODEL_ID)
    ids = llm._commit_llm_session_ids
    assert ids["translate"].startswith("subsvibe-translate-")
    assert ids["romaji"].startswith("subsvibe-romaji-")
    assert ids["translate"].split("subsvibe-translate-")[1] == ids["romaji"].split("subsvibe-romaji-")[1]

    # No commit backend: no commit ASR route (and no server call).
    assert resolve_commit_route("") is None

    print("ok")


if __name__ == "__main__":
    main()
