"""Print the public IP seen through the configured HTTP(S)_PROXY.

Run directly (needs network; manual check, not a pass/fail gate):
    source scripts/env.sh && python tests/test_proxy_ip.py

Uses the same HTTP stacks as the client: urllib.request (server-management
calls in client/transcribe.py) and httpx (what the openai SDK sends through).
Both read HTTPS_PROXY / HTTP_PROXY / NO_PROXY from the environment, so the
reported IP should be the proxy's egress address, not your own.
"""

import json
import os
import urllib.request

import openai

IP_URL = "https://api.ipify.org?format=json"


def ip_via_urllib() -> str:
    with urllib.request.urlopen(IP_URL, timeout=10) as resp:
        return json.load(resp)["ip"]


def ip_via_httpx() -> str:
    # DefaultHttpxClient is the httpx client the openai SDK builds by default;
    # it honours the proxy env vars (trust_env=True).
    with openai.DefaultHttpxClient(timeout=10) as client:
        resp = client.get(IP_URL)
        resp.raise_for_status()
        return resp.json()["ip"]


def main() -> None:
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY"):
        print(f"{var}={os.environ.get(var) or os.environ.get(var.lower()) or '(unset)'}")
    print(f"urllib.request -> {ip_via_urllib()}")
    print(f"httpx (openai) -> {ip_via_httpx()}")


if __name__ == "__main__":
    main()
