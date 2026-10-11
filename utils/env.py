"""Environment overrides for constant defaults, shared by client and server.

Every tunable is a module-level constant somewhere; these helpers read the
variable that may override it. An unset or empty variable leaves the constant
in force, so an env.sh can list every knob with an empty value as
documentation without changing anything.
"""
from __future__ import annotations

import os


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else default
