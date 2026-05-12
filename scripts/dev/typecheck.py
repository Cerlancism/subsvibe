"""Quick syntax check across all SubsVibe Python source.

Uses stdlib `ast.parse` only - no extra deps, just catches parse errors.

For real static type checking (unused imports, type mismatches, attribute
errors), install pyright into the venv and run it over the same roots:
    pip install pyright
    python -m pyright client server utils tests
"""
from __future__ import annotations

import ast
import pathlib
import sys

ROOTS = ("client", "server", "utils", "tests")


def main() -> int:
    errors = 0
    checked = 0
    for root in ROOTS:
        for path in pathlib.Path(root).rglob("*.py"):
            checked += 1
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                errors += 1
                print(f"{path}:{exc.lineno}:{exc.offset}: {exc.msg}")

    print(f"checked {checked} files, {errors} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
