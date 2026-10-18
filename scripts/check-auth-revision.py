#!/usr/bin/env python3
"""Require the same immutable auth commit in all three build inputs."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("pyproject.toml", "requirements-lock.txt", "Dockerfile")
AUTH = re.compile(
    r"phase-platform-auth\s+@\s+git\+https://github\.com/"
    r"archaeon-ai/phase-platform-auth@([^\s\"']+)"
)


def main() -> int:
    revisions = {}
    for name in FILES:
        # Ignore explanatory comments, so they cannot mask a missing input.
        active_lines = "\n".join(
            line for line in (ROOT / name).read_text().splitlines()
            if not line.lstrip().startswith("#")
        )
        refs = AUTH.findall(active_lines)
        if len(refs) != 1 or not re.fullmatch(r"[0-9a-f]{40}", refs[0]):
            print(f"FAIL: {name}: expected exactly one full auth commit, got {refs}", file=sys.stderr)
            return 1
        revisions[name] = refs[0]

    if len(set(revisions.values())) != 1:
        print(f"FAIL: auth revisions differ: {revisions}", file=sys.stderr)
        return 1

    print(f"Auth revision aligned: {revisions[FILES[0]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
