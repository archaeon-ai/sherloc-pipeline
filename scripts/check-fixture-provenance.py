#!/usr/bin/env python3
"""Enforce the fixture-provenance ledger (tests/fixtures/PROVENANCE.json).

This is a public repository: every tracked data file must be covered by a
ledger entry declaring its provenance category, and Mars 2020 instrument
data must fall within the instrument's archived PDS release envelope
(``pds_envelope.<instrument>.max_sol``). Fail-closed: a tracked data file
with no ledger entry is a violation.

Usage:
  scripts/check-fixture-provenance.py            # default: --staged
  scripts/check-fixture-provenance.py --staged   # git-staged files (pre-commit hook)
  scripts/check-fixture-provenance.py --tree     # every tracked file (CI / audit)

Both modes fail on stale ledger entries (entries matching no file in the
git index), so staged deletions keep the ledger honest at commit time.
``--staged`` limits the per-file checks to files staged for commit
(including rename destinations); ``--tree`` checks every indexed file.

Exits 0 if clean, 1 on violation, 2 on usage/ledger schema errors.
See CONTRIBUTING.md "Fixture provenance". Stdlib-only by design so the
pre-commit hook needs no virtualenv.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

LEDGER_PATH = "tests/fixtures/PROVENANCE.json"

# File extensions treated as data (case-insensitive). Code, docs, and repo
# config (.py/.md/.toml/.yaml/...) are out of scope: values embedded in code
# are caught in normal review, and YAML in this repo is config, not data.
DATA_EXTENSIONS = {
    ".csv", ".tsv", ".json", ".jsonl",
    ".npy", ".npz", ".parquet", ".feather",
    ".db", ".sqlite", ".sqlite3",
    ".img", ".xml", ".fits", ".dat", ".bin",
    ".h5", ".hdf5", ".pkl", ".pickle",
    ".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff",
}

VALID_CATEGORIES = {
    "pds-archived",
    "raw-images-feed",
    "synthetic",
    "reference-standard",
    "operator-derived-archived",
    "metadata",
}

# Categories whose declared sol is validated against the PDS envelope.
# pds-archived REQUIRES instrument + sol; operator-derived-archived is
# envelope-checked only when a sol is declared.
ENVELOPE_CHECKED = {"pds-archived", "operator-derived-archived"}

# High-confidence sol tokens in paths. Deliberately narrow: ambiguous
# trailing numerics (e.g. "..._trimmed_mean_1266.csv") are NOT extracted —
# the ledger entry is authoritative for those.
_SOL_SEGMENT_RE = re.compile(r"^sol[_-]?(\d{1,4})(?:$|[_-])", re.IGNORECASE)
_PDS_PRODUCT_RE = re.compile(r"^[A-Z]{2,4}\d?_{1,2}(\d{4})_\d{9,10}")
_SOL_PREFIX_RE = re.compile(r"^(\d{4})_")


def extract_sols(path: str) -> set[int]:
    """Extract high-confidence sol numbers embedded in a repo-relative path."""
    sols: set[int] = set()
    parts = Path(path).parts
    for segment in parts:
        m = _SOL_SEGMENT_RE.match(segment)
        if m:
            sols.add(int(m.group(1)))
    basename = parts[-1] if parts else ""
    for pattern in (_PDS_PRODUCT_RE, _SOL_PREFIX_RE):
        m = pattern.match(basename)
        if m:
            sols.add(int(m.group(1)))
    return sols


def is_data_file(path: str) -> bool:
    return Path(path).suffix.lower() in DATA_EXTENSIONS


def load_ledger(repo_root: Path) -> dict:
    """Load and schema-validate the ledger. Raises ValueError on defects."""
    ledger_file = repo_root / LEDGER_PATH
    if not ledger_file.is_file():
        raise ValueError(f"ledger not found at {LEDGER_PATH}")
    try:
        ledger = json.loads(ledger_file.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{LEDGER_PATH} is not valid JSON: {exc}") from exc

    envelope = ledger.get("pds_envelope")
    if not isinstance(envelope, dict) or not envelope:
        raise ValueError("pds_envelope must be a non-empty object")
    for instrument, block in envelope.items():
        if not isinstance(block, dict) or not isinstance(block.get("max_sol"), int):
            raise ValueError(f"pds_envelope.{instrument}.max_sol must be an integer")
        if block["max_sol"] <= 0:
            raise ValueError(f"pds_envelope.{instrument}.max_sol must be positive")

    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries must be a non-empty list")
    for entry in entries:
        path = entry.get("path")
        category = entry.get("category")
        if not isinstance(path, str) or not path:
            raise ValueError(f"entry missing path: {entry!r}")
        if category not in VALID_CATEGORIES:
            raise ValueError(f"{path}: invalid category {category!r}")
        sol = entry.get("sol")
        if sol is not None and not isinstance(sol, int):
            raise ValueError(f"{path}: sol must be an integer")
        instrument = entry.get("instrument")
        if instrument is not None and instrument not in envelope:
            raise ValueError(
                f"{path}: instrument {instrument!r} has no pds_envelope block"
            )
        if category == "pds-archived":
            if instrument is None or sol is None:
                raise ValueError(
                    f"{path}: pds-archived entries require 'instrument' and 'sol'"
                )
        if sol is not None and category in ENVELOPE_CHECKED and instrument is None:
            raise ValueError(
                f"{path}: envelope-checked entry with a sol requires 'instrument'"
            )
    return ledger


def find_entry(path: str, entries: list[dict]) -> dict | None:
    """Longest-prefix match: exact file path, or directory entry ending in '/'."""
    best: dict | None = None
    best_len = -1
    for entry in entries:
        epath = entry["path"]
        if path == epath or (epath.endswith("/") and path.startswith(epath)):
            if len(epath) > best_len:
                best = entry
                best_len = len(epath)
    return best


def check_files(files: list[str], ledger: dict) -> list[str]:
    """Return violation messages for the given repo-relative data files."""
    violations: list[str] = []
    envelope = ledger["pds_envelope"]
    entries = ledger["entries"]

    for path in files:
        entry = find_entry(path, entries)
        if entry is None:
            violations.append(
                f"{path}: no provenance entry in {LEDGER_PATH} — every tracked "
                "data file needs a category (see CONTRIBUTING.md)"
            )
            continue

        category = entry["category"]
        sol = entry.get("sol")
        instrument = entry.get("instrument")

        if category in ENVELOPE_CHECKED and sol is not None:
            max_sol = envelope[instrument]["max_sol"]
            if sol > max_sol:
                violations.append(
                    f"{path}: sol {sol} exceeds the {instrument} archived "
                    f"envelope (max_sol {max_sol}) — not publicly releasable"
                )

        embedded = extract_sols(path)
        if embedded:
            if sol is None:
                violations.append(
                    f"{path}: path embeds sol {sorted(embedded)} but ledger "
                    f"entry '{entry['path']}' declares no sol"
                )
            elif embedded != {sol}:
                violations.append(
                    f"{path}: path embeds sol {sorted(embedded)} but ledger "
                    f"entry '{entry['path']}' declares sol {sol}"
                )
    return violations


def stale_entries(tracked: list[str], ledger: dict) -> list[str]:
    """Ledger entries matching no tracked file (deleted fixtures)."""
    violations = []
    tracked_set = set(tracked)
    for entry in ledger["entries"]:
        epath = entry["path"]
        if epath.endswith("/"):
            if not any(t.startswith(epath) for t in tracked_set):
                violations.append(
                    f"{LEDGER_PATH}: stale entry '{epath}' matches no tracked file"
                )
        elif epath not in tracked_set:
            violations.append(
                f"{LEDGER_PATH}: stale entry '{epath}' matches no tracked file"
            )
    return violations


def _git_lines(repo_root: Path, *args: str) -> list[str]:
    out = subprocess.run(
        ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout
    return [line for line in out.splitlines() if line]


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "--staged"
    if mode not in ("--staged", "--tree"):
        print(f"Usage: {argv[0]} [--staged|--tree]", file=sys.stderr)
        return 2

    repo_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    )

    try:
        ledger = load_ledger(repo_root)
    except ValueError as exc:
        print(f"FAIL: fixture-provenance ledger defect: {exc}", file=sys.stderr)
        return 2

    # The git index is the commit being made (pre-commit) and the tracked
    # tree (CI), so stale-entry detection runs in BOTH modes against it —
    # a staged deletion that orphans a ledger entry fails at commit time.
    indexed = _git_lines(repo_root, "ls-files")

    if mode == "--staged":
        # ACMR: rename destinations are listed under their NEW path, so a
        # rename from a ledgered to an unledgered path is still checked.
        candidates = _git_lines(
            repo_root, "diff", "--cached", "--name-only", "--diff-filter=ACMR"
        )
    else:
        candidates = indexed

    data_files = [f for f in candidates if is_data_file(f)]

    violations = check_files(data_files, ledger)
    violations.extend(stale_entries(indexed, ledger))

    if violations:
        print("FAIL: fixture-provenance violations:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    print(
        f"fixture-provenance: OK ({len(data_files)} data file(s) checked, "
        f"mode {mode})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
