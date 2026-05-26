"""Shared fixtures and skip markers for integration tests.

Tests that depend on operator-side data (cached PDS Sol 921 archive,
Loupe phase.db) are skipped automatically when that data isn't present.

Operators with the data should set:

  SHERLOC_PDS_CACHE_DIR=/path/to/pds        # contains sol_0921/data_processed/
  SHERLOC_DB=/path/to/phase.db              # Loupe-derived database

Defaults are repo-relative (./pds, ./phase.db). On a fresh public clone
neither will exist, so the data-dependent tests are skipped without
breaking the suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PDS_CACHE_DIR = Path(os.getenv("SHERLOC_PDS_CACHE_DIR", "./pds")).resolve()
SOL_921_DIR = PDS_CACHE_DIR / "sol_0921" / "data_processed"
LOUPE_DB = Path(os.getenv("SHERLOC_DB", "./phase.db")).resolve()

requires_sol921_data = pytest.mark.skipif(
    not SOL_921_DIR.exists(),
    reason=f"Sol 921 PDS cache not present at {SOL_921_DIR} "
    "(set SHERLOC_PDS_CACHE_DIR to the directory containing sol_0921/)",
)

requires_loupe_db = pytest.mark.skipif(
    not LOUPE_DB.exists(),
    reason=f"Loupe phase.db not present at {LOUPE_DB} (set SHERLOC_DB)",
)
