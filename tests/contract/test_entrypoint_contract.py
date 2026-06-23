"""Contract test: docker-entrypoint.sh runs config_check + alembic before uvicorn.

Pins DEPLOYMENT_CONTRACT.md §7 (boot sequence) + §5.1 (PHASE_DATABASE_PATH
entrypoint export).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "docker-entrypoint.sh"


def _text() -> str:
    return ENTRYPOINT.read_text()


def test_entrypoint_exists():
    assert ENTRYPOINT.is_file()


def test_strict_mode_first_non_comment_line():
    text = _text()
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#") and not ln.startswith("#!")
    ]
    assert lines, "entrypoint has no executable lines"
    assert lines[0] == "set -euo pipefail", (
        f"first non-comment line must be `set -euo pipefail`, got {lines[0]!r}"
    )


def test_boot_order_config_check_then_alembic_then_dispatch():
    text = _text()
    config_check_idx = text.find("python -m sherloc_pipeline.web.config_check")
    alembic_idx = text.find("alembic upgrade head")
    web_case_idx = re.search(r"^\s*web\)\s*", text, re.MULTILINE)
    assert config_check_idx != -1, "missing `python -m sherloc_pipeline.web.config_check`"
    assert alembic_idx != -1, "missing `alembic upgrade head`"
    assert web_case_idx is not None, "missing `web)` case dispatch"
    assert config_check_idx < alembic_idx, "config_check must run before alembic"
    assert alembic_idx < web_case_idx.start(), "alembic must run before case dispatch"


def test_web_case_dispatches_uvicorn_with_factory():
    text = _text()
    # exec uvicorn sherloc_pipeline.web.app:create_app --factory --host 0.0.0.0 --port 8000
    assert re.search(
        r"exec\s+uvicorn\s+sherloc_pipeline\.web\.app:create_app\s+--factory\s+--host\s+0\.0\.0\.0\s+--port\s+8000",
        text,
    ), "web) case must exec uvicorn against the create_app factory on port 8000"


def test_phase_database_path_exported_from_sherloc_db():
    """Q1 fix (v4.1.14): entrypoint exports PHASE_DATABASE_PATH from
    SHERLOC_DB when unset. Without this, consumers who set only
    SHERLOC_DB run alembic against whatever default PHASE_DATABASE_PATH
    resolves to inside alembic/env.py — silently divergent migrations."""
    text = _text()
    # Permissive match: export must reference SHERLOC_DB inside a
    # PHASE_DATABASE_PATH default expansion. Brace style + `:-` defaults
    # for `set -u` safety (e.g. `${SHERLOC_DB:-}`) are both acceptable.
    assert re.search(
        r'export\s+PHASE_DATABASE_PATH\s*=\s*"?\$\{PHASE_DATABASE_PATH:-[^}]*SHERLOC_DB[^"]*}"?',
        text,
    ), (
        "missing entrypoint export "
        '`export PHASE_DATABASE_PATH="${PHASE_DATABASE_PATH:-${SHERLOC_DB:-}}"`'
    )
