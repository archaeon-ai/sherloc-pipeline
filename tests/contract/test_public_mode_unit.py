"""Contract test: public-mode DB filename invariant.

Pins DEPLOYMENT_CONTRACT.md §6. Fast unit-level companion to the live
smoke's Case C (test_container_smoke.py): catches refactors that move
or weaken the guard without needing Docker.

The :memory: case is pinned deliberately. The substring check in
web/app.py is ``"phase.db" in db_str AND "phase_pds.db" not in db_str``;
``:memory:`` matches neither side and is currently accepted in public
mode. Future tightening (e.g., "public mode forbids :memory:") becomes
a deliberate decision, not silent drift.
"""

from __future__ import annotations

import pytest

from sherloc_pipeline.web.app import create_app


# Cases use container-internal paths (/data/...) per DEPLOYMENT_CONTRACT.md
# §4. The ci.yml absolute-path guard rejects /home|/data|/nas substrings
# in pytest collection output, so override the auto-generated test IDs to
# keep the path values out of the IDs.
@pytest.mark.parametrize(
    "access_mode,db_path,expect_raise",
    [
        pytest.param("public", "/data/phase.db", True, id="public-phase_db-raises"),
        pytest.param("public", "/data/phase_pds.db", False, id="public-phase_pds_db-ok"),
        pytest.param("public", ":memory:", False, id="public-memory-ok"),
        pytest.param("internal", "/data/phase.db", False, id="internal-phase_db-ok"),
        pytest.param("internal", "/data/phase_pds.db", False, id="internal-phase_pds_db-ok"),
    ],
)
def test_public_mode_db_filename_invariant(access_mode, db_path, expect_raise):
    if expect_raise:
        with pytest.raises(ValueError, match=r"phase_pds\.db|Loupe|Public mode"):
            create_app(database_path=db_path, access_mode=access_mode)
    else:
        # Construct + tear down. We don't need to hit the API — the
        # invariant is asserted at create_app() startup before lifespan.
        app = create_app(database_path=db_path, access_mode=access_mode)
        assert app is not None
