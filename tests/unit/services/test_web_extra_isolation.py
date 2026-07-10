"""Ingestion import path must not require the ``web`` extra.

The contract-smoke workflow's host side installs only ``.[dev]`` — no
fastapi. ``tests/conftest.py`` imports ``IngestionService``, so the
ingestion service chain (which imports ``core.r2_keys`` for locator
derivation) must be importable without fastapi installed. Run in a
subprocess so other tests' imports cannot contaminate sys.modules;
assert fastapi was not *loaded* (a faithful proxy for "not required").
"""

import subprocess
import sys
import textwrap


def _run_isolated(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"isolation subprocess failed:\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


class TestWebExtraIsolation:
    def test_ingestion_imports_do_not_load_fastapi(self):
        _run_isolated(
            """
            import sys
            import sherloc_pipeline.core.r2_keys
            import sherloc_pipeline.services.ingestion
            import sherloc_pipeline.services.image_ingestion
            import sherloc_pipeline.services.pds_ingestion
            leaked = [m for m in sys.modules if m.startswith("fastapi")]
            assert not leaked, f"fastapi leaked into ingestion path: {leaked}"
            """
        )

    def test_derive_rel_locator_works_without_fastapi_loaded(self):
        """The go-forward locator writer is fastapi-free end to end."""
        _run_isolated(
            """
            import sys
            from sherloc_pipeline.core.r2_keys import derive_rel_locator
            rel = derive_rel_locator(
                "/data/sherloc/data/loupe/sol_0921/ws/img/a.PNG"
            )
            assert rel == "loupe/sol_0921/ws/img/a.PNG", rel
            leaked = [m for m in sys.modules if m.startswith("fastapi")]
            assert not leaked, f"fastapi leaked: {leaked}"
            """
        )
