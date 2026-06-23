"""Unit tests for `sherloc init` (§12.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestSherlocInit:
    """Smoke tests for the init command."""

    def test_creates_sherloc_home_tree(self, runner, tmp_path):
        db_path = tmp_path / "phase.db"
        result = runner.invoke(
            app,
            [
                "init",
                "--sherloc-home", str(tmp_path),
                "--database", str(db_path),
                "--mode", "empty",
            ],
        )
        assert result.exit_code == 0, result.output
        for sub in ("data", "outputs", ".cache/sherloc"):
            assert (tmp_path / sub).is_dir(), f"missing: {sub}"
        assert db_path.exists(), "database file was not created"

    def test_pds_mode_hint(self, runner, tmp_path):
        result = runner.invoke(
            app,
            [
                "init",
                "--sherloc-home", str(tmp_path),
                "--database", str(tmp_path / "phase.db"),
                "--mode", "pds",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "pds-download" in result.output

    def test_loupe_mode_hint(self, runner, tmp_path):
        result = runner.invoke(
            app,
            [
                "init",
                "--sherloc-home", str(tmp_path),
                "--database", str(tmp_path / "phase.db"),
                "--mode", "loupe",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "ingest" in result.output

    def test_invalid_mode_rejected(self, runner, tmp_path):
        result = runner.invoke(
            app,
            [
                "init",
                "--sherloc-home", str(tmp_path),
                "--mode", "garbage",
            ],
        )
        assert result.exit_code != 0
        assert "--mode must be one of" in result.output

    def test_idempotent(self, runner, tmp_path):
        """Re-running init on an already-initialized tree must succeed."""
        args = [
            "init",
            "--sherloc-home", str(tmp_path),
            "--database", str(tmp_path / "phase.db"),
            "--mode", "empty",
        ]
        first = runner.invoke(app, args)
        second = runner.invoke(app, args)
        assert first.exit_code == 0, first.output
        assert second.exit_code == 0, second.output

    def test_json_payload_shape(self, runner, tmp_path):
        import json as json_mod

        result = runner.invoke(
            app,
            [
                "--json",
                "init",
                "--sherloc-home", str(tmp_path),
                "--database", str(tmp_path / "phase.db"),
                "--mode", "pds",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json_mod.loads(result.output.strip().splitlines()[-1])
        assert payload["command"] == "init"
        assert payload["result"]["mode"] == "pds"
        assert payload["result"]["sherloc_home"] == str(Path(tmp_path).resolve())
        assert "pds-download" in payload["result"]["next_step"]
