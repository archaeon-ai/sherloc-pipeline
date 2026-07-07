"""Contract test: the ``--json`` CLI envelope conforms to its committed golden.

Failures here mean the ``CLIResult`` / ``CLIError`` / ``PipelineManifest``
envelope shape drifted from the frozen contract in ``cli_contract.golden.json``.
To move the contract surface, edit ``models/schemas/cli.py`` AND regenerate the
golden together::

    python -m sherloc_pipeline.models.schemas.cli_contract

A shape change requires bumping ``schema_version`` first — the regen tool
enforces this — so that consumers vendoring the envelope can detect the change.
"""

from __future__ import annotations

import json

import pytest

from sherloc_pipeline.models.schemas import cli_contract


def test_cli_envelope_matches_committed_golden():
    """The live models must match the frozen golden exactly (no silent drift)."""
    problems = cli_contract.diff()
    assert not problems, (
        "CLI --json envelope drifted from its committed contract:\n  "
        + "\n  ".join(problems)
        + "\n\nRegenerate (after deciding the semver impact):\n"
        "  python -m sherloc_pipeline.models.schemas.cli_contract"
    )


def test_golden_records_current_schema_version():
    """The golden's recorded version tracks the models' shared default."""
    golden = cli_contract.load_golden()
    assert golden["schema_version"] == cli_contract.SCHEMA_VERSION


def test_shape_fingerprint_ignores_version_value():
    """Two contracts that differ only by version string share a fingerprint,
    so the version-bump coupling can be reasoned about independently of shape."""
    live = cli_contract.build_live_contract()
    schemas = json.loads(json.dumps(live["models"]))
    # Bump the version default in every model; the shape fingerprint must not move.
    for schema in schemas.values():
        schema["properties"]["schema_version"]["default"] = "9.9.9"
    assert cli_contract._shape_fingerprint(schemas) == live["shape_fingerprint"]


def _write_golden(path, *, shape_fingerprint, schema_version, models=None):
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "shape_fingerprint": shape_fingerprint,
                "models": models if models is not None else {},
            }
        )
    )


def test_regen_refuses_shape_change_without_version_bump(monkeypatch, tmp_path):
    """A shape change with an unchanged schema_version must be REFUSED, so a
    breaking/additive change can never reuse the same version silently."""
    golden = tmp_path / "cli_contract.golden.json"
    _write_golden(golden, shape_fingerprint="OLD_SHAPE", schema_version="1.0.0")
    monkeypatch.setattr(cli_contract, "GOLDEN_PATH", golden)
    monkeypatch.setattr(
        cli_contract,
        "build_live_contract",
        lambda: {
            "schema_version": "1.0.0",  # NOT bumped
            "shape_fingerprint": "NEW_SHAPE",  # shape changed
            "models": {"CLIResult": {"changed": True}},
        },
    )
    with pytest.raises(cli_contract.VersionBumpRequired):
        cli_contract.regen(enforce_bump=True)


def test_regen_allows_shape_change_with_version_bump(monkeypatch, tmp_path):
    """A shape change accompanied by a version bump regenerates cleanly."""
    golden = tmp_path / "cli_contract.golden.json"
    _write_golden(golden, shape_fingerprint="OLD_SHAPE", schema_version="1.0.0")
    monkeypatch.setattr(cli_contract, "GOLDEN_PATH", golden)
    new_contract = {
        "schema_version": "2.0.0",  # bumped
        "shape_fingerprint": "NEW_SHAPE",  # shape changed
        "models": {"CLIResult": {"changed": True}},
    }
    monkeypatch.setattr(cli_contract, "build_live_contract", lambda: new_contract)
    cli_contract.regen(enforce_bump=True)
    written = json.loads(golden.read_text())
    assert written["schema_version"] == "2.0.0"
    assert written["shape_fingerprint"] == "NEW_SHAPE"


def test_regen_allows_pure_version_bump(monkeypatch, tmp_path):
    """A version bump with no shape change is allowed (not a refusal case)."""
    golden = tmp_path / "cli_contract.golden.json"
    _write_golden(golden, shape_fingerprint="SAME_SHAPE", schema_version="1.0.0")
    monkeypatch.setattr(cli_contract, "GOLDEN_PATH", golden)
    monkeypatch.setattr(
        cli_contract,
        "build_live_contract",
        lambda: {
            "schema_version": "1.1.0",
            "shape_fingerprint": "SAME_SHAPE",
            "models": {},
        },
    )
    cli_contract.regen(enforce_bump=True)  # must not raise
    assert json.loads(golden.read_text())["schema_version"] == "1.1.0"
