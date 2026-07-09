"""Tests for the --despike-method CLI option.

Covers the override helper (config-singleton mutation, the --trim-pct
pattern), help-text exposure on all three commands, and Typer's
parse-time rejection of invalid values with the choice list.
"""

import pytest
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import DespikeMethod, _apply_despike_method_override, app
from sherloc_pipeline.config import get_config, reset_config

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_config():
    """Reset config singleton after each test."""
    yield
    reset_config()


class TestDespikeMethodEnum:
    """Closed set, mirrors VALID_DESPIKE_METHODS."""

    def test_values_match_service_constant(self):
        from sherloc_pipeline.services.preprocessing import VALID_DESPIKE_METHODS

        assert tuple(m.value for m in DespikeMethod) == VALID_DESPIKE_METHODS


class TestApplyDespikeMethodOverride:
    """Unit tests for _apply_despike_method_override helper."""

    def test_none_is_noop(self):
        original = get_config().preprocessing['despike'].get('method')
        _apply_despike_method_override(None)
        assert get_config().preprocessing['despike'].get('method') == original

    @pytest.mark.parametrize("method", list(DespikeMethod))
    def test_valid_override_sets_config(self, method):
        _apply_despike_method_override(method)
        assert get_config().preprocessing['despike']['method'] == method.value

    def test_override_outranks_shipped_default(self):
        """CLI > config precedence: the shipped config says
        ml; the override flips the resolved method."""
        from sherloc_pipeline.services.preprocessing import resolve_despike_method

        assert resolve_despike_method(None, get_config()) == "ml"
        _apply_despike_method_override(DespikeMethod.modz)
        assert resolve_despike_method(None, get_config()) == "modz"

    def test_override_survives_missing_despike_block(self):
        """setdefault path: a config without a despike block still accepts
        the override."""
        get_config().preprocessing.pop('despike', None)
        _apply_despike_method_override(DespikeMethod.none)
        assert get_config().preprocessing['despike']['method'] == "none"


class TestCliHelpText:
    """--despike-method exposed on all three commands."""

    @pytest.mark.parametrize("command", ["full-pipeline", "process-new", "plot"])
    def test_help_shows_despike_method(self, command):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert "--despike-method" in result.output


class TestCliDispatch:
    """A valid --despike-method on the
    pipeline commands reaches the service layer's method resolution —
    not merely the parser. full-pipeline is stubbed at the service
    boundary and the effective method is resolved exactly the way
    run_scan resolves it (resolve_despike_method against the shared
    config singleton)."""

    @pytest.mark.parametrize(
        ("argv_method", "expected"),
        [(["--despike-method", "ml"], "ml"),
         (["--despike-method", "modz"], "modz"),
         (["--despike-method", "none"], "none"),
         ([], "ml")],  # no flag: shipped config default
        ids=["ml", "modz", "none", "default"],
    )
    def test_full_pipeline_dispatches_effective_method(
        self, monkeypatch, argv_method, expected
    ):
        from sherloc_pipeline.services.base import ServiceResult
        from sherloc_pipeline.services.preprocessing import resolve_despike_method

        seen: dict = {}

        class StubPipelineService:
            def __init__(self, *args, **kwargs):
                pass

            def run_full_pipeline(self, **kwargs):
                # What run_scan will resolve at despike time (the CLI
                # override mutates the config singleton, pipeline passes
                # despike_method=None through).
                seen["effective"] = resolve_despike_method(None, get_config())
                return ServiceResult(
                    summary="stub run", artifacts=[], warnings=[], metadata={}
                )

        monkeypatch.setattr(
            "sherloc_pipeline.cli.app.PipelineService", StubPipelineService
        )
        result = runner.invoke(
            app, ["full-pipeline", "0921", "Amherst_Point", "detail_1",
                  *argv_method],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert seen["effective"] == expected

    def test_process_new_applies_override_before_processing(self):
        """process-new shares the identical config-singleton mechanism;
        the override is applied before any path/ingest side effects."""
        result = runner.invoke(
            app, ["process-new", "/nonexistent/sol_0000",
                  "--despike-method", "modz"],
        )
        assert result.exit_code != 0  # path validation fails afterwards
        assert get_config().preprocessing['despike']['method'] == "modz"

    def test_process_new_dispatches_effective_method(
        self, monkeypatch, tmp_path, fixtures_path
    ):
        """Successful process-new command-to-service dispatch: the scan
        list comes from the database, the (stubbed) pipeline service is
        invoked per scan, and the effective despike method at dispatch
        time matches the flag."""
        from sherloc_pipeline.database.connection import get_session
        from sherloc_pipeline.database.models import ScanORM
        from sherloc_pipeline.services.base import ServiceResult
        from sherloc_pipeline.services.ingestion import IngestionService
        from sherloc_pipeline.services.preprocessing import resolve_despike_method

        db_path = tmp_path / "test.db"
        ingestion = IngestionService(
            database_path=db_path, include_spectra=False
        )
        ingestion.ingest_workspace(
            fixtures_path
            / "loupe/sol_0921/detail_1"
            / "SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
        )
        with get_session(ingestion.engine) as session:
            scan = session.query(ScanORM).first()
            scan.target = "Amherst Point"
            scan.target_type = "mars_target"

        seen: list = []

        class StubPipelineService:
            def __init__(self, *args, **kwargs):
                pass

            def run_full_pipeline(self, **kwargs):
                seen.append(
                    (kwargs.get("scan"),
                     resolve_despike_method(None, get_config()))
                )
                return ServiceResult(
                    summary="stub run", artifacts=[], warnings=[], metadata={}
                )

        monkeypatch.setattr(
            "sherloc_pipeline.cli.app.PipelineService", StubPipelineService
        )
        result = runner.invoke(
            app,
            ["process-new", str(fixtures_path / "loupe" / "sol_0921"),
             "--skip-ingest",
             "--database", str(db_path),
             "--data-dir", str(fixtures_path / "loupe"),
             "--results-dir", str(tmp_path / "results"),
             "--despike-method", "modz"],
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert seen == [("detail_1", "modz")]


class TestCliInvalidValue:
    """Invalid values rejected at parse time with a
    nonzero exit and the choice list — before any processing."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["full-pipeline", "0921", "Amherst_Point", "detail_1",
             "--despike-method", "median"],
            ["process-new", "/nonexistent/sol_0000.zip",
             "--despike-method", "median"],
            ["plot", "--sol", "921", "--target", "Amherst_Point",
             "--scan", "detail_1", "--despike-method", "median"],
        ],
        ids=["full-pipeline", "process-new", "plot"],
    )
    def test_invalid_value_rejected_with_choices(self, argv):
        result = runner.invoke(app, argv)
        assert result.exit_code != 0
        output = result.output
        for valid in ("ml", "modz", "none"):
            assert valid in output
