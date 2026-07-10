"""Despike method selection and resolution.

The 4-cell precedence matrix {CLI set/unset} x {config set/unset}, the
closed value set, and invalid-value rejection at both layers before any
processing side effects.
"""

from types import SimpleNamespace

import pytest

from sherloc_pipeline.services.errors import PreprocessingError
from sherloc_pipeline.services.preprocessing import (
    DEFAULT_DESPIKE_METHOD,
    VALID_DESPIKE_METHODS,
    resolve_despike_method,
)


def _cfg(despike: dict | None) -> SimpleNamespace:
    preprocessing = {} if despike is None else {"despike": despike}
    return SimpleNamespace(preprocessing=preprocessing)


class TestClosedSet:
    def test_valid_methods_closed_set(self):
        assert VALID_DESPIKE_METHODS == ("ml", "modz", "none")

    def test_default_is_ml(self):
        assert DEFAULT_DESPIKE_METHOD == "ml"


class TestPrecedenceMatrix:
    """{CLI set/unset} x {config set/unset}."""

    def test_cli_set_config_set_cli_wins(self):
        cfg = _cfg({"method": "modz"})
        assert resolve_despike_method("none", cfg) == "none"

    def test_cli_set_config_unset_cli_wins(self):
        cfg = _cfg(None)
        assert resolve_despike_method("modz", cfg) == "modz"

    def test_cli_unset_config_set_config_wins(self):
        cfg = _cfg({"method": "modz"})
        assert resolve_despike_method(None, cfg) == "modz"

    def test_cli_unset_config_unset_default_ml(self):
        cfg = _cfg(None)
        assert resolve_despike_method(None, cfg) == "ml"

    def test_config_despike_block_without_method_key_defaults(self):
        cfg = _cfg({"window_size": 7})
        assert resolve_despike_method(None, cfg) == "ml"

    def test_attr_style_config(self):
        despike = SimpleNamespace(method="none")
        cfg = SimpleNamespace(preprocessing=SimpleNamespace(despike=despike))
        assert resolve_despike_method(None, cfg) == "none"

    def test_missing_preprocessing_section_defaults(self):
        cfg = SimpleNamespace()
        assert resolve_despike_method(None, cfg) == "ml"


class TestShippedConfig:
    """The packaged config.yaml carries the documented defaults."""

    def test_shipped_default_method_is_ml(self):
        from sherloc_pipeline.config import load_config

        cfg = load_config()
        assert cfg.preprocessing["despike"]["method"] == "ml"
        assert resolve_despike_method(None, cfg) == "ml"

    def test_unwired_fluorescence_despike_key_removed(self):
        """``fluorescence_fitting.despike_enabled`` was read by nothing;
        it is removed (not deprecated) with a CHANGELOG compat note."""
        from sherloc_pipeline.config import load_config

        cfg = load_config()
        assert "despike_enabled" not in cfg.fluorescence_fitting


class TestInvalidValues:
    """Rejection lists all valid values; nonzero exit code."""

    def test_invalid_cli_value_rejected(self):
        with pytest.raises(PreprocessingError) as excinfo:
            resolve_despike_method("median", _cfg(None))
        message = str(excinfo.value)
        for valid in VALID_DESPIKE_METHODS:
            assert valid in message
        assert excinfo.value.exit_code != 0

    def test_invalid_config_value_rejected(self):
        with pytest.raises(PreprocessingError) as excinfo:
            resolve_despike_method(None, _cfg({"method": "fancy"}))
        message = str(excinfo.value)
        for valid in VALID_DESPIKE_METHODS:
            assert valid in message
        assert excinfo.value.exit_code != 0

    def test_invalid_cli_outranks_valid_config(self):
        """An explicit bad CLI value must fail, not silently fall back."""
        with pytest.raises(PreprocessingError):
            resolve_despike_method("typo", _cfg({"method": "modz"}))

    def test_run_scan_rejects_invalid_method_before_side_effects(self, tmp_path):
        """Validation precedes any processing side
        effects — no results directory contents are created."""
        from sherloc_pipeline.services.preprocessing import PreprocessingService

        service = PreprocessingService()
        results_dir = tmp_path / "results"
        with pytest.raises(PreprocessingError):
            service.run_scan(
                sol="0921",
                target="Whatever",
                scan="detail_1",
                data_dir=tmp_path / "nonexistent_data",
                results_dir=results_dir,
                despike_method="bogus",
            )
        assert not results_dir.exists() or not any(results_dir.iterdir())
