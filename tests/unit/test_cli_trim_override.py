"""Tests for --trim-pct CLI override."""

import click
import pytest
import typer
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import _apply_trim_pct_override, app
from sherloc_pipeline.config import get_config, reset_config
from sherloc_pipeline.core.utils import format_trim_label


@pytest.fixture(autouse=True)
def _clean_config():
    """Reset config singleton after each test."""
    yield
    reset_config()


class TestApplyTrimPctOverride:
    """Unit tests for _apply_trim_pct_override helper."""

    def test_none_is_noop(self):
        original = get_config().preprocessing['trim_mean_baseline_pct']
        _apply_trim_pct_override(None)
        assert get_config().preprocessing['trim_mean_baseline_pct'] == original

    def test_valid_override_sets_config(self):
        _apply_trim_pct_override(8.0)
        assert get_config().preprocessing['trim_mean_baseline_pct'] == 0.08

    def test_zero_accepted(self):
        _apply_trim_pct_override(0.0)
        assert get_config().preprocessing['trim_mean_baseline_pct'] == 0.0

    def test_negative_rejected(self):
        with pytest.raises(click.exceptions.Exit):
            _apply_trim_pct_override(-1.0)

    def test_over_50_rejected(self):
        with pytest.raises(click.exceptions.Exit):
            _apply_trim_pct_override(51.0)

    def test_boundary_50_accepted(self):
        _apply_trim_pct_override(50.0)
        assert get_config().preprocessing['trim_mean_baseline_pct'] == 0.50

    def test_fractional_accepted(self):
        _apply_trim_pct_override(8.5)
        assert get_config().preprocessing['trim_mean_baseline_pct'] == 0.085


class TestCliHelpText:
    """Verify --trim-pct appears in help output."""

    runner = CliRunner()

    def test_full_pipeline_help_shows_trim_pct(self):
        result = self.runner.invoke(app, ["full-pipeline", "--help"])
        assert "--trim-pct" in result.output

    def test_process_new_help_shows_trim_pct(self):
        result = self.runner.invoke(app, ["process-new", "--help"])
        assert "--trim-pct" in result.output


class TestTrimLabelIntegration:
    """Verify format_trim_label works with non-default baselines."""

    def test_8pct_on_100pt(self):
        assert format_trim_label(100, 0.08) == "8p_trim_mean"

    def test_8pct_on_25pt(self):
        # floor(25*0.08) = 2 >= 1, no dynamic bump needed
        assert format_trim_label(25, 0.08) == "8p_trim_mean"

    def test_8pct_on_10pt(self):
        # floor(10*0.08) = 0, dynamic bump to >= 1 point per tail
        assert format_trim_label(10, 0.08) == "10p_trim_mean"

    def test_0pct_passthrough(self):
        assert format_trim_label(25, 0.0) == "0p_trim_mean"
