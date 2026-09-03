"""The evidence sweep must report bound pinning against the requested tolerance (#38).

``--floor-tolerance`` selects which rows are swept AND is the epsilon the veto
measures bound pinning with. If only the first were wired up, every sweep run
with a non-default tolerance would report a bound-pinning count computed against
the 0.5 cm-1 default — silently wrong evidence for a threshold-ratification
decision.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sherloc_pipeline.core.hydration_veto import (
    FLAG_FWHM_FLOOR_PINNED,
    evaluate_hydration_peak,
)

_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "hydration_cr_veto_report.py"
)


@pytest.fixture(scope="module")
def report():
    spec = importlib.util.spec_from_file_location("hydration_cr_veto_report", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(report, *extra):
    return report._parse_args(["--db", "unused.db", *extra])


def test_default_tolerance_reaches_the_config(report):
    config = report._config_from_args(_args(report))
    assert config.fwhm_floor_epsilon_cm1 == 0.5


def test_custom_tolerance_reaches_the_config(report):
    config = report._config_from_args(
        _args(report, "--floor", "50", "--floor-tolerance", "5")
    )
    assert config.fwhm_floor_cm1 == 50.0
    assert config.fwhm_floor_epsilon_cm1 == 5.0


def test_custom_tolerance_changes_the_reported_pinning_verdict(report):
    """A width 3 cm-1 above the floor is pinned at tolerance 5, not at 0.5."""
    x = np.linspace(2800.0, 3900.0, 128)
    y = np.full_like(x, 100.0)
    mask = np.zeros(x.shape, dtype=bool)

    def pinned(tolerance):
        config = report._config_from_args(
            _args(report, "--floor", "50", "--floor-tolerance", str(tolerance))
        )
        verdict = evaluate_hydration_peak(3300.0, 53.0, x, y, y, mask, config)
        return FLAG_FWHM_FLOOR_PINNED in verdict.flags

    assert pinned(0.5) is False
    assert pinned(5.0) is True
