"""Organics rerun artifact hygiene.

The organics fitter writes one of two mutually exclusive per-point exports: a
D+G table (``*_organics_dg_peaks.csv``) or a G-only one
(``*_organics_g_peaks.csv``). ``FittingService._discover_peak_csvs`` prefers the
DG file when both are present, so a rerun that changes the outcome must delete
the export it did not write. Otherwise ``persist-peaks`` rediscovers the
previous run's peaks and writes them to the database as if they were current.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from sherloc_pipeline.models.fitting import FitResult, PeakFit
from sherloc_pipeline.services.fitting import _fit_point_organics

SOL, TARGET, SCAN = "0851", "Lake_Haiyaha", "detail_1"
POINT = 0


def _peak(center: float, fwhm: float) -> PeakFit:
    return PeakFit(
        m_cm1=center, a=100.0, fwhm=fwhm, sigma=fwhm / 2.355,
        area=100.0 * fwhm, snr=20.0,
        pass_snr=True, pass_fwhm=True, pass_r2=True,
        sharpness_ratio=1.0, pass_sharpness=True,
    )


def _result(peaks) -> FitResult:
    return FitResult(peaks=list(peaks), r2=0.95, rss=1.0, dof=10, warnings=[])


def _paths(out_dir: Path):
    stem = f"{SOL}_{TARGET}_{SCAN}_R1_point{POINT}_organics"
    return (
        out_dir / f"{stem}_dg_peaks.csv",
        out_dir / f"{stem}_dg_fit.png",
        out_dir / f"{stem}_g_peaks.csv",
        out_dir / f"{stem}_g_fit.png",
    )


def _seed_previous_dg_run(out_dir: Path) -> None:
    """Stand in for a previous run that exported a D+G detection."""
    dg_csv, dg_png, _, _ = _paths(out_dir)
    dg_csv.write_text("m_cm1,a,fwhm,snr\n1605.0,100.0,60.0,20.0\n")
    dg_png.write_bytes(b"stale png")


def _run(out_dir: Path, fit_results):
    """Run the organics point worker with `fit_spectrum` returning canned fits."""
    x = np.linspace(1250.0, 1850.0, 256)
    y = np.zeros_like(x)
    org_mask = np.ones_like(x, dtype=bool)

    calls = iter(fit_results)

    def fake_fit_spectrum(*args, **kwargs):
        return next(calls), np.zeros_like(x)

    with patch("sherloc_pipeline.core.fitting.fit_spectrum", fake_fit_spectrum), \
         patch("sherloc_pipeline.visualization.fitting_plots.plot_fit_overlay"):
        return _fit_point_organics(
            {"point_idx": POINT, "y": y},
            x=x,
            fit_cfg_org={"min_snr": 3.0},
            g_roi=(1500.0, 1700.0),
            d_roi=(1250.0, 1500.0),
            org_roi=(1250.0, 1850.0),
            org_plot=(1250.0, 1850.0),
            org_mask=org_mask,
            g_acc_lo=40.0, g_acc_hi=100.0,
            d_acc_lo=100.0, d_acc_hi=200.0,
            persist_min_snr=3.0,
            organics_fwhm_mins={},
            use_norm_input=False,
            rebaseline_cfg={},
            out_dir=str(out_dir),
            sol=SOL, target=TARGET, scan=SCAN,
        )


@pytest.fixture
def out_dir(tmp_path):
    d = tmp_path / "organics_fit"
    d.mkdir(parents=True)
    return d


def test_dg_to_g_rerun_removes_the_stale_dg_export(out_dir):
    """A rerun that falls back to G-only must drop the previous DG export.

    `_discover_peak_csvs` prefers DG, so leaving it behind would persist the
    old two-band result instead of this run's single band.
    """
    _seed_previous_dg_run(out_dir)
    dg_csv, dg_png, g_csv, _g_png = _paths(out_dir)

    # G gate passes; the D+G fit accepts nothing -> G-only fallback.
    result = _run(out_dir, [_result([_peak(1605.0, 60.0)]), _result([])])

    assert not dg_csv.exists(), "stale DG table would be preferred over this run's G table"
    assert not dg_png.exists(), "stale DG overlay would be read as current"
    assert g_csv.exists()
    assert [row["band"] for row in result["accepted_peaks"]] == ["G"]


def test_g_to_dg_rerun_removes_the_stale_g_export(out_dir):
    """The reverse transition drops the G-only export."""
    _, _, g_csv, g_png = _paths(out_dir)
    g_csv.write_text("m_cm1,a,fwhm,snr\n1605.0,100.0,60.0,20.0\n")
    g_png.write_bytes(b"stale png")

    dg_csv, _dg_png, _, _ = _paths(out_dir)
    _run(out_dir, [
        _result([_peak(1605.0, 60.0)]),
        _result([_peak(1410.0, 150.0), _peak(1605.0, 60.0)]),
    ])

    assert dg_csv.exists()
    assert not g_csv.exists(), "two exports for one point leave the outcome ambiguous"
    assert not g_png.exists()


def test_positive_to_zero_rerun_removes_both_exports(out_dir):
    """A rerun with no G detection must leave nothing for persistence to find.

    The completion marker then records `accepted_peaks: 0`, and persistence
    treats the scan as a real zero result and clears its rows -- which is only
    safe because no stale CSV survives.
    """
    _seed_previous_dg_run(out_dir)
    dg_csv, dg_png, g_csv, g_png = _paths(out_dir)

    result = _run(out_dir, [_result([])])

    assert not any(p.exists() for p in (dg_csv, dg_png, g_csv, g_png))
    assert result["accepted_peaks"] == []
    assert result["summary_row"]["g_detected"] is False
