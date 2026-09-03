"""The hydration cosmic-ray veto applied consistently across all three paths (#38).

Covers ``services.fitting._fit_point_hydration`` (CLI/pipeline), and
``services.map_fitting._fit_raman_domain`` (web map-mode quick fit). The web
point-fit endpoint is covered in ``tests/unit/web/test_processing.py``.

Every path is asserted twice: flag OFF must reproduce pre-#38 behaviour exactly,
flag ON must remove the cosmic ray while leaving a broad, authentic band alone.
"""

from pathlib import Path

import numpy as np
import pytest

from sherloc_pipeline.core.hydration_veto import HydrationVetoConfig
from sherloc_pipeline.services.fitting import (
    _despike_params_from_config,
    _fit_point_hydration,
)
from sherloc_pipeline.services.map_fitting import _fit_raman_domain

FWHM_FLOOR = 50.0


def _hydration_axis(n=256):
    """A detector-realistic R1 axis (~8.6 cm-1/channel).

    Wide enough to contain the 2000-2100 cm-1 noise window the fitter uses, and
    spaced like the real detector so a two-channel spike has the width that
    actually reproduces the reported defect.
    """
    return np.linspace(1800.0, 4000.0, n)


def _cosmic_ray_only_spectrum(x, center=3300.0, amplitude=6000.0, seed=38):
    """Flat noise with a two-channel spike inside the OH window — no real band.

    This reproduces the filed defect: the spike clears R2 >= 0.25, clears the
    F-test and the sharpness gate, and is reported as an OH-stretch feature with
    its FWHM pinned at the 50 cm-1 floor.
    """
    y = 100.0 + np.random.default_rng(seed).normal(0.0, 4.0, size=x.shape)
    idx = int(np.argmin(np.abs(x - center)))
    y[idx] += amplitude
    y[idx + 1] += amplitude * 0.9
    return y


def _broad_hydration_spectrum(x, center=3400.0, fwhm=280.0, amplitude=900.0, seed=7):
    """A wide OH stretch band — the known-authentic control."""
    sigma = fwhm / 2.3548
    y = 100.0 + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return y + np.random.default_rng(seed).normal(0.0, 4.0, size=x.shape)


CR_CENTER = 3800.0
BAND_CENTER = 3400.0


def _mixed_spectrum(x, seed=11):
    """An authentic broad OH band AND a cosmic ray, well separated.

    Both components are fit and both clear the acceptance filters, so the veto
    has to remove exactly one of them — the case where a stale model curve is
    visible rather than merely inconsistent.
    """
    y = _broad_hydration_spectrum(x, center=BAND_CENTER, seed=seed)
    idx = int(np.argmin(np.abs(x - CR_CENTER)))
    y[idx] += 6000.0
    y[idx + 1] += 5400.0
    return y


def _fit_cfg_oh():
    return {
        "max_peaks": 2,
        "fit_fwhm_min_initial_cm1": FWHM_FLOOR,
        "filter_fwhm_min_cm1": FWHM_FLOOR,
        "fwhm_max_cm1": 300.0,
        "min_snr": 3.0,
        "min_seed_snr": 3.0,
        "r_squared_min": 0.25,
        "peak_separation_cm1": 25,
        "noise_estimation": {"window": [2000.0, 2100.0]},
        "parsimony": {"model_selection": "ftest", "ftest_alpha": 0.01},
        "posthoc_filters": {},
    }


def _run_worker(x, y, tmp_path, veto_cfg):
    oh_roi = (2800.0, 3900.0)
    oh_plot = (2600.0, 4000.0)
    return _fit_point_hydration(
        {"point_idx": 1, "y": y},
        x=x,
        fit_cfg_oh=_fit_cfg_oh(),
        oh_roi=oh_roi,
        oh_plot=oh_plot,
        oh_mask=(x >= oh_roi[0]) & (x <= oh_roi[1]),
        plot_mask_oh=(x >= oh_plot[0]) & (x <= oh_plot[1]),
        n_edge=5,
        min_snr=3.0,
        r2_min=0.25,
        center_lo=3000.0,
        center_hi=3900.0,
        out_dir=str(tmp_path),
        sol="0921",
        target="Test_Target",
        scan="detail_1",
        veto_cfg=veto_cfg,
    )


class TestPipelineWorker:
    """services/fitting.py::_fit_point_hydration — the full implementation."""

    def test_flag_off_accepts_the_cosmic_ray(self, tmp_path):
        # The defect as filed: a CR inside 2800-3900 passes R2 + F-test and is
        # reported as hydration, with its width pinned at the floor.
        x = _hydration_axis()
        y = _cosmic_ray_only_spectrum(x)
        result = _run_worker(x, y, tmp_path, None)

        assert result["accepted_peaks"], "baseline behaviour must still accept the CR"
        widths = [row["fwhm_cm1"] for row in result["accepted_peaks"]]
        assert min(widths) <= FWHM_FLOOR + 0.5

    def test_disabled_config_is_identical_to_no_config(self, tmp_path):
        x = _hydration_axis()
        y = _cosmic_ray_only_spectrum(x)
        off = _run_worker(x, y, tmp_path, HydrationVetoConfig(enabled=False))
        none = _run_worker(x, y, tmp_path, None)

        assert off["accepted_peaks"] == none["accepted_peaks"]
        assert off["summary_row"] == none["summary_row"]
        # Flag-off rows carry no veto columns at all.
        for row in off["accepted_peaks"]:
            assert "cr_vetoed" not in row

    def test_flag_on_vetoes_the_cosmic_ray(self, tmp_path):
        x = _hydration_axis()
        y = _cosmic_ray_only_spectrum(x)
        cfg = HydrationVetoConfig(enabled=True, fwhm_floor_cm1=FWHM_FLOOR)
        result = _run_worker(x, y, tmp_path, cfg)

        assert result["accepted_peaks"] == []
        assert result["summary_row"]["oh_detected"] is False
        assert any("vetoed as cosmic ray" in w for w in result["warnings"])

    def test_flag_on_spares_a_broad_authentic_band(self, tmp_path):
        x = _hydration_axis()
        y = _broad_hydration_spectrum(x)
        cfg = HydrationVetoConfig(enabled=True, fwhm_floor_cm1=FWHM_FLOOR)
        result = _run_worker(x, y, tmp_path, cfg)

        assert result["accepted_peaks"], "authentic broad OH band must survive"
        for row in result["accepted_peaks"]:
            assert row["cr_vetoed"] is False
            assert row["fwhm_floor_pinned"] is False

    def test_flag_action_keeps_the_peak_and_annotates_it(self, tmp_path):
        x = _hydration_axis()
        y = _cosmic_ray_only_spectrum(x)
        cfg = HydrationVetoConfig(
            enabled=True, action="flag", fwhm_floor_cm1=FWHM_FLOOR
        )
        result = _run_worker(x, y, tmp_path, cfg)

        assert result["accepted_peaks"]
        row = result["accepted_peaks"][0]
        assert row["cr_vetoed"] is False
        assert row["cr_mask_hit"] or row["cr_amplitude_drop_ratio"] > 0.5
        assert row["fwhm_floor_pinned"] is True


class TestPipelineOverlayAfterVeto:
    """A mixed authentic+cosmic fit must not render the vetoed component.

    Pruning ``accepted_peaks`` alone leaves the overlay and the exported R2
    describing the original fit, so the PNG would still draw the rejected
    cosmic-ray Gaussian as part of an otherwise-accepted hydration fit.
    """

    @staticmethod
    def _capture(monkeypatch):
        captured = {}

        def _fake_plot(x, y, mask, result, model, path, **kwargs):
            captured["result"] = result
            captured["model"] = np.asarray(model, dtype=float)
            Path(path).write_bytes(b"")

        monkeypatch.setattr(
            "sherloc_pipeline.visualization.fitting_plots.plot_fit_overlay",
            _fake_plot,
        )
        return captured

    def test_mixed_spectrum_overlay_drops_the_vetoed_component(
        self, tmp_path, monkeypatch
    ):
        x = _hydration_axis()
        y = _mixed_spectrum(x)
        cfg = HydrationVetoConfig(enabled=True, fwhm_floor_cm1=FWHM_FLOOR)

        captured = self._capture(monkeypatch)
        result = _run_worker(x, y, tmp_path, cfg)

        # Precondition: the authentic band survives, so a plot is produced.
        assert result["accepted_peaks"], "the broad OH band must survive the veto"
        assert any("vetoed as cosmic ray" in w for w in result["warnings"])
        assert "model" in captured, "an accepted fit must still be plotted"

        cr_idx = int(np.argmin(np.abs(x - CR_CENTER)))
        band_idx = int(np.argmin(np.abs(x - BAND_CENTER)))
        model = captured["model"]

        # The vetoed component is gone from the plotted model...
        assert model[cr_idx] < 0.05 * model[band_idx]
        # ...and from the result the overlay labels its peaks from.
        assert all(
            abs(p.m_cm1 - CR_CENTER) > 3 * FWHM_FLOOR for p in captured["result"].peaks
        )
        # The exported R2 is the surviving fit's, not the original's.
        assert result["summary_row"]["oh_r2"] == pytest.approx(
            captured["result"].r2
        )

    def test_flag_off_overlay_still_contains_the_cosmic_ray(
        self, tmp_path, monkeypatch
    ):
        """Baseline contract: with the flag off nothing about the plot changes."""
        x = _hydration_axis()
        y = _mixed_spectrum(x)

        captured = self._capture(monkeypatch)
        _run_worker(x, y, tmp_path, None)

        cr_idx = int(np.argmin(np.abs(x - CR_CENTER)))
        band_idx = int(np.argmin(np.abs(x - BAND_CENTER)))
        assert captured["model"][cr_idx] > 0.05 * captured["model"][band_idx]


def _map_config(enabled, action="reject"):
    return {
        "fitting": {
            "hydration_fit_range": [2800, 3900],
            "hydration_fwhm_min_cm1": FWHM_FLOOR,
            "hydration_fwhm_max_cm1": 300,
            "hydration_max_peaks": 2,
            "hydration_min_snr": 3.0,
            "hydration_r2_min": 0.25,
            "hydration_ftest_alpha": 0.01,
            "hydration_cr_veto": {"enabled": enabled, "action": action},
        }
    }


class TestMapModeQuickFit:
    """services/map_fitting.py hydration domain — web map-mode quick fit."""

    def test_flag_off_reports_the_cosmic_ray(self):
        x = _hydration_axis()
        y = _cosmic_ray_only_spectrum(x)
        result = _fit_raman_domain(
            x, y, _map_config(False), "hydration", preprocessed=True
        )
        assert result.peaks, "baseline behaviour must still report the CR"
        for peak in result.peaks:
            assert "cr_vetoed" not in peak

    def test_flag_on_vetoes_the_cosmic_ray(self):
        x = _hydration_axis()
        y = _cosmic_ray_only_spectrum(x)
        result = _fit_raman_domain(
            x, y, _map_config(True), "hydration", preprocessed=True,
            raw_intensity_r1=y,
        )
        assert result.peaks == []
        assert result.status == "below_threshold"

    def test_flag_on_spares_a_broad_authentic_band(self):
        x = _hydration_axis()
        y = _broad_hydration_spectrum(x)
        result = _fit_raman_domain(
            x, y, _map_config(True), "hydration", preprocessed=True,
            raw_intensity_r1=y,
        )
        assert result.peaks, "authentic broad OH band must survive"
        assert all(p["cr_vetoed"] is False for p in result.peaks)

    def test_mixed_spectrum_reports_the_post_veto_r2(self):
        """R2 rides on every peak row, so it must describe the surviving model."""
        x = _hydration_axis()
        y = _mixed_spectrum(x)

        off = _fit_raman_domain(
            x, y, _map_config(False), "hydration", preprocessed=True
        )
        on = _fit_raman_domain(
            x, y, _map_config(True), "hydration", preprocessed=True,
            raw_intensity_r1=y,
        )

        assert len(off.peaks) == 2, "both components must be reported flag-off"
        assert len(on.peaks) == 1, "exactly the cosmic ray must be vetoed"
        assert all(abs(pk["center_cm1"] - CR_CENTER) > FWHM_FLOOR for pk in on.peaks)
        # The kept band's R2 is recomputed without the vetoed component.
        assert on.peaks[0]["r2"] != off.peaks[0]["r2"]

    def test_veto_does_not_touch_other_domains(self):
        # The veto is hydration-only: a narrow mineral band must be unaffected
        # even with the flag on.
        x = np.linspace(700.0, 1200.0, 400)
        y = 100.0 + 800.0 * np.exp(-0.5 * ((x - 1090.0) / 14.0) ** 2)
        y = y + np.random.default_rng(3).normal(0.0, 4.0, size=x.shape)
        config = _map_config(True)
        config["fitting"].update({"r1_fit_range": [700, 1200]})

        result = _fit_raman_domain(x, y, config, "minerals", preprocessed=True)

        assert result.peaks
        for peak in result.peaks:
            assert "cr_vetoed" not in peak


class TestDespikeParamsFromConfig:
    def test_reads_the_preprocessing_section(self):
        class _Cfg:
            preprocessing = {"despike": {"window_size": 9, "zscore_threshold": 4.5}}

        params = _despike_params_from_config(_Cfg())
        assert params.window_size == 9
        assert params.zscore_threshold == pytest.approx(4.5)

    def test_falls_back_to_defaults(self):
        params = _despike_params_from_config(object())
        assert params.window_size == 7
        assert params.zscore_threshold == pytest.approx(6.0)
