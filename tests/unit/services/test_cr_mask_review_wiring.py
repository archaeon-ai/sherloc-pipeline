"""Cosmic-ray review flag wiring: this run's R1 masks -> fit_spectrum -> peak table.

The R1 selection tests are synthetic: they pin the channel-to-row mapping to the
one the R1 despike frame uses (apply_masks_to_r1_frame), including channels that
must be dropped (outside the certified window, or not exported to the frame).
"""
import numpy as np
import pandas as pd
import pytest

from sherloc_pipeline.core.mask_application import (
    apply_masks_to_r1_frame,
    derive_region_channel_masks,
    r1_mask_wavenumbers,
)
from sherloc_pipeline.services.fitting import (
    _attach_cr_mask_wavenumbers,
    _fit_point_minerals,
    _optional_bool,
)
from sherloc_pipeline.services.pipeline import _run_r1_cr_masks

N_CHANNELS = 2148
R1_WINDOW = (52, 575)


@pytest.fixture(scope="module")
def config():
    from sherloc_pipeline.config import get_config

    return get_config()


def _selection(lo, hi):
    sel = np.zeros(N_CHANNELS, dtype=bool)
    sel[lo:hi] = True
    return sel


class TestR1Selection:
    def test_rows_follow_the_frame_channel_selection(self):
        # Frame exports channels 60..99 (40 rows); window is [52, 575).
        sel = _selection(60, 100)
        shift = 800.0 + np.arange(40) * 2.0
        # 61 -> row 1, 99 -> row 39; 55 is in-window but not exported;
        # 600 is R2; 10 is below the window. Input order is not channel order.
        got = r1_mask_wavenumbers([99, 600, 61, 55, 10], shift, sel, R1_WINDOW)
        assert got.tolist() == [shift[1], shift[39]]

    def test_window_is_enforced_even_where_the_frame_exports(self):
        sel = _selection(40, 80)  # frame reaches below the certified window
        shift = np.arange(40, dtype=float)
        got = r1_mask_wavenumbers([45, 52, 79], shift, sel, R1_WINDOW)
        assert got.tolist() == [shift[52 - 40], shift[79 - 40]]

    def test_no_channels_gives_no_wavenumbers(self):
        sel = _selection(60, 100)
        shift = np.arange(40, dtype=float)
        assert r1_mask_wavenumbers(None, shift, sel, R1_WINDOW).size == 0
        assert r1_mask_wavenumbers([], shift, sel, R1_WINDOW).size == 0
        assert r1_mask_wavenumbers([600], shift, sel, R1_WINDOW).size == 0

    def test_row_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="rows but the channel mask selects"):
            r1_mask_wavenumbers([61], np.arange(39, dtype=float), _selection(60, 100), R1_WINDOW)

    def test_same_rows_as_the_r1_despike_frame(self, config):
        _, channel_masks = derive_region_channel_masks(config, N_CHANNELS)
        n_rows = int(channel_masks["R1"].sum())
        shift = np.linspace(150.0, 4200.0, n_rows)
        frame = pd.DataFrame({"raman_shift": shift, 0: np.ones(n_rows)})
        channels = [52, 300, 574, 575, 1000]
        _, spike_rows = apply_masks_to_r1_frame(
            frame, {"R1": {0: np.array(channels)}}, channel_masks["R1"],
            {"R1": R1_WINDOW}, "linear",
        )
        got = r1_mask_wavenumbers(channels, shift, channel_masks["R1"], R1_WINDOW)
        assert got.tolist() == shift[spike_rows[0].to_numpy()].tolist()
        assert len(got) == 3


class TestAttach:
    def _inputs(self):
        return [{"point_idx": 0, "y": None}, {"point_idx": 1, "y": None}]

    def test_no_masks_attaches_nothing(self, config):
        inputs = self._inputs()
        assert _attach_cr_mask_wavenumbers(inputs, None, np.arange(523.0), config) is None
        assert all("cr_mask_cm1" not in inp for inp in inputs)

    def test_masks_attach_per_point_and_absent_point_is_empty(self, config):
        inputs = self._inputs()
        x = 100.0 + np.arange(523.0)
        warning = _attach_cr_mask_wavenumbers(inputs, {0: [52, 53, 900]}, x, config)
        assert warning is None
        assert inputs[0]["cr_mask_cm1"] == [100.0, 101.0]
        assert inputs[1]["cr_mask_cm1"] == []

    def test_mismatched_input_rows_leave_the_flag_unset_with_a_warning(self, config):
        inputs = self._inputs()
        warning = _attach_cr_mask_wavenumbers(inputs, {0: [60]}, np.arange(400.0), config)
        assert warning.startswith("cr_mask_review not set: R1 frame has 400 rows")
        assert all("cr_mask_cm1" not in inp for inp in inputs)


def _gauss(x, c, a, f):
    return a * np.exp(-4 * np.log(2) * (x - c) ** 2 / f ** 2)


def _worker_peaks(tmp_path, cr_mask_cm1):
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "src/sherloc_pipeline/config.yaml").read_text()
    )["fitting"]
    x = np.linspace(500.0, 2200.0, 1800)
    y = np.random.default_rng(7).normal(0.0, 10.0, x.size) + _gauss(x, 1015.0, 600.0, 30.0)
    tmp_path.mkdir(parents=True, exist_ok=True)
    point = {"point_idx": 3, "y": y}
    if cr_mask_cm1 is not None:
        point["cr_mask_cm1"] = cr_mask_cm1
    out = _fit_point_minerals(
        point, x=x, fit_cfg=cfg, fit_roi=(700.0, 1200.0), plot_roi=(700.0, 1300.0),
        out_dir=str(tmp_path), sol="0001", target="T", scan="detail_1", region="R1",
        snr_min_cfg=3.0, fwhm_min_cfg=25.0,
    )
    csv = tmp_path / "0001_T_detail_1_R1_point3_fit_peaks.csv"
    assert csv in out["artifacts"]
    return pd.read_csv(csv)


class TestWorker:
    def test_worker_passes_masks_to_the_fit(self, tmp_path):
        assert _worker_peaks(tmp_path / "hit", [1012.0]).cr_mask_review.tolist() == [True]
        assert _worker_peaks(tmp_path / "miss", [1150.0]).cr_mask_review.tolist() == [False]
        assert _worker_peaks(tmp_path / "empty", []).cr_mask_review.tolist() == [False]

    def test_no_masks_leaves_the_column_blank(self, tmp_path):
        assert _worker_peaks(tmp_path, None).cr_mask_review.isna().all()


class TestRunMasks:
    def test_this_runs_r1_masks_by_point(self):
        md = {"preprocessing": {"despike": {"masks": {
            "0": {"R1": [60, 61], "R2": [700]}, "1": {"R1": []},
        }}}}
        assert _run_r1_cr_masks(md) == {0: [60, 61], 1: []}

    @pytest.mark.parametrize("md", [
        {},
        {"preprocessing": None},
        {"preprocessing": {"despike": {}}},
        {"preprocessing": {"despike": {"masks": {}}}},
        {"preprocessing": {"despike": {"masks": {"0": {"R2": [700]}}}}},
    ])
    def test_no_r1_masks_this_run_gives_none(self, md):
        assert _run_r1_cr_masks(md) is None


@pytest.mark.parametrize("value, expected", [
    (True, True), (False, False), ("True", True), ("False", False),
    (float("nan"), None), (None, None), ("", None), (np.float64("nan"), None),
])
def test_optional_bool(value, expected):
    assert _optional_bool(value) is expected
