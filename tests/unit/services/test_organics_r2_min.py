"""Organics keep their own R2 minimum.

The mineral ``r_squared_min`` is 0.0. The
organics fit config is copied from the mineral one, so without its own
``organics_r2_min`` the dropped minimum reached organics by inheritance and
a G fit with R2 below 0.25 was kept. These tests drive the real
``FittingService.fit_organics`` with the shipped config over a synthetic scan.
"""

import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import yaml

from sherloc_pipeline.config import load_config
from sherloc_pipeline.services.fitting import FittingService
from sherloc_pipeline.services.runtime import RuntimeContext

SOL = "0921"
TARGET = "Test_Target"
SCAN = "detail_1"

SHIPPED = yaml.safe_load(
    (Path(__file__).resolve().parents[3] / "src/sherloc_pipeline/config.yaml").read_text()
)["fitting"]


def _g_point(window_noise: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A G-like band at 1600 cm-1 (FWHM 60) on quiet noise outside the G window.

    Extra noise inside 1450-1750 cm-1 lowers the single-band fit's R2 while the
    SNR (from the quiet 2000-2100 cm-1 window) stays high.
    """
    x = np.linspace(800.0, 2200.0, 1400)
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 5.0, x.size)
    window = (x > 1450.0) & (x < 1750.0)
    y[window] += rng.normal(0.0, window_noise, int(window.sum()))
    y += 45.0 * np.exp(-4.0 * np.log(2.0) * (x - 1600.0) ** 2 / 60.0 ** 2)
    return x, y


def _fit(tmp_path, point_y: dict[str, np.ndarray], x: np.ndarray):
    base = tmp_path / "results" / SCAN
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"raman_shift": x, **point_y}).to_csv(
        base / f"{SOL}_{TARGET}_{SCAN}_R1_normalized_baselined.csv", index=False
    )
    cfg = copy.deepcopy(load_config())
    cfg.fitting = dict(cfg.fitting or {})
    cfg.fitting["parallel_workers"] = 1
    context = RuntimeContext.bootstrap(
        data_dir=tmp_path / "data", results_dir=tmp_path / "results", config=cfg
    )
    scan_ctx = MagicMock()
    scan_ctx.base_data_dir = base
    scan_ctx.results_dir = base
    ingestion = MagicMock()
    ingestion.get_results_path.return_value = base
    with patch("sherloc_pipeline.services.fitting.resolve_scan_context",
               return_value=scan_ctx), \
         patch("sherloc_pipeline.core.data_ingestion.DataIngestion",
               return_value=ingestion):
        FittingService(context=context).fit_organics(sol=SOL, target=TARGET, scan=SCAN)
    return pd.read_csv(base / f"{SOL}_{TARGET}_{SCAN}_R1_organics_summary.csv")


def test_shipped_config_keeps_minerals_at_none_and_organics_at_025():
    assert SHIPPED["r_squared_min"] == 0.0
    assert SHIPPED["organics_r2_min"] == 0.25


def test_low_r2_g_fit_is_not_detected_but_a_good_one_is(tmp_path):
    x, low = _g_point(window_noise=40.0)   # G fit R2 about 0.18
    _, good = _g_point(window_noise=0.0)   # G fit R2 near 1
    summary = _fit(tmp_path, {"0": low, "1": good}, x).set_index("point")

    assert 0.0 < summary.loc[0, "g_r2"] < 0.25
    assert not bool(summary.loc[0, "g_detected"]), (
        "a G fit below the organics R2 minimum was kept: the mineral "
        "r_squared_min reached organics"
    )
    assert summary.loc[1, "g_r2"] >= 0.25
    assert bool(summary.loc[1, "g_detected"])


def test_map_organics_fit_uses_the_organics_r2_min(monkeypatch):
    """The map refit's organics config takes organics_r2_min, not r_squared_min.

    The map path keeps peaks on pass_snr and pass_fwhm only, so this setting
    changes no map output today; the test pins the config wiring.
    """
    from sherloc_pipeline.services import map_fitting

    seen = {}

    def spy(x, y, cfg, roi=None, **kwargs):
        seen["r_squared_min"] = cfg["r_squared_min"]
        raise RuntimeError("stop after capturing the config")

    monkeypatch.setattr(map_fitting, "fit_spectrum", spy)
    x, y = _g_point(window_noise=0.0)
    config = {"fitting": {**SHIPPED}}
    result = map_fitting._fit_raman_domain(x, y, config, "organics", preprocessed=True)

    assert result.status == "missing"
    assert config["fitting"]["r_squared_min"] == 0.0
    assert seen["r_squared_min"] == 0.25
