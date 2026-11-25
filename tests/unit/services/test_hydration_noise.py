"""Noise used by batch, map and single-spectrum Raman fits."""
from types import SimpleNamespace

import numpy as np
import pytest

from sherloc_pipeline.core import fitting
from sherloc_pipeline.services import map_fitting
from sherloc_pipeline.services.fitting import _fit_point_hydration
from sherloc_pipeline.web.routes.processing import process_fit
from sherloc_pipeline.web.schemas import FitRequest


def workbench_fit(x, y, config, domain, roi):
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=config)))
    body = FitRequest(wavenumber=x.tolist(), intensity=y.tolist(),
                      params={"fitting": {"domain": domain, "wavenumber_range": roi}})
    return process_fit(request, body)


@pytest.mark.parametrize(("window", "roi"), [
    ((2000, 2100), (2800, 3900)),  # tilted interval
    ((1900, 1980), (2800, 3900)),  # alternative configured interval
    ((2000, 2020), (2800, 3900)),  # two channels, use the quiet part of the seed
    ((2000, 2020), (3100, 3900)),  # no quiet channels, use the whole seed
])
def test_hydration_noise_on_same_input(monkeypatch, tmp_path, window, roi):
    x = np.arange(1900., 3901., 20.)
    y = .5 * (x - 2000) + np.cos(np.arange(x.size)) + fitting.gaussian(x, 3400, 50, 120)
    cfg = {"hydration_noise_window": window, "noise_estimation": {"window": [2200, 2400]},
           "hydration_fit_range": roi}
    mask = (x >= roi[0]) & (x <= roi[1])
    noises = []

    def record_noise(x, seed, cfg, *, roi, noise_std, linear_baseline):
        noises.append(noise_std)
        return fitting.FitResult([], 0., 0., 0, []), np.zeros_like(seed)

    monkeypatch.setattr(fitting, "fit_spectrum", record_noise)
    _fit_point_hydration({"point_idx": 0, "y": y}, x=x, oh_roi=roi, oh_plot=roi,
                         fit_cfg_oh=cfg, oh_mask=mask, plot_mask_oh=mask, n_edge=5,
                         min_snr=0., r2_min=1., center_lo=3000., center_hi=3900.,
                         out_dir=str(tmp_path), sol="1", target="sample", scan="scan")
    map_fitting._fit_raman_domain(x, y, {"fitting": cfg}, "hydration", preprocessed=True)
    workbench_fit(x, y, SimpleNamespace(fitting=cfg), "hydration", roi)
    batch, mapped, workbench = noises
    assert (mapped, workbench) == (batch, batch)


@pytest.mark.parametrize(("domain", "roi"), [("minerals", (700, 1200)), ("organics", (1250, 1850))])
def test_other_raman_noise_uses_window_sd(monkeypatch, domain, roi):
    x = np.arange(700., 3901., 10.)
    y = .5 * x + np.cos(x)
    cfg = {"hydration_noise_window": [2000, 2100], "noise_estimation": {"window": [2150, 2200]}}
    noises = []
    estimate = fitting.compute_noise_std

    def record_noise(x, y, cfg):
        noise = estimate(x, y, cfg=cfg)
        noises.append(noise)
        return noise

    monkeypatch.setattr(fitting, "compute_noise_std", record_noise)
    map_fitting._fit_raman_domain(x, y, {"fitting": cfg}, domain, preprocessed=True)
    workbench_fit(x, y, SimpleNamespace(fitting=cfg), domain, roi)
    expected = np.std(y[(x >= 2150) & (x <= 2200)])
    assert noises == [expected, expected]


@pytest.mark.parametrize(("slope", "window", "previous_sd"), [
    (0., (2000, 2100), float.fromhex("0x1.6a09e667f3bcdp+0")),
    (.5, (2000, 2100), float.fromhex("0x1.6a09e667f3bcdp+0")),
    (.5, (2000, 2000), 5.),
])
def test_batch_noise_reference_bits(slope, window, previous_sd):
    # The two readings at the center scatter equally about the fitted line.
    x = np.array([2000., 2050., 2050., 2100.])
    y = slope * (x - 2050) + [0., -2., 2., 0.]
    x_oh = np.array([2800., 2850., 2950., 3000., 3300.])
    seed = np.array([-5., 5., -5., 5., 20.])
    assert fitting.hydration_noise(x, y, window, x_oh, seed).hex() == previous_sd.hex()
