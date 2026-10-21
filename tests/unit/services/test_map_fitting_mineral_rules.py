"""Map mode honors configured mineral rules (issue #125)."""

from types import SimpleNamespace

import numpy as np

from sherloc_pipeline.services import map_fitting

# Synthetic peak; 1010 cm-1 falls in the packaged sulf1_v1 rule.
_PEAK_CENTER = 1010.0


def _stub_fit(monkeypatch):
    peak = SimpleNamespace(
        m_cm1=_PEAK_CENTER, snr=10.0, fwhm=40.0, a=100.0, area=4000.0,
        pass_snr=True, pass_fwhm=True,
    )
    result = SimpleNamespace(peaks=[peak], r2=0.9)
    monkeypatch.setattr(map_fitting, "fit_spectrum", lambda *a, **k: (result, None))


def _mineral_assignment(config):
    x = np.linspace(700.0, 1200.0, 50)
    y = np.zeros_like(x)
    res = map_fitting._fit_raman_domain(x, y, config, "minerals", preprocessed=True)
    return [p["assignment"] for p in res.peaks]


def test_packaged_rules_apply_without_override(monkeypatch):
    _stub_fit(monkeypatch)
    assert _mineral_assignment({"fitting": {}}) == ["sulf1_v1"]


def test_inline_mineral_rules_override_map_assignment(monkeypatch):
    _stub_fit(monkeypatch)
    config = {"fitting": {"mineral_rules": [{"label": "custom", "lo": 1000.0, "hi": 1015.0}]}}
    assert _mineral_assignment(config) == ["custom"]


def test_library_path_rules_override_map_assignment(monkeypatch, tmp_path):
    _stub_fit(monkeypatch)
    lib = tmp_path / "rules.yaml"
    lib.write_text("- {label: from_file, lo: 1005.0, hi: 1012.0}\n")
    config = {"fitting": {"library_path": str(lib)}}
    assert _mineral_assignment(config) == ["from_file"]
