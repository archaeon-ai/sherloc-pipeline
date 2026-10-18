"""Direct callers must not multiply fits through repeated domains."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sherloc_pipeline.services import map_fitting


def test_direct_caller_fits_each_unique_domain_once_per_point(monkeypatch):
    points = [SimpleNamespace(id=f"synthetic-point-{i}", point_index=i) for i in range(2)]
    session = Mock()
    query = session.query.return_value.filter.return_value
    query.order_by.return_value.all.return_value = points
    query.all.return_value = []  # No photodiode normalization data.

    axis = np.linspace(0.0, 1.0, 8)
    arrays = {point.id: np.linspace(1.0, 2.0, 8) for point in points}
    monkeypatch.setattr(
        map_fitting, "_load_point_spectra", Mock(return_value=arrays)
    )
    monkeypatch.setattr(
        map_fitting, "_load_point_spectra_multi", Mock(return_value={})
    )
    monkeypatch.setattr(
        map_fitting, "calculate_loupe_wavelength_wavenumber",
        Mock(return_value=(axis, axis)),
    )
    monkeypatch.setattr(
        map_fitting, "get_region_wavelength_mask",
        Mock(return_value=np.ones(8, dtype=bool)),
    )
    monkeypatch.setattr(
        map_fitting, "build_weight_vector_from_windows",
        Mock(return_value=np.ones(8)),
    )
    preprocess = Mock(side_effect=lambda x, y, *args, **kwargs: y)
    monkeypatch.setattr(map_fitting, "_preprocess_r1_intensity", preprocess)
    raman_fit = Mock(return_value=map_fitting.DomainResult(status="measured"))
    fluor_fit = Mock(return_value=map_fitting.DomainResult(status="measured"))
    monkeypatch.setattr(map_fitting, "_fit_raman_domain", raman_fit)
    monkeypatch.setattr(map_fitting, "_fit_fluorescence_domain", fluor_fit)

    domains = [
        "organics", "minerals", "organics", "hydration",
        "minerals", "fluorescence", "hydration", "fluorescence",
    ]
    original = domains.copy()
    results = []
    logs = []
    summary = map_fitting.MapFitService(config={}).run_map_fit(
        session=session,
        scan_id="synthetic-scan",
        domains=domains,
        point_indices=None,
        point_coords={i: (float(i), float(i)) for i in range(2)},
        on_point_fitted=results.append,
        on_progress=lambda *args: None,
        on_log=lambda index, message: logs.append((index, message)),
        cancel_event=threading.Event(),
    )

    assert [call.args[3] for call in raman_fit.call_args_list] == [
        "organics", "minerals", "hydration",
        "organics", "minerals", "hydration",
    ]
    assert [call.args[1] for call in fluor_fit.call_args_list] == [p.id for p in points]
    assert preprocess.call_count == len(points)
    assert summary.total_points == len(points)
    assert summary.detections == {
        "organics": 2, "minerals": 2, "hydration": 2, "fluorescence": 2,
    }
    assert all(list(result.results) == list(summary.detections) for result in results)
    assert logs[0][1] == (
        "Fitting 2 points x 4 domain(s): organics, minerals, hydration, fluorescence"
    )
    assert domains == original  # Do not mutate the caller's list.
