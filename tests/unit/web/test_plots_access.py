"""Plot authorization with invented scan stubs and no spectrum data."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib
import matplotlib.pyplot as plt
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from sherloc_pipeline.core import calibration
from sherloc_pipeline.database.models import ScanORM
from sherloc_pipeline.web.routes import plots


@pytest.fixture
def plot_stubs(monkeypatch):
    """Stub rendering and calibration; never generate spectroscopy values."""
    figure, axes = MagicMock(), MagicMock()
    figure.savefig.side_effect = lambda buffer, **kwargs: buffer.write(b"invented-plot")
    subplots = MagicMock(return_value=(figure, axes))
    use = MagicMock()
    calibrate = MagicMock(return_value=(MagicMock(), MagicMock()))
    mask = MagicMock(return_value=object())
    monkeypatch.setattr(matplotlib, "use", use)
    monkeypatch.setattr(plt, "subplots", subplots)
    monkeypatch.setattr(plt, "close", MagicMock())
    monkeypatch.setattr(calibration, "calculate_loupe_wavelength_wavenumber", calibrate)
    monkeypatch.setattr(calibration, "get_region_wavelength_mask", mask)
    return SimpleNamespace(subplots=subplots, use=use, calibrate=calibrate, mask=mask)


def _plot_app(access_mode, scan):
    """Mount the real router with the same request/app state as the app factory."""
    session = MagicMock()
    query = MagicMock()
    query.filter.return_value.first.return_value = scan

    def scan_query(model):
        assert model is ScanORM, "Plot authorization must precede any spectrum query"
        return query

    session.query.side_effect = scan_query
    app = FastAPI()
    app.state.access_mode = access_mode

    @app.middleware("http")
    async def attach_session(request, call_next):
        request.state.db = session
        return await call_next(request)

    app.include_router(plots.router)
    return app, session


@pytest.mark.asyncio
@pytest.mark.parametrize("plot_type", ["spectrogram", "averaged_spectrum"])
async def test_public_loupe_plot_is_forbidden_before_rendering(
    plot_type, plot_stubs, monkeypatch
):
    # App state is authoritative even if the ambient environment says internal.
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "internal")
    scan = SimpleNamespace(data_source="loupe", target="Invented", scan_name="sample")
    app, session = _plot_app("public", scan)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/plots/invented-scan", params={"plot_type": plot_type}
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "This resource is not available in public mode"
    session.query.assert_called_once_with(ScanORM)
    plot_stubs.use.assert_not_called()
    plot_stubs.calibrate.assert_not_called()
    plot_stubs.mask.assert_not_called()
    plot_stubs.subplots.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("access_mode,data_source", [("public", "pds4"), ("internal", "loupe")])
async def test_authorized_plot_still_renders(access_mode, data_source, plot_stubs):
    scan = SimpleNamespace(data_source=data_source, target="Invented", scan_name="sample")
    app, session = _plot_app(access_mode, scan)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/plots/invented-scan")

    assert response.status_code == 200
    assert response.content == b"invented-plot"
    assert response.headers["content-type"] == "image/png"
    session.query.assert_called_once_with(ScanORM)
    plot_stubs.subplots.assert_called_once()


@pytest.mark.asyncio
async def test_missing_plot_scan_stays_404(plot_stubs):
    app, session = _plot_app("public", None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/plots/absent-invented-scan")

    assert response.status_code == 404
    assert response.json()["detail"] == "Scan not found"
    session.query.assert_called_once_with(ScanORM)
    plot_stubs.use.assert_not_called()
    plot_stubs.subplots.assert_not_called()
