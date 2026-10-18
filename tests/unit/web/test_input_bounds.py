"""Request work bounds using invented identifiers and arrays only."""

from unittest.mock import Mock

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from sherloc_pipeline.web.routes import map as map_routes
from sherloc_pipeline.web.routes import processing
from sherloc_pipeline.web.schemas import BaselineParamsSchema, MapFitRequest


INVALID_DOMAINS = [
    ["minerals", "minerals"],
    ["minerals", "organics", "hydration", "fluorescence", "minerals"],
    ["unknown"],
    [],
]


@pytest.mark.parametrize("domains", INVALID_DOMAINS)
def test_map_domains_rejected(domains):
    with pytest.raises(ValidationError) as exc_info:
        MapFitRequest(scan_id="synthetic-scan", domains=domains)
    assert exc_info.value.errors()[0]["loc"][0] == "domains"


@pytest.mark.parametrize(
    "domains",
    [["minerals"], ["hydration", "fluorescence", "organics", "minerals"]],
)
def test_valid_map_domains_preserve_order(domains):
    body = MapFitRequest(scan_id="synthetic-scan", domains=domains)
    assert body.domains == domains


@pytest.mark.parametrize("max_iter", [0, 51])
def test_baseline_iterations_rejected(max_iter):
    with pytest.raises(ValidationError) as exc_info:
        BaselineParamsSchema(max_iter=max_iter)
    assert exc_info.value.errors()[0]["loc"] == ("max_iter",)


@pytest.mark.parametrize("max_iter", [1, 50])
def test_baseline_iteration_boundaries_accepted(max_iter):
    assert BaselineParamsSchema(max_iter=max_iter).max_iter == max_iter


@pytest.mark.asyncio
@pytest.mark.parametrize("domains", INVALID_DOMAINS)
async def test_invalid_map_domains_422_before_route_work(monkeypatch, domains):
    # Use the production router, with no application DB/executor configured.
    # A request must fail validation before even obtaining a DB session.
    get_session = Mock(side_effect=AssertionError("route work must not start"))
    monkeypatch.setattr(map_routes, "_get_session", get_session)
    app = FastAPI()
    app.include_router(map_routes.router)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/map/fit", json={"scan_id": "synthetic-scan", "domains": domains}
        )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert isinstance(errors, list)
    assert errors[0]["loc"][:2] == ["body", "domains"]
    get_session.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("max_iter", [0, 51])
@pytest.mark.parametrize("endpoint", ["baseline", "fit"])
async def test_invalid_iterations_422_before_numerical_work(
    monkeypatch, max_iter, endpoint
):
    baseline_fit = Mock(side_effect=AssertionError("numerical work must not start"))
    peak_fit = Mock(side_effect=AssertionError("numerical work must not start"))
    monkeypatch.setattr(processing, "fit_baseline", baseline_fit)
    monkeypatch.setattr(processing, "fit_spectrum", peak_fit)
    app = FastAPI()
    app.state.config = {}
    app.include_router(processing.router)
    params = {"max_iter": max_iter}
    if endpoint == "fit":
        params = {"baseline": params}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/api/process/{endpoint}",
            json={
                "wavenumber": np.linspace(0.0, 1.0, 8).tolist(),
                "intensity": np.linspace(1.0, 2.0, 8).tolist(),
                "params": params,
            },
        )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert isinstance(errors, list)
    assert errors[0]["loc"][-1] == "max_iter"
    baseline_fit.assert_not_called()
    peak_fit.assert_not_called()
