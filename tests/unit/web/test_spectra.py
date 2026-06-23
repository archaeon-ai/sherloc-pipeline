"""Spectra route tests."""

import pytest

from tests.unit.web.conftest import SCAN_UUID


@pytest.mark.asyncio
async def test_average_spectrum(client):
    resp = await client.get(f"/api/spectra/{SCAN_UUID}/average")
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert data["scan_id"] == SCAN_UUID
    assert data["region"] == "R1"
    assert data["n_points_averaged"] == 3
    assert data["baseline_corrected"] is False
    assert data["n_channels"] == len(data["wavenumber"])
    assert data["n_channels"] == len(data["intensity"])
    # R1 region should have 523 channels
    assert data["n_channels"] == 523
    assert data["provenance"]["calibration_version"] == "loupe_v5.1.5a"
    assert data["provenance"]["wavenumber_unit"] == "cm-1"


@pytest.mark.asyncio
async def test_average_spectrum_invalid_region(client):
    resp = await client.get(f"/api/spectra/{SCAN_UUID}/average", params={"region": "R4"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_average_spectrum_not_found(client):
    resp = await client.get("/api/spectra/00000000-0000-0000-0000-000000000099/average")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_point_spectrum(client):
    resp = await client.get(f"/api/spectra/{SCAN_UUID}/point/0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert data["point_index"] == 0
    assert data["region"] == "R1"
    assert data["spectrum_type"] == "dark_subtracted"
    assert data["n_channels"] == 523
    assert data["photodiode_mean"] == 4800.0


@pytest.mark.asyncio
async def test_point_spectrum_out_of_range(client):
    resp = await client.get(f"/api/spectra/{SCAN_UUID}/point/999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_subset_average(client):
    resp = await client.post(
        f"/api/spectra/{SCAN_UUID}/subset",
        json={"point_indices": [0, 1, 2], "region": "R1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert data["n_points_averaged"] == 3
    assert data["point_indices"] == [0, 1, 2]
    assert data["n_channels"] == 523


@pytest.mark.asyncio
async def test_subset_empty_indices(client):
    resp = await client.post(
        f"/api/spectra/{SCAN_UUID}/subset",
        json={"point_indices": []},
    )
    assert resp.status_code == 422  # Pydantic validation


@pytest.mark.asyncio
async def test_subset_out_of_range(client):
    resp = await client.post(
        f"/api/spectra/{SCAN_UUID}/subset",
        json={"point_indices": [0, 999]},
    )
    assert resp.status_code == 400
