"""Display annotations served by GET /api/spectra/anomalous-pixels."""

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from sherloc_pipeline.core.anomalous_pixels import _parse_anomalous_pixels_csv, load_anomalous_pixels
from sherloc_pipeline.core.calibration import (
    calculate_loupe_wavelength_wavenumber,
    get_region_wavelength_mask,
)
from tests.unit.web.conftest import _FakeConfig

N_CHANNELS = 2148
S3 = {106, 108, 125, 160, 174, 178, 256, 337, 361, 373, 393, 426, 442, 470, 473}
S4 = {
    640, 641, 689, 995, 996, 1070, 1106, 1133, 1134, 1330, 1456, 1505, 1506,
    1615, 1651, 1743, 1775, 1780, 1864, 1877, 2091,
}


def test_published_channels_and_tables():
    records = load_anomalous_pixels()
    assert [r.channel for r in records] == sorted(S3 | S4)
    assert {r.channel for r in records if r.table == "S3"} == S3
    assert {r.channel for r in records if r.table == "S4"} == S4


@pytest.mark.parametrize("rows", [
    "160,S5", "-1,S3", "2148,S4", "160,S3\n160,S4", "x,S3", "160", "160,S3,extra",
], ids=["table", "negative", "upper-bound", "duplicate", "noninteger", "short", "long"])
def test_malformed_row_raises(rows):
    with pytest.raises(ValueError):
        _parse_anomalous_pixels_csv("channel,table\n" + rows)


def _selection(region: str) -> np.ndarray:
    """Absolute channel indices the region's served array spans."""
    wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels=N_CHANNELS)
    if region == "R123":
        return np.arange(N_CHANNELS)
    return np.where(get_region_wavelength_mask(wavelength, region))[0]


@pytest_asyncio.fixture()
async def client(monkeypatch):
    # The anomalous pixel endpoint is static (no scan, no DB), so no seeded engine is
    # needed — an empty in-memory app suffices. create_app builds its own
    # engine when none is passed.
    from sherloc_pipeline.web.app import create_app
    from sherloc_pipeline.web.auth import _reset_validator_for_tests

    monkeypatch.setenv("SHERLOC_AUTH_MODE", "dev")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "internal")
    monkeypatch.setenv("SHERLOC_DB", ":memory:")
    monkeypatch.delenv("SHERLOC_AUTH0_DOMAIN", raising=False)
    _reset_validator_for_tests()

    app = create_app(config=_FakeConfig())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    _reset_validator_for_tests()


@pytest.mark.asyncio
async def test_r1_channel_137_not_anomalous_pixel(client):
    data = (await client.get("/api/spectra/anomalous-pixels", params={"region": "R1"})).json()
    assert 137 not in {it["channel"] for it in data["anomalous_pixels"]}


@pytest.mark.asyncio
async def test_r1_position_and_wavenumber_mapping(client):
    """Channel 160 is the most frequent anomalous pixel in the paper."""
    _, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=N_CHANNELS)
    channels = _selection("R1")
    data = (await client.get("/api/spectra/anomalous-pixels", params={"region": "R1"})).json()
    rec = next(it for it in data["anomalous_pixels"] if it["channel"] == 160)
    assert rec["table"] == "S3"
    assert rec["position"] != 160
    assert wavenumber[channels][rec["position"]] == wavenumber[160]


@pytest.mark.parametrize("region", ["R1", "R2", "R3"])
@pytest.mark.asyncio
async def test_positions_match_served_selection(client, region):
    """Every returned position is the served index of its absolute channel."""
    data = (await client.get("/api/spectra/anomalous-pixels", params={"region": region})).json()
    assert data["region"] == region
    channels = _selection(region)
    published = S3 if region == "R1" else S4
    expected_table = "S3" if region == "R1" else "S4"
    assert data["n_channels"] == len(channels)
    assert data["anomalous_pixels"] == [
        {"position": pos, "channel": int(channel), "table": expected_table}
        for pos, channel in enumerate(channels) if channel in published
    ]


@pytest.mark.asyncio
async def test_r123_identity_and_full_table(client):
    data = (await client.get("/api/spectra/anomalous-pixels", params={"region": "R123"})).json()
    assert data["region"] == "R123"
    assert data["n_channels"] == N_CHANNELS
    assert [it["position"] for it in data["anomalous_pixels"]] == sorted(S3 | S4)
    assert all(it["position"] == it["channel"] for it in data["anomalous_pixels"])


@pytest.mark.asyncio
async def test_item_shape_and_domains(client):
    data = (await client.get("/api/spectra/anomalous-pixels", params={"region": "R123"})).json()
    for it in data["anomalous_pixels"]:
        assert set(it) == {"position", "channel", "table"}
        assert it["table"] == ("S3" if it["channel"] in S3 else "S4")


@pytest.mark.asyncio
async def test_default_region_is_r1(client):
    """Omitting ``region`` defaults to R1 (matches the spectra endpoints)."""
    data = (await client.get("/api/spectra/anomalous-pixels")).json()
    assert data["region"] == "R1"


@pytest.mark.parametrize("region", ["R4", "raman", "", "r1"])
@pytest.mark.asyncio
async def test_invalid_region_400(client, region):
    resp = await client.get("/api/spectra/anomalous-pixels", params={"region": region})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_anomalous_pixels_schema_version(client):
    data = (await client.get("/api/spectra/anomalous-pixels", params={"region": "R1"})).json()
    assert data["schema_version"] == "2.0.0"


@pytest.mark.asyncio
async def test_no_despike_fields_in_response(client):
    """The annotation layer is separate from despiking: the response carries no
    despike provenance fields whatsoever.
    """
    data = (await client.get("/api/spectra/anomalous-pixels", params={"region": "R1"})).json()
    for forbidden in (
        "despike_applied",
        "despike_method",
        "masked_positions",
        "masked_channels",
    ):
        assert forbidden not in data


@pytest.mark.asyncio
async def test_former_route_is_not_served(client):
    """The former /api/spectra/badpix path answers as any unknown path does (no alias)."""
    former = await client.get("/api/spectra/badpix")
    unknown = await client.get("/api/spectra/unknown-route")
    assert former.status_code == unknown.status_code
    assert former.content == unknown.content
