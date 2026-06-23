"""``GET /api/spectra/badpix`` known-noisy channel annotation layer (issue #9).

The badpix endpoint serves a STATIC curated list of known-noisy detector
channels, mapped to positions INTO the served array for a region view. It is
an annotation surface only — wholly separate from cosmic-ray despiking, with
no DB involvement. These tests pin:

- position-mapping correctness per region: a known channel maps to the served
  position whose wavenumber matches the absolute channel's wavenumber;
- the carbonate nu1 apex (channel 137) is NOT badpix — it carries real carbonate
  and is dark-quiet; the 2026-06-17 dark-veto remediation removed it (with the
  sulfate nu1 channels) as epsilon-rate-criterion false positives;
- R123 identity (full-stitch served array → position == channel);
- per-region item set matches the curated table restricted to that region;
- response shape ({position, channel, tier, source}) + tier/source domains;
- invalid region → 400 (matches the spectra endpoints' contract);
- additive only — ``schema_version`` stays ``1.0.0``; no scan/DB needed.
"""

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from sherloc_pipeline.core.badpix import VALID_BADPIX_SOURCES, load_badpix_channels
from sherloc_pipeline.core.calibration import (
    calculate_loupe_wavelength_wavenumber,
    get_region_wavelength_mask,
)
from tests.unit.web.conftest import _FakeConfig

N_CHANNELS = 2148
# Carbonate nu1 apex (~1086.7 cm-1): REAL carbonate signal, dark-quiet — NOT a
# bad pixel. The prior epsilon-rate criterion mis-flagged this active-only band;
# the 2026-06-17 dark-veto remediation removed it (and the sulfate nu1 channels).
CARBONATE_APEX_CHANNEL = 137
# A genuine R1 defect that remains (JB25-attested, dark-firing, ~1307.1 cm-1),
# used for the position-mapping checks that 137 previously anchored.
KEPT_R1_DEFECT_CHANNEL = 160


def _selection(region: str) -> np.ndarray:
    """Absolute channel indices the region's served array spans."""
    wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels=N_CHANNELS)
    if region == "R123":
        return np.arange(N_CHANNELS)
    return np.where(get_region_wavelength_mask(wavelength, region))[0]


def _served_position(region: str, abs_channel: int) -> int:
    sel = _selection(region)
    return int(np.where(sel == abs_channel)[0][0])


def _served_wavenumber(region: str) -> np.ndarray:
    wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=N_CHANNELS)
    if region == "R123":
        return wavenumber
    return wavenumber[get_region_wavelength_mask(wavelength, region)]


@pytest_asyncio.fixture()
async def client(monkeypatch):
    # The badpix endpoint is static (no scan, no DB), so no seeded engine is
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


# ---------------------------------------------------------------------------
# (a) position-mapping correctness per region
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r1_carbonate_apex_not_badpix(client):
    """The carbonate nu1 apex (137) carries real carbonate and is dark-quiet, so
    it must NOT be in the badpix list. Pins the 2026-06-17 dark-veto remediation
    that removed it — the prior epsilon-rate criterion was inverted (it flagged
    active-only real bands and missed the true RTS defects flanking them)."""
    data = (await client.get("/api/spectra/badpix", params={"region": "R1"})).json()
    apex = [it for it in data["badpix"] if it["channel"] == CARBONATE_APEX_CHANNEL]
    assert apex == [], "carbonate apex 137 is real signal, must NOT be in the badpix list"


@pytest.mark.asyncio
async def test_r1_position_and_wavenumber_mapping(client):
    """A known channel maps to the served position whose wavenumber matches the
    absolute channel's wavenumber. R1's served selection starts mid-detector, so
    position != channel."""
    _, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=N_CHANNELS)
    data = (await client.get("/api/spectra/badpix", params={"region": "R1"})).json()
    rec = [it for it in data["badpix"] if it["channel"] == KEPT_R1_DEFECT_CHANNEL]
    assert len(rec) == 1, "expected the kept R1 defect channel in the list"
    rec = rec[0]
    expected_pos = _served_position("R1", KEPT_R1_DEFECT_CHANNEL)
    assert rec["position"] == expected_pos
    assert rec["position"] != rec["channel"]
    full_wn = _served_wavenumber("R1")
    assert abs(full_wn[expected_pos] - wavenumber[KEPT_R1_DEFECT_CHANNEL]) < 0.01


@pytest.mark.parametrize("region", ["R1", "R2", "R3"])
@pytest.mark.asyncio
async def test_positions_match_served_selection(client, region):
    """Every returned position is the served index of its absolute channel."""
    data = (await client.get("/api/spectra/badpix", params={"region": region})).json()
    assert data["region"] == region
    for it in data["badpix"]:
        assert it["position"] == _served_position(region, it["channel"])
    # positions are sorted + within the served array bounds.
    positions = [it["position"] for it in data["badpix"]]
    assert positions == sorted(positions)
    assert all(0 <= p < data["n_channels"] for p in positions)


@pytest.mark.asyncio
async def test_per_region_item_set_matches_curated_table(client):
    """The endpoint returns exactly the curated channels that fall in-region."""
    recs = load_badpix_channels()
    for region in ("R1", "R2", "R3"):
        in_region = sorted(r.channel for r in recs if r.region == region)
        data = (await client.get("/api/spectra/badpix", params={"region": region})).json()
        got = sorted(it["channel"] for it in data["badpix"])
        assert got == in_region


# ---------------------------------------------------------------------------
# (b) R123 composite — positions equal absolute channels (full-stitch identity)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r123_identity_and_full_table(client):
    data = (await client.get("/api/spectra/badpix", params={"region": "R123"})).json()
    assert data["region"] == "R123"
    assert data["n_channels"] == N_CHANNELS
    # R123 serves the full 2148-channel stitch → position == channel, and ALL
    # curated channels (every region) are returned.
    assert len(data["badpix"]) == len(load_badpix_channels())
    for it in data["badpix"]:
        assert it["position"] == it["channel"]


# ---------------------------------------------------------------------------
# (c) response shape + value domains
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_item_shape_and_domains(client):
    data = (await client.get("/api/spectra/badpix", params={"region": "R123"})).json()
    assert set(data["badpix"][0].keys()) == {"position", "channel", "tier", "source"}
    for it in data["badpix"]:
        assert it["tier"] in (1, 2)
        assert it["source"] in VALID_BADPIX_SOURCES
        assert 0 <= it["channel"] < N_CHANNELS


@pytest.mark.asyncio
async def test_default_region_is_r1(client):
    """Omitting ``region`` defaults to R1 (matches the spectra endpoints)."""
    data = (await client.get("/api/spectra/badpix")).json()
    assert data["region"] == "R1"


# ---------------------------------------------------------------------------
# (d) invalid region + additive schema version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("region", ["R4", "raman", "", "r1"])
@pytest.mark.asyncio
async def test_invalid_region_400(client, region):
    resp = await client.get("/api/spectra/badpix", params={"region": region})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_schema_version_unchanged(client):
    data = (await client.get("/api/spectra/badpix", params={"region": "R1"})).json()
    assert data["schema_version"] == "1.0.0"


# ---------------------------------------------------------------------------
# (e) does NOT interact with despike — no scan, no DB needed to serve it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_despike_fields_in_response(client):
    """The annotation layer is separate from despiking: the response carries no
    despike provenance fields whatsoever.
    """
    data = (await client.get("/api/spectra/badpix", params={"region": "R1"})).json()
    for forbidden in (
        "despike_applied",
        "despike_method",
        "masked_positions",
        "masked_channels",
    ):
        assert forbidden not in data
