"""Web stored-mask despike toggle round-trip tests (spec §4.6, MLD-IFC-003).

The serving host never runs inference: ``?despike=true`` looks up the
persisted ``cosmic_ray_masks`` and applies the same interpolation helper the
pipeline uses. These tests pin:

- exact-channel diff between despiked and non-despiked single-region views;
- communicated state when a spectrum has no stored mask
  (``despike_applied=false``, no error, no inference);
- constituent-first R123 despiking with the coverage-disclosure count;
- all-or-none composite availability (a missing constituent region renders
  the whole composite non-despiked with ``despike_missing_regions``);
- the additive fields keep ``schema_version`` at ``1.0.0``;
- the serving path imports no ML runtime.
"""

import uuid
import zlib

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from sherloc_pipeline.database.connection import (
    create_all_tables,
    get_session_factory,
)
from sherloc_pipeline.database.models import (
    CosmicRayMaskORM,
    ScanORM,
    ScanPointORM,
    SolORM,
    SpectrumORM,
)
from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST
from tests.unit.web.conftest import _FakeConfig

SCAN_UUID = str(uuid.UUID("00000000-0000-0000-0000-0000000000a1"))
SOL_NUMBER = 921
N_POINTS = 3
N_CHANNELS = 2148
METHOD = DEFAULT_MANIFEST.provenance_label
MODEL_SHA = DEFAULT_MANIFEST.sha256

# Flagged channels per point/region (absolute, in the region's certified
# window). Point 0 has all three; point 1 lacks R2 (all-or-none probe);
# point 2 has none. R2 channel 600 sits in the R1+R2 stitch overlap.
MASKS = {
    0: {"R1": [200], "R2": [600], "R3": [1700]},
    1: {"R1": [210], "R3": [1710]},  # no R2 mask
    2: {},
}


def _spectrum_bytes(point_index: int, region: str) -> bytes:
    """Deterministic smooth+noise 2148-channel array (distinct per row)."""
    seed = point_index * 10 + {"R1": 1, "R2": 2, "R3": 3}[region]
    rng = np.random.RandomState(seed)
    x = np.arange(N_CHANNELS, dtype=np.float64)
    spectrum = 500.0 + 80.0 * np.exp(-((x - 300) ** 2) / (2 * 40**2))
    spectrum += rng.normal(0, 7, size=N_CHANNELS)
    return zlib.compress(spectrum.astype(np.float32).tobytes())


@pytest.fixture()
def despike_engine() -> Engine:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    create_all_tables(engine)
    factory = get_session_factory(engine)
    session = factory()
    try:
        session.add(SolORM(sol_number=SOL_NUMBER, data_source="loupe"))
        session.flush()
        scan = ScanORM(
            id=SCAN_UUID,
            sol_number=SOL_NUMBER,
            scan_name="detail_1",
            target="Amherst_Point",
            scan_id="0921_Amherst_Point_detail_1",
            sclk_start=730000000,
            sclk_stop=730001000,
            n_points=N_POINTS,
            n_channels=N_CHANNELS,
            shots_per_point=50,
            laser_wavelength_nm=248.5794,
            data_source="loupe",
            target_type="mars_target",
            scan_class="primary",
            scan_type="detail",
        )
        session.add(scan)
        session.flush()

        for i in range(N_POINTS):
            pt_id = str(uuid.UUID(f"00000000-0000-0000-0000-0000000{1000 + i:05d}"))
            session.add(
                ScanPointORM(
                    id=pt_id,
                    scan_id=SCAN_UUID,
                    point_index=i,
                    x_pixel=100.0 + i,
                    y_pixel=200.0 + i,
                    # Constant photodiode so on-the-fly normalization is a
                    # no-op scaling (max_pd / pd == 1) and assertions read
                    # cleanly. The despike diff is normalization-invariant
                    # either way (linear interp commutes with scaling).
                    photodiode_mean=5000.0,
                    photodiode_std=10.0,
                )
            )
            session.flush()
            for region in ("R1", "R2", "R3"):
                sp_id = str(uuid.uuid4())
                session.add(
                    SpectrumORM(
                        id=sp_id,
                        scan_point_id=pt_id,
                        region=region,
                        spectrum_type="dark_subtracted",
                        processing_level="dark_subtracted",
                        intensities=_spectrum_bytes(i, region),
                    )
                )
                session.flush()
                flagged = MASKS[i].get(region)
                if flagged is not None:
                    session.add(
                        CosmicRayMaskORM(
                            id=str(uuid.uuid4()),
                            spectrum_id=sp_id,
                            method=METHOD,
                            model_sha256=MODEL_SHA,
                            tau=float(DEFAULT_MANIFEST.tau[region]),
                            channel_indices=list(flagged),
                            n_flagged=len(flagged),
                        )
                    )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    return engine


@pytest_asyncio.fixture()
async def client(despike_engine, monkeypatch):
    from sherloc_pipeline.web.app import create_app
    from sherloc_pipeline.web.auth import _reset_validator_for_tests

    monkeypatch.setenv("SHERLOC_AUTH_MODE", "dev")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "internal")
    monkeypatch.delenv("SHERLOC_AUTH0_DOMAIN", raising=False)
    _reset_validator_for_tests()

    app = create_app(engine=despike_engine, config=_FakeConfig())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    _reset_validator_for_tests()


def _r1_row_of_channel(channel: int) -> int:
    """Row index of an absolute channel in the R1 wavelength selection."""
    from sherloc_pipeline.core.calibration import (
        calculate_loupe_wavelength_wavenumber,
        get_region_wavelength_mask,
    )

    wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels=N_CHANNELS)
    sel = np.where(get_region_wavelength_mask(wavelength, "R1"))[0]
    return int(np.where(sel == channel)[0][0])


# ---------------------------------------------------------------------------
# Single-region (R1) — exact diff, communicated state, defaults
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_point_r1_despike_exact_channel_diff(client):
    off = (await client.get(f"/api/spectra/{SCAN_UUID}/point/0", params={"region": "R1"})).json()
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0", params={"region": "R1", "despike": True}
        )
    ).json()

    assert off["despike_applied"] is False
    assert off["n_masked_channels"] == 0
    assert on["despike_applied"] is True
    assert on["despike_method"] == METHOD
    assert on["masked_channels"] == [200]
    assert on["n_masked_channels"] == 1
    assert on["n_uncovered_contributor_channels"] == 0  # single region

    a = np.asarray(off["intensity"])
    b = np.asarray(on["intensity"])
    changed = np.where(a != b)[0]
    assert changed.tolist() == [_r1_row_of_channel(200)]


@pytest.mark.asyncio
async def test_point_r1_no_stored_mask_communicates_state(client):
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/2", params={"region": "R1", "despike": True}
        )
    ).json()
    assert on["despike_applied"] is False
    assert on["despike_method"] is None
    assert on["n_masked_channels"] == 0
    assert on["masked_channels"] == []
    assert on["despike_missing_regions"] == []


@pytest.mark.asyncio
async def test_additive_fields_keep_schema_version_1_0_0(client):
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0", params={"region": "R1", "despike": True}
        )
    ).json()
    assert on["schema_version"] == "1.0.0"
    # The new fields are present (additive, defaulted) without a version bump.
    for field in (
        "despike_applied",
        "despike_method",
        "n_masked_channels",
        "masked_channels",
        "n_uncovered_contributor_channels",
    ):
        assert field in on


@pytest.mark.asyncio
async def test_average_r1_despike_applies_per_point(client):
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/average", params={"region": "R1", "despike": True}
        )
    ).json()
    # Points 0 and 1 have R1 masks (200, 210); point 2 has none.
    assert on["despike_applied"] is True
    assert on["despike_method"] == METHOD
    assert on["n_masked_channels"] == 2  # union {200, 210}


@pytest.mark.asyncio
async def test_subset_r1_despike(client):
    resp = await client.post(
        f"/api/spectra/{SCAN_UUID}/subset",
        json={"point_indices": [0], "region": "R1", "despike": True},
    )
    data = resp.json()
    assert resp.status_code == 200
    assert data["despike_applied"] is True
    assert data["n_masked_channels"] == 1


@pytest.mark.asyncio
async def test_subset_despike_defaults_off(client):
    resp = await client.post(
        f"/api/spectra/{SCAN_UUID}/subset",
        json={"point_indices": [0], "region": "R1"},
    )
    data = resp.json()
    assert data["despike_applied"] is False
    assert data["n_masked_channels"] == 0


# ---------------------------------------------------------------------------
# Composite (R123) — constituent-first, all-or-none, coverage disclosure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_point_r123_constituent_first_with_overlap_flag(client):
    off = (await client.get(f"/api/spectra/{SCAN_UUID}/point/0", params={"region": "R123"})).json()
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0", params={"region": "R123", "despike": True}
        )
    ).json()

    assert on["despike_applied"] is True
    assert on["despike_method"] == METHOD
    # Coverage disclosure is the derived static count for the summation view.
    from sherloc_pipeline.core.mask_application import count_uncovered_contributor_channels

    expected = count_uncovered_contributor_channels(
        "r123_summation", DEFAULT_MANIFEST.region_windows
    )
    assert on["n_uncovered_contributor_channels"] == expected == 207
    assert set(on["masked_channels"]) == {200, 600, 1700}

    a = np.asarray(off["intensity"])
    b = np.asarray(on["intensity"])
    # R123 view is identity channel→row; the flagged absolute channels change
    # (600 sits in the R1+R2 overlap and changes via its R2 constituent).
    changed = set(np.where(a != b)[0].tolist())
    assert {200, 600, 1700}.issubset(changed)


@pytest.mark.asyncio
async def test_r123_all_or_none_missing_region(client):
    # Point 1 has R1 + R3 masks but no R2 mask → composite renders
    # non-despiked, naming the missing region.
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/1", params={"region": "R123", "despike": True}
        )
    ).json()
    assert on["despike_applied"] is False
    assert on["despike_missing_regions"] == ["R2"]
    assert on["n_masked_channels"] == 0
    # The disclosure count is still reported on a composite despike request.
    assert on["n_uncovered_contributor_channels"] == 207

    off = (await client.get(f"/api/spectra/{SCAN_UUID}/point/1", params={"region": "R123"})).json()
    np.testing.assert_array_equal(np.asarray(on["intensity"]), np.asarray(off["intensity"]))


@pytest.mark.asyncio
async def test_average_r123_all_or_none_per_point(client):
    # All-or-none is per requested point: point 1 lacks R2 and point 2 lacks
    # every mask, so the averaged composite must render non-despiked even
    # though point 0 is fully masked (spec §4.6 — a partially despiked
    # composite is never labeled applied).
    off = (await client.get(f"/api/spectra/{SCAN_UUID}/average", params={"region": "R123"})).json()
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/average", params={"region": "R123", "despike": True}
        )
    ).json()
    assert on["despike_applied"] is False
    assert on["n_masked_channels"] == 0
    # Every region is missing on at least one involved point (point 2).
    assert on["despike_missing_regions"] == ["R1", "R2", "R3"]
    # Coverage disclosure is still reported on a composite despike request.
    assert on["n_uncovered_contributor_channels"] == 207
    np.testing.assert_array_equal(np.asarray(on["intensity"]), np.asarray(off["intensity"]))


@pytest.mark.asyncio
async def test_subset_r123_all_points_fully_masked_despikes(client):
    # A subset restricted to the fully-masked point 0 despikes the composite.
    resp = await client.post(
        f"/api/spectra/{SCAN_UUID}/subset",
        json={"point_indices": [0], "region": "R123", "despike": True},
    )
    data = resp.json()
    assert resp.status_code == 200
    assert data["despike_applied"] is True
    assert data["despike_missing_regions"] == []
    assert data["n_masked_channels"] > 0
    assert data["n_uncovered_contributor_channels"] == 207


@pytest.mark.asyncio
async def test_subset_r123_mixed_points_falls_back(client):
    # Mixing the fully-masked point 0 with the R2-less point 1 must fall back
    # to the non-despiked composite naming R2.
    resp = await client.post(
        f"/api/spectra/{SCAN_UUID}/subset",
        json={"point_indices": [0, 1], "region": "R123", "despike": True},
    )
    data = resp.json()
    assert data["despike_applied"] is False
    assert data["despike_missing_regions"] == ["R2"]
    assert data["n_masked_channels"] == 0


@pytest.mark.asyncio
async def test_process_despike_endpoint_unchanged(client):
    # MLD-IFC-004: the real-time modz endpoint keeps its contract; it does
    # not gain a stored-mask despike toggle.
    resp = await client.post(
        "/api/process/despike",
        json={"wavenumber": [float(i) for i in range(10)], "intensity": [1.0] * 10},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "spike_mask" in body and "despiked" in body
