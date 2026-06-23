"""``masked_positions`` served-index markers for ML despike (issue #8).

v4.3.0 (issue #6) shipped the ``none | ml | modz`` despike method selector. The
ML path returned only TEXTUAL provenance (``n_masked_channels``) plus the point
endpoint's ABSOLUTE ``masked_channels`` — neither lets the Workbench place the
red-triangle spike markers, which need positions INTO the served array.

Issue #8 adds ``masked_positions: int[]`` to all three spectra responses
(average, point, subset): indices into the served wavenumber/intensity arrays
where the stored-mask (``ml``) despike replaced channels. These tests pin:

- positions present + correct on ``ml`` (a known masked absolute channel maps to
  the right served index, verified by the wavenumber at that position);
- empty on ``none`` / ``modz`` (the Workbench computes its own modz mask);
- single-region (R1/R2/R3) + R123 composite + point + average + subset coverage;
- R123 positions equal the absolute channels (identity — the served array is the
  full 2148-channel stitch);
- additive only — ``masked_channels`` (absolute) is untouched and
  ``schema_version`` stays ``1.0.0``.
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

from sherloc_pipeline.core.calibration import (
    calculate_loupe_wavelength_wavenumber,
    get_region_wavelength_mask,
)
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

SCAN_UUID = str(uuid.UUID("00000000-0000-0000-0000-0000000000c1"))
SOL_NUMBER = 921
N_POINTS = 2
N_CHANNELS = 2148
METHOD = DEFAULT_MANIFEST.provenance_label
MODEL_SHA = DEFAULT_MANIFEST.sha256

# One masked absolute channel per detector region, each inside its certified
# manifest window (R1 [52,575), R2 [575,1677), R3 [1677,2140)). Same channels
# on both points so the average/subset union is deterministic.
MASKED_CHANNEL = {"R1": 205, "R2": 800, "R3": 1700}


def _selection(region: str) -> np.ndarray:
    """Absolute channel indices the given region's served array spans."""
    wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels=N_CHANNELS)
    if region == "R123":
        return np.arange(N_CHANNELS)
    return np.where(get_region_wavelength_mask(wavelength, region))[0]


def _served_position(region: str, abs_channel: int) -> int:
    """Served-array index of an absolute channel for the region (test oracle)."""
    sel = _selection(region)
    return int(np.where(sel == abs_channel)[0][0])


def _served_wavenumber(region: str) -> np.ndarray:
    """The served wavenumber axis for a region (what the endpoint returns)."""
    wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=N_CHANNELS)
    if region == "R123":
        return wavenumber
    return wavenumber[get_region_wavelength_mask(wavelength, region)]


def _spectrum_bytes(point_index: int, region: str) -> bytes:
    seed = point_index * 10 + {"R1": 1, "R2": 2, "R3": 3}[region]
    rng = np.random.RandomState(seed)
    x = np.arange(N_CHANNELS, dtype=np.float64)
    spectrum = 500.0 + 80.0 * np.exp(-((x - 300) ** 2) / (2 * 40**2))
    spectrum += rng.normal(0, 7, size=N_CHANNELS)
    return zlib.compress(spectrum.astype(np.float32).tobytes())


@pytest.fixture()
def positions_engine() -> Engine:
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
            pt_id = str(uuid.uuid4())
            session.add(
                ScanPointORM(
                    id=pt_id,
                    scan_id=SCAN_UUID,
                    point_index=i,
                    x_pixel=100.0 + i,
                    y_pixel=200.0 + i,
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
                session.add(
                    CosmicRayMaskORM(
                        id=str(uuid.uuid4()),
                        spectrum_id=sp_id,
                        method=METHOD,
                        model_sha256=MODEL_SHA,
                        tau=float(DEFAULT_MANIFEST.tau[region]),
                        channel_indices=[MASKED_CHANNEL[region]],
                        n_flagged=1,
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
async def client(positions_engine, monkeypatch):
    from sherloc_pipeline.web.app import create_app
    from sherloc_pipeline.web.auth import _reset_validator_for_tests

    monkeypatch.setenv("SHERLOC_AUTH_MODE", "dev")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "internal")
    monkeypatch.delenv("SHERLOC_AUTH0_DOMAIN", raising=False)
    _reset_validator_for_tests()

    app = create_app(engine=positions_engine, config=_FakeConfig())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    _reset_validator_for_tests()


# ---------------------------------------------------------------------------
# (a) ml positions present + correct (the masked channel lands at the served
#     index whose wavenumber matches the absolute channel's wavenumber)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("region", ["R1", "R2", "R3"])
@pytest.mark.asyncio
async def test_point_ml_positions_single_region(client, region):
    data = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0",
            params={"region": region, "despike_method": "ml"},
        )
    ).json()
    assert data["despike_applied"] is True
    abs_ch = MASKED_CHANNEL[region]
    expected_pos = _served_position(region, abs_ch)
    # masked_channels stays ABSOLUTE; masked_positions is the served index.
    assert data["masked_channels"] == [abs_ch]
    assert data["masked_positions"] == [expected_pos]
    # The two coordinate systems genuinely differ on the offset regions.
    if region != "R123":
        assert expected_pos != abs_ch
    # The served wavenumber at that position equals the absolute channel's
    # wavenumber — i.e. the marker would sit on the right spot.
    served_wn = np.asarray(data["wavenumber"])
    full_wn = _served_wavenumber(region)
    np.testing.assert_allclose(
        served_wn[expected_pos], full_wn[expected_pos], rtol=0, atol=1e-9
    )


@pytest.mark.asyncio
async def test_average_ml_positions_union(client):
    data = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/average",
            params={"region": "R1", "despike_method": "ml"},
        )
    ).json()
    assert data["despike_applied"] is True
    # Both points masked the same R1 channel → union is the one served position.
    expected_pos = _served_position("R1", MASKED_CHANNEL["R1"])
    assert data["masked_positions"] == [expected_pos]
    assert data["n_masked_channels"] == 1
    # average response has no absolute masked_channels field (point-only).
    assert "masked_channels" not in data


@pytest.mark.asyncio
async def test_subset_ml_positions(client):
    data = (
        await client.post(
            f"/api/spectra/{SCAN_UUID}/subset",
            json={"point_indices": [0, 1], "region": "R1", "despike_method": "ml"},
        )
    ).json()
    assert data["despike_applied"] is True
    expected_pos = _served_position("R1", MASKED_CHANNEL["R1"])
    assert data["masked_positions"] == [expected_pos]
    assert "masked_channels" not in data


# ---------------------------------------------------------------------------
# (b) R123 composite — positions equal absolute channels (full-stitch identity)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_point_ml_positions_r123_identity(client):
    data = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0",
            params={"region": "R123", "despike_method": "ml"},
        )
    ).json()
    assert data["despike_applied"] is True
    # All three regions contribute one masked channel each; the served R123
    # array is the full 2148-stitch, so positions == absolute channels.
    expected = sorted(MASKED_CHANNEL.values())
    assert data["masked_channels"] == expected
    assert data["masked_positions"] == expected
    # Served axis is the full length, so each position is a valid index.
    assert len(data["wavenumber"]) == N_CHANNELS


@pytest.mark.asyncio
async def test_average_ml_positions_r123_identity(client):
    data = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/average",
            params={"region": "R123", "despike_method": "ml"},
        )
    ).json()
    assert data["despike_applied"] is True
    assert data["masked_positions"] == sorted(MASKED_CHANNEL.values())


# ---------------------------------------------------------------------------
# (c) empty on none / modz
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_point_none_positions_empty(client):
    data = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0",
            params={"region": "R1", "despike_method": "none"},
        )
    ).json()
    assert data["despike_applied"] is False
    assert data["masked_positions"] == []
    assert data["masked_channels"] == []


@pytest.mark.asyncio
async def test_point_modz_positions_empty(client):
    # modz applies (R1) and reports a count, but masked_positions stays empty:
    # the Workbench builds the modz marker mask client-side, not from this.
    data = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0",
            params={"region": "R1", "despike_method": "modz"},
        )
    ).json()
    assert data["despike_method"] == "modz"
    assert data["masked_positions"] == []


@pytest.mark.asyncio
async def test_average_modz_positions_empty(client):
    data = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/average",
            params={"region": "R1", "despike_method": "modz"},
        )
    ).json()
    assert data["despike_method"] == "modz"
    assert data["masked_positions"] == []


@pytest.mark.asyncio
async def test_subset_none_positions_empty(client):
    data = (
        await client.post(
            f"/api/spectra/{SCAN_UUID}/subset",
            json={"point_indices": [0], "region": "R1", "despike_method": "none"},
        )
    ).json()
    assert data["masked_positions"] == []


# ---------------------------------------------------------------------------
# (d) additive — schema_version unchanged, field always present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_masked_positions_keeps_schema_version_1_0_0(client):
    data = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0",
            params={"region": "R1", "despike_method": "ml"},
        )
    ).json()
    assert data["schema_version"] == "1.0.0"
    assert "masked_positions" in data
