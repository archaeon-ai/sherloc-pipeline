"""Despike method-selector web tests (issue #6).

v4.2.x shipped a stored-mask ML despike on the three spectra read endpoints
behind a ``despike: bool`` toggle. Issue #6 generalizes that to a coarse
``despike_method`` selector (``none | ml | modz``) while keeping the legacy
bool working forever. These tests pin the additive, backward-compatible
contract:

- ``despike_method=ml`` is byte-identical to the legacy ``despike=true``;
- precedence (explicit ``despike_method`` wins over the legacy bool);
- the new live ``modz`` path: R1 applied + ``despike_method="modz"`` +
  ``despike_params_used`` populated; non-R1 served non-despiked with the
  un-covered region disclosed;
- invalid method → 422;
- scan-detail ``ml_mask_count`` present, correct, and 0-safe on a mask-less
  scan;
- the subset body field carries all of the above.

The serving host never runs ML inference on any path: ``ml`` reads stored
masks, ``modz`` runs the stdlib/scipy rolling-median despike — neither
imports ``ml_despike`` inference or onnxruntime.
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

SCAN_UUID = str(uuid.UUID("00000000-0000-0000-0000-0000000000b1"))
# A second scan with NO cosmic-ray masks, to pin ml_mask_count == 0 safety.
SCAN_NO_MASKS_UUID = str(uuid.UUID("00000000-0000-0000-0000-0000000000b2"))
SOL_NUMBER = 921
N_POINTS = 2
N_CHANNELS = 2148
METHOD = DEFAULT_MANIFEST.provenance_label
MODEL_SHA = DEFAULT_MANIFEST.sha256

# One R1 mask per point so ml ≡ legacy-bool parity has something to apply.
ML_MASKS = {0: {"R1": [205]}, 1: {"R1": [205]}}

# The absolute R1 channel into which the fixture injects an obvious spike so
# the live modz path has a real cosmic ray to remove. Chosen to sit well
# outside the modz laser (600-700) and sulfate (1014-1020) exclusion windows.
_SPIKE_CHANNEL = 230


def _r1_selection():
    wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels=N_CHANNELS)
    return np.where(get_region_wavelength_mask(wavelength, "R1"))[0]


def _spectrum_bytes(point_index: int, region: str, spike: bool) -> bytes:
    """Deterministic smooth+noise array; optionally inject one R1 spike."""
    seed = point_index * 10 + {"R1": 1, "R2": 2, "R3": 3}[region]
    rng = np.random.RandomState(seed)
    x = np.arange(N_CHANNELS, dtype=np.float64)
    spectrum = 500.0 + 80.0 * np.exp(-((x - 300) ** 2) / (2 * 40**2))
    spectrum += rng.normal(0, 7, size=N_CHANNELS)
    if spike and region == "R1":
        spectrum[_SPIKE_CHANNEL] += 6000.0
    return zlib.compress(spectrum.astype(np.float32).tobytes())


def _add_scan(session, scan_uuid: str, scan_name: str, with_masks: bool, spike: bool):
    scan = ScanORM(
        id=scan_uuid,
        sol_number=SOL_NUMBER,
        scan_name=scan_name,
        target="Amherst_Point",
        scan_id=f"0921_Amherst_Point_{scan_name}",
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
                scan_id=scan_uuid,
                point_index=i,
                x_pixel=100.0 + i,
                y_pixel=200.0 + i,
                # Constant photodiode → on-the-fly normalization is a no-op
                # scaling, so assertions read cleanly.
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
                    intensities=_spectrum_bytes(i, region, spike),
                )
            )
            session.flush()
            if with_masks:
                flagged = ML_MASKS[i].get(region)
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


@pytest.fixture()
def selector_engine() -> Engine:
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
        _add_scan(session, SCAN_UUID, "detail_1", with_masks=True, spike=True)
        _add_scan(session, SCAN_NO_MASKS_UUID, "detail_2", with_masks=False, spike=True)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    return engine


@pytest_asyncio.fixture()
async def client(selector_engine, monkeypatch):
    from sherloc_pipeline.web.app import create_app
    from sherloc_pipeline.web.auth import _reset_validator_for_tests

    monkeypatch.setenv("SHERLOC_AUTH_MODE", "dev")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "internal")
    monkeypatch.delenv("SHERLOC_AUTH0_DOMAIN", raising=False)
    _reset_validator_for_tests()

    app = create_app(engine=selector_engine, config=_FakeConfig())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    _reset_validator_for_tests()


# ---------------------------------------------------------------------------
# (a) despike_method=ml ≡ despike=true parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_point_ml_equals_legacy_bool(client):
    legacy = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0",
            params={"region": "R1", "despike": True},
        )
    ).json()
    explicit = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0",
            params={"region": "R1", "despike_method": "ml"},
        )
    ).json()

    assert legacy["despike_applied"] is True
    # The response despike_method preserves the PRECISE stored provenance
    # string (not the coarse request enum "ml").
    assert legacy["despike_method"] == METHOD
    assert explicit["despike_method"] == METHOD
    assert explicit["despike_applied"] is True
    assert explicit["masked_channels"] == legacy["masked_channels"] == [205]
    assert explicit["n_masked_channels"] == legacy["n_masked_channels"] == 1
    # ml path never populates despike_params_used.
    assert explicit.get("despike_params_used") is None
    np.testing.assert_array_equal(
        np.asarray(explicit["intensity"]), np.asarray(legacy["intensity"])
    )


@pytest.mark.asyncio
async def test_average_ml_equals_legacy_bool(client):
    legacy = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/average",
            params={"region": "R1", "despike": True},
        )
    ).json()
    explicit = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/average",
            params={"region": "R1", "despike_method": "ml"},
        )
    ).json()
    assert explicit["despike_applied"] is True
    assert explicit["despike_method"] == METHOD == legacy["despike_method"]
    assert explicit["n_masked_channels"] == legacy["n_masked_channels"]
    assert explicit.get("despike_params_used") is None
    np.testing.assert_array_equal(
        np.asarray(explicit["intensity"]), np.asarray(legacy["intensity"])
    )


# ---------------------------------------------------------------------------
# (b) precedence: explicit despike_method wins over legacy bool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_precedence_none_beats_legacy_true(client):
    # despike=true would normally mean ml; despike_method=none overrides it.
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0",
            params={"region": "R1", "despike": True, "despike_method": "none"},
        )
    ).json()
    off = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0", params={"region": "R1"}
        )
    ).json()
    assert on["despike_applied"] is False
    assert on["despike_method"] is None
    assert on["n_masked_channels"] == 0
    assert on.get("despike_params_used") is None
    np.testing.assert_array_equal(
        np.asarray(on["intensity"]), np.asarray(off["intensity"])
    )


@pytest.mark.asyncio
async def test_precedence_modz_beats_legacy_false(client):
    # despike defaults false (→none); explicit despike_method=modz overrides.
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0",
            params={"region": "R1", "despike_method": "modz"},
        )
    ).json()
    assert on["despike_applied"] is True
    assert on["despike_method"] == "modz"


# ---------------------------------------------------------------------------
# (c) modz R1: applied + provenance + params_used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_point_modz_r1_applied(client):
    off = (
        await client.get(f"/api/spectra/{SCAN_UUID}/point/0", params={"region": "R1"})
    ).json()
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0",
            params={"region": "R1", "despike_method": "modz"},
        )
    ).json()
    assert on["despike_applied"] is True
    assert on["despike_method"] == "modz"
    # The injected spike is removed → at least one channel replaced.
    assert on["n_masked_channels"] >= 1
    assert on["despike_missing_regions"] == []
    assert on["n_uncovered_contributor_channels"] == 0
    # The new additive field carries the config-default modz parameters.
    params = on["despike_params_used"]
    assert isinstance(params, dict)
    assert params["window_size"] == 7
    assert params["zscore_threshold"] == 6.0
    assert params["interpolation_method"] == "linear"

    # The served array actually changed at the injected-spike row.
    sel = _r1_selection()
    spike_row = int(np.where(sel == _SPIKE_CHANNEL)[0][0])
    a = np.asarray(off["intensity"])
    b = np.asarray(on["intensity"])
    assert a[spike_row] != b[spike_row]
    assert b[spike_row] < a[spike_row]  # spike interpolated down


@pytest.mark.asyncio
async def test_average_modz_r1_applied_with_params(client):
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/average",
            params={"region": "R1", "despike_method": "modz"},
        )
    ).json()
    assert on["despike_applied"] is True
    assert on["despike_method"] == "modz"
    assert on["n_masked_channels"] >= 1
    assert on["despike_missing_regions"] == []
    assert on["despike_params_used"]["window_size"] == 7


@pytest.mark.asyncio
async def test_modz_works_without_stored_masks(client):
    # modz is a live compute — it works on the mask-less scan too.
    on = (
        await client.get(
            f"/api/spectra/{SCAN_NO_MASKS_UUID}/point/0",
            params={"region": "R1", "despike_method": "modz"},
        )
    ).json()
    assert on["despike_applied"] is True
    assert on["despike_method"] == "modz"
    assert on["n_masked_channels"] >= 1


# ---------------------------------------------------------------------------
# (d) modz non-R1 → not applied + missing_regions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_point_modz_r2_not_applied(client):
    off = (
        await client.get(f"/api/spectra/{SCAN_UUID}/point/0", params={"region": "R2"})
    ).json()
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0",
            params={"region": "R2", "despike_method": "modz"},
        )
    ).json()
    assert on["despike_applied"] is False
    assert on["despike_method"] is None
    assert on["n_masked_channels"] == 0
    assert on["despike_missing_regions"] == ["R2"]
    assert on.get("despike_params_used") is None
    # Served array is unchanged (non-despiked).
    np.testing.assert_array_equal(
        np.asarray(on["intensity"]), np.asarray(off["intensity"])
    )


@pytest.mark.asyncio
async def test_point_modz_r123_not_applied_lists_all_contributors(client):
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0",
            params={"region": "R123", "despike_method": "modz"},
        )
    ).json()
    assert on["despike_applied"] is False
    assert on["despike_missing_regions"] == ["R1", "R2", "R3"]
    assert on["n_masked_channels"] == 0
    assert on.get("despike_params_used") is None


@pytest.mark.asyncio
async def test_average_modz_r123_not_applied(client):
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/average",
            params={"region": "R123", "despike_method": "modz"},
        )
    ).json()
    assert on["despike_applied"] is False
    assert on["despike_missing_regions"] == ["R1", "R2", "R3"]
    assert on["n_masked_channels"] == 0


# ---------------------------------------------------------------------------
# (e) invalid method → 422
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_method_point_422(client):
    resp = await client.get(
        f"/api/spectra/{SCAN_UUID}/point/0",
        params={"region": "R1", "despike_method": "bogus"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_invalid_method_average_422(client):
    resp = await client.get(
        f"/api/spectra/{SCAN_UUID}/average",
        params={"region": "R1", "despike_method": "MODZ"},  # case-sensitive
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_invalid_method_subset_422(client):
    resp = await client.post(
        f"/api/spectra/{SCAN_UUID}/subset",
        json={"point_indices": [0], "region": "R1", "despike_method": "nope"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# (f) ml_mask_count on scan detail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_detail_ml_mask_count_present_and_correct(client):
    resp = await client.get(f"/api/scans/{SCAN_UUID}")
    assert resp.status_code == 200
    scan = resp.json()["scan"]
    assert "ml_mask_count" in scan
    # One R1 mask per point × 2 points.
    assert scan["ml_mask_count"] == 2


@pytest.mark.asyncio
async def test_scan_detail_ml_mask_count_zero_safe(client):
    resp = await client.get(f"/api/scans/{SCAN_NO_MASKS_UUID}")
    assert resp.status_code == 200
    scan = resp.json()["scan"]
    assert scan["ml_mask_count"] == 0


# ---------------------------------------------------------------------------
# (g) subset body field
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subset_ml_equals_legacy_bool(client):
    legacy = (
        await client.post(
            f"/api/spectra/{SCAN_UUID}/subset",
            json={"point_indices": [0], "region": "R1", "despike": True},
        )
    ).json()
    explicit = (
        await client.post(
            f"/api/spectra/{SCAN_UUID}/subset",
            json={"point_indices": [0], "region": "R1", "despike_method": "ml"},
        )
    ).json()
    assert explicit["despike_applied"] is True
    assert explicit["despike_method"] == METHOD == legacy["despike_method"]
    assert explicit["n_masked_channels"] == legacy["n_masked_channels"] == 1
    assert explicit.get("despike_params_used") is None
    np.testing.assert_array_equal(
        np.asarray(explicit["intensity"]), np.asarray(legacy["intensity"])
    )


@pytest.mark.asyncio
async def test_subset_modz_r1_applied_with_params(client):
    resp = await client.post(
        f"/api/spectra/{SCAN_UUID}/subset",
        json={"point_indices": [0], "region": "R1", "despike_method": "modz"},
    )
    data = resp.json()
    assert resp.status_code == 200
    assert data["despike_applied"] is True
    assert data["despike_method"] == "modz"
    assert data["n_masked_channels"] >= 1
    assert data["despike_params_used"]["window_size"] == 7


@pytest.mark.asyncio
async def test_subset_modz_r2_not_applied(client):
    resp = await client.post(
        f"/api/spectra/{SCAN_UUID}/subset",
        json={"point_indices": [0], "region": "R2", "despike_method": "modz"},
    )
    data = resp.json()
    assert data["despike_applied"] is False
    assert data["despike_missing_regions"] == ["R2"]
    assert data["n_masked_channels"] == 0
    assert data.get("despike_params_used") is None


@pytest.mark.asyncio
async def test_subset_precedence_none_beats_legacy_true(client):
    resp = await client.post(
        f"/api/spectra/{SCAN_UUID}/subset",
        json={
            "point_indices": [0],
            "region": "R1",
            "despike": True,
            "despike_method": "none",
        },
    )
    data = resp.json()
    assert data["despike_applied"] is False
    assert data["n_masked_channels"] == 0


# ---------------------------------------------------------------------------
# schema_version stays 1.0.0 (additive fields only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_modz_keeps_schema_version_1_0_0(client):
    on = (
        await client.get(
            f"/api/spectra/{SCAN_UUID}/point/0",
            params={"region": "R1", "despike_method": "modz"},
        )
    ).json()
    assert on["schema_version"] == "1.0.0"
    assert "despike_params_used" in on
