"""Native-axis route contracts using only invented scans and arrays."""

import zlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from sherloc_pipeline.database.connection import create_all_tables
from sherloc_pipeline.database.models import ScanORM, ScanPointORM, SolORM, SpectrumORM
from sherloc_pipeline.services import spectrum_axes
from sherloc_pipeline.web.routes import spectra

SCAN_ID = "synthetic-scan"
PRODUCT_ID = "ss__9999_0000000000_000rrs__synthetic"
WAVELENGTH = np.linspace(249, 359, 2148).astype(np.float32)
EXCITATION = 247.0


def packed(values):
    return zlib.compress(np.asarray(values, dtype=np.float32).tobytes())


@pytest.fixture
def axis_db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    create_all_tables(engine)
    with Session(engine) as session:
        session.add(SolORM(sol_number=9999, data_source="pds4"))
        session.flush()
        session.add(ScanORM(
            id=SCAN_ID, sol_number=9999, scan_name="synthetic", target="Invented",
            scan_id=PRODUCT_ID, sclk_start=0, n_points=2, n_channels=2148,
            laser_wavelength_nm=EXCITATION, data_source="pds4", target_type="mars_target",
        ))
        session.flush()
        for index in range(2):
            session.add(ScanPointORM(id=f"synthetic-point-{index}", scan_id=SCAN_ID, point_index=index))
            session.flush()
            for number, region in enumerate(("R1", "R2", "R3"), 1):
                session.add(SpectrumORM(
                    id=f"synthetic-{index}-{region}", scan_point_id=f"synthetic-point-{index}",
                    region=region, spectrum_type="laser_normalized", processing_level="normalized",
                    intensities=packed(np.full(2148, number + index * 2)),
                    wavelengths=packed(WAVELENGTH), wavelength_source="pds_embedded",
                ))
        session.commit()
    yield engine
    engine.dispose()


@pytest_asyncio.fixture
async def axis_client(axis_db):
    app = FastAPI()
    app.state.config = SimpleNamespace(preprocessing={})
    app.state.access_mode = "internal"

    @app.middleware("http")
    async def db_session(request, call_next):
        with Session(axis_db) as session:
            request.state.db = session
            return await call_next(request)

    app.include_router(spectra.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://synthetic") as client:
        yield client


async def request_trace(client, mode, region):
    if mode == "subset":
        return await client.post(f"/api/spectra/{SCAN_ID}/subset", json={
            "point_indices": [0, 1], "region": region, "averaging_method": "mean",
        })
    suffix = "point/0" if mode == "point" else "average"
    return await client.get(f"/api/spectra/{SCAN_ID}/{suffix}", params={
        "region": region, "averaging_method": "mean",
    })


@pytest.mark.asyncio
@pytest.mark.parametrize("region", ["R1", "R2", "R3", "R123"])
@pytest.mark.parametrize("mode", ["point", "average", "subset"])
async def test_native_axis_and_stitch_survive_routes(axis_client, mode, region):
    response = await request_trace(axis_client, mode, region)
    assert response.status_code == 200, response.text
    data = response.json()
    bounds = {"R1": (250, 282), "R2": (282, 337.8), "R3": (337.8, 357.4)}
    mask = np.ones(2148, dtype=bool) if region == "R123" else (
        (WAVELENGTH >= bounds[region][0]) & (WAVELENGTH <= bounds[region][1])
    )
    expected = np.full(2148, {"R1": 1, "R2": 2, "R3": 3, "R123": 1}[region], dtype=float)
    if region == "R123":
        expected[565:690] = 3
        expected[690:1668] = 2
        expected[1668:1690] = 5
        expected[1690:] = 3
    if mode != "point":
        expected += 1
        if region == "R123":
            expected[565:690] += 1
            expected[1668:1690] += 1
    np.testing.assert_array_equal(data["wavelength"], WAVELENGTH[mask])
    np.testing.assert_allclose(data["wavenumber"], 1e7 * (1 / EXCITATION - 1 / WAVELENGTH[mask].astype(float)))
    np.testing.assert_array_equal(data["intensity"], expected[mask])
    assert data["channel_indices"] == np.flatnonzero(mask).tolist()
    assert data["n_channels"] == len(data["intensity"]) == len(data["wavelength"])
    assert data["provenance"]["axis_source"] == "pds_embedded"
    assert data["provenance"]["calibration_version"] is None
    assert data["provenance"]["excitation_nm"] == EXCITATION
    assert len(data["provenance"]["axis_sha256"]) == 64


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_axis", [None, b"corrupt", packed([251, 252]), packed([251, np.nan]),
                                      packed([252, 251]), packed([0, 251]), packed([251, 251])])
async def test_invalid_pds_axis_is_explicit_error(axis_client, axis_db, bad_axis):
    with Session(axis_db) as session:
        row = session.get(SpectrumORM, "synthetic-0-R1")
        row.wavelengths = bad_axis
        if bad_axis is not None and bad_axis not in (b"corrupt", packed([251, 252])):
            row.intensities = packed([1, 2])
        session.commit()
    response = await request_trace(axis_client, "point", "R1")
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,changed", [("average", "synthetic-1-R1"), ("subset", "synthetic-1-R1"),
                                          ("point", "synthetic-0-R2")])
async def test_incompatible_contributors_are_rejected(axis_client, axis_db, mode, changed):
    with Session(axis_db) as session:
        session.get(SpectrumORM, changed).wavelengths = packed(WAVELENGTH + 1)
        session.commit()
    response = await request_trace(axis_client, mode, "R123" if mode == "point" else "R1")
    assert response.status_code == 422
    assert "Incompatible" in response.json()["detail"]


def test_rounded_duplicates_and_loupe_fallback(monkeypatch):
    scan = SimpleNamespace(data_source="pds4", laser_wavelength_nm=EXCITATION)
    row = SimpleNamespace(intensities=packed([1, 2, 3]), wavelengths=packed([251, 251, 252]),
                          wavelength_source="pds_embedded")
    axes = spectrum_axes.resolve_spectrum_axes(scan, [row], "R1")
    np.testing.assert_array_equal(axes.wavelength, [251, 251, 252])
    row.wavelengths = None
    row.wavelength_source = None
    scan.data_source = "loupe"
    calibrate = MagicMock(return_value=(np.array([251., 252., 253.]), np.array([11., 12., 13.])))
    monkeypatch.setattr(spectrum_axes, "calculate_loupe_wavelength_wavenumber", calibrate)
    axes = spectrum_axes.resolve_spectrum_axes(scan, [row], "R1")
    calibrate.assert_called_once_with(3)
    assert axes.source == "loupe_polynomial"
    assert axes.excitation_basis == "loupe_calibration_default"
    np.testing.assert_array_equal(axes.wavenumber, [11, 12, 13])


@pytest.mark.asyncio
async def test_ml_positions_follow_embedded_region_mask(axis_client, axis_db, monkeypatch):
    with Session(axis_db) as session:
        session.get(ScanORM, SCAN_ID).data_source = "loupe"
        for row in session.query(SpectrumORM).all():
            row.spectrum_type = "dark_subtracted"
        session.commit()
    monkeypatch.setattr(spectra, "_fetch_stored_masks", lambda session, ids: {
        sid: [SimpleNamespace(channel_indices=[24], method="synthetic", created_at=None)] for sid in ids
    })
    monkeypatch.setattr(spectra, "_select_stored_mask", lambda masks: masks[0] if masks else None)
    monkeypatch.setattr(spectra, "_despike_region_windows", lambda: {"R1": (0, 100)})
    response = await axis_client.get(f"/api/spectra/{SCAN_ID}/point/0", params={"despike_method": "ml"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["masked_channels"] == [24]
    assert data["masked_positions"] == [data["channel_indices"].index(24)]
    assert data["provenance"]["processing_chain"] == [{
        "step": "despike", "method": "synthetic", "scope": "per_contributor", "interpolation": "linear",
    }]


def test_static_badpix_callers_keep_regional_loupe_mapping(monkeypatch):
    monkeypatch.setattr(spectra, "calculate_loupe_wavelength_wavenumber", lambda **kwargs: (
        np.array([249., 251., 300.]), np.array([1., 2., 3.]),
    ))
    monkeypatch.setattr(spectra, "load_badpix_channels", lambda: [
        SimpleNamespace(channel=1, tier=1, source="both"),
        SimpleNamespace(channel=2, tier=2, source="both"),
    ])
    spectra._badpix_items_for_region.cache_clear()
    try:
        result = spectra.get_badpix_channels("R1")
        assert result.n_channels == 1
        assert [(item.channel, item.position) for item in result.badpix] == [(1, 0)]
        result = spectra.get_badpix_channels("R123")
        assert [(item.channel, item.position) for item in result.badpix] == [(1, 1), (2, 2)]
    finally:
        spectra._badpix_items_for_region.cache_clear()


@pytest.mark.parametrize("excitation", [None, 0, float("nan")])
def test_invalid_recorded_excitation_is_rejected(excitation):
    scan = SimpleNamespace(data_source="pds4", laser_wavelength_nm=excitation)
    row = SimpleNamespace(intensities=packed([1, 2]), wavelengths=packed([251, 252]),
                          wavelength_source="pds_embedded")
    with pytest.raises(ValueError, match="excitation"):
        spectrum_axes.resolve_spectrum_axes(scan, [row], "R1")


def test_only_documented_polynomial_segment_break_is_permitted():
    scan = SimpleNamespace(data_source="loupe", laser_wavelength_nm=EXCITATION)
    values = np.concatenate([np.linspace(250, 280, 501), [279, 281]])
    row = SimpleNamespace(intensities=packed(np.ones(503)), wavelengths=packed(values),
                          wavelength_source="loupe_polynomial")
    axes = spectrum_axes.resolve_spectrum_axes(scan, [row], "R1")
    np.testing.assert_array_equal(axes.wavelength, values.astype(np.float32))
    row.wavelength_source = "pds_embedded"
    with pytest.raises(ValueError, match="ordering"):
        spectrum_axes.resolve_spectrum_axes(scan, [row], "R1")


@pytest.mark.asyncio
async def test_defaulted_scan_excitation_discloses_nominal_basis(axis_client, axis_db):
    with Session(axis_db) as session:
        scan = ScanORM(
            id="synthetic-default-scan", sol_number=9999, scan_name="synthetic-default",
            target="Invented", scan_id="ss__9999_0000000000_000rrs__synthetic_default",
            sclk_start=0, n_points=1, n_channels=2, data_source="pds4", target_type="mars_target",
        )
        session.add(scan)
        session.flush()
        session.add(ScanPointORM(id="synthetic-default-point", scan_id=scan.id, point_index=0))
        session.flush()
        session.add(SpectrumORM(
            id="synthetic-default-spectrum", scan_point_id="synthetic-default-point", region="R1",
            spectrum_type="laser_normalized", processing_level="normalized",
            intensities=packed([1, 2]), wavelengths=packed([251, 252]), wavelength_source="pds_embedded",
        ))
        session.commit()
        chosen_excitation = scan.laser_wavelength_nm
    response = await axis_client.get("/api/spectra/synthetic-default-scan/point/0")
    assert response.status_code == 200
    data = response.json()
    assert data["provenance"]["excitation_nm"] == chosen_excitation
    assert data["provenance"]["excitation_basis"] == "scan_record_or_nominal_default"
    np.testing.assert_allclose(data["wavenumber"], 1e7 * (1 / chosen_excitation - 1 / np.array([251, 252])))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["point", "average", "subset"])
async def test_pds_provenance_is_backend_owned(axis_client, axis_db, mode):
    from sherloc_pipeline import __version__

    with Session(axis_db) as session:
        session.get(ScanORM, SCAN_ID).pds4_metadata = {"lidvid": PRODUCT_ID, "version": "9.0", "software_version": "synthetic-9"}
        session.commit()
    response = await request_trace(axis_client, mode, "R2")
    assert response.status_code == 200
    provenance = response.json()["provenance"]
    assert "loupe" not in str(provenance).lower()
    assert provenance["data_origin"] == "pds4"
    assert provenance["source"]["product_id"] == PRODUCT_ID
    assert provenance["source"]["product_version"] == "9.0"
    assert provenance["source"]["source_sha256"] is None
    assert provenance["source"]["scan_excitation_nm"] == EXCITATION
    assert provenance["source"]["scan_excitation_basis"] == "scan_record_or_nominal_default"
    assert provenance["source_processing"]["software_version"] == "synthetic-9"
    assert provenance["application"]["version"] == __version__
    assert provenance["application"]["commit"] is None
    assert provenance["intensity_unit"] is None
    assert provenance["wavelength_unit"] == "nm"
    assert provenance["selection"]["mode"] == mode
    expected_indices = [0] if mode == "point" else [0, 1]
    assert provenance["selection"]["contributor_indices"] == expected_indices
    assert [row["point_index"] for row in provenance["source_spectra"]] == expected_indices
    assert provenance["source_processing"]["spectrum_type"] == "laser_normalized"
    assert [step["step"] for step in provenance["processing_chain"]] == ([] if mode == "point" else ["aggregate"])
    assert len(provenance["source_digest"]) == 64


@pytest.mark.asyncio
async def test_digest_tracks_native_data_and_metadata(axis_client, axis_db):
    async def digest():
        response = await request_trace(axis_client, "average", "R1")
        assert response.status_code == 200
        return response.json()["provenance"]["source_digest"]

    original = await digest()
    assert await digest() == original
    with Session(axis_db) as session:
        # The changed channel is outside the served R1 window, but belongs to
        # the native input and must still change its reproducible identity.
        values = np.ones(2148)
        values[-1] = 17
        session.get(SpectrumORM, "synthetic-0-R1").intensities = packed(values)
        session.commit()
    changed_data = await digest()
    assert changed_data != original
    with Session(axis_db) as session:
        session.get(ScanORM, SCAN_ID).pds4_metadata = {"version": "9.1"}
        session.commit()
    assert await digest() != changed_data


@pytest.mark.asyncio
async def test_subset_discloses_missing_contributors(axis_client, axis_db):
    with Session(axis_db) as session:
        session.delete(session.get(SpectrumORM, "synthetic-1-R1"))
        session.commit()
    response = await request_trace(axis_client, "subset", "R1")
    assert response.status_code == 200
    provenance = response.json()["provenance"]
    assert provenance["selection"] == {
        "mode": "subset", "indices": [0, 1], "n_points": 1,
        "contributor_indices": [0], "omitted_indices": [1],
    }
    assert [row["spectrum_id"] for row in provenance["source_spectra"]] == ["synthetic-0-R1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("region", ["R1", "R123"])
async def test_subset_rejects_duplicate_indices(axis_client, region):
    response = await axis_client.post(f"/api/spectra/{SCAN_ID}/subset", json={
        "point_indices": [0, 0, 1], "region": region, "averaging_method": "mean",
    })
    assert response.status_code == 422
    assert "unique" in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["loupe", None])
async def test_loupe_and_unknown_origin_are_not_invented(axis_client, axis_db, origin):
    with Session(axis_db) as session:
        scan = session.get(ScanORM, SCAN_ID)
        scan.data_source = origin
        # Deliberately differ from the fixed legacy fallback's excitation.
        scan.laser_wavelength_nm = 245
        for row in session.query(SpectrumORM).all():
            row.spectrum_type = "dark_subtracted"
            row.wavelengths = None
            row.wavelength_source = None
        for point in session.query(ScanPointORM).all():
            point.photodiode_mean = point.point_index + 1
        session.commit()
    response = await request_trace(axis_client, "average", "R123")
    assert response.status_code == 200
    provenance = response.json()["provenance"]
    assert provenance["data_origin"] == origin
    assert provenance["source"]["scan_excitation_nm"] == 245
    assert provenance["excitation_basis"] == "loupe_calibration_default"
    assert provenance["excitation_nm"] != 245
    assert provenance["source"]["product_id"] is None
    assert [step["step"] for step in provenance["processing_chain"]] == [
        "r123_stitch", "laser_normalization", "aggregate",
    ]
    assert provenance["processing_chain"][1]["factors"] == {"0": 2, "1": 1}


@pytest.mark.asyncio
async def test_processing_chain_reports_only_applied_steps(axis_client, monkeypatch):
    from sherloc_pipeline.core import baseline

    monkeypatch.setattr(spectra, "_apply_modz_to_served_r1", lambda intensity, wn, params: (intensity - 1, 2))
    monkeypatch.setattr(baseline, "fit_baseline", lambda series, params: (series - 2, series * 0 + 2))
    response = await axis_client.get(f"/api/spectra/{SCAN_ID}/average", params={
        "despike_method": "modz", "baseline_corrected": True, "averaging_method": "mean",
    })
    assert response.status_code == 200
    data = response.json()
    chain = data["provenance"]["processing_chain"]
    assert [step["step"] for step in chain] == ["aggregate", "despike", "baseline"]
    assert chain[1]["method"] == "modz"
    assert chain[1]["params"] == data["despike_params_used"]
    assert chain[2]["method"] == "aspls"
    assert data["baseline_corrected"] is True

    def fail_baseline(*args):
        raise ValueError("Synthetic failure")

    monkeypatch.setattr(baseline, "fit_baseline", fail_baseline)
    response = await axis_client.get(f"/api/spectra/{SCAN_ID}/average", params={"baseline_corrected": True})
    assert response.status_code == 200
    assert response.json()["baseline_corrected"] is False
    assert [step["step"] for step in response.json()["provenance"]["processing_chain"]] == ["aggregate"]
