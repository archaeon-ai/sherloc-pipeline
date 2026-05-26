"""Shared fixtures for web API tests."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from sherloc_pipeline.database.connection import create_all_tables
from sherloc_pipeline.database.models import (
    Base,
    ScanORM,
    ScanPointORM,
    SolORM,
    SpectrumORM,
)


# ---------------------------------------------------------------------------
# Minimal config stub that satisfies the web layer without loading config.yaml
# ---------------------------------------------------------------------------

@dataclass
class _WavelengthCal:
    raman_coefficients: List[float] = field(
        default_factory=lambda: [-7.85e-06, 6.524e-02, 2.4669e+02]
    )
    fluorescence_coefficients: List[float] = field(
        default_factory=lambda: [-5.65724e-06, 6.33627e-02, 2.47474e+02]
    )
    cutoff_channel: int = 500
    laser_wavelength: float = 248.5794
    n_channels: int = 2148


@dataclass
class _FakeConfig:
    wavelength: _WavelengthCal = field(default_factory=_WavelengthCal)
    fitting: Dict[str, Any] = field(default_factory=lambda: {
        "r1_fit_range": [700, 1200],
        "max_peaks": 5,
        "min_snr": 3.0,
        "fit_fwhm_min_initial_cm1": 22,
        "fwhm_max_cm1": 90,
        "slit_width_cm1_default": 34.1,
        "slit_pref_weight": 0.2,
        "low_fwhm_edge_penalty": 0.1,
        "peak_separation_cm1": 25,
        "r_squared_min": 0.25,
        "parsimony": {"model_selection": "ftest", "ftest_alpha": 0.01},
    })
    fluorescence_fitting: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "trim_mean_baseline_pct": 0.02,
        "baseline": {"lam": 1e6, "iters": 10},
    })
    spatial: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    database: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sample data constants
# ---------------------------------------------------------------------------

SCAN_UUID = str(uuid.UUID("00000000-0000-0000-0000-000000000001"))
POINT_UUID = str(uuid.UUID("00000000-0000-0000-0000-000000000010"))
SPECTRUM_UUID = str(uuid.UUID("00000000-0000-0000-0000-000000000100"))
SOL_NUMBER = 921
N_POINTS = 3
N_CHANNELS = 2148


def _make_spectrum_bytes() -> bytes:
    """Create a realistic 2148-channel dark-subtracted spectrum.

    Returns zlib-compressed float32 bytes matching the production DB format
    (see ``_extract_intensities`` in ``web/routes/spectra.py``).
    """
    import zlib

    rng = np.random.RandomState(42)
    # Smooth spectrum with a bump around channel 200 (mineral region in R1)
    x = np.arange(N_CHANNELS, dtype=np.float64)
    spectrum = 500.0 + 100.0 * np.exp(-((x - 200) ** 2) / (2 * 30**2))
    spectrum += rng.normal(0, 5, size=N_CHANNELS)
    return zlib.compress(spectrum.astype(np.float32).tobytes())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_engine() -> Engine:
    """In-memory SQLite engine with sample data.

    Uses StaticPool so ALL connections (including those created by the
    middleware via get_session_factory) share the same in-memory database.
    """
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

    from sherloc_pipeline.database.connection import get_session_factory

    factory = get_session_factory(engine)
    session = factory()
    try:
        # Sol
        sol = SolORM(sol_number=SOL_NUMBER, data_source="loupe")
        session.add(sol)
        session.flush()

        # Scan
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

        # Points + spectra
        for i in range(N_POINTS):
            pt_id = str(uuid.UUID(f"00000000-0000-0000-0000-{10 + i:012d}"))
            pt = ScanPointORM(
                id=pt_id,
                scan_id=SCAN_UUID,
                point_index=i,
                x_pixel=100.0 + i * 10,
                y_pixel=200.0 + i * 5,
                photodiode_mean=4800.0 + i,
                photodiode_std=12.0,
            )
            session.add(pt)
            session.flush()

            sp_id = str(uuid.UUID(f"00000000-0000-0000-0000-{100 + i:012d}"))
            sp = SpectrumORM(
                id=sp_id,
                scan_point_id=pt_id,
                region="R1",
                spectrum_type="dark_subtracted",
                processing_level="dark_subtracted",
                intensities=_make_spectrum_bytes(),
            )
            session.add(sp)

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return engine


@pytest.fixture()
def fake_config() -> _FakeConfig:
    return _FakeConfig()


@pytest_asyncio.fixture()
async def client(
    test_engine: Engine,
    fake_config: _FakeConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncClient:
    """Async HTTP client wired to the test app.

    Runs in ``SHERLOC_AUTH_MODE=dev`` so route-logic tests can exercise
    the data API without minting tokens. Auth-gating behaviour is covered
    by the dedicated test files (``test_scans_auth.py``,
    ``test_auth_routes.py``, ``test_auth0_validator.py``,
    ``test_auth_factory.py``). Per spec §13.5, dev mode skips token
    validation and returns synthetic ``sherloc:internal`` claims, so
    every protected route resolves with the operator-trust identity.
    """
    from sherloc_pipeline.web.app import create_app
    from sherloc_pipeline.web.auth import _reset_validator_for_tests

    monkeypatch.setenv("SHERLOC_AUTH_MODE", "dev")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "internal")
    monkeypatch.delenv("SHERLOC_AUTH0_DOMAIN", raising=False)
    monkeypatch.delenv("SHERLOC_CF_TEAM_DOMAIN", raising=False)
    monkeypatch.delenv("SHERLOC_CF_AUDIENCE", raising=False)
    _reset_validator_for_tests()

    app = create_app(engine=test_engine, config=fake_config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    _reset_validator_for_tests()
