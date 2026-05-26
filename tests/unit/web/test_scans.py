"""Scan route tests."""

from datetime import datetime, timezone

import pytest

from tests.unit.web.conftest import SCAN_UUID, SOL_NUMBER


@pytest.mark.asyncio
async def test_list_scans(client):
    resp = await client.get("/api/scans")
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert data["total"] == 1
    assert data["offset"] == 0
    assert data["limit"] == 50
    assert len(data["scans"]) == 1
    scan = data["scans"][0]
    assert scan["id"] == SCAN_UUID
    assert scan["sol_number"] == SOL_NUMBER
    assert scan["target"] == "Amherst_Point"
    assert scan["scan_name"] == "detail_1"
    assert scan["n_points"] == 3
    assert scan["scan_class"] == "primary"
    assert scan["scan_type"] == "detail"
    assert scan["target_type"] == "mars_target"


@pytest.mark.asyncio
async def test_list_scans_filter_sol(client):
    resp = await client.get("/api/scans", params={"sol": 999})
    data = resp.json()
    assert data["total"] == 0
    assert len(data["scans"]) == 0


@pytest.mark.asyncio
async def test_list_scans_filter_target(client):
    resp = await client.get("/api/scans", params={"target": "Amherst Point"})
    data = resp.json()
    assert data["total"] == 1


@pytest.mark.asyncio
async def test_list_scans_filter_scan_class(client):
    resp = await client.get("/api/scans", params={"scan_class": "primary"})
    data = resp.json()
    assert data["total"] == 1


@pytest.mark.asyncio
async def test_list_scans_invalid_scan_class(client):
    resp = await client.get("/api/scans", params={"scan_class": "invalid"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_list_scans_pagination(client):
    resp = await client.get("/api/scans", params={"offset": 0, "limit": 1})
    data = resp.json()
    assert data["limit"] == 1
    assert len(data["scans"]) == 1


@pytest.mark.asyncio
async def test_get_scan_detail(client):
    resp = await client.get(f"/api/scans/{SCAN_UUID}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    scan = data["scan"]
    assert scan["id"] == SCAN_UUID
    assert scan["n_channels"] == 2148
    assert scan["shots_per_point"] == 50
    assert scan["laser_wavelength_nm"] == 248.5794


@pytest.mark.asyncio
async def test_get_scan_not_found(client):
    resp = await client.get("/api/scans/00000000-0000-0000-0000-000000000099")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_scan_points(client):
    resp = await client.get(f"/api/scans/{SCAN_UUID}/points")
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert data["scan_id"] == SCAN_UUID
    assert data["n_points"] == 3
    assert len(data["points"]) == 3
    p0 = data["points"][0]
    assert p0["point_index"] == 0
    assert p0["x_pixel"] == 100.0
    assert p0["photodiode_mean"] == 4800.0


@pytest.mark.asyncio
async def test_get_scan_points_not_found(client):
    resp = await client.get("/api/scans/00000000-0000-0000-0000-000000000099/points")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_scan_points_aci_pixel_from_cache(test_engine, client):
    """Issue #15: Workbench overlay needs x_aci_pixel populated.

    Map mode's coordinate resolver (core.coordinates.resolve_display_coordinates)
    caches resolved ACI pixels in map_display_coordinates. The scan-points
    endpoint must surface those cached values so the Workbench AciViewer can
    render the overlay (same source as Map mode, no FS dependency on Loupe
    workspace files).
    """
    from sherloc_pipeline.database.connection import get_session_factory
    from sherloc_pipeline.database.models import MapDisplayCoordinateORM, ScanPointORM

    factory = get_session_factory(test_engine)
    session = factory()
    try:
        point_ids = [
            pt.id
            for pt in session.query(ScanPointORM)
            .filter(ScanPointORM.scan_id == SCAN_UUID)
            .order_by(ScanPointORM.point_index)
            .all()
        ]
        # Seed map_display_coordinates with synthetic ACI-pixel values
        # distinct from x_pixel so the test can prove the route reads from
        # the cache rather than echoing x_pixel/y_pixel.
        now = datetime.now(timezone.utc)
        for idx, pt_id in enumerate(point_ids):
            session.add(
                MapDisplayCoordinateORM(
                    scan_point_id=pt_id,
                    aci_x=500.0 + idx * 25,
                    aci_y=800.0 + idx * 7,
                    transform_method="scanner_calibration",
                    computed_at=now,
                )
            )
        session.commit()
    finally:
        session.close()

    resp = await client.get(f"/api/scans/{SCAN_UUID}/points")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_points"] == 3

    for idx, point in enumerate(sorted(data["points"], key=lambda p: p["point_index"])):
        assert point["x_aci_pixel"] == pytest.approx(500.0 + idx * 25)
        assert point["y_aci_pixel"] == pytest.approx(800.0 + idx * 7)


@pytest.mark.asyncio
async def test_get_scan_points_aci_pixel_graceful_when_unresolvable(client):
    """When the resolver raises (e.g. fixture Loupe points have no
    coordinate_frame and no cache), the endpoint must still return 200 with
    x_pixel/y_pixel intact and x_aci_pixel left null — the overlay simply
    will not render for that scan rather than the entire request failing.
    """
    resp = await client.get(f"/api/scans/{SCAN_UUID}/points")
    assert resp.status_code == 200
    data = resp.json()
    p0 = data["points"][0]
    assert p0["x_pixel"] == 100.0
    assert p0["x_aci_pixel"] is None
    assert p0["y_aci_pixel"] is None


# ---------------------------------------------------------------------------
# Workbench Colorized-button gating: scan detail surfaces R2 variant existence
# so the SPA can disable the button when no colorized sibling is present
# (avoids the prior silent-fallback UX where clicking returned grayscale).
# ---------------------------------------------------------------------------


def _seed_aci_context_image(test_engine, file_path: str) -> None:
    """Add one ACI context_images row for SCAN_UUID with the given file_path.

    The colorized-variant probe in `get_scan` only fires when an ACI row
    exists, so tests that exercise the gating need to seed one. Uses the
    same StaticPool engine the client fixture wraps, so the inserted row
    is visible inside the request handler.
    """
    import uuid as _uuid

    from sherloc_pipeline.database.connection import get_session_factory
    from sherloc_pipeline.database.models import ContextImageORM

    factory = get_session_factory(test_engine)
    session = factory()
    try:
        session.add(
            ContextImageORM(
                id=str(_uuid.UUID("00000000-0000-0000-0000-000000000200")),
                scan_id=SCAN_UUID,
                image_type="ACI",
                file_path=file_path,
            )
        )
        session.commit()
    finally:
        session.close()


@pytest.mark.asyncio
async def test_get_scan_colorized_aci_available_default_false(client):
    """No ACI context_images row → field defaults to False (the
    Workbench will hide / disable the Colorized button).
    """
    resp = await client.get(f"/api/scans/{SCAN_UUID}")
    assert resp.status_code == 200
    assert resp.json()["scan"]["colorized_aci_available"] is False


@pytest.mark.asyncio
async def test_get_scan_colorized_aci_available_true_when_variant_exists(
    test_engine, client, monkeypatch
):
    """ACI row present + R2 reports a sol_NNNN_colorized/ sibling → True."""
    _seed_aci_context_image(
        test_engine,
        "/work/sherloc/PHASE-data/loupe/sol_0921/Amherst_Point_aci.png",
    )
    # Stub the R2 HEAD probe to assert the route consults it and to
    # avoid network. Patch in the routes module where it's imported.
    monkeypatch.setattr(
        "sherloc_pipeline.web.routes.scans.colorized_variant_exists",
        lambda _path: True,
    )

    resp = await client.get(f"/api/scans/{SCAN_UUID}")
    assert resp.status_code == 200
    assert resp.json()["scan"]["colorized_aci_available"] is True


@pytest.mark.asyncio
async def test_get_scan_colorized_aci_available_false_when_variant_missing(
    test_engine, client, monkeypatch
):
    """ACI row present but no R2 sibling → False (the sparse-coverage
    case that motivated this fix — 170 of 205 historical sols hit it).
    """
    _seed_aci_context_image(
        test_engine,
        "/work/sherloc/PHASE-data/loupe/sol_0614/Uganik_Island_aci.png",
    )
    monkeypatch.setattr(
        "sherloc_pipeline.web.routes.scans.colorized_variant_exists",
        lambda _path: False,
    )

    resp = await client.get(f"/api/scans/{SCAN_UUID}")
    assert resp.status_code == 200
    assert resp.json()["scan"]["colorized_aci_available"] is False


@pytest.mark.asyncio
async def test_get_scan_colorized_probe_targets_base_aci_not_angle_range(
    test_engine, client, monkeypatch
):
    """PR #31 Codex Round 1 F1: with multiple ACI rows where the first
    is an angle-range variant (``_145-185``), the probe must target the
    BASE row's file_path — the same row that GET /api/images/{id}/aci
    actually serves. Earlier ``.first()``-based code path would have
    probed the angle-range row and answered with stale data, making
    the Workbench button state disagree with the served image bytes.
    """
    import uuid as _uuid

    from sherloc_pipeline.database.connection import get_session_factory
    from sherloc_pipeline.database.models import ContextImageORM

    factory = get_session_factory(test_engine)
    session = factory()
    try:
        # Insert the angle-range variant FIRST so a naive ``.first()``
        # query would pick the wrong row.
        session.add(
            ContextImageORM(
                id=str(_uuid.UUID("00000000-0000-0000-0000-000000000301")),
                scan_id=SCAN_UUID,
                image_type="ACI",
                file_path=(
                    "/work/sherloc/PHASE-data/loupe/sol_0921/"
                    "Amherst_Point_aci_145-185.png"
                ),
            )
        )
        session.add(
            ContextImageORM(
                id=str(_uuid.UUID("00000000-0000-0000-0000-000000000302")),
                scan_id=SCAN_UUID,
                image_type="ACI",
                file_path=(
                    "/work/sherloc/PHASE-data/loupe/sol_0921/"
                    "Amherst_Point_aci.png"
                ),
            )
        )
        session.commit()
    finally:
        session.close()

    # Capture which file_path the probe actually receives so we can
    # assert against the route's selection, not just the boolean
    # result. A naive ``.first()`` would invoke us with the
    # ``_145-185`` filename; the helper must invoke us with the base.
    seen_paths: list[str] = []

    def _capture(path: str) -> bool:
        seen_paths.append(path)
        return True

    monkeypatch.setattr(
        "sherloc_pipeline.web.routes.scans.colorized_variant_exists",
        _capture,
    )

    resp = await client.get(f"/api/scans/{SCAN_UUID}")
    assert resp.status_code == 200
    assert resp.json()["scan"]["colorized_aci_available"] is True
    assert len(seen_paths) == 1, f"expected single probe, got {seen_paths!r}"
    # The probe MUST have seen the base file_path (no angle-range
    # suffix), matching what GET /api/images/{id}/aci would serve.
    assert seen_paths[0].endswith("Amherst_Point_aci.png")
    assert "_145-185" not in seen_paths[0]


@pytest.mark.asyncio
async def test_get_scan_colorized_probe_skipped_when_access_denied(
    test_engine, monkeypatch
):
    """PR #31 Codex Round 1 F2: in public mode, a Loupe scan-detail
    request must reject via validate_scan_access() BEFORE the
    opportunistic R2 colorized probe fires. Stubs the probe to raise
    if called — the test passes only when the access check short-
    circuits ahead of it.
    """
    from httpx import ASGITransport, AsyncClient

    # Seed an ACI row so a buggy ordering would actually attempt the
    # probe; without it, select_served_aci returns None and the probe
    # is skipped for an unrelated reason.
    _seed_aci_context_image(
        test_engine,
        "/work/sherloc/PHASE-data/loupe/sol_0921/Amherst_Point_aci.png",
    )

    def _fail_if_called(_path):  # pragma: no cover — must not run
        raise AssertionError(
            "colorized_variant_exists must NOT fire for an "
            "access-denied scan"
        )

    monkeypatch.setattr(
        "sherloc_pipeline.web.routes.scans.colorized_variant_exists",
        _fail_if_called,
    )

    # Build an app in public mode against the fixture engine (whose
    # SCAN_UUID has data_source="loupe", so it MUST 403).
    from sherloc_pipeline.web.app import create_app
    from sherloc_pipeline.web.auth import _reset_validator_for_tests

    monkeypatch.setenv("SHERLOC_AUTH_MODE", "dev")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "public")
    monkeypatch.delenv("SHERLOC_AUTH0_DOMAIN", raising=False)
    monkeypatch.delenv("SHERLOC_CF_TEAM_DOMAIN", raising=False)
    monkeypatch.delenv("SHERLOC_CF_AUDIENCE", raising=False)
    _reset_validator_for_tests()

    from tests.unit.web.conftest import _FakeConfig

    app = create_app(
        engine=test_engine, config=_FakeConfig(), access_mode="public"
    )
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(f"/api/scans/{SCAN_UUID}")
    finally:
        _reset_validator_for_tests()

    assert resp.status_code == 403
