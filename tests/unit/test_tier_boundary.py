"""Tier/config boundary using only invented identities and local synthetic records."""

import copy
import json
import os
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from sherloc_pipeline.database.models import Base, ScanORM, SolORM
from sherloc_pipeline.services.public_release import PublicReleaseIndex, backfill_public_release
from sherloc_pipeline.web import auth, config_check
from sherloc_pipeline.web.app import create_app
from sherloc_pipeline.web.data_access import DataAccessService
from sherloc_pipeline.web.feature_policy import load_feature_policy

PRODUCT = "ss__9999_0000000000_000rrs__synthetic"
DIGEST = "a" * 64


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("SHERLOC_", "AWS_")) or key == "PHASE_TIER":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SHERLOC_AUTH_MODE", "dev")
    monkeypatch.setenv("SHERLOC_DB", ":memory:")
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("network forbidden")))


def write_index(tmp_path, released=True):
    path = tmp_path / "release.json"
    path.write_text(json.dumps({
        "schema_version": "public-release-1", "release_id": "synthetic-release-0001",
        "products": [{"product_id": PRODUCT, "version_id": "synthetic-v1",
                      "sha256": DIGEST, "released": released}],
    }))
    return path


def make_scan(scan_id="synthetic-scan", data_source="pds4", metadata=None):
    return ScanORM(
        id=scan_id, sol_number=9999, scan_name=scan_id, target="Synthetic",
        scan_id=PRODUCT, sclk_start=0, n_points=0, n_channels=7,
        laser_wavelength_nm=211.0, data_source=data_source,
        target_type="mars_target", pds4_metadata=metadata,
    )


@pytest.fixture
def engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(SolORM(sol_number=9999, data_source="pds4"))
        session.commit()
    yield engine
    engine.dispose()


def fake_config():
    return SimpleNamespace(
        fitting={}, fluorescence_fitting={}, preprocessing={},
        wavelength=SimpleNamespace(laser_wavelength=211.0, cutoff_channel=2, n_channels=7,
                                   raman_coefficients=[1.0], fluorescence_coefficients=[2.0]),
    )


@pytest.fixture
def app(engine):
    app = create_app(engine=engine, config=fake_config(), access_mode="public")
    yield app
    app.state.job_queue.shutdown()
    app.state.map_executor.shutdown(wait=False)


def approved_policy():
    return {"references": {"mode": "allowlist", "release_id": DIGEST, "subset_id": DIGEST,
                           "serving_policy_version": DIGEST, "allowed_trace_ids": [DIGEST]}}


@pytest.mark.parametrize("policy", [
    {"referneces": {}}, {"references": {"mod": "off"}},
    {"identification": {"enabled": "false"}}, {"identification": {"enabled": 1}},
    {"identification": {"enabled": True}},
    {"identification": {"experimental_candidates": True}},
    {"identification": {"max_concurrency": 0}},
    {"references": {"mode": "on"}}, {"references": {"mode": "all_approved"}},
])
def test_invalid_policy_fails_both_startup_paths(monkeypatch, policy):
    monkeypatch.setenv("SHERLOC_FEATURE_POLICY", json.dumps(policy))
    assert any("feature policy" in error for error in config_check.validate())
    with pytest.raises(ValueError):
        create_app(engine=Mock(), config=Mock())


def test_public_cannot_enable_references(monkeypatch):
    monkeypatch.setenv("SHERLOC_FEATURE_POLICY", json.dumps(approved_policy()))
    assert load_feature_policy("internal").references.mode == "allowlist"
    with pytest.raises(ValueError, match="public references.mode"):
        create_app(engine=Mock(), config=Mock(), access_mode="public")


def test_method_must_bind_exact_release_and_subset(monkeypatch):
    policy = approved_policy()
    policy["identification"] = {"enabled": True, "allowed_method_releases": [{
        "method_release_id": DIGEST, "reference_release_id": DIGEST,
        "reference_subset_id": "b" * 64,
    }]}
    monkeypatch.setenv("SHERLOC_FEATURE_POLICY", json.dumps(policy))
    with pytest.raises(ValueError, match="matching"):
        load_feature_policy("internal")
    policy["identification"]["allowed_method_releases"][0]["reference_subset_id"] = DIGEST
    monkeypatch.setenv("SHERLOC_FEATURE_POLICY", json.dumps(policy))
    assert load_feature_policy("internal").identification.enabled


def test_unknown_feature_env_fails_startup(monkeypatch):
    monkeypatch.setenv("SHERLOC_FEATURE_REFERNCES", "off")
    with pytest.raises(ValueError, match="unknown feature setting"):
        load_feature_policy("internal")


def test_defaults_off():
    policy = load_feature_policy("public")
    assert policy.references.mode == "off"
    assert not policy.identification.enabled
    assert not policy.identification.experimental_candidates


@pytest.mark.parametrize("mutation", [
    {}, {"eligible": False}, {"eligible": "true"}, {"eligible": 1},
    {"schema_version": "unknown"}, {"product_id": "synthetic-other"},
    {"version_id": "synthetic-v2"}, {"source_sha256": "x" * 64},
    {"index_sha256": None}, {"release_id": 123}, {"version_id": 123},
])
def test_list_and_direct_agree(engine, tmp_path, mutation):
    meta = PublicReleaseIndex(write_index(tmp_path)).apply(PRODUCT, {"version": "synthetic-v1"})
    meta["public_release"].update(mutation)
    scan = make_scan(metadata=meta)
    gate = DataAccessService("public")
    with Session(engine) as session:
        session.add(scan)
        session.commit()
        listed = gate.filter_scans_query(session.query(ScanORM)).all()
        if listed:
            gate.validate_scan_access(scan)
        else:
            with pytest.raises(HTTPException):
                gate.validate_scan_access(scan)
        assert bool(listed) == (mutation == {})


def test_ingestion_index_and_backfill_fail_closed(engine, tmp_path):
    index = PublicReleaseIndex(write_index(tmp_path))
    with Session(engine) as session:
        scan = make_scan(metadata={"version": "synthetic-v1", "untouched": "synthetic"})
        session.add(scan)
        session.flush()
        assert backfill_public_release(session, index) == 1
        DataAccessService("public").validate_scan_access(scan)
        assert scan.pds4_metadata["untouched"] == "synthetic"
        backfill_public_release(session, PublicReleaseIndex(write_index(tmp_path, released=False)))
        with pytest.raises(HTTPException):
            DataAccessService("public").validate_scan_access(scan)
        assert index.apply(PRODUCT, {"version": "unknown"}) == {"version": "unknown"}


@pytest.mark.parametrize("source", [None, "unknown", "loupe", "pds4"])
@pytest.mark.asyncio
async def test_every_direct_surface_denies_unknown_release(app, engine, source):
    with Session(engine) as session:
        session.add(make_scan(data_source=source, metadata={"version": "synthetic-v1"}))
        session.commit()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://synthetic") as client:
        listed = await client.get("/api/scans")
        assert listed.status_code == 200
        assert listed.json()["total"] == 0
        for path in [
            "/api/scans/synthetic-scan", "/api/scans/synthetic-scan/points",
            "/api/spectra/synthetic-scan/average", "/api/spectra/synthetic-scan/point/0",
            "/api/plots/synthetic-scan", "/api/plots/synthetic-scan?plot_type=fit_overlay",
            "/api/images/synthetic-scan/aci", "/api/map/layers/synthetic-scan",
            "/api/map/data/synthetic-scan?domain=minerals",
        ]:
            response = await client.get(path)
            assert response.status_code == 403, (path, response.text)
        assert (await client.post("/api/spectra/synthetic-scan/subset", json={"point_indices": [0]})).status_code == 403
        assert (await client.post("/api/map/fit", json={"scan_id": "synthetic-scan", "domains": ["minerals"]})).status_code == 403


@pytest.mark.asyncio
async def test_released_list_detail_and_config(app, engine, tmp_path):
    meta = PublicReleaseIndex(write_index(tmp_path)).apply(PRODUCT, {"version": "synthetic-v1"})
    with Session(engine) as session:
        session.add(make_scan(metadata=meta))
        session.commit()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://synthetic") as client:
        assert (await client.get("/api/scans")).json()["total"] == 1
        assert (await client.get("/api/scans/synthetic-scan")).status_code == 200
        assert (await client.get("/api/plots/synthetic-scan")).status_code == 422
        bootstrap = (await client.get("/api/config/bootstrap")).json()
        assert set(bootstrap) == {"schema_version", "auth"}
        config = await client.get("/api/config")
        assert config.status_code == 200
        assert config.json()["tier"] == "public"
        assert not any(config.json()["capabilities"].values())
        assert config.headers["cache-control"] == "private, no-store"


@pytest.mark.parametrize("case,status", [("missing", 401), ("expired", 401), ("wrong-tier", 403), ("team", 200)])
@pytest.mark.asyncio
async def test_authenticated_config_uses_effective_tier(engine, monkeypatch, case, status):
    app = create_app(engine=engine, config=fake_config(), access_mode="internal")
    # Deliberately disagree with the effective app setting: the app must win.
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "public")
    monkeypatch.setenv("SHERLOC_AUTH_MODE", "auth0")
    validator = Mock()
    if case == "expired":
        validator.validate.side_effect = auth.AuthError("synthetic-expired")
    else:
        validator.validate.return_value = SimpleNamespace(sub="synthetic-user", roles=["phase:team-member"] if case == "team" else [])
    monkeypatch.setattr(auth, "get_validator", lambda: validator)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://synthetic") as client:
            headers = {} if case == "missing" else {"Authorization": "Bearer synthetic-token"}
            assert (await client.get("/api/config", headers=headers)).status_code == status
            assert (await client.get("/api/config/bootstrap")).status_code == 200
    finally:
        app.state.job_queue.shutdown()
        app.state.map_executor.shutdown(wait=False)


def test_pds_build_scan_persists_local_evidence(tmp_path):
    from sherloc_pipeline.services.pds_ingestion import PDSIngestionService

    service = object.__new__(PDSIngestionService)
    service._public_release_index = PublicReleaseIndex(write_index(tmp_path))
    group = SimpleNamespace(sol=9999, sclk=0, obs_id="synthetic", scan_type=None, sequence_code=None)
    metadata = SimpleNamespace(
        logical_identifier=PRODUCT, sclk_start_int=0, spacecraft_clock_stop=None,
        n_spectra=1, site_drive_str=None,
        to_pds4_metadata_dict=lambda: {"version": "synthetic-v1"},
    )
    scan = service._build_scan(group, metadata, target="Synthetic")
    DataAccessService("public").validate_scan_access(scan)
    changed = copy.deepcopy(scan)
    changed.pds4_metadata["version"] = "synthetic-v2"
    with pytest.raises(HTTPException):
        DataAccessService("public").validate_scan_access(changed)
