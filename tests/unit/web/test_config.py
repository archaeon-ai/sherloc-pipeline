"""Config endpoint tests."""

import pytest


@pytest.mark.asyncio
async def test_config_response(client):
    resp = await client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert "config_hash" in data
    assert data["config_hash"].startswith("sha256:")
    assert "fitting" in data
    assert "preprocessing" in data
    assert "calibration" in data
    assert data["calibration"]["version"] == "loupe_v5.1.5a"
    assert data["calibration"]["laser_wavelength_nm"] == 248.5794


# ---------------------------------------------------------------------------
# Feature flag: PDS Browser (issue #21)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_features_pds_browser_default_enabled(client, monkeypatch):
    """No env var → tab visible (legacy + dev default)."""
    monkeypatch.delenv("SHERLOC_FEATURE_PDS_BROWSER", raising=False)
    resp = await client.get("/api/config")
    assert resp.status_code == 200
    assert resp.json()["features"]["pds_browser"] is True


@pytest.mark.asyncio
async def test_features_pds_browser_enabled_explicit(client, monkeypatch):
    """Explicit `enabled` matches the unset default."""
    monkeypatch.setenv("SHERLOC_FEATURE_PDS_BROWSER", "enabled")
    resp = await client.get("/api/config")
    assert resp.status_code == 200
    assert resp.json()["features"]["pds_browser"] is True


@pytest.mark.asyncio
async def test_features_pds_browser_disabled(client, monkeypatch):
    """Production override: `disabled` hides the tab."""
    monkeypatch.setenv("SHERLOC_FEATURE_PDS_BROWSER", "disabled")
    resp = await client.get("/api/config")
    assert resp.status_code == 200
    assert resp.json()["features"]["pds_browser"] is False


@pytest.mark.asyncio
async def test_features_pds_browser_disabled_case_insensitive(client, monkeypatch):
    """`DISABLED` / `Disabled` also opt out — case-insensitive match."""
    monkeypatch.setenv("SHERLOC_FEATURE_PDS_BROWSER", "DISABLED")
    resp = await client.get("/api/config")
    assert resp.json()["features"]["pds_browser"] is False


@pytest.mark.asyncio
async def test_features_pds_browser_typo_stays_enabled(client, monkeypatch):
    """Typos (`disable`, `off`) leave the flag ON — fail-open semantics
    so a deploy-template typo can't accidentally hide a feature.
    Only the literal `disabled` (case-insensitive) opts out."""
    monkeypatch.setenv("SHERLOC_FEATURE_PDS_BROWSER", "off")
    resp = await client.get("/api/config")
    assert resp.json()["features"]["pds_browser"] is True
