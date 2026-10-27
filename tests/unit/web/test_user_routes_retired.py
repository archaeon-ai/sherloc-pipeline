"""The retired /api/user/* routes are a 404 on every app and never write a row.

Synthesized fixtures only: a stub validator stands in for Auth0 / CF Access
so a credentialed request is one the data API accepts (checked against
/api/scans), and a stub SPA build proves the catch-all does not answer.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from sherloc_pipeline.database.connection import create_all_tables, get_engine
from sherloc_pipeline.database.models import (
    ClassificationProfileORM,
    UserORM,
    UserPreferenceORM,
)
from sherloc_pipeline.web import auth
from sherloc_pipeline.web.app import create_app
from tests.unit.public_release_helpers import synthetic_index

RETIRED = [
    ("GET", "/api/user"),
    ("GET", "/api/user/preferences"),
    ("PUT", "/api/user/preferences/synthetic_key"),
    ("DELETE", "/api/user/preferences/synthetic_key"),
    ("GET", "/api/user/profiles"),
    ("POST", "/api/user/profiles"),
    ("PUT", "/api/user/profiles/synthetic-id"),
    ("DELETE", "/api/user/profiles/synthetic-id"),
    ("GET", "/api/user/profiles/synthetic-id/export"),
    ("POST", "/api/user/profiles/import"),
    ("GET", "/api/user/classification/defaults"),
]
BODY = {"value": 1, "name": "synthetic", "base": "default", "overrides": []}


def _env(monkeypatch, auth_mode, access_mode):
    monkeypatch.setenv("SHERLOC_AUTH_MODE", auth_mode)
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", access_mode)
    if auth_mode == "auth0":
        monkeypatch.setenv("SHERLOC_AUTH0_DOMAIN", "synthetic.auth0.invalid")
        monkeypatch.setenv("SHERLOC_AUTH0_AUDIENCE", "https://synthetic.invalid/api")
        monkeypatch.setenv("SHERLOC_AUTH0_IDENTITY_CLAIM_URI", "https://synthetic.invalid/identity")
        monkeypatch.setenv("SHERLOC_AUTH0_SPA_CLIENT_ID", "synthetic-spa")
    else:
        monkeypatch.setenv("SHERLOC_CF_TEAM_DOMAIN", "synthetic.cloudflareaccess.invalid")
        monkeypatch.setenv("SHERLOC_CF_AUDIENCE", "synthetic-aud")
    claims = SimpleNamespace(
        sub="synthetic-user", email="synthetic@example.invalid", roles=["phase:team-member"]
    )
    validator = SimpleNamespace(validate=lambda token: claims)
    monkeypatch.setattr(auth, "get_validator", lambda: validator)


def _build(tmp_path, access_mode):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>spa-shell</html>")
    name = "phase_pds.db" if access_mode == "public" else "phase.db"
    engine = get_engine(str(tmp_path / name))
    create_all_tables(engine)
    app = create_app(engine=engine, access_mode=access_mode, frontend_dist=dist)
    return engine, TestClient(app)


@pytest.mark.parametrize("access_mode", ["internal", "public"])
@pytest.mark.parametrize("auth_mode", ["auth0", "cf-access"])
def test_retired_user_routes_are_404_and_write_nothing(monkeypatch, tmp_path, auth_mode, access_mode):
    synthetic_index(tmp_path, monkeypatch)
    _env(monkeypatch, auth_mode, access_mode)
    header = "Authorization" if auth_mode == "auth0" else "Cf-Access-Jwt-Assertion"
    token = "Bearer synthetic-token" if auth_mode == "auth0" else "synthetic-token"
    engine, client = _build(tmp_path, access_mode)
    with client:
        # Control: the stub credential is accepted by the live data API.
        assert client.get("/api/scans", headers={header: token}).status_code == 200
        for headers in ({}, {header: token}):
            for method, path in RETIRED:
                kwargs = {"json": BODY} if method in ("PUT", "POST") else {}
                resp = client.request(method, path, headers=headers, **kwargs)
                assert resp.status_code == 404, (method, path, headers, resp.status_code)
                assert "spa-shell" not in resp.text
    with engine.connect() as conn:
        for model in (UserORM, UserPreferenceORM, ClassificationProfileORM):
            assert conn.execute(select(func.count()).select_from(model)).scalar_one() == 0


def test_retired_user_routes_are_not_in_the_schema(monkeypatch, tmp_path):
    monkeypatch.setenv("SHERLOC_AUTH_MODE", "dev")
    _, client = _build(tmp_path, "internal")
    with client:
        paths = client.get("/openapi.json").json()["paths"]
    assert not [p for p in paths if p.startswith("/api/user")]
