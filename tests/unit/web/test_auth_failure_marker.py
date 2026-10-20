"""Authentication denials carry an explicit session-invalidating marker."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from sherloc_pipeline.web import auth
from sherloc_pipeline.web.routes import user


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["science", "user-role", "user-anonymous"])
async def test_auth_403_marks_session_failure(monkeypatch, surface):
    monkeypatch.setenv("SHERLOC_AUTH_MODE", "auth0")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "internal")
    validator = Mock()
    validator.validate.return_value = SimpleNamespace(sub="synthetic-user", roles=[])
    monkeypatch.setattr(auth, "get_validator", lambda: validator)
    monkeypatch.setattr(user, "get_validator", lambda: validator)
    app = FastAPI()

    @app.get("/synthetic")
    def denied():
        if surface == "science":
            auth._authenticate_token("synthetic-token", "auth0", "internal")
        else:
            request = SimpleNamespace(
                state=SimpleNamespace(db=Mock()),
                headers={} if surface == "user-anonymous" else {"Authorization": "Bearer synthetic-token"},
            )
            user._require_user(request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://synthetic") as client:
        response = await client.get("/synthetic")
    assert response.status_code == 403
    assert response.headers["X-Sherloc-Auth-Failure"] == "authorization"
