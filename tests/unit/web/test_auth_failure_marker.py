"""Authentication denials carry an explicit session-invalidating marker."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from sherloc_pipeline.web import auth


@pytest.mark.asyncio
async def test_auth_403_marks_session_failure(monkeypatch):
    monkeypatch.setenv("SHERLOC_AUTH_MODE", "auth0")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "internal")
    validator = Mock()
    validator.validate.return_value = SimpleNamespace(sub="synthetic-user", roles=[])
    monkeypatch.setattr(auth, "get_validator", lambda: validator)
    app = FastAPI()

    @app.get("/synthetic")
    def denied():
        auth._authenticate_token("synthetic-token", "auth0", "internal")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://synthetic") as client:
        response = await client.get("/synthetic")
    assert response.status_code == 403
    assert response.headers["X-Sherloc-Auth-Failure"] == "authorization"
