"""Synthetic ASGI byte admission before DB, authentication and JSON parsing."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine

from sherloc_pipeline.web.app import IDENTIFY_BODY_LIMIT, create_app
from sherloc_pipeline.web.auth import require_authenticated_request


@pytest.fixture(params=["team", "public"])
def limited_app(request, monkeypatch):
    tier = request.param
    monkeypatch.setenv("PHASE_TIER", tier)
    monkeypatch.setenv("SHERLOC_FEATURE_POLICY", "{}")
    engine = create_engine("sqlite://")
    app = create_app(engine=engine, config=SimpleNamespace(),
                     access_mode="internal" if tier == "team" else "public")
    yield app
    app.state.job_queue.shutdown()
    app.state.map_executor.shutdown(wait=False)
    engine.dispose()


@pytest.mark.parametrize("delivery", ["length", "chunked", "understated"])
def test_identify_rejects_oversize_before_parsing_or_admission(limited_app, monkeypatch, delivery):
    parse = Mock(side_effect=AssertionError("JSON parsed before body admission"))
    database = Mock(side_effect=AssertionError("DB opened before body admission"))
    monkeypatch.setattr(Request, "json", parse)
    monkeypatch.setattr("sherloc_pipeline.database.connection.get_session_factory", database)
    headers = []
    if delivery == "length":
        headers = [(b"content-length", str(IDENTIFY_BODY_LIMIT + 1).encode())]
    elif delivery == "chunked":
        headers = [(b"transfer-encoding", b"chunked")]
    else:
        headers = [(b"content-length", b"1")]
    consumed = 0
    sent = []

    async def receive():
        nonlocal consumed
        consumed += 1
        assert consumed <= 17, "Oversize request was drained after refusal"
        return {"type": "http.request", "body": b"x" * 65536, "more_body": True}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/api/identify", "headers": headers,
             "query_string": b"", "scheme": "http", "server": ("synthetic", 80),
             "client": ("synthetic", 1), "http_version": "1.1"}
    asyncio.run(limited_app(scope, receive, send))
    assert sent[0]["status"] == 413
    assert (b"cache-control", b"private, no-store") in sent[0]["headers"]
    assert json.loads(sent[1]["body"])["error"] == "payload_too_large"
    assert consumed == (0 if delivery == "length" else 17)
    parse.assert_not_called()
    database.assert_not_called()


def test_identify_permitted_template_ceiling_reaches_auth(limited_app):
    def denied():
        raise HTTPException(401, "Synthetic authentication refusal")

    limited_app.dependency_overrides[require_authenticated_request] = denied
    body = {"scan_id": "synthetic", "selection": {"mode": "point", "index": 0}, "region": "R1",
            "spectrum_type": "laser_normalized", "expected_source_digest": "a" * 64,
            "reference_release_id": "b" * 64, "reference_subset_id": "c" * 64,
            "method_release_id": "d" * 64, "template_ids": ["e" * 64] * 10000}
    encoded = json.dumps(body).encode()
    assert 256 * 1024 < len(encoded) < IDENTIFY_BODY_LIMIT
    with TestClient(limited_app) as client:
        response = client.post("/api/identify", content=encoded, headers={"Content-Type": "application/json"})
    assert response.status_code == 401
    assert response.headers["cache-control"] == "private, no-store"
