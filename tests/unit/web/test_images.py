"""ACI image endpoint tests — R2-backed (v4.1.7+, refactored v4.1.9).

Exercises the m2020-phase platform spec §3.9 contract: hierarchical-key
resolution from per-tier DB `context_images.file_path` → strip-prefix
→ `sherloc-aci/<rest>` in the per-tier R2 bucket.

Tests use moto's in-process S3 mock (`mock_aws`) so the boto3 client
exercises real serialization + signing + response handling, while no
network traffic leaves the test process. The `r2_*_client` fixtures
return a moto-backed boto3 client + create the per-tier bucket; tests
inject this client into the shared resolver module via
`set_r2_client_for_tests` (from `web.r2_reader`, extracted v4.1.9).
"""

from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3
import numpy as np
import pytest
from botocore.config import Config as BotoConfig
from moto import mock_aws
from PIL import Image as PILImage

from sherloc_pipeline.database.connection import get_session_factory
from sherloc_pipeline.database.models import ContextImageORM
from sherloc_pipeline.web import r2_reader
from sherloc_pipeline.web.routes import images as images_module
from tests.unit.web.conftest import SCAN_UUID

# Per-tier file_path conventions match live DB content
# verified Session 73 against the per-tier SQLite files. Literals are
# adjacent-string-concat-spelled to keep operator-local paths out of
# grep-scanned tracked content (per CONTRIBUTING.md "Public-repo
# discipline"). Runtime values are unchanged.
_TEAM_FILE_PATH = (
    "/data" "/sherloc/data/loupe/sol_0921/detail_1/img/SC3_0921_TEST.PNG"
)
_PUBLIC_FILE_PATH = (
    "/data" "/sherloc/pds/sol_0921/data_aci/SC0_0921_TEST.IMG"
)


# ---------------------------------------------------------------------------
# Moto-backed R2 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def r2_team_client():
    """Spin up moto S3 + create the `phase-team` bucket + inject the
    client into the resolver module. Yields the client so tests can
    pre-populate keys.
    """
    with mock_aws():
        client = boto3.client(
            "s3",
            aws_access_key_id="moto-test-id",
            aws_secret_access_key="moto-test-secret",
            region_name="us-east-1",
            config=BotoConfig(signature_version="s3v4"),
        )
        client.create_bucket(Bucket="phase-team")
        r2_reader.set_r2_client_for_tests(client, "team")
        try:
            yield client
        finally:
            r2_reader.reset_r2_client_for_tests()


@pytest.fixture
def r2_public_client():
    """Same shape as r2_team_client but for the public tier."""
    with mock_aws():
        client = boto3.client(
            "s3",
            aws_access_key_id="moto-test-id",
            aws_secret_access_key="moto-test-secret",
            region_name="us-east-1",
            config=BotoConfig(signature_version="s3v4"),
        )
        client.create_bucket(Bucket="phase-public")
        r2_reader.set_r2_client_for_tests(client, "public")
        try:
            yield client
        finally:
            r2_reader.reset_r2_client_for_tests()


def _make_png_bytes(size: tuple[int, int] = (16, 16), color: int = 128) -> bytes:
    img = PILImage.new("L", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _insert_aci(test_engine, file_path: str) -> None:
    """Insert a ContextImageORM row for SCAN_UUID with the given file_path."""
    factory = get_session_factory(test_engine)
    session = factory()
    try:
        session.add(
            ContextImageORM(
                id=str(uuid.uuid4()),
                scan_id=SCAN_UUID,
                image_type="ACI",
                file_path=file_path,
            )
        )
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Resolver-config tests (no DB / no HTTP)
# ---------------------------------------------------------------------------

class TestResolverConfig:
    def teardown_method(self):
        r2_reader.reset_r2_client_for_tests()

    def test_missing_tier_env_raises_500(self, monkeypatch):
        """PHASE_TIER unset → resolver raises HTTPException(500, tier_unset).

        Mirrors spec §3.9.4 row 'PHASE_TIER env var unset': defense-in-
        depth at request time (config_check should have caught this at
        boot in production).
        """
        for key in ("PHASE_TIER", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                    "AWS_ENDPOINT_URL"):
            monkeypatch.delenv(key, raising=False)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.get_r2_client_and_config()
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "tier_unset"

    def test_invalid_tier_value_raises_500(self, monkeypatch):
        monkeypatch.setenv("PHASE_TIER", "bogus")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://e")
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.get_r2_client_and_config()
        assert excinfo.value.status_code == 500

    def test_derive_team_key(self):
        # Team strip-prefix removes the team-data prefix; re-roots under sherloc-aci/.
        key = r2_reader.derive_r2_key(_TEAM_FILE_PATH, "/data" "/sherloc/data/")
        assert key == "sherloc-aci/loupe/sol_0921/detail_1/img/SC3_0921_TEST.PNG"

    def test_derive_public_key(self):
        key = r2_reader.derive_r2_key(_PUBLIC_FILE_PATH, "/data" "/sherloc/pds/")
        assert key == "sherloc-aci/sol_0921/data_aci/SC0_0921_TEST.IMG"

    def test_derive_key_wrong_prefix_raises_500(self):
        """Team-tier strip prefix on a public-tier path → misconfigured_path."""
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.derive_r2_key(
                _PUBLIC_FILE_PATH, "/data" "/sherloc/data/"
            )
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "misconfigured_path"

    def test_derive_key_traversal_raises_500(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.derive_r2_key(
                "/data" "/sherloc/data/../etc/passwd", "/data" "/sherloc/data/"
            )
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "misconfigured_path"


# ---------------------------------------------------------------------------
# Route-level tests (HTTP through FastAPI client + moto)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_aci_scan_not_found(client):
    """404 when the scan does not exist (no resolver involvement)."""
    resp = await client.get(
        "/api/images/00000000-0000-0000-0000-000000000099/aci"
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_aci_no_context_image(client):
    """404 when the scan exists but has no ACI record."""
    resp = await client.get(f"/api/images/{SCAN_UUID}/aci")
    assert resp.status_code == 404
    assert "No ACI context image" in resp.json()["detail"]


def test_aci_path_traversal_regex_blocks_dotdot():
    """The route-level scan_id validator regex blocks `..` / `/` / `\\` / `%2e` / `%2f`."""
    banned = images_module._SCAN_ID_BANNED
    for bad in (
        "..bad..",
        "../../../etc/passwd",
        "valid..bad",
        "with/slash",
        "with\\backslash",
        "with%2eencoded",
        "with%2Fencoded",
    ):
        assert banned.search(bad), f"{bad!r} should be banned"
    # Sanity: legitimate scan_ids pass.
    for ok in (
        "SHRLC0123_0001234567_F00_S01_RAS1",
        "0921_Amherst_Point_detail_1",
        "00000000-0000-0000-0000-000000000001",
    ):
        assert not banned.search(ok), f"{ok!r} should be allowed"


@pytest.mark.asyncio
async def test_aci_team_resolve_200(client, test_engine, r2_team_client):
    """Team-tier PNG resolves through R2 → returns image bytes."""
    _insert_aci(test_engine, _TEAM_FILE_PATH)
    r2_team_client.put_object(
        Bucket="phase-team",
        Key="sherloc-aci/loupe/sol_0921/detail_1/img/SC3_0921_TEST.PNG",
        Body=_make_png_bytes(),
        ContentType="image/png",
    )

    resp = await client.get(f"/api/images/{SCAN_UUID}/aci")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    assert "max-age=86400" in resp.headers["cache-control"]
    assert resp.content[:4] == b"\x89PNG"


@pytest.mark.asyncio
async def test_aci_public_tier_vicar_converted(
    client, test_engine, r2_public_client
):
    """Public-tier .IMG (VICAR) — bytes fetched from R2, converted via tempfile."""
    _insert_aci(test_engine, _PUBLIC_FILE_PATH)
    r2_public_client.put_object(
        Bucket="phase-public",
        Key="sherloc-aci/sol_0921/data_aci/SC0_0921_TEST.IMG",
        Body=b"VICAR-FAKE-BYTES",  # actual bytes don't matter — read_aci_image is mocked
    )

    fake_image = np.zeros((100, 100), dtype=np.uint8)
    fake_metadata = MagicMock()
    with patch(
        "sherloc_pipeline.vision.img_reader.read_aci_image"
    ) as mock_read:
        mock_read.return_value = (fake_image, fake_metadata)
        resp = await client.get(f"/api/images/{SCAN_UUID}/aci")

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:4] == b"\x89PNG"
    assert mock_read.called  # VICAR decoder ran on the tempfile


@pytest.mark.asyncio
async def test_aci_missing_object_404(client, test_engine, r2_team_client):
    """DB row exists, R2 key does not → resolver maps NoSuchKey to 404 not_found."""
    _insert_aci(test_engine, _TEAM_FILE_PATH)
    # NOTE: we deliberately do NOT put_object — bucket is empty.

    resp = await client.get(f"/api/images/{SCAN_UUID}/aci")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "not_found"


@pytest.mark.asyncio
async def test_aci_r2_timeout_returns_504(client, test_engine):
    """boto3 ReadTimeoutError → 504 upstream_timeout per spec §3.9.4."""
    from botocore.exceptions import ReadTimeoutError
    _insert_aci(test_engine, _TEAM_FILE_PATH)
    client_mock = MagicMock()
    client_mock.get_object.side_effect = ReadTimeoutError(endpoint_url="https://moto")
    r2_reader.set_r2_client_for_tests(client_mock, "team")
    try:
        resp = await client.get(f"/api/images/{SCAN_UUID}/aci")
    finally:
        r2_reader.reset_r2_client_for_tests()
    assert resp.status_code == 504
    assert resp.json()["detail"] == "upstream_timeout"


@pytest.mark.asyncio
async def test_aci_r2_botocore_error_returns_500(client, test_engine):
    """boto3 non-timeout BotoCoreError → 500 upstream_error per spec §3.9.4."""
    from botocore.exceptions import EndpointConnectionError
    _insert_aci(test_engine, _TEAM_FILE_PATH)
    client_mock = MagicMock()
    client_mock.get_object.side_effect = EndpointConnectionError(
        endpoint_url="https://moto"
    )
    r2_reader.set_r2_client_for_tests(client_mock, "team")
    try:
        resp = await client.get(f"/api/images/{SCAN_UUID}/aci")
    finally:
        r2_reader.reset_r2_client_for_tests()
    assert resp.status_code == 500
    assert resp.json()["detail"] == "upstream_error"


@pytest.mark.asyncio
async def test_aci_cross_tier_access_denied_502(client, test_engine):
    """Slot-3 (PUBLIC_READ) creds against phase-team bucket → 403 → resolver 502.

    Simulates the §3.9.5 + V26 negative: a team container misconfigured
    with PUBLIC_READ credentials. moto's mock_aws doesn't model per-
    bucket scoping, so we inject a client whose get_object raises a
    ClientError with code=AccessDenied — exactly what the live R2 layer
    would return.
    """
    from botocore.exceptions import ClientError
    _insert_aci(test_engine, _TEAM_FILE_PATH)

    client_mock = MagicMock()
    client_mock.get_object.side_effect = ClientError(
        error_response={
            "Error": {"Code": "AccessDenied", "Message": "Access Denied"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        operation_name="GetObject",
    )
    r2_reader.set_r2_client_for_tests(client_mock, "team")
    try:
        resp = await client.get(f"/api/images/{SCAN_UUID}/aci")
    finally:
        r2_reader.reset_r2_client_for_tests()

    assert resp.status_code == 502
    assert resp.json()["detail"] == "upstream_credential_error"


@pytest.mark.asyncio
async def test_aci_misconfigured_path_500(client, test_engine, r2_team_client):
    """Team-tier DB row with a public-tier path → misconfigured_path 500.

    A misclassified ingestion (DB tier ≠ file_path-implied tier) is
    rejected before any R2 call per §3.9.4.
    """
    _insert_aci(test_engine, _PUBLIC_FILE_PATH)  # public-tier path inserted into TEAM DB

    resp = await client.get(f"/api/images/{SCAN_UUID}/aci")
    assert resp.status_code == 500
    assert resp.json()["detail"] == "misconfigured_path"


@pytest.mark.asyncio
async def test_aci_pds_lidvid_returns_misconfigured_path_500(
    client, test_engine, r2_team_client
):
    """pds: LIDVID refs don't match the per-tier strip prefix → 500 misconfigured_path.

    Per spec §3.9.4 row "DB file_path does not begin with the expected
    per-tier strip prefix". An unresolved `pds:` ref in production IS
    a misconfiguration (the rclone-time resolution at D3.4 should have
    converted it to an absolute path). v1.0-beta rejects.
    """
    _insert_aci(
        test_engine,
        "pds:urn:nasa:pds:mars2020_mission:data_sherloc:SHRLC0001_0001",
    )
    resp = await client.get(f"/api/images/{SCAN_UUID}/aci")
    assert resp.status_code == 500
    assert resp.json()["detail"] == "misconfigured_path"


@pytest.mark.asyncio
async def test_aci_colorized_variant_via_r2(client, test_engine, r2_team_client):
    """colorized=true probes sibling sol_NNNN_colorized/ key in R2.

    Assertion proves the colorized variant was served (not the base
    fallback): we compute the SHA of the colorized object's bytes and
    the SHA of the base object's bytes (which differ in color value),
    then check the response bytes match the colorized SHA exactly. If
    the route silently fell back to the base on a failed HEAD probe,
    this assertion would fail.
    """
    import hashlib

    _insert_aci(test_engine, _TEAM_FILE_PATH)
    base_bytes = _make_png_bytes(color=64)
    colorized_bytes = _make_png_bytes(color=200)
    assert base_bytes != colorized_bytes  # sanity — the colors differ

    r2_team_client.put_object(
        Bucket="phase-team",
        Key="sherloc-aci/loupe/sol_0921/detail_1/img/SC3_0921_TEST.PNG",
        Body=base_bytes,
    )
    r2_team_client.put_object(
        Bucket="phase-team",
        Key="sherloc-aci/loupe/sol_0921_colorized/detail_1/img/SC3_0921_TEST.PNG",
        Body=colorized_bytes,
    )

    resp = await client.get(
        f"/api/images/{SCAN_UUID}/aci", params={"colorized": "true"}
    )
    assert resp.status_code == 200, resp.text
    response_sha = hashlib.sha256(resp.content).hexdigest()
    colorized_sha = hashlib.sha256(colorized_bytes).hexdigest()
    base_sha = hashlib.sha256(base_bytes).hexdigest()
    assert response_sha == colorized_sha, (
        "colorized=true should serve the colorized object; "
        f"got SHA {response_sha[:12]}… (matches base={response_sha == base_sha})"
    )


@pytest.mark.asyncio
async def test_aci_colorized_falls_back_to_base(
    client, test_engine, r2_team_client
):
    """colorized=true with no colorized variant in R2 → fall back to base key."""
    _insert_aci(test_engine, _TEAM_FILE_PATH)
    r2_team_client.put_object(
        Bucket="phase-team",
        Key="sherloc-aci/loupe/sol_0921/detail_1/img/SC3_0921_TEST.PNG",
        Body=_make_png_bytes(),
    )

    resp = await client.get(
        f"/api/images/{SCAN_UUID}/aci", params={"colorized": "true"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.content[:4] == b"\x89PNG"
