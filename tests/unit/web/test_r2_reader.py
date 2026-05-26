"""Tests for ``web/r2_reader.py`` — the shared R2-reader module (v4.1.9+).

Exercises the m2020-phase platform spec §3.9.8 contract:
``get_working_file(file_path, filename) → bytes`` for Loupe-workspace
companion files (``spatial.csv``, ``loupe.csv``). Uses moto's in-process
S3 mock (``mock_aws``) so the boto3 client exercises real serialization +
signing while no network traffic leaves the test process.

Mirrors the per-tier resolve / cross-tier-creds / misconfigured-path /
traversal coverage from ``test_images.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import boto3
import pytest
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, ReadTimeoutError
from fastapi import HTTPException
from moto import mock_aws

from sherloc_pipeline.web import r2_reader

# Per-tier file_path conventions match live DB content
# verified Session 73 against the per-tier SQLite files. The Loupe
# workspace directory is two levels up from the ACI file (drops
# ``img/<aci-product>.{PNG,IMG}``).
_TEAM_FILE_PATH = (
    "/data" "/sherloc/data/loupe/sol_0921/detail_1/"
    "SrlcSpecSpecSohRaw_TEST_Loupe_working/img/SC3_0921_TEST.PNG"
)
_PUBLIC_FILE_PATH = (
    "/data" "/sherloc/pds/sol_0921/detail_1/"
    "SrlcSpecSpecSohRaw_TEST_Loupe_working/img/SC0_0921_TEST.IMG"
)

# Derived R2 keys for the Loupe workspace files (spec §3.9.8.2 examples).
_TEAM_SPATIAL_KEY = (
    "sherloc-aci/loupe/sol_0921/detail_1/"
    "SrlcSpecSpecSohRaw_TEST_Loupe_working/spatial.csv"
)
_TEAM_LOUPE_KEY = (
    "sherloc-aci/loupe/sol_0921/detail_1/"
    "SrlcSpecSpecSohRaw_TEST_Loupe_working/loupe.csv"
)
_PUBLIC_SPATIAL_KEY = (
    "sherloc-aci/sol_0921/detail_1/"
    "SrlcSpecSpecSohRaw_TEST_Loupe_working/spatial.csv"
)

_SAMPLE_SPATIAL_CSV = b"x,y\n0.0,0.0\n0.5,0.5\n1.0,1.0\n"
_SAMPLE_LOUPE_CSV = b"laser_x,809.0\nlaser_y,664.0\n"


@pytest.fixture
def r2_team_client():
    """Moto S3 + phase-team bucket + inject client into the shared module."""
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


# ---------------------------------------------------------------------------
# get_working_file — happy path (team + public)
# ---------------------------------------------------------------------------

class TestGetWorkingFile:
    def test_get_working_file_team_resolve_spatial(self, r2_team_client):
        """Team-tier spatial.csv resolves through R2 → returns bytes verbatim."""
        r2_team_client.put_object(
            Bucket="phase-team", Key=_TEAM_SPATIAL_KEY, Body=_SAMPLE_SPATIAL_CSV
        )
        result = r2_reader.get_working_file(_TEAM_FILE_PATH, "spatial.csv")
        assert result == _SAMPLE_SPATIAL_CSV

    def test_get_working_file_team_resolve_loupe(self, r2_team_client):
        """Team-tier loupe.csv resolves through R2 → returns bytes verbatim."""
        r2_team_client.put_object(
            Bucket="phase-team", Key=_TEAM_LOUPE_KEY, Body=_SAMPLE_LOUPE_CSV
        )
        result = r2_reader.get_working_file(_TEAM_FILE_PATH, "loupe.csv")
        assert result == _SAMPLE_LOUPE_CSV

    def test_get_working_file_public_resolve(self, r2_public_client):
        """Public-tier spatial.csv resolves through R2 → returns bytes."""
        r2_public_client.put_object(
            Bucket="phase-public",
            Key=_PUBLIC_SPATIAL_KEY,
            Body=_SAMPLE_SPATIAL_CSV,
        )
        result = r2_reader.get_working_file(_PUBLIC_FILE_PATH, "spatial.csv")
        assert result == _SAMPLE_SPATIAL_CSV

    def test_get_working_file_missing_404(self, r2_team_client):
        """R2 key not present → HTTPException(404, not_found).

        Mirrors spec §3.9.8.3 — companion-file missing maps to 404 at the
        resolver level. The route handler (`map.py:get_map_layers`)
        catches this via `CoordinatesUnavailableError` and re-raises as
        400 with an explicit "Loupe workspace files not found in R2"
        message — that mapping is exercised in test_coordinates.py.
        """
        # NOTE: bucket exists (fixture creates it) but no put_object — key absent.
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.get_working_file(_TEAM_FILE_PATH, "spatial.csv")
        assert excinfo.value.status_code == 404
        assert excinfo.value.detail == "not_found"

    def test_get_working_file_misconfigured_path(self, r2_team_client):
        """Team-tier creds with public-tier file_path → misconfigured_path 500.

        `_PUBLIC_FILE_PATH` does not begin with the team strip-prefix
        (the per-tier ``TIER_TO_STRIP_PREFIX["team"]`` value), so the
        resolver rejects without an R2 call per spec §3.9.4 / §3.9.8.3.
        """
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.get_working_file(_PUBLIC_FILE_PATH, "spatial.csv")
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "misconfigured_path"

    def test_get_working_file_path_traversal_rejected(self, r2_team_client):
        """A file_path with `..` segments cannot derive a traversal key."""
        traversal_path = "/data" "/sherloc/data/../etc/passwd/img/aci.PNG"
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.get_working_file(traversal_path, "spatial.csv")
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "misconfigured_path"

    @pytest.mark.parametrize(
        "disallowed_filename",
        [
            # Slash-containing names that the post-interpolation guard
            # would NOT catch (no leading slash, no ``..``, no backslash).
            "img/aci.PNG",
            "other_workspace/spatial.csv",
            # Names outside the §3.9.8.1 allowlist.
            "other.csv",
            "loupe.csv.bak",
            "config.json",
            "",
        ],
    )
    def test_get_working_file_disallowed_filename_500(
        self, r2_team_client, disallowed_filename
    ):
        """Filename not in WORKSPACE_FILENAMES → misconfigured_path 500.

        Per spec §3.9.8.1 the resolver accepts only ``spatial.csv`` and
        ``loupe.csv``; rejecting anything else BEFORE the R2 GET prevents
        this helper from being inadvertently turned into a general
        workspace-object reader (Codex PR #11 R1 F1).
        """
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.get_working_file(_TEAM_FILE_PATH, disallowed_filename)
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "misconfigured_path"

    def test_get_working_file_cross_tier_403_502(self):
        """boto3 raises AccessDenied → resolver 502 upstream_credential_error.

        Simulates the §3.9.5 negative: a team container misconfigured
        with public-tier (slot 3) credentials attempts a team-bucket GET.
        moto's mock_aws doesn't model per-bucket scoping, so inject a
        client whose `get_object` raises a ClientError(AccessDenied) —
        exactly what live R2 would return.
        """
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
            with pytest.raises(HTTPException) as excinfo:
                r2_reader.get_working_file(_TEAM_FILE_PATH, "spatial.csv")
            assert excinfo.value.status_code == 502
            assert excinfo.value.detail == "upstream_credential_error"
        finally:
            r2_reader.reset_r2_client_for_tests()

    def test_derive_workspace_key_team(self, r2_team_client):
        """derive_workspace_key returns the canonical R2 key for team-tier file_path."""
        key = r2_reader.derive_workspace_key(_TEAM_FILE_PATH, "spatial.csv")
        assert key == _TEAM_SPATIAL_KEY

    def test_derive_workspace_key_public(self, r2_public_client):
        """derive_workspace_key returns the canonical R2 key for public-tier file_path."""
        key = r2_reader.derive_workspace_key(_PUBLIC_FILE_PATH, "spatial.csv")
        assert key == _PUBLIC_SPATIAL_KEY

    def test_derive_workspace_key_rejects_disallowed_filename(self, r2_team_client):
        """derive_workspace_key applies the same WORKSPACE_FILENAMES allowlist."""
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.derive_workspace_key(_TEAM_FILE_PATH, "other.csv")
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "misconfigured_path"

    # -----------------------------------------------------------------
    # Legacy-alias acceptance (v4.1.17, PR #__) — production DBs may carry
    # pre-v4.1.9 `file_path` rows with a NAS-mount prefix. ``derive_r2_key``
    # and ``derive_workspace_key`` accept those as aliases of the canonical
    # team-tier prefix, deriving the SAME ``sherloc-aci/<rel>`` R2 key.
    # Live audit 2026-05-24 found 3 such rows on the team-tier VPS DB:
    # Mauchsberg sol 1810 + Pafuri Gate sol 1798. They had been 500-ing
    # on every ACI route since v4.1.9 deployed.
    # -----------------------------------------------------------------

    def test_get_working_file_team_resolves_legacy_nas_alias(self, r2_team_client):
        """A legacy NAS-prefixed ``file_path`` resolves to the same R2 key as the canonical sibling."""
        legacy_path = (
            "/nas/" "000_sherloc/data/loupe/sol_1810/line_1/"
            "SrlcSpecSpecSohRaw_TEST_Loupe_working/img/SC2_1810_TEST.PNG"
        )
        canonical_key = (
            "sherloc-aci/loupe/sol_1810/line_1/"
            "SrlcSpecSpecSohRaw_TEST_Loupe_working/spatial.csv"
        )
        r2_team_client.put_object(
            Bucket="phase-team", Key=canonical_key, Body=_SAMPLE_SPATIAL_CSV
        )
        body = r2_reader.get_working_file(legacy_path, "spatial.csv")
        assert body == _SAMPLE_SPATIAL_CSV

    def test_derive_workspace_key_team_accepts_legacy_nas_alias(self, r2_team_client):
        """derive_workspace_key strips the legacy alias to the same R2 key."""
        legacy_path = (
            "/nas/" "000_sherloc/data/loupe/sol_0921/detail_1/"
            "SrlcSpecSpecSohRaw_TEST_Loupe_working/img/SC3_0921_TEST.PNG"
        )
        key = r2_reader.derive_workspace_key(legacy_path, "spatial.csv")
        assert key == _TEAM_SPATIAL_KEY  # identical to canonical-prefix sibling

    def test_get_working_file_public_rejects_team_legacy_alias(self, r2_public_client):
        """Public tier does not inherit team-tier legacy aliases (tier isolation)."""
        team_legacy_path = (
            "/nas/" "000_sherloc/data/loupe/sol_1810/line_1/"
            "SrlcSpecSpecSohRaw_TEST_Loupe_working/img/SC2_1810_TEST.PNG"
        )
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.get_working_file(team_legacy_path, "spatial.csv")
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "misconfigured_path"

    def test_legacy_alias_still_enforces_path_traversal_guard(self, r2_team_client):
        """A legacy-aliased path with `..` segments still 500s — guard applies post-strip."""
        traversal_legacy = (
            "/nas/" "000_sherloc/data/../etc/passwd/img/aci.PNG"
        )
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.get_working_file(traversal_legacy, "spatial.csv")
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "misconfigured_path"

    def test_derive_r2_key_team_accepts_legacy_nas_alias(self):
        """Direct ``derive_r2_key`` call resolves a legacy NAS path to the canonical R2 key.

        Mirrors the ACI image route's resolution step (`/api/images/{id}/aci`
        and `colorized_variant_exists`), independent of the workspace
        companion-file path. PR #39 Codex R1 F4.
        """
        legacy_path = (
            "/nas/" "000_sherloc/data/loupe/sol_1810/line_1/"
            "SrlcSpecSpecSohRaw_TEST_Loupe_working/img/SC2_1810_TEST.PNG"
        )
        team_strip = r2_reader.TIER_TO_STRIP_PREFIX["team"]
        key = r2_reader.derive_r2_key(legacy_path, team_strip)
        assert key == (
            "sherloc-aci/loupe/sol_1810/line_1/"
            "SrlcSpecSpecSohRaw_TEST_Loupe_working/img/SC2_1810_TEST.PNG"
        )

    def test_derive_r2_key_explicit_active_tier_overrides_inference(self):
        """``active_tier`` kwarg lets callers bypass strip-prefix-based tier inference.

        Regression guard for PR #39 Codex R1 F1 — when the canonical prefix
        doesn't match, the active-tier scope determines which alias set to
        consult. Explicit ``active_tier="public"`` MUST NOT consult team
        aliases even when the strip-prefix value happens to match team's.
        """
        legacy_team_path = (
            "/nas/" "000_sherloc/data/loupe/sol_1810/line_1/img/aci.PNG"
        )
        team_strip = r2_reader.TIER_TO_STRIP_PREFIX["team"]
        # Implicit (back-compat) call: inference finds team via strip-prefix
        # match → team aliases consulted → resolves.
        key = r2_reader.derive_r2_key(legacy_team_path, team_strip)
        assert key.startswith("sherloc-aci/loupe/sol_1810/")
        # Explicit active_tier="public" → only public aliases (empty by
        # default) → no alias matches → misconfigured_path.
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.derive_r2_key(legacy_team_path, team_strip, active_tier="public")
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "misconfigured_path"

    def test_legacy_alias_env_var_appends_not_replaces_default(self, monkeypatch):
        """Setting ``PHASE_TEAM_LEGACY_STRIP_ALIASES`` MUST NOT drop the built-in default.

        Regression guard for PR #39 Codex R1 F2 — the env var is additive
        (colon-separated, $PATH-style). The built-in production alias
        (the in-tree default per ``_DEFAULT_TEAM_LEGACY_ALIASES``) stays
        in scope regardless.
        """
        import importlib

        monkeypatch.setenv("PHASE_TEAM_LEGACY_STRIP_ALIASES", "/some/other/legacy/root/")
        # Reimport the module so the env-var-derived constant rebuilds.
        from sherloc_pipeline.core import r2_keys as r2_keys_mod
        importlib.reload(r2_keys_mod)
        try:
            team_aliases = r2_keys_mod.TIER_TO_LEGACY_STRIP_ALIASES["team"]
            # Built-in default still present
            assert any("000_sherloc" in p for p in team_aliases)
            # Plus the env-provided extra
            assert "/some/other/legacy/root/" in team_aliases
        finally:
            # Restore module state so subsequent tests don't see the override.
            monkeypatch.delenv("PHASE_TEAM_LEGACY_STRIP_ALIASES", raising=False)
            importlib.reload(r2_keys_mod)

    def test_get_working_file_timeout_504(self):
        """boto3 ReadTimeoutError → 504 upstream_timeout per spec §3.9.4."""
        client_mock = MagicMock()
        client_mock.get_object.side_effect = ReadTimeoutError(endpoint_url="https://moto")
        r2_reader.set_r2_client_for_tests(client_mock, "team")
        try:
            with pytest.raises(HTTPException) as excinfo:
                r2_reader.get_working_file(_TEAM_FILE_PATH, "spatial.csv")
            assert excinfo.value.status_code == 504
            assert excinfo.value.detail == "upstream_timeout"
        finally:
            r2_reader.reset_r2_client_for_tests()


# ---------------------------------------------------------------------------
# is_r2_mode — env-var pre-check (used by route layer to branch R2 vs FS)
# ---------------------------------------------------------------------------

class TestIsR2Mode:
    """is_r2_mode() — spec-compliant per Codex PR #11 R2 F3.

    The predicate gates FS-fallback vs R2-routing in the route layer.
    Per m2020-phase spec §3.9.6 + §3.9.8.5 the FS fallback is permitted
    ONLY when PHASE_TIER is unset (legacy dev / local main worktree).
    Any container that has PHASE_TIER set is a production / staging
    deployment and MUST route through the R2 path, where invalid tier
    values + missing AWS_* credentials surface as
    HTTPException(500, "tier_unset") per §3.9.4 — NOT as silent FS
    fallback.
    """

    def test_is_r2_mode_false_without_phase_tier(self, monkeypatch):
        """No PHASE_TIER → FS fallback enabled (legacy dev / local)."""
        for key in ("PHASE_TIER", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                    "AWS_ENDPOINT_URL"):
            monkeypatch.delenv(key, raising=False)
        assert r2_reader.is_r2_mode() is False

    def test_is_r2_mode_true_with_tier_but_no_aws_creds(self, monkeypatch):
        """PHASE_TIER set + AWS_* missing → R2 path (so tier_unset 500 surfaces).

        Spec §3.9.4: missing AWS_* env in a tier-configured container is
        a production misconfiguration and MUST fail loudly as
        ``tier_unset`` — NOT silently fall back to local FS (which the
        production container does not have mounted per §3.9.6).
        """
        monkeypatch.setenv("PHASE_TIER", "team")
        for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                    "AWS_ENDPOINT_URL"):
            monkeypatch.delenv(key, raising=False)
        assert r2_reader.is_r2_mode() is True

    def test_is_r2_mode_true_with_invalid_tier(self, monkeypatch):
        """Invalid PHASE_TIER value → still routes through R2 (tier_unset 500).

        Per Codex PR #11 R2 F3 recommendation: "If PHASE_TIER is present
        but invalid, ... route through the R2 resolver/config path so
        get_r2_client_and_config() raises tier_unset."
        """
        monkeypatch.setenv("PHASE_TIER", "bogus")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://e")
        assert r2_reader.is_r2_mode() is True

    def test_is_r2_mode_true_with_full_config(self, monkeypatch):
        monkeypatch.setenv("PHASE_TIER", "team")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://e")
        assert r2_reader.is_r2_mode() is True

    def test_is_r2_mode_true_with_empty_string_phase_tier(self, monkeypatch):
        """``PHASE_TIER=""`` → R2 path (so tier_unset 500 surfaces).

        Per Codex PR #11 R3 F5 + spec §3.9.4: an env file that exports
        ``PHASE_TIER=`` (empty string) is a production misconfiguration.
        Treating empty as "unset" would silently select FS fallback —
        which the production container has no mount for. Empty string
        is NOT in {team, public} so the §3.9.4 row applies; the predicate
        routes to R2 so ``get_r2_client_and_config`` raises tier_unset.
        """
        monkeypatch.setenv("PHASE_TIER", "")
        assert r2_reader.is_r2_mode() is True
