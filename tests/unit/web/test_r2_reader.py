"""Tests for ``web/r2_reader.py`` — the shared R2-reader module (v4.1.9+).

Exercises the R2 companion-file contract:
``get_working_file(rel_locator, filename) → bytes`` for Loupe-workspace
companion files (``spatial.csv``, ``loupe.csv``). Uses moto's in-process
S3 mock (``mock_aws``) so the boto3 client exercises real serialization +
signing while no network traffic leaves the test process.

Keys derive from the stored relative locator
(``context_images.r2_rel_key``) by concatenation — there is no per-tier
strip-prefix table or legacy-alias machinery anymore (retired with the
v5.4.0 locator migration). Tier isolation is credential-side (bucket-
scoped tokens) plus per-tier databases.
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

# Relative locators (the ``r2_rel_key`` column value). The Loupe
# workspace directory is two levels up from the ACI file (drops
# ``img/<aci-product>.{PNG,IMG}``).
_TEAM_LOCATOR = (
    "loupe/sol_0921/detail_1/"
    "SrlcSpecSpecSohRaw_TEST_Loupe_working/img/SC3_0921_TEST.PNG"
)
_PUBLIC_LOCATOR = (
    "sol_0921/detail_1/"
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
# derive_r2_key — pure locator → key concatenation
# ---------------------------------------------------------------------------

class TestDeriveR2Key:
    def test_team_locator_concatenates(self):
        assert r2_reader.derive_r2_key(_TEAM_LOCATOR) == (
            "sherloc-aci/" + _TEAM_LOCATOR
        )

    def test_public_locator_concatenates(self):
        assert r2_reader.derive_r2_key(_PUBLIC_LOCATOR) == (
            "sherloc-aci/" + _PUBLIC_LOCATOR
        )

    @pytest.mark.parametrize(
        "bad_locator",
        [
            None,  # row predates backfill or matched no known layout
            "",
            # Unresolved PDS reference: no R2 identity until the download
            # step resolves it (same 500 the pre-locator serve path gave).
            "pds:urn:nasa:pds:mars2020_imgops:data_aci_imgops:x::1.0",
            "loupe/sol_1/../../../etc/passwd",  # traversal
            "/absolute/path/img/a.PNG",  # absolute → not a locator
            "loupe\\sol_1\\img\\a.PNG",  # backslash
        ],
    )
    def test_bad_locator_misconfigured_path_500(self, bad_locator):
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.derive_r2_key(bad_locator)
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "misconfigured_path"


# ---------------------------------------------------------------------------
# get_working_file — happy path (team + public)
# ---------------------------------------------------------------------------

class TestGetWorkingFile:
    def test_get_working_file_team_resolve_spatial(self, r2_team_client):
        """Team-tier spatial.csv resolves through R2 → returns bytes verbatim."""
        r2_team_client.put_object(
            Bucket="phase-team", Key=_TEAM_SPATIAL_KEY, Body=_SAMPLE_SPATIAL_CSV
        )
        result = r2_reader.get_working_file(_TEAM_LOCATOR, "spatial.csv")
        assert result == _SAMPLE_SPATIAL_CSV

    def test_get_working_file_team_resolve_loupe(self, r2_team_client):
        """Team-tier loupe.csv resolves through R2 → returns bytes verbatim."""
        r2_team_client.put_object(
            Bucket="phase-team", Key=_TEAM_LOUPE_KEY, Body=_SAMPLE_LOUPE_CSV
        )
        result = r2_reader.get_working_file(_TEAM_LOCATOR, "loupe.csv")
        assert result == _SAMPLE_LOUPE_CSV

    def test_get_working_file_public_resolve(self, r2_public_client):
        """Public-tier spatial.csv resolves through R2 → returns bytes."""
        r2_public_client.put_object(
            Bucket="phase-public",
            Key=_PUBLIC_SPATIAL_KEY,
            Body=_SAMPLE_SPATIAL_CSV,
        )
        result = r2_reader.get_working_file(_PUBLIC_LOCATOR, "spatial.csv")
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
            r2_reader.get_working_file(_TEAM_LOCATOR, "spatial.csv")
        assert excinfo.value.status_code == 404
        assert excinfo.value.detail == "not_found"

    @pytest.mark.parametrize(
        "bad_locator",
        [
            None,  # NULL r2_rel_key (row matched no known ingestion layout)
            "",
            "pds:urn:nasa:pds:mars2020_imgops:data_aci_imgops:x::1.0",
            "loupe/sol_1/../../../etc/passwd/img/aci.PNG",  # traversal
            "img/a.PNG",  # too shallow for a <workspace>/img/<file> tree
        ],
    )
    def test_get_working_file_bad_locator_500(self, r2_team_client, bad_locator):
        """Unusable locator → misconfigured_path 500 BEFORE any R2 call.

        Covers the NULL-locator row (pre-backfill / unknown-layout), the
        unresolved ``pds:`` sentinel, path traversal, and locators too
        shallow to contain a Loupe workspace — spec §3.9.4 / §3.9.8.3.
        """
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.get_working_file(bad_locator, "spatial.csv")
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
        workspace-object reader.
        """
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.get_working_file(_TEAM_LOCATOR, disallowed_filename)
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "misconfigured_path"

    def test_get_working_file_cross_tier_403_502(self):
        """boto3 raises AccessDenied → resolver 502 upstream_credential_error.

        Simulates the §3.9.5 negative: a team container misconfigured
        with public-tier (slot 3) credentials attempts a team-bucket GET.
        With locator-derived keys, cross-tier misconfiguration is caught
        entirely credential-side — exactly this 403 → 502 mapping.
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
                r2_reader.get_working_file(_TEAM_LOCATOR, "spatial.csv")
            assert excinfo.value.status_code == 502
            assert excinfo.value.detail == "upstream_credential_error"
        finally:
            r2_reader.reset_r2_client_for_tests()

    def test_derive_workspace_key_team(self):
        """derive_workspace_key returns the canonical R2 key for a team locator."""
        key = r2_reader.derive_workspace_key(_TEAM_LOCATOR, "spatial.csv")
        assert key == _TEAM_SPATIAL_KEY

    def test_derive_workspace_key_public(self):
        """derive_workspace_key returns the canonical R2 key for a public locator."""
        key = r2_reader.derive_workspace_key(_PUBLIC_LOCATOR, "spatial.csv")
        assert key == _PUBLIC_SPATIAL_KEY

    def test_derive_workspace_key_rejects_disallowed_filename(self):
        """derive_workspace_key applies the same WORKSPACE_FILENAMES allowlist."""
        with pytest.raises(HTTPException) as excinfo:
            r2_reader.derive_workspace_key(_TEAM_LOCATOR, "other.csv")
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "misconfigured_path"

    def test_get_working_file_timeout_504(self):
        """boto3 ReadTimeoutError → 504 upstream_timeout per spec §3.9.4."""
        client_mock = MagicMock()
        client_mock.get_object.side_effect = ReadTimeoutError(endpoint_url="https://moto")
        r2_reader.set_r2_client_for_tests(client_mock, "team")
        try:
            with pytest.raises(HTTPException) as excinfo:
                r2_reader.get_working_file(_TEAM_LOCATOR, "spatial.csv")
            assert excinfo.value.status_code == 504
            assert excinfo.value.detail == "upstream_timeout"
        finally:
            r2_reader.reset_r2_client_for_tests()


# ---------------------------------------------------------------------------
# is_r2_mode — env-var pre-check (used by route layer to branch R2 vs FS)
# ---------------------------------------------------------------------------

class TestIsR2Mode:
    """is_r2_mode() — spec-compliant predicate.

    The predicate gates FS-fallback vs R2-routing in the route layer.
    The FS fallback is permitted
    ONLY when PHASE_TIER is unset (legacy local-filesystem dev).
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

        If PHASE_TIER is present but invalid, route through the R2
        resolver/config path so get_r2_client_and_config() raises
        tier_unset.
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

        Per spec §3.9.4: an env file that exports ``PHASE_TIER=`` (empty
        string) is a production misconfiguration.
        Treating empty as "unset" would silently select FS fallback —
        which the production container has no mount for. Empty string
        is NOT in {team, public} so the §3.9.4 row applies; the predicate
        routes to R2 so ``get_r2_client_and_config`` raises tier_unset.
        """
        monkeypatch.setenv("PHASE_TIER", "")
        assert r2_reader.is_r2_mode() is True


class TestColorizeSolSegment:
    """Pure ``sol_NNNN → sol_NNNN_colorized`` swap (issue #8).

    Shared by ``find_colorized_key`` (image key) and the coordinate
    resolver (workspace spatial.csv/loupe.csv locator), so it must swap
    only the bare ``sol_NNNN`` segment regardless of the surrounding path.
    """

    def test_swaps_first_bare_sol_segment(self):
        from sherloc_pipeline.core.r2_keys import colorize_sol_segment

        assert (
            colorize_sol_segment("sherloc-aci/loupe/sol_1213/ws/img/a.PNG")
            == "sherloc-aci/loupe/sol_1213_colorized/ws/img/a.PNG"
        )

    def test_swaps_in_relative_locator(self):
        from sherloc_pipeline.core.r2_keys import colorize_sol_segment

        assert (
            colorize_sol_segment(
                "loupe/sol_0921/d1/ws/img/SC3_0921_T.PNG"
            )
            == "loupe/sol_0921_colorized/d1/ws/img/SC3_0921_T.PNG"
        )

    def test_does_not_touch_sol_substring_in_filename(self):
        """``_0921`` inside a product filename is NOT a bare sol segment."""
        from sherloc_pipeline.core.r2_keys import colorize_sol_segment

        out = colorize_sol_segment("x/sol_5/SC3_0921_sol_thing.PNG")
        # Only the standalone ``sol_5`` segment is rewritten.
        assert out == "x/sol_5_colorized/SC3_0921_sol_thing.PNG"

    def test_returns_none_when_no_sol_segment(self):
        from sherloc_pipeline.core.r2_keys import colorize_sol_segment

        assert colorize_sol_segment("/data/foo/bar/baz.PNG") is None
        assert colorize_sol_segment("sherloc-aci/loupe/solitary/x.PNG") is None

    def test_swaps_only_first_sol_segment(self):
        from sherloc_pipeline.core.r2_keys import colorize_sol_segment

        # Defensive: a path with two sol segments rewrites only the first
        # (mirrors find_colorized_key's first-match behavior).
        assert (
            colorize_sol_segment("a/sol_1/b/sol_2/c")
            == "a/sol_1_colorized/b/sol_2/c"
        )


class TestDeriveRelLocator:
    """Structural locator derivation for go-forward ingestion writers.

    ``derive_rel_locator`` anchors on the ``sol_NNNN`` segment so the
    locator is derivable from the path shape regardless of which machine
    or mount ingestion read from. The cases mirror every file_path shape
    observed in the production databases at migration time.
    """

    def test_canonical_loupe_tree(self):
        from sherloc_pipeline.core.r2_keys import derive_rel_locator

        assert derive_rel_locator(
            "/data/sherloc/data/loupe/sol_0921/detail_1/ws/img/a.PNG"
        ) == "loupe/sol_0921/detail_1/ws/img/a.PNG"

    def test_legacy_nas_loupe_tree(self):
        from sherloc_pipeline.core.r2_keys import derive_rel_locator

        assert derive_rel_locator(
            "/nas/000_sherloc/data/loupe/sol_1810/line_1/ws/img/b.PNG"
        ) == "loupe/sol_1810/line_1/ws/img/b.PNG"

    def test_relocated_loupe_tree_still_derives(self):
        """Any mount point works — the anchor is structural, not a prefix."""
        from sherloc_pipeline.core.r2_keys import derive_rel_locator

        assert derive_rel_locator(
            "/mnt/elsewhere/loupe/sol_0100/survey_1/ws/img/c.PNG"
        ) == "loupe/sol_0100/survey_1/ws/img/c.PNG"

    def test_colorized_loupe_tree(self):
        from sherloc_pipeline.core.r2_keys import derive_rel_locator

        assert derive_rel_locator(
            "/data/sherloc/data/loupe/sol_1213_colorized/ws/img/d.PNG"
        ) == "loupe/sol_1213_colorized/ws/img/d.PNG"

    def test_pds_cache_tree(self):
        from sherloc_pipeline.core.r2_keys import derive_rel_locator

        assert derive_rel_locator(
            "/data/sherloc/pds/sol_0712/data_aci/e.IMG"
        ) == "sol_0712/data_aci/e.IMG"

    def test_unknown_layout_returns_none(self):
        from sherloc_pipeline.core.r2_keys import derive_rel_locator

        assert derive_rel_locator("/somewhere/else/entirely/f.PNG") is None
        # sol segment present but neither loupe- nor data_aci-shaped
        assert derive_rel_locator("/backup/sol_0921/raw/g.PNG") is None
