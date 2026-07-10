"""Tests for ``core.r2_keys.resolve_disk_path`` — the disk-edge inverse.

``resolve_disk_path`` maps a stored relative locator
(``context_images.r2_rel_key``) back to an on-disk path, prepending the
deployment data root instead of the R2 key prefix. It is the
processing-side counterpart to ``derive_r2_key`` (the R2 edge) and the
inverse of ``derive_rel_locator`` (ingestion-time derivation).

Two tiers, two roots:

- Loupe workspace tree (team): ``loupe/sol_NNNN/…`` → ``<data_root>/…``
- PDS ACI cache tree (public): ``sol_NNNN/data_aci/…`` → ``<pds_cache_dir>/…``

Guards mirror ``derive_r2_key``'s traversal/absolute rejection; unknown
shapes, ``pds:`` sentinels, NULL locators, and unmounted (None) roots all
resolve to ``None`` so the caller raises its own missing-file-equivalent
error.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from sherloc_pipeline.core.r2_keys import derive_rel_locator, resolve_disk_path

# Representative live-shape pairs (verified 100% coverage on both prod DBs):
# an absolute file_path, its stored locator, and the deployment root the
# locator hangs off. On live pop-os data_root=/data/sherloc/data and
# pds.cache_dir=/data/sherloc/pds.
_DATA_ROOT = "/data/sherloc/data"
_PDS_CACHE_DIR = "/data/sherloc/pds"

_TEAM_FILE_PATH = (
    "/data/sherloc/data/loupe/sol_0059/algan_335/"
    "SrlcSpecSpecSohRaw_0672350000-16000-1_Loupe_working/img/SC3_0059_X.PNG"
)
_TEAM_LOCATOR = (
    "loupe/sol_0059/algan_335/"
    "SrlcSpecSpecSohRaw_0672350000-16000-1_Loupe_working/img/SC3_0059_X.PNG"
)

_PUBLIC_FILE_PATH = "/data/sherloc/pds/sol_0059/data_aci/SC3_0059_X.IMG"
_PUBLIC_LOCATOR = "sol_0059/data_aci/SC3_0059_X.IMG"


class TestLoupeShape:
    def test_loupe_locator_resolves_under_data_root(self):
        got = resolve_disk_path(
            _TEAM_LOCATOR, data_root=_DATA_ROOT, pds_cache_dir=_PDS_CACHE_DIR
        )
        assert got == Path(_TEAM_FILE_PATH)

    def test_loupe_colorized_variant_resolves_under_data_root(self):
        # core.coordinates colorizes the *locator* before resolving; the
        # sol_NNNN anchor accepts the _colorized variant.
        colorized = _TEAM_LOCATOR.replace("sol_0059", "sol_0059_colorized")
        got = resolve_disk_path(
            colorized, data_root=_DATA_ROOT, pds_cache_dir=_PDS_CACHE_DIR
        )
        assert got == Path(_DATA_ROOT) / PurePosixPath(colorized)

    def test_loupe_none_data_root_returns_none(self):
        assert (
            resolve_disk_path(
                _TEAM_LOCATOR, data_root=None, pds_cache_dir=_PDS_CACHE_DIR
            )
            is None
        )

    def test_data_root_accepts_path_object(self):
        got = resolve_disk_path(
            _TEAM_LOCATOR, data_root=Path(_DATA_ROOT), pds_cache_dir=None
        )
        assert got == Path(_TEAM_FILE_PATH)


class TestPdsCacheShape:
    def test_pds_cache_locator_resolves_under_cache_dir(self):
        got = resolve_disk_path(
            _PUBLIC_LOCATOR, data_root=_DATA_ROOT, pds_cache_dir=_PDS_CACHE_DIR
        )
        assert got == Path(_PUBLIC_FILE_PATH)

    def test_pds_cache_none_cache_dir_returns_none(self):
        assert (
            resolve_disk_path(
                _PUBLIC_LOCATOR, data_root=_DATA_ROOT, pds_cache_dir=None
            )
            is None
        )


class TestSentinelAndNullAndUnknown:
    def test_pds_scheme_sentinel_returns_none(self):
        # Unresolved PDS reference: not yet downloaded — caller branches on
        # the pds: scheme exactly as before.
        assert (
            resolve_disk_path(
                "pds:urn:nasa:pds:mars2020_imgops:data_aci_imgops:x::1.0",
                data_root=_DATA_ROOT,
                pds_cache_dir=_PDS_CACHE_DIR,
            )
            is None
        )

    @pytest.mark.parametrize("locator", [None, ""])
    def test_missing_locator_returns_none(self, locator):
        assert (
            resolve_disk_path(
                locator, data_root=_DATA_ROOT, pds_cache_dir=_PDS_CACHE_DIR
            )
            is None
        )

    @pytest.mark.parametrize(
        "locator",
        [
            "loupe",  # too shallow (no sol segment)
            "sol_0059",  # bare sol, no data_aci / not under loupe
            "sol_0059/img/x.PNG",  # public-ish but not data_aci
            "loupe/detail_1/img/x.PNG",  # loupe but 2nd segment not sol_NNNN
            "results/sol_0059/x.csv",  # neither tree
            "sol_59x/data_aci/x.IMG",  # sol anchor must be sol_\d+
        ],
    )
    def test_unknown_shape_returns_none(self, locator):
        assert (
            resolve_disk_path(
                locator, data_root=_DATA_ROOT, pds_cache_dir=_PDS_CACHE_DIR
            )
            is None
        )


class TestTraversalGuard:
    @pytest.mark.parametrize(
        "locator",
        [
            "loupe/sol_0059/../../../etc/passwd",  # parent traversal
            "/data/sherloc/data/loupe/sol_0059/img/x.PNG",  # absolute
            "loupe\\sol_0059\\img\\x.PNG",  # backslash
            "sol_0059/data_aci/../../../etc/passwd",  # traversal, pds tree
        ],
    )
    def test_traversal_or_absolute_returns_none(self, locator):
        # Mirrors derive_r2_key's misconfigured_path rejection: a locator
        # serving would reject must never resolve to a disk path.
        assert (
            resolve_disk_path(
                locator, data_root=_DATA_ROOT, pds_cache_dir=_PDS_CACHE_DIR
            )
            is None
        )


class TestRoundTripAgainstLiveShapes:
    """resolver(locator) reproduces the original absolute path, and the
    ingestion-time derive_rel_locator(file_path) reproduces the locator —
    the two edges compose to identity for both live tier shapes."""

    @pytest.mark.parametrize(
        "file_path, locator, data_root, pds_cache_dir",
        [
            (_TEAM_FILE_PATH, _TEAM_LOCATOR, _DATA_ROOT, _PDS_CACHE_DIR),
            (_PUBLIC_FILE_PATH, _PUBLIC_LOCATOR, _DATA_ROOT, _PDS_CACHE_DIR),
        ],
    )
    def test_resolve_reproduces_absolute_path(
        self, file_path, locator, data_root, pds_cache_dir
    ):
        resolved = resolve_disk_path(
            locator, data_root=data_root, pds_cache_dir=pds_cache_dir
        )
        assert str(resolved) == file_path

    @pytest.mark.parametrize(
        "file_path, locator",
        [
            (_TEAM_FILE_PATH, _TEAM_LOCATOR),
            (_PUBLIC_FILE_PATH, _PUBLIC_LOCATOR),
        ],
    )
    def test_derive_then_resolve_is_identity(self, file_path, locator):
        # ingestion edge: file_path → locator
        assert derive_rel_locator(file_path) == locator
        # disk edge: locator → file_path (with the matching root)
        resolved = resolve_disk_path(
            locator, data_root=_DATA_ROOT, pds_cache_dir=_PDS_CACHE_DIR
        )
        assert str(resolved) == file_path
