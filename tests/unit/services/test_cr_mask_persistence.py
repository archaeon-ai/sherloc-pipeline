"""CRMaskService persistence: round-trip, idempotency, skip path,
provenance equality (MLD-SYS-008, MLD-SYS-012 AC2, MLD-SYS-006 AC2 DB
leg).
"""

import pytest

from sherloc_pipeline.database.connection import get_session
from sherloc_pipeline.database.models import (
    CosmicRayMaskORM,
    ScanORM,
    ScanPointORM,
    SpectrumORM,
)
from sherloc_pipeline.services.cr_masks import CRMaskService
from sherloc_pipeline.services.errors import CRMaskError
from sherloc_pipeline.services.ingestion import IngestionService

WORKSPACE_REL = (
    "loupe/sol_0921/detail_1/SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
)

PROVENANCE = {
    "method": "ml_v1.3_tau_matched",
    "model_sha256": "ab" * 32,
    "tau": {
        "R1": 0.29882812500038747,
        "R2": 0.2656250000008831,
        "R3": 0.2656250000008831,
    },
    "ort_version": "1.26.0",
}


def _metadata(masks: dict) -> dict:
    n_flagged: dict = {}
    for region_masks in masks.values():
        for region, channels in region_masks.items():
            n_flagged[region] = n_flagged.get(region, 0) + len(channels)
    return {**PROVENANCE, "n_flagged": n_flagged, "masks": masks}


#: Three points; includes an empty mask (point 1 R1) and empty regions —
#: the stored set must reconstruct this dict exactly.
MASKS_3REGION = {
    "0": {"R1": [60, 100, 574], "R2": [600, 1200], "R3": [1700]},
    "1": {"R1": [], "R2": [800], "R3": []},
    "2": {"R1": [52], "R2": [], "R3": [2139]},
}

MASKS_R1_ONLY = {
    "0": {"R1": [60, 100]},
    "1": {"R1": []},
}


def _ingest(tmp_path, fixtures_path, mode: str):
    db_path = tmp_path / "test.db"
    service = IngestionService(
        database_path=db_path,
        include_spectra=True,
        ingestion_mode=mode,
    )
    service.ingest_workspace(fixtures_path / WORKSPACE_REL)
    # The directly-ingested fixture workspace has no .lpe-derived target;
    # set the DB value the way production ingestion would.
    with get_session(service.engine) as session:
        scan = session.query(ScanORM).first()
        scan.target = "Test Target"
    return db_path, service.engine


@pytest.fixture
def all_regions_db(tmp_path, fixtures_path):
    return _ingest(tmp_path, fixtures_path, "all_regions")


@pytest.fixture
def r1_only_db(tmp_path, fixtures_path):
    return _ingest(tmp_path, fixtures_path, "R1_only")


def _stored_masks(engine, method=None):
    """Reconstruct {point: {region: channels}} + provenance rows from DB."""
    out: dict = {}
    rows = []
    with get_session(engine) as session:
        query = (
            session.query(CosmicRayMaskORM, SpectrumORM, ScanPointORM)
            .join(SpectrumORM, CosmicRayMaskORM.spectrum_id == SpectrumORM.id)
            .join(ScanPointORM, SpectrumORM.scan_point_id == ScanPointORM.id)
        )
        if method:
            query = query.filter(CosmicRayMaskORM.method == method)
        for mask, spectrum, point in query.all():
            out.setdefault(str(point.point_index), {})[spectrum.region] = list(
                mask.channel_indices
            )
            rows.append(
                {
                    "region": spectrum.region,
                    "method": mask.method,
                    "model_sha256": mask.model_sha256,
                    "tau": mask.tau,
                    "n_flagged": mask.n_flagged,
                }
            )
    return out, rows


class TestRoundTrip:
    def test_exact_mask_round_trip(self, all_regions_db, tmp_path):
        """MLD-SYS-008 AC1/AC2: every (point, region) mask — including
        empty ones — reconstructs exactly from the stored records."""
        db_path, engine = all_regions_db
        result = CRMaskService().persist_masks(
            sol="0921",
            target="Test_Target",
            scan="detail_1",
            despike_metadata=_metadata(MASKS_3REGION),
            database_path=db_path,
        )
        assert result.metadata["masks_inserted"] == 9
        assert result.metadata["regions_skipped"] == 0

        stored, rows = _stored_masks(engine)
        assert stored == {
            point: dict(region_masks)
            for point, region_masks in MASKS_3REGION.items()
        }

    def test_provenance_equality_on_rows(self, all_regions_db):
        """MLD-SYS-012 AC2: identity/digest/threshold equal the
        ServiceResult-level provenance on every persisted row."""
        db_path, engine = all_regions_db
        metadata = _metadata(MASKS_3REGION)
        CRMaskService().persist_masks(
            sol="0921",
            target="Test_Target",
            scan="detail_1",
            despike_metadata=metadata,
            database_path=db_path,
        )
        _, rows = _stored_masks(engine)
        assert rows
        for row in rows:
            assert row["method"] == metadata["method"]
            assert row["model_sha256"] == metadata["model_sha256"]
            assert row["tau"] == metadata["tau"][row["region"]]

    def test_n_flagged_matches_channel_count(self, all_regions_db):
        db_path, engine = all_regions_db
        CRMaskService().persist_masks(
            sol="0921",
            target="Test_Target",
            scan="detail_1",
            despike_metadata=_metadata(MASKS_3REGION),
            database_path=db_path,
        )
        with get_session(engine) as session:
            for mask in session.query(CosmicRayMaskORM).all():
                assert mask.n_flagged == len(mask.channel_indices)

    def test_get_masks_for_spectra_read_path(self, all_regions_db):
        db_path, engine = all_regions_db
        CRMaskService().persist_masks(
            sol="0921",
            target="Test_Target",
            scan="detail_1",
            despike_metadata=_metadata(MASKS_3REGION),
            database_path=db_path,
        )
        with get_session(engine) as session:
            spectrum_ids = [
                row.spectrum_id
                for row in session.query(CosmicRayMaskORM).all()
            ]
            masks = CRMaskService.get_masks_for_spectra(
                session, spectrum_ids, method="ml_v1.3_tau_matched"
            )
            assert set(masks) == set(spectrum_ids)
            # Pydantic domain models with validated invariants
            for mask_list in masks.values():
                for mask in mask_list:
                    assert mask.n_flagged == len(mask.channel_indices)
            # Method filter excludes everything else
            assert (
                CRMaskService.get_masks_for_spectra(
                    session, spectrum_ids, method="other_method"
                )
                == {}
            )


class TestGetMasksForScan:
    """Scan-addressed read path for plot (MLD-IFC-007)."""

    def test_reconstructs_masks_by_region_and_point(self, all_regions_db):
        db_path, _ = all_regions_db
        CRMaskService().persist_masks(
            sol="0921",
            target="Test_Target",
            scan="detail_1",
            despike_metadata=_metadata(MASKS_3REGION),
            database_path=db_path,
        )
        masks = CRMaskService.get_masks_for_scan(
            sol="0921", target="Test_Target", scan="detail_1",
            database_path=db_path,
        )
        expected: dict = {}
        for point_key, region_masks in MASKS_3REGION.items():
            for region, channels in region_masks.items():
                expected.setdefault(region, {})[int(point_key)] = list(channels)
        assert masks == expected
        # Empty stored masks still count as "screened" coverage
        assert masks["R1"][1] == []

    def test_missing_database_returns_empty(self, tmp_path):
        masks = CRMaskService.get_masks_for_scan(
            sol="0921", target="Test_Target", scan="detail_1",
            database_path=tmp_path / "absent.db",
        )
        assert masks == {}

    def test_premigration_database_returns_empty(self, tmp_path):
        """A database from before the cosmic_ray_masks migration must read
        as "no stored masks", never error (MLD-IFC-007) — the default
        SHERLOC_DB_PATH may point at such a file during upgrades."""
        import sqlite3

        db_path = tmp_path / "old.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE scans (id TEXT PRIMARY KEY)")
        masks = CRMaskService.get_masks_for_scan(
            sol="0921", target="Test_Target", scan="detail_1",
            database_path=db_path,
        )
        assert masks == {}

    def test_non_database_file_returns_empty(self, tmp_path):
        """An unrelated file at the database path reads as no masks."""
        db_path = tmp_path / "not_a.db"
        db_path.write_bytes(b"this is not a sqlite database")
        masks = CRMaskService.get_masks_for_scan(
            sol="0921", target="Test_Target", scan="detail_1",
            database_path=db_path,
        )
        assert masks == {}

    def test_unknown_scan_returns_empty(self, all_regions_db):
        db_path, _ = all_regions_db
        masks = CRMaskService.get_masks_for_scan(
            sol="0921", target="Test_Target", scan="line_9",
            database_path=db_path,
        )
        assert masks == {}

    def test_method_filter(self, all_regions_db):
        db_path, _ = all_regions_db
        CRMaskService().persist_masks(
            sol="0921", target="Test_Target", scan="detail_1",
            despike_metadata=_metadata(MASKS_3REGION), database_path=db_path,
        )
        assert (
            CRMaskService.get_masks_for_scan(
                sol="0921", target="Test_Target", scan="detail_1",
                database_path=db_path, method="other_method",
            )
            == {}
        )
        assert CRMaskService.get_masks_for_scan(
            sol="0921", target="Test_Target", scan="detail_1",
            database_path=db_path, method="ml_v1.3_tau_matched",
        )


class TestIdempotency:
    def test_re_run_replaces_rows(self, all_regions_db):
        """Delete-then-insert per (spectrum, method): same row count, no
        IntegrityError, fresh channel lists win."""
        db_path, engine = all_regions_db
        service = CRMaskService()
        service.persist_masks(
            sol="0921", target="Test_Target", scan="detail_1",
            despike_metadata=_metadata(MASKS_3REGION), database_path=db_path,
        )
        modified = {
            point: {region: list(channels) for region, channels in rm.items()}
            for point, rm in MASKS_3REGION.items()
        }
        modified["0"]["R1"] = [55, 56]
        result = service.persist_masks(
            sol="0921", target="Test_Target", scan="detail_1",
            despike_metadata=_metadata(modified), database_path=db_path,
        )
        assert result.metadata["masks_inserted"] == 9

        stored, _ = _stored_masks(engine)
        assert stored["0"]["R1"] == [55, 56]
        with get_session(engine) as session:
            assert session.query(CosmicRayMaskORM).count() == 9


class TestSkipPath:
    def test_persistence_follows_parent_row_existence(self, r1_only_db):
        """MLD-SYS-008 AC3: full-region masks against an R1_only-ingested
        DB persist R1 only; R2/R3 are skipped with a count, never
        partially written."""
        db_path, engine = r1_only_db
        result = CRMaskService().persist_masks(
            sol="0921",
            target="Test_Target",
            scan="detail_1",
            despike_metadata=_metadata(MASKS_3REGION),
            database_path=db_path,
        )
        assert result.metadata["masks_inserted"] == 3  # R1 x 3 points
        assert result.metadata["regions_skipped"] == 6  # R2+R3 x 3 points

        stored, rows = _stored_masks(engine)
        assert {row["region"] for row in rows} == {"R1"}

    def test_r1_alone_metadata_produces_r1_rows_only(self, all_regions_db):
        """MLD-SYS-006 AC2 (DB leg): an R1-alone run's metadata yields
        zero R2/R3 mask rows even when parent rows exist for them."""
        db_path, engine = all_regions_db
        CRMaskService().persist_masks(
            sol="0921",
            target="Test_Target",
            scan="detail_1",
            despike_metadata={
                **PROVENANCE,
                "tau": {"R1": PROVENANCE["tau"]["R1"]},
                "n_flagged": {"R1": 2},
                "masks": MASKS_R1_ONLY,
            },
            database_path=db_path,
        )
        _, rows = _stored_masks(engine)
        assert rows
        assert {row["region"] for row in rows} == {"R1"}


class TestFailLoud:
    def test_missing_database_raises(self, tmp_path):
        with pytest.raises(CRMaskError, match="Database not found"):
            CRMaskService().persist_masks(
                sol="0921", target="X", scan="detail_1",
                despike_metadata=_metadata(MASKS_R1_ONLY),
                database_path=tmp_path / "missing.db",
            )

    def test_missing_provenance_field_raises(self, all_regions_db):
        db_path, _ = all_regions_db
        incomplete = _metadata(MASKS_R1_ONLY)
        del incomplete["model_sha256"]
        with pytest.raises(CRMaskError, match="model_sha256"):
            CRMaskService().persist_masks(
                sol="0921", target="Test_Target", scan="detail_1",
                despike_metadata=incomplete, database_path=db_path,
            )

    def test_unknown_scan_raises(self, all_regions_db):
        db_path, _ = all_regions_db
        with pytest.raises(CRMaskError, match="Scan not found"):
            CRMaskService().persist_masks(
                sol="0852", target="Test_Target", scan="detail_9",
                despike_metadata=_metadata(MASKS_R1_ONLY),
                database_path=db_path,
            )

    @pytest.mark.parametrize(
        "bad_channels,reason",
        [
            ([100, 60], "unsorted"),
            ([60, 60], "duplicate"),
            ([60, 5000], "out of range"),
            ([-1, 60], "negative"),
        ],
    )
    def test_malformed_channel_indices_rejected_at_write_boundary(
        self, all_regions_db, bad_channels, reason
    ):
        """The pydantic domain model guards the write boundary: malformed
        metadata must never create rows violating the persistence
        contract (channel range/ordering invariants)."""
        db_path, engine = all_regions_db
        with pytest.raises(CRMaskError, match="Invalid mask"):
            CRMaskService().persist_masks(
                sol="0921", target="Test_Target", scan="detail_1",
                despike_metadata=_metadata({"0": {"R1": bad_channels}}),
                database_path=db_path,
            )
        with get_session(engine) as session:
            assert session.query(CosmicRayMaskORM).count() == 0
