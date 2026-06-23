"""
Tests for the cosmic_ray_masks satellite table (ML despike, spec §4.4).

Covers (per MLD-SYS-008/009, MLD-IFC-005):
- ORM <-> pydantic round-trip equality (incl. exact channel_indices).
- Pydantic validation: range / sorted / duplicate / n_flagged mismatch /
  extra-field rejection.
- DB-level FK CASCADE from scan and from spectrum deletion.
- Unique constraint on (spectrum_id, method).
- Re-ingest CASCADE leaves zero orphaned mask rows (MLD-SYS-009 AC1).
"""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from sherloc_pipeline.database.connection import (
    get_engine,
    get_session,
    create_all_tables,
)
from sherloc_pipeline.database.models import (
    SolORM,
    ScanORM,
    ScanPointORM,
    SpectrumORM,
    CosmicRayMaskORM,
)
from sherloc_pipeline.models import (
    CosmicRayMask,
    Spectrum,
)
from sherloc_pipeline.services.ingestion import IngestionService


# Provenance constants shared across tests.
ML_METHOD = "ml_v1.3_tau_matched"
MODEL_SHA = "9668a0b2ca257ce333d57e3f76598dda8cb5c1839e2fde6bd955086d959be0ba"
R1_TAU = 0.29882812500038747


@pytest.fixture
def engine():
    """Create an in-memory SQLite database for testing."""
    eng = get_engine(":memory:")
    create_all_tables(eng)
    return eng


@pytest.fixture
def session(engine):
    """Create a database session for testing."""
    with get_session(engine) as sess:
        yield sess


def _make_spectrum_chain(session, *, region="R1", spectrum_type="dark_subtracted"):
    """Create a sol -> scan -> scan_point -> spectrum chain.

    Returns the spectrum id (str). Caller may flush/commit as needed.
    """
    sol_number = 921
    if session.query(SolORM).filter_by(sol_number=sol_number).first() is None:
        session.add(SolORM(sol_number=sol_number, data_source="loupe"))
        session.flush()

    scan_id = str(uuid.uuid4())
    session.add(
        ScanORM(
            id=scan_id,
            sol_number=sol_number,
            scan_name="cr_mask_test",
            scan_id=f"scan-{scan_id}",
            sclk_start=100000,
            n_points=1,
            n_channels=2148,
            shots_per_point=10,
            target_type="rock",
        )
    )
    session.flush()

    point_id = str(uuid.uuid4())
    session.add(
        ScanPointORM(id=point_id, scan_id=scan_id, point_index=0)
    )
    session.flush()

    spectrum_id = str(uuid.uuid4())
    session.add(
        SpectrumORM(
            id=spectrum_id,
            scan_point_id=point_id,
            region=region,
            spectrum_type=spectrum_type,
            processing_level="normalized",
            intensities=Spectrum.compress_array([1.0, 2.0, 3.0]),
        )
    )
    session.flush()
    return scan_id, spectrum_id


# ---------------------------------------------------------------------------
# (a) ORM <-> pydantic round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_orm_pydantic_round_trip(self):
        """from_pydantic -> to_pydantic preserves every field exactly."""
        spectrum_id = uuid.uuid4()
        channels = [52, 120, 405, 511, 574]
        mask = CosmicRayMask(
            spectrum_id=spectrum_id,
            method=ML_METHOD,
            model_sha256=MODEL_SHA,
            tau=R1_TAU,
            channel_indices=channels,
            n_flagged=len(channels),
        )

        orm = CosmicRayMaskORM.from_pydantic(mask)
        back = orm.to_pydantic()

        assert back.id == mask.id
        assert back.spectrum_id == spectrum_id
        assert back.method == ML_METHOD
        assert back.model_sha256 == MODEL_SHA
        assert back.tau == pytest.approx(R1_TAU)
        assert back.channel_indices == channels  # exact list
        assert back.n_flagged == len(channels)
        # ORM has no updated_at column; round-trip leaves it None.
        assert back.updated_at is None
        assert back.created_at == mask.created_at

    def test_round_trip_empty_mask(self):
        """A zero-flag mask round-trips (n_flagged == 0)."""
        mask = CosmicRayMask(
            spectrum_id=uuid.uuid4(),
            method=ML_METHOD,
            model_sha256=MODEL_SHA,
            tau=R1_TAU,
            channel_indices=[],
            n_flagged=0,
        )
        back = CosmicRayMaskORM.from_pydantic(mask).to_pydantic()
        assert back.channel_indices == []
        assert back.n_flagged == 0


# ---------------------------------------------------------------------------
# (b) Pydantic validation
# ---------------------------------------------------------------------------


class TestPydanticValidation:
    def _kwargs(self, **overrides):
        base = dict(
            spectrum_id=uuid.uuid4(),
            method=ML_METHOD,
            model_sha256=MODEL_SHA,
            tau=R1_TAU,
            channel_indices=[10, 20, 30],
            n_flagged=3,
        )
        base.update(overrides)
        return base

    def test_unsorted_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            CosmicRayMask(**self._kwargs(channel_indices=[30, 10, 20]))

    def test_duplicates_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            CosmicRayMask(**self._kwargs(channel_indices=[10, 10, 30]))

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            CosmicRayMask(**self._kwargs(channel_indices=[-1, 10, 30]))

    def test_out_of_range_high_rejected(self):
        # 2148 is the first invalid absolute index (plane is 0..2147).
        with pytest.raises(ValueError, match="out of range"):
            CosmicRayMask(
                **self._kwargs(channel_indices=[10, 20, 2148], n_flagged=3)
            )

    def test_max_valid_index_accepted(self):
        mask = CosmicRayMask(
            **self._kwargs(channel_indices=[10, 20, 2147], n_flagged=3)
        )
        assert mask.channel_indices[-1] == 2147

    def test_n_flagged_mismatch_rejected(self):
        with pytest.raises(ValueError, match="n_flagged"):
            CosmicRayMask(**self._kwargs(channel_indices=[10, 20, 30], n_flagged=2))

    def test_extra_field_rejected(self):
        with pytest.raises(ValueError):
            CosmicRayMask(**self._kwargs(unexpected="x"))

    def test_short_sha_rejected(self):
        with pytest.raises(ValueError):
            CosmicRayMask(**self._kwargs(model_sha256="abc"))

    def test_long_method_rejected(self):
        with pytest.raises(ValueError):
            CosmicRayMask(**self._kwargs(method="m" * 41))


# ---------------------------------------------------------------------------
# (c) DB-level FK CASCADE
# ---------------------------------------------------------------------------


class TestForeignKeyCascade:
    def test_scan_delete_cascades_to_masks(self, session):
        """Deleting the SCAN cascades scan->point->spectrum->mask."""
        scan_id, spectrum_id = _make_spectrum_chain(session)
        session.add(
            CosmicRayMaskORM.from_pydantic(
                CosmicRayMask(
                    spectrum_id=uuid.UUID(spectrum_id),
                    method=ML_METHOD,
                    model_sha256=MODEL_SHA,
                    tau=R1_TAU,
                    channel_indices=[100, 200],
                    n_flagged=2,
                )
            )
        )
        session.flush()
        assert session.query(CosmicRayMaskORM).count() == 1

        scan = session.query(ScanORM).filter_by(id=scan_id).first()
        session.delete(scan)
        session.flush()

        assert session.query(CosmicRayMaskORM).count() == 0

    def test_spectrum_delete_cascades_to_masks(self, session):
        """Deleting the parent spectrum directly cascades to its masks."""
        _, spectrum_id = _make_spectrum_chain(session)
        session.add(
            CosmicRayMaskORM.from_pydantic(
                CosmicRayMask(
                    spectrum_id=uuid.UUID(spectrum_id),
                    method=ML_METHOD,
                    model_sha256=MODEL_SHA,
                    tau=R1_TAU,
                    channel_indices=[5],
                    n_flagged=1,
                )
            )
        )
        session.flush()
        assert session.query(CosmicRayMaskORM).count() == 1

        spectrum = session.query(SpectrumORM).filter_by(id=spectrum_id).first()
        session.delete(spectrum)
        session.flush()

        assert session.query(CosmicRayMaskORM).count() == 0


# ---------------------------------------------------------------------------
# (d) Unique constraint
# ---------------------------------------------------------------------------


class TestUniqueConstraint:
    def test_duplicate_spectrum_method_raises(self, session):
        _, spectrum_id = _make_spectrum_chain(session)
        for _ in range(2):
            session.add(
                CosmicRayMaskORM(
                    id=str(uuid.uuid4()),
                    spectrum_id=spectrum_id,
                    method=ML_METHOD,
                    model_sha256=MODEL_SHA,
                    tau=R1_TAU,
                    channel_indices=[1, 2],
                    n_flagged=2,
                )
            )
        with pytest.raises(IntegrityError):
            session.flush()
        # Leave the session clean so the get_session context manager's
        # commit-on-exit does not trip PendingRollbackError.
        session.rollback()

    def test_different_methods_coexist(self, session):
        _, spectrum_id = _make_spectrum_chain(session)
        session.add(
            CosmicRayMaskORM(
                id=str(uuid.uuid4()),
                spectrum_id=spectrum_id,
                method=ML_METHOD,
                model_sha256=MODEL_SHA,
                tau=R1_TAU,
                channel_indices=[1, 2],
                n_flagged=2,
            )
        )
        session.add(
            CosmicRayMaskORM(
                id=str(uuid.uuid4()),
                spectrum_id=spectrum_id,
                method="modz",
                model_sha256=MODEL_SHA,
                tau=R1_TAU,
                channel_indices=[3, 4],
                n_flagged=2,
            )
        )
        session.flush()
        assert (
            session.query(CosmicRayMaskORM)
            .filter_by(spectrum_id=spectrum_id)
            .count()
            == 2
        )


# ---------------------------------------------------------------------------
# (e) Re-ingest CASCADE (MLD-SYS-009 AC1)
# ---------------------------------------------------------------------------


class TestReingestCascade:
    WORKSPACE_REL = (
        "loupe/sol_0921/detail_1/"
        "SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
    )

    def test_force_reingest_wipes_masks_no_orphans(self, fixtures_path, tmp_path):
        """Force re-ingest removes pre-existing masks; zero orphans remain."""
        workspace = fixtures_path / self.WORKSPACE_REL
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=True,
            ingestion_mode="all_regions",
        )

        # First ingest.
        result1 = service.ingest_workspace(workspace)
        assert result1.metadata["success"]
        assert result1.metadata["spectra_ingested"] > 0

        # Attach masks to a few DARK_SUBTRACTED spectra.
        with get_session(service.engine) as session:
            spectra = (
                session.query(SpectrumORM)
                .filter_by(spectrum_type="dark_subtracted")
                .limit(3)
                .all()
            )
            assert len(spectra) >= 1
            attached_ids = [s.id for s in spectra]
            for sp in spectra:
                session.add(
                    CosmicRayMaskORM(
                        id=str(uuid.uuid4()),
                        spectrum_id=sp.id,
                        method=ML_METHOD,
                        model_sha256=MODEL_SHA,
                        tau=R1_TAU,
                        channel_indices=[100, 200, 300],
                        n_flagged=3,
                    )
                )

        with get_session(service.engine) as session:
            assert session.query(CosmicRayMaskORM).count() == len(attached_ids)

        # Force re-ingest: derived data (incl. spectra) is wiped and
        # regenerated, so the masks attached to the old spectrum rows
        # must cascade away.
        result2 = service.ingest_workspace(workspace, force=True)
        assert result2.metadata["success"]

        with get_session(service.engine) as session:
            # All pre-existing mask rows are gone.
            for old_id in attached_ids:
                assert (
                    session.query(CosmicRayMaskORM)
                    .filter_by(spectrum_id=old_id)
                    .count()
                    == 0
                )

            # Referential-integrity: no mask references a missing spectrum.
            orphan_count = session.execute(
                text(
                    "SELECT COUNT(*) FROM cosmic_ray_masks "
                    "WHERE spectrum_id NOT IN (SELECT id FROM spectra)"
                )
            ).scalar()
            assert orphan_count == 0
