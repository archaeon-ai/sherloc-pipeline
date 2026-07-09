"""End-to-end despike method dispatch through run_scan on the committed
sol_0921 fixture workspace.

- ``none`` equivalence: identical shared outputs to a modz
  run minus every despike artifact — the former despike_r1=False
  behavior under the unified selector.
- ``modz``: legacy artifact set unchanged (the byte-level golden anchor
  is the local slow suite).
- ``ml`` (stub model): full detect → replace → record → persist →
  round-trip chain.
"""

import json

import pytest

from sherloc_pipeline.database.connection import get_session
from sherloc_pipeline.database.models import (
    CosmicRayMaskORM,
    ScanORM,
    ScanPointORM,
    SpectrumORM,
)
from sherloc_pipeline.services.cr_masks import CRMaskService
from sherloc_pipeline.services.ingestion import IngestionService

WORKSPACE_REL = (
    "loupe/sol_0921/detail_1/SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
)
TARGET = "Despike_E2E"


def _run_scan(tmp_path, fixtures_path, method, subdir):
    from sherloc_pipeline.services.preprocessing import PreprocessingService

    results_dir = tmp_path / subdir
    service = PreprocessingService()
    result = service.run_scan(
        sol="0921",
        target=TARGET,
        scan="detail_1",
        data_dir=fixtures_path / "loupe",
        results_dir=results_dir,
        generate_plots=False,
        despike_method=method,
        baseline_r1=False,
    )
    return result, results_dir


def _relative_csvs(results_dir):
    return {
        p.relative_to(results_dir): p for p in results_dir.rglob("*.csv")
    }


class TestNoneEquivalence:
    def test_none_matches_modz_minus_despike_artifacts(
        self, tmp_path, fixtures_path
    ):
        """A `none` run produces no despike artifacts and
        every output it does produce is byte-identical to the modz run's
        counterpart (modz only ADDS despike artifacts)."""
        none_result, none_dir = _run_scan(
            tmp_path, fixtures_path, "none", "results_none"
        )
        modz_result, modz_dir = _run_scan(
            tmp_path, fixtures_path, "modz", "results_modz"
        )

        assert "despike" not in none_result.metadata
        assert "despike" not in modz_result.metadata  # modz records none

        none_csvs = _relative_csvs(none_dir)
        modz_csvs = _relative_csvs(modz_dir)

        assert not any(
            "despiked" in str(rel) for rel in none_csvs
        ), "none run must not produce despiked artifacts"
        assert any(
            "despiked" in str(rel) for rel in modz_csvs
        ), "modz run must produce the legacy despiked CSV"
        assert not (none_dir / TARGET).rglob("cr_masks.json") or not list(
            none_dir.rglob("cr_masks.json")
        )

        # Every CSV the none run produced exists in the modz run and is
        # byte-identical (despiking only adds artifacts; shared outputs
        # are computed from non-despiked frames in the legacy chain).
        assert none_csvs, "none run produced no CSV outputs?"
        for rel, none_path in none_csvs.items():
            assert rel in modz_csvs, f"missing from modz run: {rel}"
            assert none_path.read_bytes() == modz_csvs[rel].read_bytes(), (
                f"none/modz divergence in shared artifact: {rel}"
            )


class TestMlEndToEnd:
    @pytest.fixture
    def ingested_db(self, tmp_path, fixtures_path):
        db_path = tmp_path / "phase.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=True,
            ingestion_mode="all_regions",
        )
        service.ingest_workspace(fixtures_path / WORKSPACE_REL)
        with get_session(service.engine) as session:
            scan = session.query(ScanORM).first()
            scan.target = TARGET.replace("_", " ")
        return db_path, service.engine

    def test_ml_run_detect_replace_record_persist_roundtrip(
        self,
        tmp_path,
        fixtures_path,
        ingested_db,
        preprocessing_service_with_stub,
        stub_manifest,
    ):
        db_path, engine = ingested_db
        results_dir = tmp_path / "results_ml"

        result = preprocessing_service_with_stub.run_scan(
            sol="0921",
            target=TARGET,
            scan="detail_1",
            data_dir=fixtures_path / "loupe",
            results_dir=results_dir,
            generate_plots=False,
            despike_method="ml",
            baseline_r1=False,
        )

        # Record: run-level provenance
        despike_meta = result.metadata["despike"]
        assert despike_meta["method"] == "ml_v1.3_tau_matched"
        assert despike_meta["model_sha256"] == stub_manifest.sha256
        assert set(despike_meta["tau"]) == {"R1", "R2", "R3"}

        # Replace: legacy-named despiked CSV artifact exists
        despiked_csvs = [
            p for p in results_dir.rglob("*.csv") if "despiked" in p.name
        ]
        assert despiked_csvs, "ml run must write the despiked R1 CSV"

        # cr_masks.json artifact in the run's results dir
        cr_json = list(results_dir.rglob("cr_masks.json"))
        assert len(cr_json) == 1
        with open(cr_json[0]) as f:
            assert json.load(f) == despike_meta

        # Persist + round-trip equality
        persist = CRMaskService().persist_masks(
            sol="0921",
            target=TARGET,
            scan="detail_1",
            despike_metadata=despike_meta,
            database_path=db_path,
        )
        n_points = len(despike_meta["masks"])
        assert persist.metadata["masks_inserted"] == 3 * n_points
        assert persist.metadata["regions_skipped"] == 0

        stored: dict = {}
        with get_session(engine) as session:
            rows = (
                session.query(CosmicRayMaskORM, SpectrumORM, ScanPointORM)
                .join(
                    SpectrumORM,
                    CosmicRayMaskORM.spectrum_id == SpectrumORM.id,
                )
                .join(
                    ScanPointORM,
                    SpectrumORM.scan_point_id == ScanPointORM.id,
                )
                .all()
            )
            for mask, spectrum, point in rows:
                stored.setdefault(str(point.point_index), {})[
                    spectrum.region
                ] = list(mask.channel_indices)
                # Provenance equality on rows
                assert mask.method == despike_meta["method"]
                assert mask.model_sha256 == despike_meta["model_sha256"]
                assert mask.tau == despike_meta["tau"][spectrum.region]

        assert stored == despike_meta["masks"]


class TestPipelinePersistenceWiring:
    """Functional test of the production persistence path: masks flow
    from the preprocessing stage through run_full_pipeline's DB-gated
    step (spec §4.5) — not via a direct CRMaskService call.

    The fitting/spatial stages are orthogonal to this sub-phase and
    monkeypatched to no-ops; preprocessing, despike, metadata flow,
    gating, and persistence are the real production code path.
    """

    @pytest.fixture
    def quiet_heavy_stages(self, monkeypatch):
        from sherloc_pipeline.services.base import ServiceResult
        from sherloc_pipeline.services.fitting import FittingService
        from sherloc_pipeline.services.spatial import SpatialService

        def empty_result(*args, **kwargs):
            return ServiceResult(
                summary="stubbed", artifacts=[], warnings=[], metadata={}
            )

        for name in ("fit_minerals", "fit_organics", "fit_hydration",
                     "fit_averages"):
            monkeypatch.setattr(FittingService, name, empty_result)
        monkeypatch.setattr(SpatialService, "render_overlay", empty_result)

    @pytest.fixture
    def ingested_db(self, tmp_path, fixtures_path):
        db_path = tmp_path / "phase.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=True,
            ingestion_mode="all_regions",
        )
        service.ingest_workspace(fixtures_path / WORKSPACE_REL)
        with get_session(service.engine) as session:
            scan = session.query(ScanORM).first()
            scan.target = TARGET.replace("_", " ")
        return db_path, service.engine

    def _run_pipeline(self, tmp_path, fixtures_path):
        from sherloc_pipeline.services.pipeline import PipelineService

        return PipelineService().run_full_pipeline(
            sol="0921",
            target=TARGET,
            scan="detail_1",
            data_dir=fixtures_path / "loupe",
            results_dir=tmp_path / "results_pipeline",
        )

    def test_run_full_pipeline_persists_masks(
        self,
        tmp_path,
        fixtures_path,
        ingested_db,
        stub_detector_factory,
        quiet_heavy_stages,
        monkeypatch,
    ):
        from sherloc_pipeline.services.preprocessing import (
            PreprocessingService,
        )

        db_path, engine = ingested_db
        monkeypatch.setenv("SHERLOC_DB_PATH", str(db_path))
        monkeypatch.setattr(
            PreprocessingService,
            "_build_ml_detector",
            lambda self: stub_detector_factory(),
        )

        result = self._run_pipeline(tmp_path, fixtures_path)

        # Despike metadata flowed preprocessing -> pipeline metadata.
        despike_meta = result.metadata["preprocessing"]["despike"]
        assert despike_meta["method"] == "ml_v1.3_tau_matched"

        # The DB-gated step ran, recorded its stage metadata, and the
        # counts reflect the full 3-region x n-point mask set.
        cr_meta = result.metadata["cr_masks"]
        n_points = len(despike_meta["masks"])
        assert cr_meta["masks_inserted"] == 3 * n_points
        assert cr_meta["regions_skipped"] == 0

        with get_session(engine) as session:
            assert (
                session.query(CosmicRayMaskORM).count() == 3 * n_points
            )
            row = session.query(CosmicRayMaskORM).first()
            assert row.method == despike_meta["method"]
            assert row.model_sha256 == despike_meta["model_sha256"]

    def test_run_full_pipeline_skips_persistence_without_db(
        self,
        tmp_path,
        fixtures_path,
        stub_detector_factory,
        quiet_heavy_stages,
        monkeypatch,
    ):
        """No DB: the step debug-logs the skip; run still succeeds and
        despike metadata is retained in run artifacts (never lost)."""
        from sherloc_pipeline.services.preprocessing import (
            PreprocessingService,
        )

        monkeypatch.setenv(
            "SHERLOC_DB_PATH", str(tmp_path / "absent" / "phase.db")
        )
        monkeypatch.setattr(
            PreprocessingService,
            "_build_ml_detector",
            lambda self: stub_detector_factory(),
        )

        result = self._run_pipeline(tmp_path, fixtures_path)

        assert "cr_masks" not in result.metadata
        assert result.metadata["preprocessing"]["despike"]["masks"]
