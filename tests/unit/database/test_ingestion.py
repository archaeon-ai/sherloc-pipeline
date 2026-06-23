"""
Tests for IngestionService.

Tests cover:
- Workspace ingestion
- Sol ingestion
- Directory ingestion
- Idempotency (re-ingestion is no-op)
- Force mode (re-ingest overwrites)
- Database statistics
- Error handling
"""

import pytest
from pathlib import Path

from sherloc_pipeline.services.ingestion import IngestionService, IngestionError
from sherloc_pipeline.database import (
    get_engine,
    get_session,
    create_all_tables,
    SolORM,
    ScanORM,
    ScanPointORM,
    SpectrumORM,
    InstrumentStateORM,
    CCDConfigurationORM,
)


class TestIngestionServiceInit:
    """Tests for IngestionService initialization."""

    def test_init_creates_database(self, tmp_path):
        """Test that service creates database file."""
        db_path = tmp_path / "test.db"
        service = IngestionService(database_path=db_path)
        assert db_path.exists()

    def test_init_creates_tables(self, tmp_path):
        """Test that service creates all tables."""
        db_path = tmp_path / "test.db"
        service = IngestionService(database_path=db_path)

        from sqlalchemy import inspect
        inspector = inspect(service.engine)
        tables = inspector.get_table_names()

        assert "sols" in tables
        assert "scans" in tables
        assert "scan_points" in tables
        assert "spectra" in tables


class TestWorkspaceIngestion:
    """Tests for workspace-level ingestion."""

    def test_ingest_workspace(self, fixtures_path, tmp_path):
        """Test ingesting a single workspace."""
        workspace = (
            fixtures_path / "loupe" / "sol_0921" / "detail_1" /
            "SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
        )
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=False,  # Faster test
        )

        result = service.ingest_workspace(workspace)

        assert result.metadata["success"]
        assert result.metadata["scans_ingested"] == 1
        assert result.metadata["points_ingested"] > 0

        # Verify in database
        with get_session(service.engine) as session:
            scans = session.query(ScanORM).all()
            assert len(scans) == 1
            assert scans[0].sol_number == 921

    def test_ingest_workspace_with_spectra(self, fixtures_path, tmp_path):
        """Test ingesting workspace with spectra data."""
        workspace = (
            fixtures_path / "loupe" / "sol_0921" / "detail_1" /
            "SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
        )
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=True,
        )

        result = service.ingest_workspace(workspace)

        assert result.metadata["success"]
        assert result.metadata["spectra_ingested"] > 0

        # Verify spectra in database
        with get_session(service.engine) as session:
            spectra_count = session.query(SpectrumORM).count()
            assert spectra_count > 0

    def test_ingest_workspace_idempotent(self, fixtures_path, tmp_path):
        """Test that re-ingesting workspace is idempotent."""
        workspace = (
            fixtures_path / "loupe" / "sol_0921" / "detail_1" /
            "SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
        )
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=False,
        )

        # First ingestion
        result1 = service.ingest_workspace(workspace)
        assert result1.metadata["success"]
        assert result1.metadata["scans_ingested"] == 1

        # Second ingestion should skip
        result2 = service.ingest_workspace(workspace)
        assert result2.metadata["success"]
        assert "skipped" in result2.summary.lower()

        # Database should still have only one scan
        with get_session(service.engine) as session:
            scans = session.query(ScanORM).all()
            assert len(scans) == 1

    def test_ingest_workspace_force(self, fixtures_path, tmp_path):
        """Test force re-ingestion."""
        workspace = (
            fixtures_path / "loupe" / "sol_0921" / "detail_1" /
            "SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
        )
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=False,
        )

        # First ingestion
        result1 = service.ingest_workspace(workspace)
        assert result1.metadata["success"]

        # Force re-ingestion
        result2 = service.ingest_workspace(workspace, force=True)
        assert result2.metadata["success"]
        assert result2.metadata["scans_ingested"] == 1

    def test_ingest_workspace_not_found(self, tmp_path):
        """Test error when workspace not found."""
        db_path = tmp_path / "test.db"
        service = IngestionService(database_path=db_path)

        with pytest.raises(IngestionError) as exc_info:
            service.ingest_workspace(tmp_path / "nonexistent")

        assert "not found" in str(exc_info.value.message).lower()

    def test_ingest_workspace_invalid(self, tmp_path):
        """Test error when path is not a valid workspace."""
        db_path = tmp_path / "test.db"
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()

        service = IngestionService(database_path=db_path)

        with pytest.raises(IngestionError) as exc_info:
            service.ingest_workspace(invalid_dir)

        assert "loupe.csv" in str(exc_info.value.message).lower()


class TestSolIngestion:
    """Tests for sol-level ingestion."""

    def test_ingest_sol(self, fixtures_path, tmp_path):
        """Test ingesting a complete sol."""
        sol_dir = fixtures_path / "loupe" / "sol_0921"
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=False,
        )

        result = service.ingest_sol(sol_dir)

        assert result.metadata["success"]
        assert result.metadata["sol_number"] == 921
        assert result.metadata["scans_ingested"] >= 1

        # Verify sol exists in database
        with get_session(service.engine) as session:
            sol = session.get(SolORM, 921)
            assert sol is not None
            assert sol.data_source == "loupe"

    def test_ingest_sol_creates_related_models(self, fixtures_path, tmp_path):
        """Test that sol ingestion creates instrument state and CCD config."""
        sol_dir = fixtures_path / "loupe" / "sol_0921"
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=False,
        )

        result = service.ingest_sol(sol_dir)
        assert result.metadata["success"]

        with get_session(service.engine) as session:
            # Check instrument states
            states = session.query(InstrumentStateORM).all()
            assert len(states) >= 1

            # Check CCD configurations
            configs = session.query(CCDConfigurationORM).all()
            assert len(configs) >= 1

    def test_ingest_sol_idempotent(self, fixtures_path, tmp_path):
        """Test that re-ingesting sol is idempotent."""
        sol_dir = fixtures_path / "loupe" / "sol_0921"
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=False,
        )

        # First ingestion
        result1 = service.ingest_sol(sol_dir)
        assert result1.metadata["success"]

        # Second ingestion should skip
        result2 = service.ingest_sol(sol_dir)
        assert result2.metadata["success"]
        assert "skipped" in result2.summary.lower()

    def test_ingest_sol_not_found(self, tmp_path):
        """Test error when sol directory not found."""
        db_path = tmp_path / "test.db"
        service = IngestionService(database_path=db_path)

        with pytest.raises(IngestionError) as exc_info:
            service.ingest_sol(tmp_path / "sol_9999")

        assert "not found" in str(exc_info.value.message).lower()


class TestDirectoryIngestion:
    """Tests for directory-level ingestion."""

    def test_ingest_directory(self, fixtures_path, tmp_path):
        """Test ingesting entire Loupe directory."""
        loupe_dir = fixtures_path / "loupe"
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=False,
        )

        result = service.ingest_directory(loupe_dir)

        assert result.metadata["success"]
        assert result.metadata["sols_processed"] >= 1
        assert result.metadata["scans_ingested"] >= 1

    def test_ingest_directory_limit(self, fixtures_path, tmp_path):
        """Test directory ingestion with limit."""
        loupe_dir = fixtures_path / "loupe"
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=False,
        )

        result = service.ingest_directory(loupe_dir, limit=1)

        assert result.metadata["success"]
        assert result.metadata["sols_processed"] == 1

    def test_ingest_directory_not_found(self, tmp_path):
        """Test error when directory not found."""
        db_path = tmp_path / "test.db"
        service = IngestionService(database_path=db_path)

        with pytest.raises(IngestionError) as exc_info:
            service.ingest_directory(tmp_path / "nonexistent")

        assert "not found" in str(exc_info.value.message).lower()

    def test_ingest_directory_no_sols(self, tmp_path):
        """Test error when no sol directories found."""
        db_path = tmp_path / "test.db"
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        service = IngestionService(database_path=db_path)

        with pytest.raises(IngestionError) as exc_info:
            service.ingest_directory(empty_dir)

        assert "no sol directories" in str(exc_info.value.message).lower()


class TestDatabaseStats:
    """Tests for database statistics methods."""

    def test_get_database_stats_empty(self, tmp_path):
        """Test stats on empty database."""
        db_path = tmp_path / "test.db"
        service = IngestionService(database_path=db_path)

        stats = service.get_database_stats()

        assert stats["sols"] == 0
        assert stats["scans"] == 0
        assert stats["scan_points"] == 0
        assert stats["spectra"] == 0

    def test_get_database_stats_with_data(self, fixtures_path, tmp_path):
        """Test stats after ingestion."""
        sol_dir = fixtures_path / "loupe" / "sol_0921"
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=False,
        )

        service.ingest_sol(sol_dir)
        stats = service.get_database_stats()

        assert stats["sols"] >= 1
        assert stats["scans"] >= 1
        assert stats["scan_points"] >= 1
        assert stats["instrument_states"] >= 1

    def test_list_sols(self, fixtures_path, tmp_path):
        """Test listing ingested sols."""
        loupe_dir = fixtures_path / "loupe"
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=False,
        )

        service.ingest_directory(loupe_dir)
        sols = service.list_sols()

        assert len(sols) >= 1
        assert all(isinstance(s, int) for s in sols)
        assert sols == sorted(sols)  # Should be sorted

    def test_get_sol_scans(self, fixtures_path, tmp_path):
        """Test getting scans for a sol."""
        sol_dir = fixtures_path / "loupe" / "sol_0921"
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=False,
        )

        service.ingest_sol(sol_dir)
        scans = service.get_sol_scans(921)

        assert len(scans) >= 1
        assert all("scan_id" in s for s in scans)
        assert all("scan_name" in s for s in scans)
        assert all("n_points" in s for s in scans)


class TestIngestionMode:
    """Tests for ingestion_mode parameter (bd-13h: R2/R3 ingestion)."""

    def test_default_mode_is_all_regions(self, tmp_path):
        """Default ingestion_mode is 'all_regions'."""
        db_path = tmp_path / "test.db"
        service = IngestionService(database_path=db_path)
        assert service.ingestion_mode == "all_regions"

    def test_invalid_mode_raises(self, tmp_path):
        """Invalid ingestion_mode raises ValueError."""
        db_path = tmp_path / "test.db"
        with pytest.raises(ValueError, match="Invalid ingestion_mode"):
            IngestionService(database_path=db_path, ingestion_mode="invalid")

    def test_all_regions_mode_accepted(self, tmp_path):
        """'all_regions' is a valid ingestion_mode."""
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path, ingestion_mode="all_regions"
        )
        assert service.ingestion_mode == "all_regions"

    def test_r1_only_ingests_single_region(self, tmp_path):
        """R1_only mode ingests one spectrum per scan point per file type."""
        # Create a workspace with multi-section spectra
        workspace = tmp_path / "sol_0100" / "detail_1" / "SrlcSpec_Loupe_working"
        workspace.mkdir(parents=True)

        (workspace / "loupe.csv").write_text(
            "original_data_file,SrlcSpecSpecSohRaw_0700000000-10000-1\n"
            "human_readable_workspace,detail_1\n"
            "n_spectra,2\n"
            "n_channels,5\n"
            "laser_wavelength,248.5794\n"
            "shots_per_spec,500\n"
            "az_scale,0.628\n"
            "el_scale,0.422\n"
            "laser_x,809\n"
            "laser_y,664\n"
            "rotation,20.0\n"
            "specProcessingApplied,None\n"
            "CNDH_PCB_TEMP_STAT_REG,26.0 C\n"
            "SE_CCD_TEMP_STAT_REG,-29.0 C\n"
            "laser_shot_counter,1000000\n"
        )

        (workspace / "spatial.csv").write_text(
            "az,el\n1041,726\n994,503\nx,y\n0.518,0.503\n0.419,0.509\n"
        )

        spectra_lines = [
            "R1_C0,R1_C1,R1_C2,R1_C3,R1_C4",
            "100,110,120,130,140",
            "101,111,121,131,141",
            "R2_C0,R2_C1,R2_C2,R2_C3,R2_C4",
            "200,210,220,230,240",
            "201,211,221,231,241",
            "R3_C0,R3_C1,R3_C2,R3_C3,R3_C4",
            "300,310,320,330,340",
            "301,311,321,331,341",
        ]
        (workspace / "darkSubSpectra.csv").write_text("\n".join(spectra_lines))

        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=True,
            ingestion_mode="R1_only",
        )

        result = service.ingest_workspace(workspace)

        assert result.metadata["success"]

        with get_session(service.engine) as session:
            spectra = session.query(SpectrumORM).all()
            # R1_only: 1 file type (darkSub) x 2 scan points x 1 region = 2 spectra
            assert len(spectra) == 2
            # All should be R1
            for s in spectra:
                assert s.region == "R1"

    def test_all_regions_ingests_three_regions(self, tmp_path):
        """all_regions mode ingests R1+R2+R3 (3 spectra per scan point per file)."""
        # Create a workspace with multi-section spectra
        workspace = tmp_path / "sol_0100" / "detail_1" / "SrlcSpec_Loupe_working"
        workspace.mkdir(parents=True)

        (workspace / "loupe.csv").write_text(
            "original_data_file,SrlcSpecSpecSohRaw_0700000000-10000-1\n"
            "human_readable_workspace,detail_1\n"
            "n_spectra,2\n"
            "n_channels,5\n"
            "laser_wavelength,248.5794\n"
            "shots_per_spec,500\n"
            "az_scale,0.628\n"
            "el_scale,0.422\n"
            "laser_x,809\n"
            "laser_y,664\n"
            "rotation,20.0\n"
            "specProcessingApplied,None\n"
            "CNDH_PCB_TEMP_STAT_REG,26.0 C\n"
            "SE_CCD_TEMP_STAT_REG,-29.0 C\n"
            "laser_shot_counter,1000000\n"
        )

        (workspace / "spatial.csv").write_text(
            "az,el\n1041,726\n994,503\nx,y\n0.518,0.503\n0.419,0.509\n"
        )

        spectra_lines = [
            "R1_C0,R1_C1,R1_C2,R1_C3,R1_C4",
            "100,110,120,130,140",
            "101,111,121,131,141",
            "R2_C0,R2_C1,R2_C2,R2_C3,R2_C4",
            "200,210,220,230,240",
            "201,211,221,231,241",
            "R3_C0,R3_C1,R3_C2,R3_C3,R3_C4",
            "300,310,320,330,340",
            "301,311,321,331,341",
        ]
        (workspace / "darkSubSpectra.csv").write_text("\n".join(spectra_lines))

        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=True,
            ingestion_mode="all_regions",
        )

        result = service.ingest_workspace(workspace)

        assert result.metadata["success"]

        with get_session(service.engine) as session:
            spectra = session.query(SpectrumORM).all()
            # all_regions: 1 file type (darkSub) x 2 scan points x 3 regions = 6 spectra
            assert len(spectra) == 6

            # Check region distribution
            regions = [s.region for s in spectra]
            assert regions.count("R1") == 2
            assert regions.count("R2") == 2
            assert regions.count("R3") == 2

    def test_all_regions_backward_compatible_with_existing_data(self, tmp_path):
        """all_regions mode produces correct region labels for each section."""
        workspace = tmp_path / "sol_0100" / "detail_1" / "SrlcSpec_Loupe_working"
        workspace.mkdir(parents=True)

        (workspace / "loupe.csv").write_text(
            "original_data_file,SrlcSpecSpecSohRaw_0700000000-10000-1\n"
            "human_readable_workspace,detail_1\n"
            "n_spectra,1\n"
            "n_channels,3\n"
            "laser_wavelength,248.5794\n"
            "shots_per_spec,100\n"
            "az_scale,0.628\n"
            "el_scale,0.422\n"
            "laser_x,809\n"
            "laser_y,664\n"
            "rotation,20.0\n"
            "specProcessingApplied,None\n"
            "CNDH_PCB_TEMP_STAT_REG,26.0 C\n"
            "SE_CCD_TEMP_STAT_REG,-29.0 C\n"
            "laser_shot_counter,1000000\n"
        )

        (workspace / "spatial.csv").write_text(
            "az,el\n1041,726\nx,y\n0.518,0.503\n"
        )

        spectra_lines = [
            "R1_C0,R1_C1,R1_C2",
            "10.0,20.0,30.0",
            "R2_C0,R2_C1,R2_C2",
            "40.0,50.0,60.0",
            "R3_C0,R3_C1,R3_C2",
            "70.0,80.0,90.0",
        ]
        (workspace / "darkSubSpectra.csv").write_text("\n".join(spectra_lines))

        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=True,
            ingestion_mode="all_regions",
        )

        result = service.ingest_workspace(workspace)
        assert result.metadata["success"]
        # 1 point x 3 regions x 1 file type = 3
        assert result.metadata["spectra_ingested"] == 3


class TestIngestionValidation:
    """Tests for data validation during ingestion."""

    def test_scan_points_count_matches(self, fixtures_path, tmp_path):
        """Test that scan point count matches scan.n_points."""
        workspace = (
            fixtures_path / "loupe" / "sol_0921" / "detail_1" /
            "SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
        )
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=False,
        )

        service.ingest_workspace(workspace)

        with get_session(service.engine) as session:
            scan = session.query(ScanORM).first()
            points_count = session.query(ScanPointORM).filter(
                ScanPointORM.scan_id == scan.id
            ).count()

            # Point count should match what's in the scan
            assert points_count == scan.n_points

    def test_scan_has_valid_sclk(self, fixtures_path, tmp_path):
        """Test that ingested scans have valid SCLK values."""
        workspace = (
            fixtures_path / "loupe" / "sol_0921" / "detail_1" /
            "SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
        )
        db_path = tmp_path / "test.db"
        service = IngestionService(
            database_path=db_path,
            include_spectra=False,
        )

        service.ingest_workspace(workspace)

        with get_session(service.engine) as session:
            scan = session.query(ScanORM).first()
            assert scan.sclk_start > 0  # Valid SCLK
            assert scan.sclk_start > 700000000  # Reasonable M2020 SCLK range
