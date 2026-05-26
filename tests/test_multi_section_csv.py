"""
Tests for multi-section CSV parsing in Loupe spectra files.

Sprint 4 introduced correct handling of Loupe CSV section structure:
- R1: Raman region (250-282 nm)
- R2: Fluorescence region 1 (282-337.8 nm)
- R3: Fluorescence region 2 (337.8-357.4 nm)

Prior to Sprint 4, only R1 was read but mislabeled as R123.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict

import pytest

from sherloc_pipeline.models.ingestion import (
    RawSpectraFile,
    LoupeWorkspaceParser,
)
from sherloc_pipeline.models.spectra import (
    SpectralRegion,
    SpectrumType,
    ProcessingLevel,
)


class TestCsvSectionDetection:
    """Tests for detecting multi-section structure in Loupe CSVs."""

    def test_count_section_rows_detects_r1_boundary(
        self, fixtures_path: Path, loupe_datasets: Dict[str, Any]
    ):
        """Verify _count_section_rows correctly counts R1 data rows."""
        # Use sol 0921 fixture which has 100 points
        ds = loupe_datasets["0921"]
        csv_path = fixtures_path / ds["path"] / "darkSubSpectra.csv"

        n_spectra = RawSpectraFile._count_section_rows(csv_path)

        # Manifest says 100 points
        assert n_spectra == ds["n_points"], (
            f"Expected {ds['n_points']} spectra in R1, got {n_spectra}"
        )

    def test_header_starts_with_r1_channel(
        self, fixtures_path: Path, loupe_datasets: Dict[str, Any]
    ):
        """Verify R1 section has R1_Channel headers."""
        ds = loupe_datasets["0921"]
        csv_path = fixtures_path / ds["path"] / "darkSubSpectra.csv"

        raw_file, _ = RawSpectraFile.from_csv_path(
            csv_path, SpectrumType.DARK_SUBTRACTED, section="R1"
        )

        # First column should start with "R1_Channel"
        assert raw_file.channel_names[0].startswith("R1_Channel"), (
            f"Expected R1_Channel header, got {raw_file.channel_names[0]}"
        )
        assert raw_file.section == "R1"


class TestSectionRowCalculation:
    """Tests for correct skiprows calculation for each section."""

    def test_r1_section_reads_correct_rows(
        self, fixtures_path: Path, loupe_datasets: Dict[str, Any]
    ):
        """Verify R1 reads from row 0 (header) and returns n_spectra data rows."""
        ds = loupe_datasets["0921"]
        csv_path = fixtures_path / ds["path"] / "darkSubSpectra.csv"

        raw_file, data = RawSpectraFile.from_csv_path(
            csv_path, SpectrumType.DARK_SUBTRACTED, section="R1"
        )

        assert len(data) == ds["n_points"], (
            f"R1 section should have {ds['n_points']} rows, got {len(data)}"
        )
        assert raw_file.n_points == ds["n_points"]

    def test_r2_section_reads_correct_rows(
        self, fixtures_path: Path, loupe_datasets: Dict[str, Any]
    ):
        """Verify R2 section reads the correct number of rows."""
        ds = loupe_datasets["0921"]
        csv_path = fixtures_path / ds["path"] / "darkSubSpectra.csv"

        raw_file, data = RawSpectraFile.from_csv_path(
            csv_path, SpectrumType.DARK_SUBTRACTED, section="R2"
        )

        assert len(data) == ds["n_points"], (
            f"R2 section should have {ds['n_points']} rows, got {len(data)}"
        )
        assert raw_file.section == "R2"
        # R2 header should have R2_Channel columns
        assert raw_file.channel_names[0].startswith("R2_Channel"), (
            f"Expected R2_Channel header, got {raw_file.channel_names[0]}"
        )

    def test_r3_section_reads_correct_rows(
        self, fixtures_path: Path, loupe_datasets: Dict[str, Any]
    ):
        """Verify R3 section reads the correct number of rows."""
        ds = loupe_datasets["0921"]
        csv_path = fixtures_path / ds["path"] / "darkSubSpectra.csv"

        raw_file, data = RawSpectraFile.from_csv_path(
            csv_path, SpectrumType.DARK_SUBTRACTED, section="R3"
        )

        assert len(data) == ds["n_points"], (
            f"R3 section should have {ds['n_points']} rows, got {len(data)}"
        )
        assert raw_file.section == "R3"
        # R3 header should have R3_Channel columns
        assert raw_file.channel_names[0].startswith("R3_Channel"), (
            f"Expected R3_Channel header, got {raw_file.channel_names[0]}"
        )

    def test_sections_have_same_channel_count(
        self, fixtures_path: Path, loupe_datasets: Dict[str, Any]
    ):
        """Verify all three sections have the same number of channels."""
        ds = loupe_datasets["0921"]
        csv_path = fixtures_path / ds["path"] / "darkSubSpectra.csv"

        r1_file, _ = RawSpectraFile.from_csv_path(csv_path, SpectrumType.DARK_SUBTRACTED, "R1")
        r2_file, _ = RawSpectraFile.from_csv_path(csv_path, SpectrumType.DARK_SUBTRACTED, "R2")
        r3_file, _ = RawSpectraFile.from_csv_path(csv_path, SpectrumType.DARK_SUBTRACTED, "R3")

        assert r1_file.n_channels == r2_file.n_channels == r3_file.n_channels, (
            f"Channel counts differ: R1={r1_file.n_channels}, "
            f"R2={r2_file.n_channels}, R3={r3_file.n_channels}"
        )
        # SHERLOC CCD has 2148 channels
        assert r1_file.n_channels == 2148, f"Expected 2148 channels, got {r1_file.n_channels}"


class TestIngestionStoresCorrectRegion:
    """Tests for correct SpectralRegion labeling in ingested spectra."""

    def test_iter_spectra_uses_r1_region(
        self, fixtures_path: Path, loupe_datasets: Dict[str, Any]
    ):
        """Verify iter_spectra produces spectra with SpectralRegion.R1 for R1 section."""
        ds = loupe_datasets["0921"]
        csv_path = fixtures_path / ds["path"] / "darkSubSpectra.csv"

        raw_file, data = RawSpectraFile.from_csv_path(
            csv_path, SpectrumType.DARK_SUBTRACTED, section="R1"
        )

        # Create fake scan point IDs
        scan_point_ids = [uuid.uuid4() for _ in range(len(data))]

        spectra = list(raw_file.iter_spectra(
            data, scan_point_ids, ProcessingLevel.RAW
        ))

        # All spectra should have region=R1
        for spectrum in spectra:
            assert spectrum.region == SpectralRegion.R1, (
                f"Expected region R1, got {spectrum.region}"
            )

    def test_iter_spectra_uses_r2_region(
        self, fixtures_path: Path, loupe_datasets: Dict[str, Any]
    ):
        """Verify iter_spectra produces spectra with SpectralRegion.R2 for R2 section."""
        ds = loupe_datasets["0921"]
        csv_path = fixtures_path / ds["path"] / "darkSubSpectra.csv"

        raw_file, data = RawSpectraFile.from_csv_path(
            csv_path, SpectrumType.DARK_SUBTRACTED, section="R2"
        )

        scan_point_ids = [uuid.uuid4() for _ in range(len(data))]
        spectra = list(raw_file.iter_spectra(
            data, scan_point_ids, ProcessingLevel.RAW
        ))

        for spectrum in spectra:
            assert spectrum.region == SpectralRegion.R2, (
                f"Expected region R2, got {spectrum.region}"
            )

    def test_iter_spectra_uses_r3_region(
        self, fixtures_path: Path, loupe_datasets: Dict[str, Any]
    ):
        """Verify iter_spectra produces spectra with SpectralRegion.R3 for R3 section."""
        ds = loupe_datasets["0921"]
        csv_path = fixtures_path / ds["path"] / "darkSubSpectra.csv"

        raw_file, data = RawSpectraFile.from_csv_path(
            csv_path, SpectrumType.DARK_SUBTRACTED, section="R3"
        )

        scan_point_ids = [uuid.uuid4() for _ in range(len(data))]
        spectra = list(raw_file.iter_spectra(
            data, scan_point_ids, ProcessingLevel.RAW
        ))

        for spectrum in spectra:
            assert spectrum.region == SpectralRegion.R3, (
                f"Expected region R3, got {spectrum.region}"
            )

    def test_workspace_parser_parse_spectra_uses_r1(
        self, fixtures_path: Path, loupe_datasets: Dict[str, Any]
    ):
        """Verify LoupeWorkspaceParser.parse_spectra correctly labels R1 spectra."""
        ds = loupe_datasets["0921"]
        workspace_path = fixtures_path / ds["path"]

        parser = LoupeWorkspaceParser(workspace_path, sol_number=921)

        # Create fake scan point IDs
        scan_point_ids = [uuid.uuid4() for _ in range(ds["n_points"])]

        spectra = parser.parse_spectra(
            SpectrumType.DARK_SUBTRACTED,
            scan_point_ids,
            ProcessingLevel.RAW,
            section="R1",
        )

        assert len(spectra) == ds["n_points"], (
            f"Expected {ds['n_points']} spectra, got {len(spectra)}"
        )

        # All should be labeled R1
        for spectrum in spectra:
            assert spectrum.region == SpectralRegion.R1, (
                f"Expected region R1, got {spectrum.region}"
            )


class TestSectionCaseInsensitivity:
    """Tests for case-insensitive section parameter handling."""

    def test_lowercase_section_works(
        self, fixtures_path: Path, loupe_datasets: Dict[str, Any]
    ):
        """Verify lowercase section parameter works."""
        ds = loupe_datasets["0921"]
        csv_path = fixtures_path / ds["path"] / "darkSubSpectra.csv"

        raw_file, data = RawSpectraFile.from_csv_path(
            csv_path, SpectrumType.DARK_SUBTRACTED, section="r1"
        )

        assert raw_file.section == "R1"
        assert len(data) == ds["n_points"]

    def test_mixed_case_section_works(
        self, fixtures_path: Path, loupe_datasets: Dict[str, Any]
    ):
        """Verify mixed case section parameter works."""
        ds = loupe_datasets["0921"]
        csv_path = fixtures_path / ds["path"] / "darkSubSpectra.csv"

        raw_file, data = RawSpectraFile.from_csv_path(
            csv_path, SpectrumType.DARK_SUBTRACTED, section="r2"
        )

        assert raw_file.section == "R2"
