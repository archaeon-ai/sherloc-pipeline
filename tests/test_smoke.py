"""
Smoke tests to verify test fixtures are loadable.

These tests validate the Phase 0 test infrastructure by verifying:
- manifest.json structure is valid
- All Loupe datasets exist and are accessible
- Background CSVs exist and have expected columns
- Reference spectra are loadable
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest


class TestManifestStructure:
    """Tests for manifest.json structure and validity."""

    def test_manifest_has_required_keys(self, manifest: Dict[str, Any]):
        """Verify manifest has required top-level keys."""
        assert "datasets" in manifest, "manifest missing 'datasets' key"
        assert "backgrounds" in manifest, "manifest missing 'backgrounds' key"
        assert "reference" in manifest, "manifest missing 'reference' key"

    def test_manifest_has_expected_datasets(self, manifest: Dict[str, Any]):
        """Verify manifest lists expected number of datasets."""
        datasets = manifest["datasets"]
        assert len(datasets) == 3, f"Expected 3 datasets, got {len(datasets)}"

    def test_dataset_entries_have_required_fields(self, manifest: Dict[str, Any]):
        """Verify each dataset entry has required fields."""
        required_fields = ["sol", "target", "scan", "ppp", "n_points", "path"]
        for ds in manifest["datasets"]:
            for field in required_fields:
                assert field in ds, f"Dataset missing field: {field}"

    def test_background_entries_have_required_fields(self, manifest: Dict[str, Any]):
        """Verify each background entry has required fields."""
        required_fields = ["path", "ppp"]
        for bg_key, bg in manifest["backgrounds"].items():
            for field in required_fields:
                assert field in bg, f"Background '{bg_key}' missing field: {field}"


class TestLoupeDatasets:
    """Tests for Loupe dataset accessibility."""

    def test_all_loupe_directories_exist(
        self, fixtures_path: Path, manifest: Dict[str, Any]
    ):
        """Verify all Loupe working directories exist."""
        for ds in manifest["datasets"]:
            loupe_dir = fixtures_path / ds["path"]
            assert loupe_dir.exists(), f"Missing Loupe directory: {ds['path']}"
            assert loupe_dir.is_dir(), f"Path is not a directory: {ds['path']}"

    def test_loupe_csv_exists_in_each_dataset(
        self, fixtures_path: Path, manifest: Dict[str, Any]
    ):
        """Verify loupe.csv exists in each dataset."""
        for ds in manifest["datasets"]:
            loupe_csv = fixtures_path / ds["path"] / "loupe.csv"
            assert loupe_csv.exists(), f"Missing loupe.csv in {ds['path']}"

    def test_dark_sub_spectra_exists(
        self, fixtures_path: Path, manifest: Dict[str, Any]
    ):
        """Verify darkSubSpectra.csv exists in each dataset."""
        for ds in manifest["datasets"]:
            spectra_csv = fixtures_path / ds["path"] / "darkSubSpectra.csv"
            assert spectra_csv.exists(), f"Missing darkSubSpectra.csv in {ds['path']}"

    def test_can_load_loupe_metadata(
        self, fixtures_path: Path, loupe_datasets: Dict[str, Dict[str, Any]]
    ):
        """Verify loupe.csv can be parsed and contains expected metadata."""
        for sol, ds in loupe_datasets.items():
            loupe_csv = fixtures_path / ds["path"] / "loupe.csv"
            
            # loupe.csv is a key-value format, not standard CSV
            with open(loupe_csv) as f:
                lines = f.readlines()
            
            # Parse as key-value pairs
            metadata = {}
            for line in lines:
                if "," in line:
                    parts = line.strip().split(",", 1)
                    if len(parts) == 2:
                        metadata[parts[0]] = parts[1]
            
            # Verify n_spectra matches manifest
            assert "n_spectra" in metadata, f"Missing n_spectra in loupe.csv for sol {sol}"
            n_spectra = int(metadata["n_spectra"])
            assert n_spectra == ds["n_points"], (
                f"n_spectra mismatch for sol {sol}: "
                f"manifest says {ds['n_points']}, loupe.csv says {n_spectra}"
            )
            
            # Verify shots_per_spec (PPP) matches manifest
            assert "shots_per_spec" in metadata, f"Missing shots_per_spec in loupe.csv for sol {sol}"
            ppp = int(metadata["shots_per_spec"])
            assert ppp == ds["ppp"], (
                f"PPP mismatch for sol {sol}: "
                f"manifest says {ds['ppp']}, loupe.csv says {ppp}"
            )


class TestBackgroundSpectra:
    """Tests for background spectra accessibility and format."""

    def test_background_files_exist(self, background_paths: Dict[str, Path]):
        """Verify all background files exist."""
        for bg_key, bg_path in background_paths.items():
            assert bg_path.exists(), f"Missing background file: {bg_key} at {bg_path}"

    def test_can_load_arm_stowed_background(self, background_paths: Dict[str, Path]):
        """Verify arm stowed background CSV is loadable with expected columns."""
        as_path = background_paths["as"]
        df = pd.read_csv(as_path)
        
        assert "raman_shift" in df.columns, "Missing 'raman_shift' column in AS background"
        assert "intensity" in df.columns, "Missing 'intensity' column in AS background"
        assert len(df) > 1000, f"AS background has fewer rows than expected: {len(df)}"

    def test_can_load_fused_silica_background(self, background_paths: Dict[str, Path]):
        """Verify fused silica background CSV is loadable with expected columns."""
        fs_path = background_paths["fs"]
        df = pd.read_csv(fs_path)
        
        # FS uses different column names
        assert "Raman shift (cm-1)" in df.columns, "Missing 'Raman shift (cm-1)' column in FS background"
        assert "Intensity" in df.columns, "Missing 'Intensity' column in FS background"
        assert len(df) > 1000, f"FS background has fewer rows than expected: {len(df)}"


class TestReferenceSpectra:
    """Tests for reference spectra accessibility."""

    def test_reference_files_exist(self, reference_paths: Dict[str, Path]):
        """Verify all reference files exist."""
        for mineral, ref_path in reference_paths.items():
            assert ref_path.exists(), f"Missing reference file for {mineral}: {ref_path}"

    def test_can_load_forsterite_reference(
        self, reference_paths: Dict[str, Path], manifest: Dict[str, Any]
    ):
        """Verify forsterite reference spectrum is loadable."""
        forsterite_path = reference_paths["forsterite"]
        
        # Get format info from manifest
        ref_info = next(r for r in manifest["reference"] if r["mineral"] == "forsterite")
        header_rows = ref_info["format"]["header_rows"]
        
        # Load with skiprows to skip metadata headers
        df = pd.read_csv(forsterite_path, skiprows=header_rows)
        
        # Should have multiple columns - we care about raman shift and intensity
        assert len(df.columns) >= 2, "Reference spectrum should have at least 2 columns"
        assert len(df) > 100, f"Reference spectrum has fewer rows than expected: {len(df)}"


class TestRuntimeContext:
    """Tests for RuntimeContext fixture."""

    def test_context_has_correct_data_root(
        self, test_context, fixtures_path: Path
    ):
        """Verify test_context points to fixtures/loupe."""
        expected_data_root = fixtures_path / "loupe"
        assert test_context.data_root == expected_data_root.resolve(), (
            f"data_root mismatch: expected {expected_data_root}, got {test_context.data_root}"
        )

    def test_context_has_results_root(self, test_context, tmp_results: Path):
        """Verify test_context has a valid results_root."""
        # The test_context uses tmp_path/results, tmp_results uses the same
        assert test_context.results_root is not None
        assert "results" in str(test_context.results_root)

