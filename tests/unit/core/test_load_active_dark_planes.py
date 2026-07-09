"""Raw ACTIVE/DARK plane loader (detection input path).

Uses the committed sol_0921 fixture workspace, which carries real
``activeSpectra.csv`` / ``darkSpectra.csv`` files in the Loupe section
layout.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from sherloc_pipeline.core.data_ingestion import DataIngestion

WORKSPACE = (
    "tests/fixtures/loupe/sol_0921/detail_1/"
    "SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
)
N_CHANNELS = 2148
N_SPECTRA = 100  # fixture loupe.csv n_spectra


@pytest.fixture(scope="module")
def workspace(fixtures_path):
    return (
        fixtures_path
        / "loupe"
        / "sol_0921"
        / "detail_1"
        / "SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
    )


@pytest.fixture(scope="module")
def ingestion(fixtures_path, tmp_path_factory):
    return DataIngestion(
        base_data_dir=fixtures_path / "loupe",
        results_dir=tmp_path_factory.mktemp("results"),
    )


class TestLoadActiveDarkPlanes:
    def test_shapes_and_dtypes(self, ingestion, workspace):
        planes = ingestion.load_active_dark_planes(workspace)
        assert set(planes) == {"active", "dark"}
        for plane in planes.values():
            assert set(plane) == {"R1", "R2", "R3"}
            for arr in plane.values():
                assert arr.shape == (N_SPECTRA, N_CHANNELS)
                assert arr.dtype == np.float64
                assert not np.isnan(arr).any()

    def test_section_slicing_matches_raw_csv(self, ingestion, workspace):
        """Row 0 of each region must equal the corresponding raw CSV row
        (R1 first data row; R2 first row after the repeated header; R3
        likewise) — guards the 3-section iloc arithmetic."""
        planes = ingestion.load_active_dark_planes(workspace)
        raw = pd.read_csv(
            workspace / "activeSpectra.csv", dtype=str, low_memory=False
        )
        for col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

        np.testing.assert_array_equal(
            planes["active"]["R1"][0], raw.iloc[0].to_numpy(dtype=float)
        )
        np.testing.assert_array_equal(
            planes["active"]["R2"][0],
            raw.iloc[N_SPECTRA + 1].to_numpy(dtype=float),
        )
        np.testing.assert_array_equal(
            planes["active"]["R3"][0],
            raw.iloc[2 * N_SPECTRA + 2].to_numpy(dtype=float),
        )

    def test_raw_planes_differ_from_normalized(self, ingestion, workspace):
        """The loader yields raw DN, not any normalized representation —
        the certified observable distinction."""
        planes = ingestion.load_active_dark_planes(workspace)
        normalized = ingestion.process_normalized_spectra(workspace)
        r1_norm = normalized["R1"][0].to_numpy(dtype=float)
        r1_raw_windowed = planes["active"]["R1"][0][52:575]
        assert r1_norm.shape[0] == r1_raw_windowed.shape[0]
        assert not np.allclose(r1_norm, r1_raw_windowed)

    def test_missing_dark_file_raises(self, workspace, tmp_path):
        partial = tmp_path / "partial_workspace"
        partial.mkdir()
        shutil.copy(workspace / "loupe.csv", partial / "loupe.csv")
        shutil.copy(
            workspace / "activeSpectra.csv", partial / "activeSpectra.csv"
        )
        ingestion = DataIngestion(
            base_data_dir=tmp_path, results_dir=tmp_path / "results"
        )
        with pytest.raises(FileNotFoundError):
            ingestion.load_active_dark_planes(partial)

    def test_missing_active_file_raises(self, workspace, tmp_path):
        partial = tmp_path / "partial_workspace"
        partial.mkdir()
        shutil.copy(workspace / "loupe.csv", partial / "loupe.csv")
        shutil.copy(workspace / "darkSpectra.csv", partial / "darkSpectra.csv")
        ingestion = DataIngestion(
            base_data_dir=tmp_path, results_dir=tmp_path / "results"
        )
        with pytest.raises(FileNotFoundError):
            ingestion.load_active_dark_planes(partial)
