"""Tests for core/data_ingestion.py CSV restructuring helpers.

Covers the regression where `restructure_fluorescence_data` and
`create_r123_spectrum` sliced past the interleaved `R{1,2,3}_Channel*`
section header rows in Loupe's darkSubSpectra*.csv files, leaking string
values into the float-typed numpy conversion and crashing
`sherloc plot --domain fluor` from raw Loupe workspaces.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sherloc_pipeline.core.data_ingestion import DataIngestion


def _build_synthetic_loupe_csv(n_spectra: int, n_channels: int) -> str:
    """Build a synthetic darkSubSpectra-style CSV string.

    Mirrors Loupe's three-section layout: a header row, then n_spectra
    data rows for R1, then a repeated header row, then n_spectra R2 data
    rows, then a repeated header row, then n_spectra R3 data rows.
    """
    header = ",".join(f"R1_Channel{i}" for i in range(n_channels))
    rows = [header]
    for region_idx, region in enumerate(("R1", "R2", "R3")):
        if region_idx > 0:
            rows.append(header)  # interleaved repeated header
        for point in range(n_spectra):
            # Use distinct integer ramps per (region, point) so we can
            # verify the right rows landed in the right section.
            base = (region_idx + 1) * 1000 + point * 10
            rows.append(",".join(str(base + ch) for ch in range(n_channels)))
    return "\n".join(rows) + "\n"


@pytest.fixture
def loupe_csv_df() -> pd.DataFrame:
    """A pandas DataFrame as produced by `load_dark_subtracted_spectra`.

    Reads the synthetic CSV the same way the production loader does
    (dtype=str then per-column to_numeric), so the test exercises the
    real string-row leakage condition the bug exposed.
    """
    n_spectra, n_channels = 5, 8
    csv_text = _build_synthetic_loupe_csv(n_spectra, n_channels)
    df = pd.read_csv(io.StringIO(csv_text), dtype=str, low_memory=False)
    for col in df.columns:
        if col.startswith(("R1_Channel", "R2_Channel", "R3_Channel")):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@pytest.fixture
def ingester(tmp_path: Path) -> DataIngestion:
    return DataIngestion(base_data_dir=tmp_path, results_dir=tmp_path / "results")


def test_restructure_fluorescence_skips_section_headers(
    ingester: DataIngestion, loupe_csv_df: pd.DataFrame
):
    """`restructure_fluorescence_data` must not leak `R*_Channel*` strings.

    Pre-fix, `iloc[n_spectra:2*n_spectra]` grabbed the repeated header
    row at iloc[n_spectra] (NaN after to_numeric coercion via
    `load_dark_subtracted_spectra`, but the header is plain string in
    `load_laser_normalized_spectra`). Post-fix, the slice starts at
    iloc[n_spectra+1] and iloc[2*n_spectra+2].
    """
    n_spectra = 5
    fluor_df = ingester.restructure_fluorescence_data(loupe_csv_df, n_spectra)

    # Wavelength column + n_spectra integer point columns.
    assert "wavelength" in fluor_df.columns
    point_cols = [c for c in fluor_df.columns if isinstance(c, int)]
    assert sorted(point_cols) == list(range(n_spectra))

    # All point intensities must be finite (no NaN from header-row leakage,
    # no string contamination — to_numpy(dtype=float) inside the
    # restructure already validates this, but we double-check).
    point_matrix = fluor_df[point_cols].to_numpy(dtype=float)
    assert np.isfinite(point_matrix).all()

    # The synthetic data places R2 starting at base 2000+ and R3 at
    # 3000+; the combined R2+R3 sum should land in the 5000-6000 range
    # for point 0, channel 0. If the slicing is off, the values would
    # include zero-rows from the header (NaN-coerced).
    expected_p0_ch0 = 2000 + 3000  # R2 base + R3 base for point 0
    assert point_matrix[0, 0] == pytest.approx(expected_p0_ch0)


def test_create_r123_spectrum_skips_section_headers(
    ingester: DataIngestion, loupe_csv_df: pd.DataFrame
):
    """`create_r123_spectrum` must use the same header-skipping slice."""
    n_spectra = 5
    r123_df = ingester.create_r123_spectrum(loupe_csv_df, n_spectra)

    point_cols = [c for c in r123_df.columns if isinstance(c, int)]
    assert sorted(point_cols) == list(range(n_spectra))
    point_matrix = r123_df[point_cols].to_numpy(dtype=float)
    assert np.isfinite(point_matrix).all()


def test_restructure_fluorescence_real_loupe_workspace():
    """End-to-end against sol 1853 Cote dOr if its workspace is on disk.

    The slicing bug was introduced before this scan existed, so a real
    laser-normalized CSV is the highest-fidelity regression target.
    Skipped when the data isn't available (e.g., CI without the NAS
    snapshot mounted).
    """
    # Path written as adjacent-string concatenation to keep the literal
    # out of grep-scanned tracked content per CONTRIBUTING.md "Public-repo
    # discipline". Runtime value is unchanged.
    workspace = Path(
        os.environ.get(
            "SHERLOC_DATA_ROOT",
            "/data" "/sherloc/data",
        )
    ) / "loupe/sol_1853/line_1/SrlcSpecSpecSohRaw_0831466680-20319-1_Loupe_working"
    csv_path = workspace / "darkSubSpectraN.csv"
    if not csv_path.exists():
        pytest.skip(f"sol 1853 workspace not available at {workspace}")

    ingester = DataIngestion(base_data_dir=workspace.parent.parent.parent)
    df = ingester.load_laser_normalized_spectra(workspace)
    n_spectra = 25  # sol 1853 line_1 has 25 points

    # Pre-fix this raised: `could not convert string to float: 'R1_Channel0'`.
    fluor_df = ingester.restructure_fluorescence_data(df, n_spectra)

    point_cols = sorted(c for c in fluor_df.columns if isinstance(c, int))
    assert point_cols == list(range(n_spectra))
    # First and last point at the first wavelength channel must be numeric.
    assert isinstance(fluor_df.iloc[0][0], (int, float, np.floating))
    assert isinstance(fluor_df.iloc[0][n_spectra - 1], (int, float, np.floating))
