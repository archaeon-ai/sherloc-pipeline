"""Laser normalization must not modify the input Loupe workspace (issue #131).

The workspace is source data. Normalization used to write darkSubSpectraN.csv
into it and round-trip loupe.csv through pandas, which turned
``specProcessingApplied,None`` into ``nanN`` and blanked ``N/A`` values.
All data here is synthetic.
"""

import hashlib

import numpy as np
import pytest

from sherloc_pipeline.core.data_ingestion import DataIngestion
from sherloc_pipeline.core.laser_normalization import process_laser_normalization

N_SPECTRA = 2
N_CHANNELS = 2148

LOUPE_WITH_TEXT = (
    f"n_spectra,{N_SPECTRA}\n"
    f"n_channels,{N_CHANNELS}\n"
    "scan_name,synthetic_scan\n"
    "specProcessingApplied,None\n"
    "CCD_VERT_COL2_HIGH,N/A\n"
)
# Every value numeric or NaN-like: pandas reads a float column, and the old
# flag rewrite raised TypeError on assignment.
LOUPE_NUMERIC_ONLY = (
    f"n_spectra,{N_SPECTRA}\n"
    f"n_channels,{N_CHANNELS}\n"
    "specProcessingApplied,None\n"
)


def write_workspace(root, loupe_text):
    work = root / "synthetic_Loupe_working"
    work.mkdir()
    (work / "loupe.csv").write_text(loupe_text)
    (work / "photodiodeRaw.csv").write_text("Shot0,Shot1\n200,200\n100,100\n")
    lines = []
    for region in (1, 2, 3):
        lines.append(",".join(f"R{region}_Channel{c}" for c in range(N_CHANNELS)))
        for point in range(N_SPECTRA):
            lines.append(",".join([f"{10 * region + point + 1}"] * N_CHANNELS))
    (work / "darkSubSpectra.csv").write_text("\n".join(lines) + "\n")
    return work


def snapshot(directory):
    return {
        p.relative_to(directory): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


@pytest.mark.parametrize("loupe_text", [LOUPE_WITH_TEXT, LOUPE_NUMERIC_ONLY], ids=["text", "numeric"])
def test_workspace_is_byte_identical_after_normalization(tmp_path, loupe_text):
    work = write_workspace(tmp_path, loupe_text)
    out = tmp_path / "results" / "scan"
    before = snapshot(work)

    written = process_laser_normalization(
        work, sol=0, target="synthetic", scan="synthetic",
        generate_plots=False, output_dir=out,
    )

    assert snapshot(work) == before
    assert written == out / "darkSubSpectraN.csv"
    assert written.is_file()


def test_reader_uses_the_normalized_file_from_the_output_dir(tmp_path):
    work = write_workspace(tmp_path, LOUPE_WITH_TEXT)
    out = tmp_path / "results" / "scan"
    written = process_laser_normalization(
        work, sol=0, target="synthetic", scan="synthetic",
        generate_plots=False, output_dir=out,
    )

    spectra = DataIngestion(base_data_dir=tmp_path).process_normalized_spectra(
        work, normalized_file=written
    )

    # Photodiode means 200 / 100 give factors 1 / 2.
    for point, factor in enumerate((1.0, 2.0)):
        np.testing.assert_allclose(spectra["R1"][point].to_numpy(dtype=float), (11 + point) * factor)


def test_scan_reader_prefers_this_pipelines_output_over_the_workspace(tmp_path):
    work = write_workspace(tmp_path, LOUPE_WITH_TEXT)
    ingestion = DataIngestion(
        base_data_dir=tmp_path, results_dir=tmp_path / "results",
        sol="0", target="synthetic", scan="detail",
    )
    assert ingestion.normalized_spectra_path(work) == work / "darkSubSpectraN.csv"

    out = ingestion.get_results_path(target="synthetic", sol="0", scan="detail")
    process_laser_normalization(
        work, sol="0", target="synthetic", scan="detail",
        generate_plots=False, output_dir=out,
    )
    assert ingestion.normalized_spectra_path(work) == out / "darkSubSpectraN.csv"
