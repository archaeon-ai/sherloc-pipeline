"""Region split in core/laser_normalization.py (issue #123).

darkSubSpectra.csv repeats a section header row before R2 and R3, so after
pandas takes the first header the other two sit at iloc[n] and iloc[2n+1].
Every array here is synthetic: point k of region r holds 100*r + k + 1 on
every channel, and the photodiode means give point k the factor 2**k.
"""

import numpy as np
import pandas as pd
import pytest

from sherloc_pipeline.core.data_ingestion import DataIngestion
from sherloc_pipeline.core.laser_normalization import process_laser_normalization

N_SPECTRA = 3
N_CHANNELS = 2148
FACTORS = np.array([1.0, 2.0, 4.0])


def raw_value(region: int, point: int) -> float:
    return 100.0 * region + point + 1


def write_workspace(root, repeat_headers: bool = True):
    work = root / "synthetic_Loupe_working"
    work.mkdir()
    (work / "loupe.csv").write_text(f"n_spectra,{N_SPECTRA}\nn_channels,{N_CHANNELS}\nspecProcessingApplied,D\n")
    # Mean photodiode 400 / 200 / 100 -> factors max/pd = 1 / 2 / 4.
    (work / "photodiodeRaw.csv").write_text("Shot0,Shot1\n400,400\n200,200\n100,100\n")
    lines = []
    for region in (1, 2, 3):
        if region == 1 or repeat_headers:
            lines.append(",".join(f"R{region}_Channel{c}" for c in range(N_CHANNELS)))
        for point in range(N_SPECTRA):
            lines.append(",".join([f"{raw_value(region, point):g}"] * N_CHANNELS))
    (work / "darkSubSpectra.csv").write_text("\n".join(lines) + "\n")
    return work


def normalize(work):
    process_laser_normalization(work, sol=0, target="synthetic", scan="synthetic", generate_plots=False)


def test_each_region_keeps_its_points_and_their_factors(tmp_path):
    work = write_workspace(tmp_path)
    normalize(work)

    lines = (work / "darkSubSpectraN.csv").read_text().splitlines()
    assert len(lines) == 3 * N_SPECTRA + 3
    headers = [0, N_SPECTRA + 1, 2 * N_SPECTRA + 2]
    for region, start in zip((1, 2, 3), headers):
        # The writer reuses the frame's column names, so every section header reads R1_Channel*.
        assert lines[start].split(",")[0] == "R1_Channel0"
        for point in range(N_SPECTRA):
            values = np.array(lines[start + 1 + point].split(","), dtype=float)
            assert values.shape == (N_CHANNELS,)
            assert not np.isnan(values).any()
            np.testing.assert_allclose(values, raw_value(region, point) * FACTORS[point])


def test_downstream_reader_returns_point_k_from_raw_point_k(tmp_path):
    work = write_workspace(tmp_path)
    normalize(work)

    spectra = DataIngestion(base_data_dir=tmp_path).process_normalized_spectra(work)

    fluorescence = spectra["fluorescence"]
    r1 = spectra["R1"]
    for point in range(N_SPECTRA):
        expected = (raw_value(2, point) + raw_value(3, point)) * FACTORS[point]
        np.testing.assert_allclose(fluorescence[point].to_numpy(dtype=float), expected)
        np.testing.assert_allclose(r1[point].to_numpy(dtype=float), raw_value(1, point) * FACTORS[point])


def test_refuses_a_file_without_the_repeated_section_headers(tmp_path):
    work = write_workspace(tmp_path, repeat_headers=False)
    with pytest.raises(ValueError, match="section header"):
        normalize(work)
    assert not (work / "darkSubSpectraN.csv").exists()


def test_refuses_a_file_whose_length_does_not_match_n_spectra(tmp_path):
    work = write_workspace(tmp_path)
    (work / "loupe.csv").write_text(f"n_spectra,{N_SPECTRA + 1}\nn_channels,{N_CHANNELS}\nspecProcessingApplied,D\n")
    with pytest.raises(ValueError, match="section header"):
        normalize(work)
    assert not (work / "darkSubSpectraN.csv").exists()
