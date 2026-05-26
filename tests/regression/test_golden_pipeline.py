"""Golden baseline regression tests for PipelineService (R-024).

Runs PipelineService.run_full_pipeline() on sol 921 detail_1 and compares
structured outputs against committed golden snapshots.

Golden snapshots live at tests/golden/sol_921_detail_1/ and are generated
by scripts/generate_golden_baseline.py.

Set UPDATE_GOLDEN_BASELINE=1 to regenerate golden files instead of asserting.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sherloc_pipeline.core.calibration import calculate_loupe_wavelength_wavenumber
from sherloc_pipeline.services.config import reset_runtime_config
from sherloc_pipeline.services.pipeline import PipelineService
from sherloc_pipeline.services.runtime import RuntimeContext

TESTS_DIR = Path(__file__).resolve().parent.parent
FIXTURES = TESTS_DIR / "fixtures"
GOLDEN_DIR = TESTS_DIR / "golden" / "sol_921_detail_1"

SOL = "0921"
TARGET = "Amherst_Point"
SCAN = "detail_1"

_BACKGROUND_FILE = (
    FIXTURES / "background" / "Arm_Stowed_post-anomaly_900ppp_trimmed_mean_1266.csv"
)

_UPDATE_MODE = os.environ.get("UPDATE_GOLDEN_BASELINE", "").strip() == "1"


def _golden_exists() -> bool:
    """Check whether all golden baseline files are present."""
    expected = [
        "calibration_arrays.npz",
        "preprocessed_spectra.npz",
        "fitted_peaks.json",
        "review_decisions.json",
        "pipeline_summary.json",
    ]
    return all((GOLDEN_DIR / f).exists() for f in expected)


def _regenerate_golden() -> None:
    """Re-run the generation script to update golden files."""
    script = TESTS_DIR.parent / "scripts" / "generate_golden_baseline.py"
    subprocess.check_call([sys.executable, str(script)])


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    """Run the full pipeline once for the module and return (result, scan_dir)."""
    if _UPDATE_MODE:
        _regenerate_golden()
        pytest.skip("UPDATE_GOLDEN_BASELINE=1: regenerated golden files, skipping assertions")

    # Reset global config to prevent state leakage from earlier tests
    # (e.g. SpectralService._apply_fitting mutates parsimony dict in-place)
    reset_runtime_config()

    tmp_results = tmp_path_factory.mktemp("golden_regression")

    context = RuntimeContext.bootstrap(
        data_dir=FIXTURES / "loupe",
        results_dir=tmp_results,
    )

    from rich.console import Console

    service = PipelineService(
        console=Console(quiet=True),
        context=context,
    )

    result = service.run_full_pipeline(
        sol=SOL,
        target=TARGET,
        scan=SCAN,
        data_dir=FIXTURES / "loupe",
        results_dir=tmp_results,
    )

    scan_dir = tmp_results / TARGET / f"{SOL}_{SCAN}"
    assert scan_dir.exists(), f"Pipeline did not produce results at {scan_dir}"

    return result, scan_dir


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    not _BACKGROUND_FILE.exists(),
    reason=f"Background file not found: {_BACKGROUND_FILE}",
)
@pytest.mark.skipif(
    not _golden_exists() and not _UPDATE_MODE,
    reason="Golden baseline files not found. Run: python scripts/generate_golden_baseline.py",
)
class TestGoldenBaselineRegression:
    """Compare pipeline outputs against golden baseline snapshots."""

    def test_calibration_exact_match(self):
        """Calibration arrays must be bit-identical (deterministic polynomial)."""
        golden = np.load(GOLDEN_DIR / "calibration_arrays.npz")

        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(2148)
        r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)
        r1_wavenumber = wavenumber[r1_mask]

        np.testing.assert_array_equal(
            wavelength, golden["wavelength"],
            err_msg="Wavelength array diverged from golden baseline",
        )
        np.testing.assert_array_equal(
            wavenumber, golden["wavenumber"],
            err_msg="Wavenumber array diverged from golden baseline",
        )
        np.testing.assert_array_equal(
            r1_wavenumber, golden["r1_wavenumber"],
            err_msg="R1 wavenumber array diverged from golden baseline",
        )

    def test_preprocessed_spectra_match(self, pipeline_output):
        """Despiked+baselined spectra must match within tight tolerance."""
        _, scan_dir = pipeline_output
        golden = np.load(GOLDEN_DIR / "preprocessed_spectra.npz")

        preproc_csv = (
            scan_dir / f"{SOL}_{TARGET}_{SCAN}_R1_normalized_despiked_baselined.csv"
        )
        assert preproc_csv.exists(), f"Preprocessed CSV not found: {preproc_csv}"

        df = pd.read_csv(preproc_csv)
        raman_shift = df["raman_shift"].values
        point_cols = [c for c in df.columns if c != "raman_shift"]
        spectra = df[point_cols].values.T

        np.testing.assert_array_equal(
            np.array(point_cols), golden["point_ids"],
            err_msg="Point IDs diverged from golden baseline",
        )
        np.testing.assert_allclose(
            raman_shift, golden["raman_shift"],
            atol=1e-10,
            err_msg="Raman shift values diverged from golden baseline",
        )
        np.testing.assert_allclose(
            spectra, golden["spectra"],
            atol=1e-10,
            err_msg="Preprocessed spectra diverged from golden baseline",
        )

    def test_fitted_peaks_match(self, pipeline_output):
        """Per-modality accepted peaks must match (exact ints, rtol=1e-6 floats)."""
        _, scan_dir = pipeline_output

        with open(GOLDEN_DIR / "fitted_peaks.json") as f:
            golden = json.load(f)

        modality_info = {
            "minerals": {
                "subdir": "minerals_fit",
                "pattern": f"{SOL}_{TARGET}_{SCAN}_R1_accepted_peaks.csv",
            },
            "organics": {
                "subdir": "organics_fit",
                "pattern": f"{SOL}_{TARGET}_{SCAN}_R1_organics_accepted_peaks.csv",
            },
            "hydration": {
                "subdir": "hydration_fit",
                "pattern": f"{SOL}_{TARGET}_{SCAN}_R1_hydration_accepted_peaks.csv",
            },
        }

        for modality, info in modality_info.items():
            csv_path = scan_dir / info["subdir"] / info["pattern"]
            golden_mod = golden[modality]

            if golden_mod["n_peaks"] == 0:
                # Golden has no peaks — current should also have none
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    assert len(df) == 0, (
                        f"{modality}: golden has 0 peaks but current has {len(df)}"
                    )
                continue

            assert csv_path.exists(), (
                f"{modality}: expected accepted peaks CSV at {csv_path}"
            )
            df = pd.read_csv(csv_path)
            current_records = json.loads(
                df.to_json(orient="records", double_precision=10)
            )

            assert len(current_records) == golden_mod["n_peaks"], (
                f"{modality}: peak count mismatch: "
                f"current={len(current_records)}, golden={golden_mod['n_peaks']}"
            )

            assert list(df.columns) == golden_mod["columns"], (
                f"{modality}: column mismatch: "
                f"current={list(df.columns)}, golden={golden_mod['columns']}"
            )

            for i, (cur, gld) in enumerate(
                zip(current_records, golden_mod["records"])
            ):
                for col in golden_mod["columns"]:
                    cur_val = cur.get(col)
                    gld_val = gld.get(col)

                    if isinstance(gld_val, float):
                        np.testing.assert_allclose(
                            cur_val,
                            gld_val,
                            rtol=1e-6,
                            err_msg=f"{modality} peak {i}, col '{col}'",
                        )
                    else:
                        assert cur_val == gld_val, (
                            f"{modality} peak {i}, col '{col}': "
                            f"current={cur_val!r}, golden={gld_val!r}"
                        )

    def test_review_decisions_match(self, pipeline_output):
        """Unified accepted peaks table must match (exact discrete, rtol=1e-6 floats)."""
        _, scan_dir = pipeline_output

        with open(GOLDEN_DIR / "review_decisions.json") as f:
            golden = json.load(f)

        review_csv = scan_dir / f"{SOL}_{TARGET}_{SCAN}_accepted_peaks.csv"

        if golden["n_rows"] == 0:
            if review_csv.exists():
                df = pd.read_csv(review_csv)
                assert len(df) == 0, (
                    f"Golden has 0 review rows but current has {len(df)}"
                )
            return

        assert review_csv.exists(), (
            f"Unified accepted peaks CSV not found: {review_csv}"
        )
        df = pd.read_csv(review_csv)
        current_records = json.loads(
            df.to_json(orient="records", double_precision=10)
        )

        assert len(current_records) == golden["n_rows"], (
            f"Review row count mismatch: "
            f"current={len(current_records)}, golden={golden['n_rows']}"
        )

        assert list(df.columns) == golden["columns"], (
            f"Review column mismatch: "
            f"current={list(df.columns)}, golden={golden['columns']}"
        )

        # Discrete columns that must match exactly
        discrete_cols = {
            "sol", "target", "scan", "modality", "point", "label_id",
            "peak_ID", "keep", "user_keep", "reviewed", "reject_reason",
        }
        # Float columns that match with tolerance
        float_cols = {"mean", "amplitude", "fwhm", "snr", "r_squared"}

        for i, (cur, gld) in enumerate(
            zip(current_records, golden["records"])
        ):
            for col in golden["columns"]:
                cur_val = cur.get(col)
                gld_val = gld.get(col)

                if col in float_cols and isinstance(gld_val, (int, float)):
                    if gld_val is not None and cur_val is not None:
                        np.testing.assert_allclose(
                            float(cur_val),
                            float(gld_val),
                            rtol=1e-6,
                            err_msg=f"Review row {i}, col '{col}'",
                        )
                else:
                    assert cur_val == gld_val, (
                        f"Review row {i}, col '{col}': "
                        f"current={cur_val!r}, golden={gld_val!r}"
                    )

    def test_pipeline_summary_match(self, pipeline_output):
        """Peak counts per modality and spectra shape must match exactly."""
        _, scan_dir = pipeline_output

        with open(GOLDEN_DIR / "pipeline_summary.json") as f:
            golden = json.load(f)

        # Verify spectra shape from current run
        preproc_csv = (
            scan_dir / f"{SOL}_{TARGET}_{SCAN}_R1_normalized_despiked_baselined.csv"
        )
        assert preproc_csv.exists()
        df = pd.read_csv(preproc_csv)
        point_cols = [c for c in df.columns if c != "raman_shift"]
        current_shape = [len(point_cols), len(df)]

        assert current_shape == golden["spectra_shape"], (
            f"Spectra shape mismatch: current={current_shape}, "
            f"golden={golden['spectra_shape']}"
        )
        assert len(point_cols) == golden["n_points"], (
            f"Point count mismatch: current={len(point_cols)}, "
            f"golden={golden['n_points']}"
        )

        # Verify per-modality peak counts
        modality_info = {
            "minerals": {
                "subdir": "minerals_fit",
                "pattern": f"{SOL}_{TARGET}_{SCAN}_R1_accepted_peaks.csv",
            },
            "organics": {
                "subdir": "organics_fit",
                "pattern": f"{SOL}_{TARGET}_{SCAN}_R1_organics_accepted_peaks.csv",
            },
            "hydration": {
                "subdir": "hydration_fit",
                "pattern": f"{SOL}_{TARGET}_{SCAN}_R1_hydration_accepted_peaks.csv",
            },
        }

        for modality, info in modality_info.items():
            csv_path = scan_dir / info["subdir"] / info["pattern"]
            if csv_path.exists():
                current_count = len(pd.read_csv(csv_path))
            else:
                current_count = 0

            golden_count = golden["peak_counts"][modality]
            assert current_count == golden_count, (
                f"{modality} peak count mismatch: "
                f"current={current_count}, golden={golden_count}"
            )

        # Total peaks
        current_total = sum(
            len(pd.read_csv(scan_dir / info["subdir"] / info["pattern"]))
            if (scan_dir / info["subdir"] / info["pattern"]).exists()
            else 0
            for info in modality_info.values()
        )
        assert current_total == golden["total_peaks"], (
            f"Total peak count mismatch: "
            f"current={current_total}, golden={golden['total_peaks']}"
        )
