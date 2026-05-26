#!/usr/bin/env python
"""Generate golden baseline snapshots for regression testing (R-024).

Runs PipelineService.run_full_pipeline() on sol 921 detail_1 test fixtures
and serializes structured outputs as reference snapshots.

Usage:
    python scripts/generate_golden_baseline.py

Outputs are written to tests/golden/sol_921_detail_1/.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden" / "sol_921_detail_1"

SOL = "0921"
TARGET = "Amherst_Point"
SCAN = "detail_1"


def main() -> int:
    # Import pipeline components
    from rich.console import Console

    from sherloc_pipeline.core.calibration import (
        calculate_loupe_wavelength_wavenumber,
    )
    from sherloc_pipeline.services.pipeline import PipelineService
    from sherloc_pipeline.services.runtime import RuntimeContext

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Calibration arrays (deterministic, no pipeline run needed) ---
    print("Generating calibration arrays...")
    wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(2148)
    r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)
    r1_wavenumber = wavenumber[r1_mask]

    np.savez_compressed(
        GOLDEN_DIR / "calibration_arrays.npz",
        wavelength=wavelength,
        wavenumber=wavenumber,
        r1_wavenumber=r1_wavenumber,
    )
    print(f"  calibration_arrays.npz: wavelength({wavelength.shape}), "
          f"wavenumber({wavenumber.shape}), r1_wavenumber({r1_wavenumber.shape})")

    # --- 2. Run pipeline ---
    print("Running full pipeline on sol 921 detail_1...")
    with tempfile.TemporaryDirectory(prefix="golden_baseline_") as tmp_dir:
        tmp_results = Path(tmp_dir)

        context = RuntimeContext.bootstrap(
            data_dir=FIXTURES / "loupe",
            results_dir=tmp_results,
        )

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

        print(f"  Pipeline result: {result.summary}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")

        # Locate scan results directory
        scan_dir = tmp_results / TARGET / f"{SOL}_{SCAN}"
        if not scan_dir.exists():
            print(f"ERROR: Expected results directory not found: {scan_dir}")
            return 1

        # --- 3. Preprocessed spectra ---
        print("Extracting preprocessed spectra...")
        preproc_pattern = f"{SOL}_{TARGET}_{SCAN}_R1_normalized_despiked_baselined.csv"
        preproc_csv = scan_dir / preproc_pattern
        if not preproc_csv.exists():
            print(f"ERROR: Preprocessed CSV not found: {preproc_csv}")
            return 1

        preproc_df = pd.read_csv(preproc_csv)
        raman_shift = preproc_df["raman_shift"].values
        point_cols = [c for c in preproc_df.columns if c != "raman_shift"]
        spectra = preproc_df[point_cols].values.T  # (n_points, n_channels)

        np.savez_compressed(
            GOLDEN_DIR / "preprocessed_spectra.npz",
            spectra=spectra,
            raman_shift=raman_shift,
            point_ids=np.array(point_cols),
        )
        print(f"  preprocessed_spectra.npz: spectra{spectra.shape}, "
              f"raman_shift({raman_shift.shape}), {len(point_cols)} points")

        # --- 4. Per-modality fitted peaks ---
        print("Extracting fitted peaks...")
        fitted_peaks = {}

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
                df = pd.read_csv(csv_path)
                fitted_peaks[modality] = {
                    "columns": list(df.columns),
                    "n_peaks": len(df),
                    "records": json.loads(df.to_json(orient="records", double_precision=10)),
                }
                print(f"  {modality}: {len(df)} peaks")
            else:
                fitted_peaks[modality] = {
                    "columns": [],
                    "n_peaks": 0,
                    "records": [],
                }
                print(f"  {modality}: no accepted peaks file found at {csv_path}")

        with open(GOLDEN_DIR / "fitted_peaks.json", "w") as f:
            json.dump(fitted_peaks, f, indent=2, sort_keys=True)

        # --- 5. Unified review decisions ---
        print("Extracting review decisions...")
        review_csv = scan_dir / f"{SOL}_{TARGET}_{SCAN}_accepted_peaks.csv"
        if review_csv.exists():
            review_df = pd.read_csv(review_csv)
            review_data = {
                "columns": list(review_df.columns),
                "n_rows": len(review_df),
                "records": json.loads(review_df.to_json(orient="records", double_precision=10)),
            }
            print(f"  review_decisions.json: {len(review_df)} rows, "
                  f"{len(review_df.columns)} columns")
        else:
            review_data = {"columns": [], "n_rows": 0, "records": []}
            print(f"  WARNING: Unified accepted peaks not found at {review_csv}")

        with open(GOLDEN_DIR / "review_decisions.json", "w") as f:
            json.dump(review_data, f, indent=2, sort_keys=True)

        # --- 6. Pipeline summary ---
        print("Building pipeline summary...")
        summary = {
            "sol": SOL,
            "target": TARGET,
            "scan": SCAN,
            "spectra_shape": list(spectra.shape),
            "raman_shift_shape": list(raman_shift.shape),
            "n_points": len(point_cols),
            "peak_counts": {
                modality: fitted_peaks[modality]["n_peaks"]
                for modality in modality_info
            },
            "total_peaks": sum(
                fitted_peaks[m]["n_peaks"] for m in modality_info
            ),
            "review_n_rows": review_data["n_rows"],
            "review_columns": review_data["columns"],
            "config_hash": result.metadata.get("config_hash", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(GOLDEN_DIR / "pipeline_summary.json", "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"  pipeline_summary.json: {summary['total_peaks']} total peaks, "
              f"spectra {spectra.shape}")

    # --- Verify ---
    print("\nGolden baseline files:")
    expected = [
        "calibration_arrays.npz",
        "preprocessed_spectra.npz",
        "fitted_peaks.json",
        "review_decisions.json",
        "pipeline_summary.json",
    ]
    all_ok = True
    for fname in expected:
        fpath = GOLDEN_DIR / fname
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"  {fname}: {size:,} bytes")
            if size == 0:
                print(f"  ERROR: {fname} is empty!")
                all_ok = False
        else:
            print(f"  ERROR: {fname} not found!")
            all_ok = False

    if all_ok:
        print("\nGolden baseline generated successfully.")
        return 0
    else:
        print("\nERROR: Some golden files are missing or empty.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
