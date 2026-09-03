#!/usr/bin/env python3
"""Evidence sweep for the hydration cosmic-ray veto (issue #38).

Re-runs the veto over hydration rows already in a pipeline database and reports
per-mechanism catch/flag counts and their overlap, so the thresholds can be
ratified from measurement rather than from argument. It is read-only: nothing is
written back to the database and no row is pruned. Pruning existing spurious
rows is a separate, operator-gated activity.

The output is deliberately aggregate — counts, and distribution summaries of the
veto's own derived ratios. No spectral intensities, centres, or target names are
printed, so the report can be produced without moving measurement values around.

Usage:
    python3 scripts/hydration_cr_veto_report.py --db /path/to/phase.db
    python3 scripts/hydration_cr_veto_report.py --db ... --scan <scan-uuid>

Write the resulting table into the "Sweep results" section of
docs/reports/HYDRATION_CR_VETO_EVIDENCE.md before proposing a default-on flip.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from sherloc_pipeline.core.calibration import (
    calculate_loupe_wavelength_wavenumber,
    get_region_wavelength_mask,
)
from sherloc_pipeline.core.hydration_veto import (
    FLAG_AMPLITUDE_DROP,
    FLAG_FWHM_FLOOR_PINNED,
    FLAG_MASK_HIT,
    HydrationVetoConfig,
    despike_for_veto,
    evaluate_hydration_peak,
)
from sherloc_pipeline.database.models import (
    FittedPeakORM,
    ScanPointORM,
    SpectrumORM,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="SQLite database path")
    parser.add_argument("--scan", default=None, help="Restrict to one scan UUID")
    parser.add_argument(
        "--floor", type=float, default=50.0, help="FWHM floor the fits were bounded by"
    )
    parser.add_argument(
        "--floor-tolerance",
        type=float,
        default=0.5,
        help="Rows within this of the floor count as floor-pinned",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Sweep every hydration row, not just the floor-pinned ones",
    )
    parser.add_argument(
        "--amplitude-drop-ratio-max", type=float, default=0.5,
    )
    parser.add_argument("--mask-min-drop-ratio", type=float, default=0.10)
    parser.add_argument("--center-window-cm1", type=float, default=15.0)
    return parser.parse_args(argv)


def _percentiles(values: list[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=float)
    return " / ".join(
        f"p{p}={np.percentile(arr, p):.3f}" for p in (10, 50, 90)
    )


def _config_from_args(args: argparse.Namespace) -> HydrationVetoConfig:
    """Build the sweep's veto config from the parsed command line.

    ``--floor-tolerance`` has to reach ``fwhm_floor_epsilon_cm1`` as well as the
    row selection: the sweep reports a bound-pinning count, and if the two came
    apart that count would be measured against the 0.5 cm-1 default rather than
    the tolerance the operator asked about.
    """
    return HydrationVetoConfig(
        enabled=True,
        action="reject",
        center_window_cm1=args.center_window_cm1,
        amplitude_drop_ratio_max=args.amplitude_drop_ratio_max,
        mask_min_drop_ratio=args.mask_min_drop_ratio,
        fwhm_floor_cm1=args.floor,
        fwhm_floor_epsilon_cm1=args.floor_tolerance,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.db.exists():
        print(f"error: no such database: {args.db}", file=sys.stderr)
        return 1

    config = _config_from_args(args)

    wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=2148)
    r1_mask = get_region_wavelength_mask(wavelength, "R1")
    x = wavenumber[r1_mask]

    engine = create_engine(f"sqlite:///{args.db}")
    counts: Counter[str] = Counter()
    mask_ratios: list[float] = []
    drop_ratios: list[float] = []

    with Session(engine) as session:
        query = (
            session.query(FittedPeakORM, SpectrumORM)
            .join(SpectrumORM, FittedPeakORM.spectrum_id == SpectrumORM.id)
            .filter(FittedPeakORM.fit_modality == "hydration")
            .filter(SpectrumORM.region == "R1")
        )
        if args.scan is not None:
            query = query.join(
                ScanPointORM, SpectrumORM.scan_point_id == ScanPointORM.id
            ).filter(ScanPointORM.scan_id == args.scan)

        for peak, spectrum in query:
            counts["rows_total"] += 1
            fwhm = peak.fwhm_cm1
            pinned = (
                fwhm is not None
                and fwhm <= args.floor + args.floor_tolerance
            )
            if pinned:
                counts["rows_floor_pinned"] += 1
            if not args.all_rows and not pinned:
                continue
            counts["rows_swept"] += 1

            try:
                y_full = np.frombuffer(
                    zlib.decompress(spectrum.intensities), dtype=np.float32
                ).astype(np.float64)
            except Exception:
                counts["rows_unreadable"] += 1
                continue
            if y_full.size < r1_mask.size:
                counts["rows_unreadable"] += 1
                continue
            y = y_full[r1_mask]

            despiked, spike_mask = despike_for_veto(x, y)
            verdict = evaluate_hydration_peak(
                peak.center_cm1, fwhm, x, y, despiked, spike_mask, config
            )

            if verdict.amplitude_drop_ratio is not None:
                drop_ratios.append(verdict.amplitude_drop_ratio)
            if verdict.mask_hit:
                counts[FLAG_MASK_HIT] += 1
                if verdict.amplitude_drop_ratio is not None:
                    mask_ratios.append(verdict.amplitude_drop_ratio)
            if FLAG_AMPLITUDE_DROP in verdict.flags:
                counts[FLAG_AMPLITUDE_DROP] += 1
            if FLAG_FWHM_FLOOR_PINNED in verdict.flags:
                counts[FLAG_FWHM_FLOOR_PINNED] += 1
            if verdict.mask_hit and FLAG_AMPLITUDE_DROP in verdict.flags:
                counts["overlap_mask_and_amplitude"] += 1
            if verdict.vetoed:
                counts["vetoed"] += 1
            else:
                counts["survived"] += 1

    print("Hydration cosmic-ray veto sweep (issue #38)")
    print(f"  database            : {args.db.name}")
    print(f"  scope               : {'all hydration rows' if args.all_rows else 'floor-pinned rows only'}")
    print(f"  fwhm floor          : {args.floor} +/- {args.floor_tolerance} cm-1")
    print(f"  amplitude threshold : {args.amplitude_drop_ratio_max}")
    print(f"  mask drop floor     : {args.mask_min_drop_ratio}")
    print(f"  centre window       : {args.center_window_cm1} cm-1")
    print()
    for key in (
        "rows_total",
        "rows_floor_pinned",
        "rows_swept",
        "rows_unreadable",
        FLAG_MASK_HIT,
        FLAG_AMPLITUDE_DROP,
        "overlap_mask_and_amplitude",
        FLAG_FWHM_FLOOR_PINNED,
        "vetoed",
        "survived",
    ):
        print(f"  {key:<28} {counts[key]}")
    print()
    print(f"  drop-ratio distribution (all swept) : {_percentiles(drop_ratios)}")
    print(f"  drop-ratio distribution (mask hits) : {_percentiles(mask_ratios)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
