#!/usr/bin/env python3
"""Benchmark: MLD-PER-002 — web stored-mask despike toggle latency.

Demonstrates that the web despike toggle satisfies **p95 ≤ 500 ms** per toggle
operation (requirement MLD-PER-002: "p95 ≤ 500 ms demonstrated
against a representative stored scan").

The benchmark has two parts, both offline and fully deterministic (synthetic
data only — no real measurement data, no network):

1. **Full path (the PER-002 verdict).** Builds a *representative stored scan*
   in an in-memory SQLite database (one scan, ``--n-points`` points, each with a
   DARK_SUBTRACTED R1 spectrum and a persisted ``cosmic_ray_masks`` row), then
   times the exact server-side work the web route performs for
   ``GET /api/spectra/{scan}/average?region=R1&despike=true``:

     - the stored-mask lookup (``CRMaskService.get_masks_for_spectra`` — the
       single indexed query over ``spectrum_id``),
     - boolean row-mask construction from each stored channel-index list,
     - the shared interpolation replacement per point
       (``_collect_single_region``), and
     - the trim-mean average over the despiked points (``_compute_average``).

   This is the representative toggle path; its p95 is the requirement verdict.
   (HTTP/JSON serialization is generic FastAPI overhead, not toggle-specific,
   and is excluded.)

2. **Component breakdown.** Times the interpolation replacement
   (``apply_mask_replacement``) on a single ≤2148-channel array in isolation,
   so the dominant arithmetic cost is visible separately.

Exit codes
----------
0 — full-path p95 ≤ 500 ms (PASS)
1 — full-path p95 > 500 ms (FAIL) or any runtime error
"""

import argparse
import json
import platform
import sys
import time
import uuid
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sherloc_pipeline.core.preprocessing import apply_mask_replacement

# ---------------------------------------------------------------------------
# Budget constant (MLD-PER-002)
# ---------------------------------------------------------------------------
P95_BUDGET_MS: float = 500.0
N_CHANNELS_FULL: int = 2148


def build_synthetic_spectrum(n_channels: int, rng: np.random.Generator) -> np.ndarray:
    """Return a synthetic dark-subtracted intensity array (smooth + noise)."""
    x = np.linspace(0.0, 1.0, n_channels)
    baseline = 500.0 + 300.0 * x - 200.0 * x**2 + 50.0 * np.sin(8.0 * np.pi * x)
    noise = rng.normal(0.0, 15.0, size=n_channels)
    return (baseline + noise).astype(np.float64)


def build_boolean_mask(n_channels: int, n_spikes: int, rng: np.random.Generator) -> np.ndarray:
    """Return a boolean mask with ``n_spikes`` True positions chosen at random."""
    spike_channels = rng.choice(n_channels, size=n_spikes, replace=False)
    mask = np.zeros(n_channels, dtype=bool)
    mask[spike_channels] = True
    return mask


def compute_stats(timings_ms: list) -> dict:
    """Summary statistics over a list of per-iteration millisecond times."""
    arr = np.array(timings_ms, dtype=float)
    return {
        "min_ms": float(np.min(arr)),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
        "n_iterations": len(timings_ms),
    }


def _print_stats(label: str, stats: dict) -> None:
    print(
        f"{label} ({stats['n_iterations']} iterations):\n"
        f"  min    {stats['min_ms']:.3f} ms\n"
        f"  mean   {stats['mean_ms']:.3f} ms\n"
        f"  median {stats['median_ms']:.3f} ms\n"
        f"  p95    {stats['p95_ms']:.3f} ms\n"
        f"  p99    {stats['p99_ms']:.3f} ms\n"
        f"  max    {stats['max_ms']:.3f} ms\n"
    )


# ---------------------------------------------------------------------------
# Representative stored scan + full-path timing
# ---------------------------------------------------------------------------


def build_representative_db(n_points: int, n_spikes: int, seed: int):
    """Build an in-memory SQLite with a representative stored scan.

    One scan, ``n_points`` points, each with a DARK_SUBTRACTED R1 spectrum and a
    persisted ``cosmic_ray_masks`` row of ``n_spikes`` in-window channels. The
    spectra are stored exactly as production stores them (zlib-compressed
    float32), so the timed path includes the real decode.
    """
    from sqlalchemy import create_engine, event
    from sqlalchemy.pool import StaticPool

    from sherloc_pipeline.database.connection import create_all_tables, get_session_factory
    from sherloc_pipeline.database.models import (
        CosmicRayMaskORM,
        ScanORM,
        ScanPointORM,
        SolORM,
        SpectrumORM,
    )
    from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _pragma(dbapi_connection, connection_record):  # pragma: no cover - trivial
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    create_all_tables(engine)
    rng = np.random.default_rng(seed)
    lo, hi = DEFAULT_MANIFEST.region_windows["R1"]
    scan_id = str(uuid.uuid4())

    session = get_session_factory(engine)()
    try:
        session.add(SolORM(sol_number=921, data_source="loupe"))
        session.flush()
        session.add(
            ScanORM(
                id=scan_id,
                sol_number=921,
                scan_name="detail_1",
                target="Benchmark_Point",
                scan_id="0921_Benchmark_Point_detail_1",
                sclk_start=730000000,
                sclk_stop=730001000,
                n_points=n_points,
                n_channels=N_CHANNELS_FULL,
                shots_per_point=50,
                laser_wavelength_nm=248.5794,
                data_source="loupe",
                target_type="mars_target",
                scan_class="primary",
                scan_type="detail",
            )
        )
        session.flush()
        for i in range(n_points):
            pt_id = str(uuid.uuid4())
            session.add(
                ScanPointORM(
                    id=pt_id,
                    scan_id=scan_id,
                    point_index=i,
                    photodiode_mean=5000.0,
                    photodiode_std=10.0,
                )
            )
            session.flush()
            sp_id = str(uuid.uuid4())
            arr = build_synthetic_spectrum(N_CHANNELS_FULL, rng).astype(np.float32)
            session.add(
                SpectrumORM(
                    id=sp_id,
                    scan_point_id=pt_id,
                    region="R1",
                    spectrum_type="dark_subtracted",
                    processing_level="dark_subtracted",
                    intensities=zlib.compress(arr.tobytes()),
                )
            )
            session.flush()
            channels = sorted(
                int(c) for c in rng.choice(np.arange(lo, hi), size=n_spikes, replace=False)
            )
            session.add(
                CosmicRayMaskORM(
                    id=str(uuid.uuid4()),
                    spectrum_id=sp_id,
                    method=DEFAULT_MANIFEST.provenance_label,
                    model_sha256=DEFAULT_MANIFEST.sha256,
                    tau=float(DEFAULT_MANIFEST.tau["R1"]),
                    channel_indices=channels,
                    n_flagged=len(channels),
                )
            )
        session.commit()
    finally:
        session.close()
    return engine, scan_id


def time_full_path(engine, scan_id: str, iterations: int, interp: str) -> dict:
    """Time the server-side stored-mask toggle path for an R1 average view.

    Reproduces what the route runs for ``?despike=true``: open a session, fetch
    stored masks (indexed query), build row masks, apply per point, average.
    """
    from sherloc_pipeline.database.connection import get_session_factory
    from sherloc_pipeline.database.models import ScanPointORM, SpectrumORM
    from sherloc_pipeline.web.routes.spectra import (
        _collect_single_region,
        _compute_average,
        _get_wavelength_wavenumber,
    )

    _, _, mask = _get_wavelength_wavenumber("R1")
    selected_channels = np.where(mask)[0]
    config = SimpleNamespace(preprocessing={"trim_mean_baseline_pct": 0.02})
    factory = get_session_factory(engine)

    timings_ms: list = []
    for _ in range(iterations):
        session = factory()
        try:
            t0 = time.perf_counter()
            rows = (
                session.query(SpectrumORM, ScanPointORM.point_index)
                .join(ScanPointORM)
                .filter(
                    ScanPointORM.scan_id == scan_id,
                    SpectrumORM.region == "R1",
                    SpectrumORM.spectrum_type == "dark_subtracted",
                )
                .all()
            )
            indexed, _outcome = _collect_single_region(
                session, "R1", rows, mask, selected_channels, True, interp
            )
            stacked = np.stack([arr for _, arr in indexed])
            _compute_average(stacked, "trim_mean", None, config)
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1_000.0)
        finally:
            session.close()
    return compute_stats(timings_ms)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark MLD-PER-002: web despike toggle p95 ≤ 500 ms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-channels", type=int, default=2148, help="Channels for the component microbench (≤2148).")
    parser.add_argument("--n-points", type=int, default=30, help="Points in the representative stored scan (full path).")
    parser.add_argument("--n-spikes", type=int, default=20, help="Flagged channels per mask (cosmic rays are sparse).")
    parser.add_argument("--iterations", type=int, default=1000, help="Timed toggle operations per measurement.")
    parser.add_argument("--interpolation-method", default="linear", help="pandas interpolation method.")
    parser.add_argument("--seed", type=int, default=20260611, help="RNG seed for reproducibility.")
    parser.add_argument("--out", type=Path, default=None, help="Optional path for a JSON verification record.")
    args = parser.parse_args()

    print(
        "Benchmark: MLD-PER-002 — web stored-mask despike toggle\n"
        f"  full path: n_points={args.n_points}  n_spikes={args.n_spikes}  "
        f"iterations={args.iterations}  method={args.interpolation_method!r}  seed={args.seed}\n"
        f"  component: n_channels={args.n_channels}\n"
    )

    # --- Full path (verdict basis): representative stored scan, real route work
    engine, scan_id = build_representative_db(args.n_points, args.n_spikes, args.seed)
    full_stats = time_full_path(engine, scan_id, args.iterations, args.interpolation_method)
    _print_stats("Full toggle path (DB lookup + row masks + apply + average)", full_stats)

    # --- Component breakdown: interpolation only
    rng = np.random.default_rng(args.seed + 1)
    series = pd.Series(build_synthetic_spectrum(args.n_channels, rng))
    bool_mask = build_boolean_mask(args.n_channels, args.n_spikes, rng)
    comp_timings = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        apply_mask_replacement(series, bool_mask, args.interpolation_method)
        comp_timings.append((time.perf_counter() - t0) * 1_000.0)
    comp_stats = compute_stats(comp_timings)
    _print_stats("Component: interpolation replacement only", comp_stats)

    p95 = full_stats["p95_ms"]
    passed = p95 <= P95_BUDGET_MS
    verdict = "PASS" if passed else "FAIL"
    print(
        f"MLD-PER-002  p95 budget = {P95_BUDGET_MS:.0f} ms  "
        f"full-path measured p95 = {p95:.3f} ms  →  {verdict}"
    )

    if args.out is not None:
        record = {
            "requirement": "MLD-PER-002",
            "description": "web despike toggle p95 <= 500 ms (representative stored scan)",
            "args": {
                "n_channels": args.n_channels,
                "n_points": args.n_points,
                "n_spikes": args.n_spikes,
                "iterations": args.iterations,
                "interpolation_method": args.interpolation_method,
                "seed": args.seed,
            },
            "host_platform": platform.platform(),
            "python_version": sys.version,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "full_path_stats_ms": full_stats,
            "component_interpolation_stats_ms": comp_stats,
            "budget_ms": P95_BUDGET_MS,
            "verdict": verdict,
            "passed": passed,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2))
        print(f"\nverification record written to: {args.out}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
