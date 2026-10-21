#!/usr/bin/env python3
"""Compare fluorescence fits between two databases (before vs after a method change).

Joins fitted_peaks (fit_modality = fluorescence) to their scan and point in each
database, then per scan point compares the number of peaks, the set of group
labels, and the centre of each group present in both. Writes a per-point table
and a per-scan summary; prints only the output paths and aggregate counts.

Usage:
    python scripts/compare_fluorescence_fits.py BEFORE.db AFTER.db OUT_DIR
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

QUERY = """
SELECT s.sol_number AS sol, s.target AS target, s.scan_name AS scan,
       sp.point_index AS point, fp.mineral_assignment AS grp, fp.center_nm AS center_nm
FROM fitted_peaks fp
JOIN spectra sx ON sx.id = fp.spectrum_id
JOIN scan_points sp ON sp.id = sx.scan_point_id
JOIN scans s ON s.id = sp.scan_id
WHERE fp.fit_modality = 'fluorescence'
"""
KEY = ["sol", "target", "scan", "point"]


def load(db: Path) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(QUERY, conn)


def per_point(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    def summarise(df: pd.DataFrame) -> pd.DataFrame:
        # One centre per group per point: the first-fitted peak if a group repeats.
        first = df.sort_values("center_nm").drop_duplicates(KEY + ["grp"])
        groups = df.groupby(KEY)["grp"].apply(lambda g: ",".join(sorted(set(g))))
        return pd.DataFrame({"n_peaks": df.groupby(KEY).size(), "groups": groups}).join(
            first.pivot_table(index=KEY, columns="grp", values="center_nm")
        )

    b, a = summarise(before), summarise(after)
    table = b.join(a, how="outer", lsuffix="_before", rsuffix="_after")
    table[["n_peaks_before", "n_peaks_after"]] = table[
        ["n_peaks_before", "n_peaks_after"]
    ].fillna(0).astype(int)
    table[["groups_before", "groups_after"]] = table[["groups_before", "groups_after"]].fillna("")
    table["count_changed"] = table["n_peaks_before"] != table["n_peaks_after"]
    table["groups_changed"] = table["groups_before"] != table["groups_after"]
    shared = sorted(
        {c[: -len("_before")] for c in table.columns if c.endswith("_before")}
        & {c[: -len("_after")] for c in table.columns if c.endswith("_after")}
        - {"n_peaks", "groups"}
    )
    for grp in shared:
        table[f"{grp}_shift_nm"] = table[f"{grp}_after"] - table[f"{grp}_before"]
    return table.reset_index()


def per_scan(table: pd.DataFrame) -> pd.DataFrame:
    shifts = [c for c in table.columns if c.endswith("_shift_nm")]
    agg = table.groupby(["sol", "target", "scan"]).agg(
        points=("point", "size"),
        peaks_before=("n_peaks_before", "sum"),
        peaks_after=("n_peaks_after", "sum"),
        points_count_changed=("count_changed", "sum"),
        points_groups_changed=("groups_changed", "sum"),
    )
    for c in shifts:
        agg[f"median_abs_{c}"] = table.groupby(["sol", "target", "scan"])[c].apply(
            lambda s: s.abs().median()
        )
    return agg.reset_index()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    before_db, after_db, out_dir = Path(argv[0]), Path(argv[1]), Path(argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    table = per_point(load(before_db), load(after_db))
    scans = per_scan(table)
    table.to_csv(out_dir / "fluorescence_compare_points.csv", index=False)
    scans.to_csv(out_dir / "fluorescence_compare_scans.csv", index=False)
    print(out_dir / "fluorescence_compare_points.csv")
    print(out_dir / "fluorescence_compare_scans.csv")
    print(
        f"scans={len(scans)} points={len(table)} "
        f"points_count_changed={int(table['count_changed'].sum())} "
        f"points_groups_changed={int(table['groups_changed'].sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
