"""scripts/compare_fluorescence_fits.py on two small synthetic databases."""

import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "compare_fluorescence_fits.py"
spec = importlib.util.spec_from_file_location("compare_fluorescence_fits", _SCRIPT)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)

SCHEMA = """
CREATE TABLE scans (id TEXT PRIMARY KEY, sol_number INT, target TEXT, scan_name TEXT);
CREATE TABLE scan_points (id TEXT PRIMARY KEY, scan_id TEXT, point_index INT);
CREATE TABLE spectra (id TEXT PRIMARY KEY, scan_point_id TEXT);
CREATE TABLE fitted_peaks (id TEXT PRIMARY KEY, spectrum_id TEXT, fit_modality TEXT,
                           mineral_assignment TEXT, center_nm REAL);
"""


def _db(path: Path, peaks: list[tuple[int, str, float]]) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO scans VALUES ('s', 9999, 'Synthetic', 'detail_1')")
    for point in range(3):
        conn.execute("INSERT INTO scan_points VALUES (?, 's', ?)", (f"p{point}", point))
        conn.execute("INSERT INTO spectra VALUES (?, ?)", (f"x{point}", f"p{point}"))
    for i, (point, group, centre) in enumerate(peaks):
        conn.execute(
            "INSERT INTO fitted_peaks VALUES (?, ?, 'fluorescence', ?, ?)",
            (f"f{i}", f"x{point}", group, centre),
        )
    conn.execute("INSERT INTO fitted_peaks VALUES ('m', 'x0', 'minerals', 'pyroxene', NULL)")
    conn.commit()
    conn.close()
    return path


def test_counts_groups_and_shifts_per_point_and_scan(tmp_path):
    before = _db(tmp_path / "before.db", [(0, "group3", 285.0), (1, "group1a", 304.0)])
    after = _db(
        tmp_path / "after.db",
        [(0, "group3", 286.0), (1, "group1a", 304.0), (1, "group1b", 325.0), (2, "group2", 340.0)],
    )

    assert compare.main([str(before), str(after), str(tmp_path / "out")]) == 0

    points = pd.read_csv(tmp_path / "out" / "fluorescence_compare_points.csv").set_index("point")
    assert list(points["n_peaks_before"]) == [1, 1, 0]
    assert list(points["n_peaks_after"]) == [1, 2, 1]
    assert list(points["groups_changed"]) == [False, True, True]
    assert points.loc[0, "group3_shift_nm"] == 1.0
    assert points.loc[1, "group1a_shift_nm"] == 0.0

    scans = pd.read_csv(tmp_path / "out" / "fluorescence_compare_scans.csv")
    row = scans.iloc[0]
    assert (row["points"], row["peaks_before"], row["peaks_after"]) == (3, 2, 4)
    assert (row["points_count_changed"], row["points_groups_changed"]) == (2, 2)
