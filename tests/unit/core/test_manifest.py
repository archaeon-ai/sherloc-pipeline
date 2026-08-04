"""Tests for core/manifest.py Loupe manifest-based scan resolution.

Covers the typo-tolerant fuzzy fallback added for issue #16: sol 1521's
raw Loupe directories (and their loupe.csv `human_readable_workspace`
field) are misspelled `meteroite` instead of `meteorite`, so exact-match
manifest resolution never lands for the DB-corrected scan_name -- and the
composite reductions (`meteroite_sum_active_median_dark`, etc.) inherit
the same typo, so the canonical composite can't be fit either.
"""
from __future__ import annotations

from pathlib import Path

from sherloc_pipeline.core.manifest import resolve_manifest_working_directory

REQUIRED_FILES = ("loupe.csv", "spatial.csv", "darkSubSpectra.csv", "photodiodeRaw.csv")


def _make_working_dir(sol_dir: Path, scan_dir_name: str, workspace: str) -> Path:
    working_dir = (
        sol_dir / scan_dir_name / f"SrlcSpecSpecSohRaw_000000-{scan_dir_name}-1_Loupe_working"
    )
    working_dir.mkdir(parents=True)
    (working_dir / "loupe.csv").write_text(f"human_readable_workspace,{workspace}\n")
    for name in REQUIRED_FILES[1:]:
        (working_dir / name).write_text("")
    return working_dir


def test_exact_match_still_resolves(tmp_path: Path) -> None:
    sol_dir = tmp_path / "sol_1697"
    expected = _make_working_dir(sol_dir, "meteorite", "meteorite")

    resolved = resolve_manifest_working_directory(
        base_data_dir=tmp_path, sol="1697", scan="meteorite"
    )

    assert resolved == expected.resolve()


def test_fuzzy_match_resolves_single_typo(tmp_path: Path) -> None:
    """sol 1521 primary meteorite scan: dir + manifest both say `meteroite`."""
    sol_dir = tmp_path / "sol_1521"
    expected = _make_working_dir(sol_dir, "meteroite", "meteroite")

    resolved = resolve_manifest_working_directory(
        base_data_dir=tmp_path, sol="1521", scan="meteorite"
    )

    assert resolved == expected.resolve()


def test_fuzzy_match_resolves_composite_scan(tmp_path: Path) -> None:
    """The reported bug: the canonical composite reduction resolves too."""
    sol_dir = tmp_path / "sol_1521"
    _make_working_dir(sol_dir, "meteroite", "meteroite")
    _make_working_dir(sol_dir, "meteroite_median_all", "meteroite_median_all")
    canonical = _make_working_dir(
        sol_dir, "meteroite_sum_active_median_dark", "meteroite_sum_active_median_dark"
    )
    _make_working_dir(
        sol_dir, "meteroite_sum_active_sum_dark", "meteroite_sum_active_sum_dark"
    )

    resolved = resolve_manifest_working_directory(
        base_data_dir=tmp_path, sol="1521", scan="meteorite_sum_active_median_dark"
    )

    assert resolved == canonical.resolve()


def test_fuzzy_match_does_not_confuse_sibling_composites(tmp_path: Path) -> None:
    """Each composite reduction must resolve to its own directory, not a sibling."""
    sol_dir = tmp_path / "sol_1521"
    _make_working_dir(sol_dir, "meteroite", "meteroite")
    median_all = _make_working_dir(
        sol_dir, "meteroite_median_all", "meteroite_median_all"
    )
    _make_working_dir(
        sol_dir, "meteroite_sum_active_median_dark", "meteroite_sum_active_median_dark"
    )
    _make_working_dir(
        sol_dir, "meteroite_sum_active_sum_dark", "meteroite_sum_active_sum_dark"
    )

    resolved = resolve_manifest_working_directory(
        base_data_dir=tmp_path, sol="1521", scan="meteorite_median_all"
    )

    assert resolved == median_all.resolve()


def test_no_match_beyond_threshold_returns_none(tmp_path: Path) -> None:
    sol_dir = tmp_path / "sol_1521"
    _make_working_dir(sol_dir, "maze", "maze")
    _make_working_dir(sol_dir, "orthofabric", "orthofabric")

    resolved = resolve_manifest_working_directory(
        base_data_dir=tmp_path, sol="1521", scan="meteorite"
    )

    assert resolved is None


def test_fuzzy_match_does_not_resolve_to_distinct_sibling_index(tmp_path: Path) -> None:
    """A close sole candidate that is a genuinely different repeat scan must not match.

    ``detail_1`` and ``detail_2`` differ only in their repeat-scan index, which
    is one edit apart -- well within the typo threshold -- but they are
    distinct scans with distinct spectral data, not a misspelling of each
    other. Requesting ``detail_1`` when only ``detail_2`` exists must refuse
    to guess rather than silently resolving to the wrong scan.
    """
    sol_dir = tmp_path / "sol_2000"
    _make_working_dir(sol_dir, "detail_2", "detail_2")

    resolved = resolve_manifest_working_directory(
        base_data_dir=tmp_path, sol="2000", scan="detail_1"
    )

    assert resolved is None


def test_fuzzy_match_does_not_resolve_to_distinct_sibling_reduction(tmp_path: Path) -> None:
    """A close sole candidate with a different reduction suffix must not match.

    ``quartz_sum_active_median_dark`` (canonical) and
    ``quartz_sum_active_sum_dark`` (alternate) differ only in the reduction
    token (``median`` vs ``sum``), which the composite-length threshold is
    generous enough to absorb as if it were a typo -- but that token is
    drawn from the pipeline's own controlled reduction vocabulary, never
    freehand-transcribed by Loupe, so a mismatch there means a genuinely
    different composite reduction, not a misspelling. Requesting the
    canonical reduction when only the alternate exists on disk must refuse
    to guess rather than silently resolving to the wrong composite.
    """
    sol_dir = tmp_path / "sol_3000"
    _make_working_dir(sol_dir, "quartz_sum_active_sum_dark", "quartz_sum_active_sum_dark")

    resolved = resolve_manifest_working_directory(
        base_data_dir=tmp_path, sol="3000", scan="quartz_sum_active_median_dark"
    )

    assert resolved is None


def test_ambiguous_fuzzy_match_refuses_to_guess(tmp_path: Path) -> None:
    """Two equally-close misspellings must not be silently disambiguated."""
    sol_dir = tmp_path / "sol_9999"
    _make_working_dir(sol_dir, "detai", "detai")  # edit distance 1 from "detail"
    _make_working_dir(sol_dir, "detail2", "detail2")  # edit distance 1 from "detail"

    resolved = resolve_manifest_working_directory(
        base_data_dir=tmp_path, sol="9999", scan="detail"
    )

    assert resolved is None
