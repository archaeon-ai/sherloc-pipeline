#!/usr/bin/env python3
"""List targetless science scans and print a read-only remediation plan.

The database is opened with SQLite ``mode=ro`` and ``query_only`` enabled.
This utility has no apply mode: it emits primary-keyed UPDATE statements for
operator review but never executes them.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from sherloc_pipeline.models.ingestion import extract_target_from_lpe
from sherloc_pipeline.models.spectra import classify_target_type


@dataclass(frozen=True)
class MissingTargetScan:
    """A detail/survey scan whose geological target is absent."""

    id: str
    sol_number: int
    scan_id: str
    scan_name: str
    scan_type: str
    n_points: int
    source_path: Optional[str]


@dataclass(frozen=True)
class Resolution:
    """A proposed target and the read-only evidence used to infer it."""

    target: Optional[str]
    source: str


def open_read_only(database: Path) -> sqlite3.Connection:
    """Open an existing SQLite database with writes disabled twice over."""
    database = database.expanduser().resolve(strict=True)
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def find_missing_scans(connection: sqlite3.Connection) -> list[MissingTargetScan]:
    """Return the deterministic census requested by the target audit."""
    rows = connection.execute(
        """
        SELECT id, sol_number, scan_id, scan_name, scan_type, n_points, source_path
        FROM scans
        WHERE target IS NULL AND scan_type IN ('detail', 'survey')
        ORDER BY sol_number, scan_id
        """
    ).fetchall()
    return [MissingTargetScan(*row) for row in rows]


def _source_sol_dir(scan: MissingTargetScan) -> Optional[Path]:
    if not scan.source_path:
        return None
    source_path = Path(scan.source_path).expanduser()
    for candidate in (source_path, *source_path.parents):
        if candidate.name.lower() == f"sol_{scan.sol_number:04d}" and candidate.is_dir():
            return candidate
    return None


def _database_context_targets(
    connection: sqlite3.Connection, sol_number: int
) -> set[str]:
    rows = connection.execute(
        """
        SELECT DISTINCT TRIM(target)
        FROM scans
        WHERE sol_number = ? AND target IS NOT NULL AND TRIM(target) <> ''
        """,
        (sol_number,),
    ).fetchall()
    return {row[0] for row in rows}


def resolve_target(
    connection: sqlite3.Connection,
    scan: MissingTargetScan,
    loupe_root: Optional[Path] = None,
) -> Resolution:
    """Resolve from the Loupe session filename, then unambiguous sol context."""
    sol_dir = _source_sol_dir(scan)
    if sol_dir is None and loupe_root is not None:
        candidate = loupe_root.expanduser() / f"sol_{scan.sol_number:04d}"
        if candidate.is_dir():
            sol_dir = candidate

    source_target = extract_target_from_lpe(sol_dir) if sol_dir else None
    context_targets = _database_context_targets(connection, scan.sol_number)

    if source_target:
        if context_targets and context_targets != {source_target}:
            candidates = ", ".join(sorted(context_targets | {source_target}))
            return Resolution(None, f"conflicting sol evidence: {candidates}")
        return Resolution(source_target, ".lpe filename")

    if len(context_targets) == 1:
        return Resolution(next(iter(context_targets)), "unique database sol context")
    if context_targets:
        return Resolution(None, "ambiguous database sol context")
    return Resolution(None, "no Loupe or database sol context")


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def update_statement(scan: MissingTargetScan, target: str) -> str:
    """Build a guarded statement keyed by both row id and source scan id."""
    target_type = classify_target_type(target, scan.scan_name)
    return (
        "UPDATE scans SET target = "
        f"{_sql_literal(target)}, target_type = {_sql_literal(target_type)} "
        f"WHERE id = {_sql_literal(scan.id)} "
        f"AND scan_id = {_sql_literal(scan.scan_id)} "
        f"AND sol_number = {scan.sol_number} AND target IS NULL;"
    )


def render_plan(
    scans: Iterable[MissingTargetScan],
    resolutions: Iterable[Resolution],
) -> str:
    """Render the census first and guarded SQL second."""
    paired = list(zip(scans, resolutions, strict=True))
    lines = [
        f"targetless science scans: {len(paired)} across "
        f"{len({scan.sol_number for scan, _ in paired})} sols",
        "sol_number\tscan_id\tscan_name\tscan_type\tn_points\tproposed_target\tevidence"
    ]
    for scan, resolution in paired:
        lines.append(
            "\t".join(
                [
                    str(scan.sol_number),
                    scan.scan_id,
                    scan.scan_name,
                    scan.scan_type,
                    str(scan.n_points),
                    resolution.target or "UNRESOLVED",
                    resolution.source,
                ]
            )
        )

    lines.extend(["", "-- REVIEW ONLY: this utility did not execute these statements."])
    for scan, resolution in paired:
        if resolution.target:
            lines.append(update_statement(scan, resolution.target))
        else:
            lines.append(
                f"-- UNRESOLVED id={_sql_literal(scan.id)}: {resolution.source}"
            )
    return "\n".join(lines)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, required=True, help="Existing SQLite database to audit"
    )
    parser.add_argument(
        "--loupe-root",
        type=Path,
        help="Optional directory containing sol_NNNN Loupe source directories",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        with open_read_only(args.database) as connection:
            scans = find_missing_scans(connection)
            resolutions = [
                resolve_target(connection, scan, args.loupe_root) for scan in scans
            ]
    except (OSError, sqlite3.Error) as exc:
        print(f"target audit failed: {exc}", file=sys.stderr)
        return 2

    print(render_plan(scans, resolutions))
    return 1 if any(resolution.target is None for resolution in resolutions) else 0


if __name__ == "__main__":
    raise SystemExit(main())
