"""Backfill ML-despiked fluorescence input rows from stored masks.

Builds ``dark_sub_despiked`` rows (services/fluor_inputs.py) for scans that
already have stored ML cosmic-ray masks, without reprocessing. Fails closed:

- By default it never writes the database it is pointed at. It copies the
  source to a new file (SQLite online backup) and backfills the copy.
  Writing in place needs both ``in_place`` and ``allow_live``.
- It holds an exclusive lock file next to the target database for the whole
  run, so two backfills cannot overlap.
- In place, it refuses while a SHERLOC CLI process that writes the database
  (pipeline runs, reclassification, ingest, fitting, persistence, backfills)
  is running against the same file. Reclassification also checks that
  ``spectra`` is unchanged across its own run.
- Each scan is one ``BEGIN IMMEDIATE`` transaction with a short busy timeout.
  If the database turns busy part-way, the run stops there and reports the
  scans already committed (``BackfillResult.stopped``) instead of waiting.
"""

from __future__ import annotations

import fcntl
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

#: CLI subcommands that write the pipeline database.
WRITER_COMMANDS = (
    "full-pipeline", "process-new", "ingest", "pds-ingest", "fit-fluor",
    "persist-peaks", "backfill", "backfill-masks", "backfill-public-release",
    "backfill-fluor-inputs", "reclassify-targets", "reclassify-scan-types",
    "reclassify-scan-classes", "reclassify-product-roles", "apply-review",
)
BUSY_TIMEOUT_MS = 5000


class BackfillRefused(RuntimeError):
    """Refused before writing anything (preflight); nothing was written."""


@dataclass
class BackfillResult:
    database: Path
    scans: int = 0
    rows_written: int = 0
    rows_removed: int = 0
    regions_without_mask: int = 0
    failed_scans: List[Tuple[int, str, str, str]] = field(default_factory=list)
    #: Set when the run stopped early (database busy); scans before it committed.
    stopped: Optional[str] = None


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


#: Database paths the CLI uses when a writer names none (relative to its cwd).
IMPLICIT_DB_NAMES = ("phase.db", "sherloc.db")


def writers_on(db_path: Path, proc_root: Path = Path("/proc")) -> List[int]:
    """PIDs of SHERLOC CLI processes that write ``db_path``.

    A process counts when its command line runs the SHERLOC CLI with a writing
    subcommand and it may target this file: an argument or ``SHERLOC_DB_PATH``
    resolving (against the process's working directory) to it, or, when it
    names no database, the implicit ``./phase.db`` / ``./sherloc.db`` in its
    working directory. A writer whose working directory cannot be read counts
    (fail closed). Without a readable ``proc_root`` nothing can be verified, so
    this raises ``BackfillRefused``.
    """
    if not proc_root.is_dir():
        raise BackfillRefused(
            f"cannot inspect running processes ({proc_root} unavailable); "
            "refusing an in-place write"
        )
    pids = []
    me = os.getpid()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [a.decode(errors="replace") for a in argv if a]
        if "sherloc" not in " ".join(args) or not any(cmd in args for cmd in WRITER_COMMANDS):
            continue
        try:
            cwd = Path(os.readlink(entry / "cwd"))
        except OSError:
            pids.append(int(entry.name))  # cannot tell where it writes: assume here
            continue
        named = [cwd / a for a in args if not a.startswith("-")]
        env_db = None
        try:
            for item in (entry / "environ").read_bytes().split(b"\0"):
                if item.startswith(b"SHERLOC_DB_PATH="):
                    env_db = cwd / item.split(b"=", 1)[1].decode(errors="replace")
        except OSError:
            pass
        candidates = named + ([env_db] if env_db else [])
        if not any(str(a).endswith(".db") for a in args) and env_db is None:
            candidates += [cwd / name for name in IMPLICIT_DB_NAMES]
        if any(_same_file(c, db_path) for c in candidates):
            pids.append(int(entry.name))
    return pids


@contextmanager
def exclusive_lock(db_path: Path) -> Iterator[Path]:
    """Hold ``<db>.fluor-inputs.lock`` exclusively, or refuse at once."""
    lock_path = db_path.with_name(db_path.name + ".fluor-inputs.lock")
    handle = open(lock_path, "w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BackfillRefused(f"another backfill holds {lock_path}") from exc
        yield lock_path
    finally:
        handle.close()


def copy_database(source: Path, dest: Path) -> None:
    """Consistent copy of a live SQLite database (online backup API)."""
    if dest.exists():
        raise BackfillRefused(f"{dest} exists; choose a new path for the copy")
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def scans_with_masks(db_path: Path, sol: Optional[int] = None) -> List[Tuple[str, int, str, str]]:
    """(scan id, sol, target, scan name) for scans that have stored masks."""
    query = """
        SELECT DISTINCT sc.id, sc.sol_number, sc.target, sc.scan_name
        FROM cosmic_ray_masks m
        JOIN spectra s ON s.id = m.spectrum_id
        JOIN scan_points sp ON sp.id = s.scan_point_id
        JOIN scans sc ON sc.id = sp.scan_id
    """
    params: Sequence[Any] = ()
    if sol is not None:
        query += " WHERE sc.sol_number = ?"
        params = (sol,)
    query += " ORDER BY sc.sol_number, sc.target, sc.scan_name"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        return [tuple(row) for row in conn.execute(query, params)]


def backfill(
    source: Path,
    config: Any,
    *,
    out: Optional[Path] = None,
    in_place: bool = False,
    allow_live: bool = False,
    sol: Optional[int] = None,
    on_scan=None,
) -> BackfillResult:
    """Backfill despiked rows into a copy of ``source`` (default) or in place."""
    from sqlalchemy import create_engine, event
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.orm import Session

    from sherloc_pipeline.services.fluor_inputs import rebuild_despiked_rows

    source = Path(source)
    if not source.exists():
        raise BackfillRefused(f"database not found: {source}")
    if in_place:
        if out is not None:
            raise BackfillRefused("give either an output copy or in_place, not both")
        if not allow_live:
            raise BackfillRefused(
                "writing in place needs allow_live (the operator's explicit go)"
            )
        target = source
    else:
        if out is None:
            raise BackfillRefused("give an output path for the copy, or in_place")
        target = Path(out)

    with exclusive_lock(target):
        if in_place:
            busy = writers_on(target)
            if busy:
                raise BackfillRefused(
                    f"SHERLOC writer process(es) {busy} are using {target}"
                )
        else:
            copy_database(source, target)

        engine = create_engine(
            f"sqlite:///{target}", connect_args={"timeout": BUSY_TIMEOUT_MS / 1000}
        )

        # SQLAlchemy's pysqlite recipe for BEGIN IMMEDIATE: take the write
        # lock at the start of each scan's transaction, not at its first write.
        @event.listens_for(engine, "connect")
        def _no_implicit_begin(dbapi_connection, _record):
            dbapi_connection.isolation_level = None

        @event.listens_for(engine, "begin")
        def _immediate(conn):
            conn.exec_driver_sql("BEGIN IMMEDIATE")

        result = BackfillResult(database=target)
        for scan_id, scan_sol, target_name, scan_name in scans_with_masks(target, sol):
            try:
                with Session(engine) as session, session.begin():
                    counts = rebuild_despiked_rows(session, scan_id, config)
            except (OperationalError, sqlite3.OperationalError) as exc:
                if "locked" in str(exc) or "busy" in str(exc):
                    result.stopped = (
                        f"database busy at {scan_sol:04d} {target_name} {scan_name}: {exc}"
                    )
                    break
                result.failed_scans.append((scan_sol, target_name, scan_name, str(exc)))
                continue
            except Exception as exc:  # one scan's failure does not stop the rest
                result.failed_scans.append((scan_sol, target_name, scan_name, str(exc)))
                continue
            result.scans += 1
            result.rows_written += counts["despiked_rows_written"]
            result.rows_removed += counts["despiked_rows_removed"]
            result.regions_without_mask += counts["regions_without_mask"]
            if on_scan is not None:
                on_scan(scan_sol, target_name, scan_name, counts)
        engine.dispose()
    return result
