#!/usr/bin/env python
"""Manual smoke for SQLite WAL multi-process safety (G18.12, §19.1 B.9).

Mirrors the production engine config from
``sherloc_pipeline.database.connection.get_engine`` (WAL, foreign_keys=ON,
busy_timeout=5000) and exercises one writer process concurrently with
multiple reader processes against a single SQLite database. Verifies that:

* The writer commits every row it intends to.
* Readers observe a monotonically non-decreasing committed row count.
* No process raises ``OperationalError`` (SQLITE_BUSY) or
  ``DatabaseError`` (SQLITE_CORRUPT).
* ``PRAGMA integrity_check`` returns ``ok`` after the run.

Usage::

    python scripts/smoke_wal_multiprocess.py

The container/bind-mount portion of G18.12 (uvicorn-in-container reading
through a Docker bind mount) is deferred to a Docker host. Linux bind
mounts are kernel-level and do not change SQLite WAL semantics, so the
host-side multi-process exercise verifies the SQLite contract; the Docker
half is left as a paired check with G18.13.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sherloc_pipeline.database.connection import get_engine  # noqa: E402
from sqlalchemy import text  # noqa: E402


WRITER_TXN_COUNT = 50
WRITER_ROWS_PER_TXN = 10
WRITER_TXN_SLEEP_S = 0.02
READER_COUNT = 4
READER_POLL_INTERVAL_S = 0.05
READER_TIMEOUT_S = 30.0
EXPECTED_TOTAL_ROWS = WRITER_TXN_COUNT * WRITER_ROWS_PER_TXN


def _init_schema(db_path: Path) -> None:
    engine = get_engine(db_path)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE smoke_wal ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "payload TEXT NOT NULL, "
                "ts REAL NOT NULL"
                ")"
            )
        )
    engine.dispose()


def _writer(db_path_str: str, result_path_str: str) -> None:
    db_path = Path(db_path_str)
    result_path = Path(result_path_str)
    summary: dict = {"role": "writer", "rows_written": 0, "errors": []}
    started = time.monotonic()
    try:
        engine = get_engine(db_path)
        for txn_idx in range(WRITER_TXN_COUNT):
            with engine.begin() as conn:
                for row_idx in range(WRITER_ROWS_PER_TXN):
                    conn.execute(
                        text(
                            "INSERT INTO smoke_wal (payload, ts) "
                            "VALUES (:payload, :ts)"
                        ),
                        {
                            "payload": f"txn={txn_idx} row={row_idx}",
                            "ts": time.time(),
                        },
                    )
                    summary["rows_written"] += 1
            time.sleep(WRITER_TXN_SLEEP_S)
        engine.dispose()
    except Exception as exc:  # noqa: BLE001 — surface any error to the parent
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    summary["elapsed_s"] = round(time.monotonic() - started, 4)
    result_path.write_text(json.dumps(summary))


def _reader(db_path_str: str, result_path_str: str, reader_id: int) -> None:
    db_path = Path(db_path_str)
    result_path = Path(result_path_str)
    summary: dict = {
        "role": "reader",
        "reader_id": reader_id,
        "polls": 0,
        "max_count_seen": 0,
        "monotonic_violation": False,
        "saw_full_count": False,
        "errors": [],
    }
    started = time.monotonic()
    try:
        engine = get_engine(db_path)
        last_seen = 0
        deadline = started + READER_TIMEOUT_S
        while time.monotonic() < deadline:
            with engine.connect() as conn:
                count = conn.execute(
                    text("SELECT COUNT(*) FROM smoke_wal")
                ).scalar_one()
            summary["polls"] += 1
            if count < last_seen:
                summary["monotonic_violation"] = True
            last_seen = count
            if count > summary["max_count_seen"]:
                summary["max_count_seen"] = count
            if count >= EXPECTED_TOTAL_ROWS:
                summary["saw_full_count"] = True
                break
            time.sleep(READER_POLL_INTERVAL_S)
        engine.dispose()
    except Exception as exc:  # noqa: BLE001
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    summary["elapsed_s"] = round(time.monotonic() - started, 4)
    result_path.write_text(json.dumps(summary))


def _post_run_checks(db_path: Path) -> dict:
    engine = get_engine(db_path)
    with engine.connect() as conn:
        integrity = conn.execute(text("PRAGMA integrity_check")).scalar_one()
        quick = conn.execute(text("PRAGMA quick_check")).scalar_one()
        journal = conn.execute(text("PRAGMA journal_mode")).scalar_one()
        final_count = conn.execute(
            text("SELECT COUNT(*) FROM smoke_wal")
        ).scalar_one()
        wal_ckpt = conn.execute(
            text("PRAGMA wal_checkpoint(TRUNCATE)")
        ).fetchone()
    engine.dispose()
    return {
        "integrity_check": integrity,
        "quick_check": quick,
        "journal_mode": journal,
        "final_row_count": final_count,
        "wal_checkpoint": list(wal_ckpt) if wal_ckpt else None,
    }


def main() -> int:
    mp.set_start_method("spawn", force=True)
    with tempfile.TemporaryDirectory(prefix="sherloc_wal_smoke_") as tmpdir:
        tmp = Path(tmpdir)
        db_path = tmp / "smoke.db"
        _init_schema(db_path)

        writer_result = tmp / "writer.json"
        reader_results = [tmp / f"reader_{i}.json" for i in range(READER_COUNT)]

        writer_proc = mp.Process(
            target=_writer, args=(str(db_path), str(writer_result))
        )
        reader_procs = [
            mp.Process(
                target=_reader,
                args=(str(db_path), str(reader_results[i]), i),
            )
            for i in range(READER_COUNT)
        ]

        wall_start = time.monotonic()
        writer_proc.start()
        for p in reader_procs:
            p.start()
        writer_proc.join()
        for p in reader_procs:
            p.join()
        wall_elapsed = round(time.monotonic() - wall_start, 4)

        writer_summary = json.loads(writer_result.read_text())
        reader_summaries = [json.loads(p.read_text()) for p in reader_results]
        post = _post_run_checks(db_path)

    report = {
        "config": {
            "writer_txn_count": WRITER_TXN_COUNT,
            "writer_rows_per_txn": WRITER_ROWS_PER_TXN,
            "expected_total_rows": EXPECTED_TOTAL_ROWS,
            "reader_count": READER_COUNT,
            "reader_timeout_s": READER_TIMEOUT_S,
        },
        "wall_elapsed_s": wall_elapsed,
        "writer": writer_summary,
        "readers": reader_summaries,
        "post_run": post,
        "writer_exit_code": writer_proc.exitcode,
        "reader_exit_codes": [p.exitcode for p in reader_procs],
    }

    print(json.dumps(report, indent=2))

    failures: list[str] = []
    if writer_proc.exitcode != 0:
        failures.append(f"writer exit code {writer_proc.exitcode}")
    for i, code in enumerate(report["reader_exit_codes"]):
        if code != 0:
            failures.append(f"reader {i} exit code {code}")
    if writer_summary["errors"]:
        failures.append(f"writer errors: {writer_summary['errors']}")
    for r in reader_summaries:
        if r["errors"]:
            failures.append(f"reader {r['reader_id']} errors: {r['errors']}")
        if r["monotonic_violation"]:
            failures.append(f"reader {r['reader_id']} saw count regress")
    if writer_summary["rows_written"] != EXPECTED_TOTAL_ROWS:
        failures.append(
            f"writer wrote {writer_summary['rows_written']} rows, "
            f"expected {EXPECTED_TOTAL_ROWS}"
        )
    if post["final_row_count"] != EXPECTED_TOTAL_ROWS:
        failures.append(
            f"final row count {post['final_row_count']}, "
            f"expected {EXPECTED_TOTAL_ROWS}"
        )
    if post["integrity_check"] != "ok":
        failures.append(f"integrity_check returned {post['integrity_check']!r}")
    if post["quick_check"] != "ok":
        failures.append(f"quick_check returned {post['quick_check']!r}")
    if str(post["journal_mode"]).lower() != "wal":
        failures.append(f"journal_mode is {post['journal_mode']!r}, expected wal")
    if not all(r["saw_full_count"] for r in reader_summaries):
        failures.append("at least one reader timed out before seeing full count")

    if failures:
        print("\nSMOKE FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("\nSMOKE PASSED — SQLite WAL multi-process safety verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
