"""Per-scan processing fields on the ``scans`` table.

Every full-pipeline run that reaches a database (``full-pipeline``,
``process-new`` and ``refit``) records, on the scan row whose peaks it
persists, how that run ended:

- ``processing_status``: ``'completed'`` when the run is complete by the
  refit rule (``refit_journal.run_failures`` returns nothing), else
  ``'failed'``. A run that fitted and accepted zero peaks is complete.
- ``processed_at``: UTC time the run ended (naive, the column has no zone).
- ``processing_config_hash``: ``refit_journal.config_digest`` of the
  effective config, the same digest a refit batch journal records. It leaves
  out the database and root paths, so the same config on another host or
  copy hashes the same.
- ``processing_pipeline_version``: the installed package version.
- ``processing_error``: the failure reasons, or NULL when completed.

The scan row is found the way peak persistence finds it (target with spaces,
then as given; first match), so the status sits on the row that holds the
run's peaks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


def record_scan_processing(
    database_path: Path,
    *,
    sol: str,
    target: str,
    scan: str,
    status: str,
    error: Optional[str],
    config_hash: str,
    pipeline_version: str,
    processed_at: Optional[datetime] = None,
) -> bool:
    """Write the processing fields of one scan. Returns False if no scan row matches."""
    if status not in (STATUS_COMPLETED, STATUS_FAILED):
        raise ValueError(f"processing status must be completed or failed, got {status!r}")
    from sherloc_pipeline.database.connection import get_engine, get_session
    from sherloc_pipeline.database.models import ScanORM

    when = processed_at or datetime.now(timezone.utc)
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)

    engine = get_engine(database_path)
    try:
        with get_session(engine) as session:
            db_target = target.replace("_", " ")
            row = session.query(ScanORM).filter_by(
                sol_number=int(sol), target=db_target, scan_name=scan,
            ).first()
            if row is None and db_target != target:
                row = session.query(ScanORM).filter_by(
                    sol_number=int(sol), target=target, scan_name=scan,
                ).first()
            if row is None:
                return False
            row.processing_status = status
            row.processed_at = when
            row.processing_config_hash = config_hash
            row.processing_pipeline_version = pipeline_version
            row.processing_error = None if status == STATUS_COMPLETED else (error or "unknown error")
        return True
    finally:
        engine.dispose()
