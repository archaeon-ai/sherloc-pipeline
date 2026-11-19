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
- ``processing_pipeline_version``: the code identity (``pipeline_identity``):
  the full git commit when the package runs from a git checkout, with
  ``-dirty`` appended when tracked files differ from it; otherwise the
  installed package version.
- ``processing_error``: the failure reasons, or NULL when completed.

The scan row is found the way peak persistence finds it (target with spaces,
then as given; first match), so the status sits on the row that holds the
run's peaks.
"""

from __future__ import annotations

import functools
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


def _git(repo: Path, *args: str) -> Optional[str]:
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def pipeline_identity(package_root: Optional[Path] = None) -> str:
    """The code identity written on each scan a run fits.

    When the package sits in a git work tree (an editable install of a
    checkout), this is the full commit of that tree's HEAD, with ``-dirty``
    appended when any tracked file differs from it. An installed wheel has no
    work tree, so the identity is the package version.
    """
    from sherloc_pipeline import __version__

    if package_root is None:
        package_root = Path(__file__).resolve().parents[1]
    top = _git(package_root, "rev-parse", "--show-toplevel")
    if not top:
        return __version__
    # The package must belong to that tree, not sit in an unrelated repository.
    tracked = _git(Path(top), "ls-files", "--error-unmatch", "--", str(package_root / "__init__.py"))
    commit = _git(Path(top), "rev-parse", "--verify", "HEAD")
    if tracked is None or not commit:
        return __version__
    status = _git(Path(top), "status", "--porcelain", "--untracked-files=no")
    if status is None or status:
        return f"{commit}-dirty"
    return commit


@functools.lru_cache(maxsize=1)
def current_pipeline_identity() -> str:
    """``pipeline_identity()`` of this process's code, computed once."""
    return pipeline_identity()


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
