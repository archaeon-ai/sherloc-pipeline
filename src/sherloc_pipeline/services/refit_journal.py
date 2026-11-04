"""Batch journal for ``sherloc refit``.

A journal is a JSON Lines file. The first line identifies the batch: the
database, data and results roots, scan selection, effective-config digest,
code digest, dependency versions and the environment variables the run reads.
Each later line names one scan that finished with every required stage run,
every fitted point persisted, and no stage error. ``--resume`` continues a batch only
when the identity is unchanged, and skips only the scans the journal lists.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
    import msvcrt

SCHEMA = "sherloc-refit-journal/1"

# Stages every successful run records (services/pipeline.py capture_stage names).
# cr_mask_persistence runs only when the despike step recorded masks.
REQUIRED_STAGES: Tuple[str, ...] = (
    "preprocessing",
    "fitting_modality",
    "fluorescence_inputs",
    "raman_persistence",
    "fluorescence_fitting",
    "review",
    "fitting_averages",
    "spatial_overlays",
    "summary",
)
RAMAN_DOMAINS: Tuple[str, ...] = ("minerals", "organics", "hydration")

# Fields that must match for --resume. started_at and the scan lines do not.
IDENTITY_FIELDS: Tuple[str, ...] = (
    "schema",
    "database",
    "data_root",
    "results_root",
    "selection",
    "config_digest",
    "code_digest",
    "runtime",
    "environment",
)

# Installed distributions whose versions can change fits, despiking or plots.
DEPENDENCIES: Tuple[str, ...] = (
    "numpy", "scipy", "pandas", "matplotlib", "pybaselines", "scikit-learn",
    "scikit-image", "onnxruntime",
)

# Environment variables read during a full-pipeline run that change its outputs.
# SHERLOC_DB_PATH is set from --database and the data and results roots are
# identity fields of their own.
ENVIRONMENT: Tuple[str, ...] = (
    "SHERLOC_HANDOFF_DIR",
    "SHERLOC_LEGACY_PLOTS",
    "SHERLOC_DETERMINISTIC_PLOTS",
    "SHERLOC_OVERLAY_DEBUG",
)

ScanKey = Tuple[int, str, str]


class JournalError(Exception):
    """The journal cannot be created, read or resumed."""


def code_digest(package_root: Optional[Path] = None) -> str:
    """SHA-256 over every file in the installed package, excluding bytecode."""
    if package_root is None:
        package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    files = sorted(
        p for p in package_root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    for path in files:
        digest.update(path.relative_to(package_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def config_digest(config: Any) -> str:
    """SHA-256 of the effective config, without the database and root paths.

    Those come from --database, --data-dir and --results-dir (or their
    defaults) and are identity fields of their own, so an ambient
    SHERLOC_DB_PATH, SHERLOC_DATA_DIR or SHERLOC_RESULTS_DIR does not change
    the digest of an otherwise identical batch.
    """
    from sherloc_pipeline.services.config import _normalize_for_hash

    normalized = _normalize_for_hash(config)
    normalized.pop("database", None)
    paths = dict(normalized.get("paths") or {})
    paths.pop("data_root", None)
    paths.pop("results_root", None)
    normalized["paths"] = paths
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def runtime_versions() -> Dict[str, Optional[str]]:
    """Python version and the installed version of each dependency (None if absent)."""
    versions: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_values() -> Dict[str, Optional[str]]:
    return {name: os.environ.get(name) for name in ENVIRONMENT}


def batch_identity(
    *,
    database: Path,
    data_root: Path,
    results_root: Path,
    selection: Dict[str, Any],
    config_digest: str,
    code: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "database": str(Path(database).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "results_root": str(Path(results_root).resolve()),
        "selection": dict(selection),
        "config_digest": config_digest,
        "code_digest": code if code is not None else code_digest(),
        "runtime": runtime_versions(),
        "environment": environment_values(),
    }


def run_failures(metadata: Dict[str, Any]) -> List[str]:
    """Reasons a ``run_full_pipeline`` result is not a complete run; empty means complete.

    A run is complete when every required stage was recorded, no stage holds a
    caught error (``exception`` or ``<name>_error`` in its extra), Raman
    persistence ran for all three domains and matched every peak CSV to a
    stored spectrum, and fluorescence found R1-R3 inputs for at least one scan
    point. A point without R1-R3 rows is a gap in the stored scan that a rerun
    would not fill; a scan with no inputs at all was not fitted. Stage
    warnings alone do not count: a scan with zero-peak points warns and is
    still complete.
    """
    stages = {s.get("name"): s for s in (metadata.get("run") or {}).get("stages") or []}
    failures = [f"{name}: did not run" for name in REQUIRED_STAGES if name not in stages]
    for name, stage in stages.items():
        extra = stage.get("extra") or {}
        for key, value in extra.items():
            if key == "exception" or key.endswith("_error"):
                failures.append(f"{name}: {value}")
    persisted = (stages.get("raman_persistence") or {}).get("extra") or {}
    if "raman_persistence" in stages:
        for domain in RAMAN_DOMAINS:
            if f"{domain}_error" in persisted:
                continue
            result = persisted.get(domain)
            if not isinstance(result, dict):
                failures.append(f"raman_persistence: {domain} did not run")
            elif result.get("csvs_without_spectrum") != 0:
                failures.append(
                    f"raman_persistence: {domain}: "
                    f"{result.get('csvs_without_spectrum', 'unknown')} peak CSV(s) have no stored spectrum"
                )
    fluor = stages.get("fluorescence_fitting")
    if fluor is not None and "exception" not in (fluor.get("extra") or {}):
        extra = fluor.get("extra") or {}
        missing, total = extra.get("points_missing_regions"), extra.get("total_scan_points")
        if not isinstance(missing, int) or not isinstance(total, int):
            failures.append("fluorescence_fitting: input counts not reported")
        elif missing >= total:
            failures.append(f"fluorescence_fitting: none of {total} point(s) have R1-R3 inputs")
    return failures


class BatchJournal:
    """An open journal. Holds an exclusive lock on the file until closed."""

    def __init__(self, path: Path, handle, identity: Dict[str, Any], done: Set[ScanKey]):
        self.path = path
        self._handle = handle
        self.identity = identity
        self.done = done

    @classmethod
    def create(cls, path: Path, identity: Dict[str, Any]) -> "BatchJournal":
        path = Path(path)
        if path.exists():
            raise JournalError(f"Journal already exists: {path}. Use --resume to continue it.")
        header = dict(identity, started_at=datetime.now(timezone.utc).isoformat())
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        except OSError as e:
            raise JournalError(f"Cannot create journal {path}: {e}")
        tmp = Path(tmp_name)
        try:
            with open(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(header, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.link(tmp, path)  # fails if a concurrent create won
        except FileExistsError:
            raise JournalError(f"Journal already exists: {path}. Use --resume to continue it.")
        except OSError as e:
            raise JournalError(f"Cannot create journal {path}: {e}")
        finally:
            tmp.unlink(missing_ok=True)
        return cls(path, cls._lock(path), identity, set())

    @classmethod
    def resume(cls, path: Path, identity: Dict[str, Any]) -> "BatchJournal":
        path = Path(path)
        if not path.is_file():
            raise JournalError(f"Journal not found: {path}")
        handle = cls._lock(path)
        try:
            done = cls._read(handle, path, identity)
        except BaseException:
            handle.close()
            raise
        return cls(path, handle, identity, done)

    @staticmethod
    def _lock(path: Path):
        try:
            # newline="\n": no CRLF translation, so text and byte offsets agree.
            handle = open(path, "r+", encoding="utf-8", newline="\n")
        except OSError as e:
            raise JournalError(f"Cannot open journal {path}: {e}")
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            handle.close()
            raise JournalError(f"Journal is in use by another process: {path}")
        handle.seek(0, os.SEEK_END)
        return handle

    @staticmethod
    def _read(handle, path: Path, identity: Dict[str, Any]) -> Set[ScanKey]:
        handle.seek(0)
        try:
            text = handle.read()
        except (OSError, UnicodeDecodeError) as e:
            raise JournalError(f"Journal is unreadable: {path}: {e}")
        lines = text.split("\n")
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError:
            raise JournalError(f"Journal header is unreadable: {path}")
        if not isinstance(header, dict) or header.get("schema") != SCHEMA:
            raise JournalError(f"Not a refit journal: {path}")
        differs = [f for f in IDENTITY_FIELDS if header.get(f) != identity.get(f)]
        if differs:
            detail = "; ".join(
                f"{f}: journal {header.get(f)!r}, now {identity.get(f)!r}" for f in differs
            )
            raise JournalError(f"Cannot resume {path}: {detail}")
        # A crash can leave the last line unterminated. It was never recorded:
        # drop it, so the next entry starts on its own line.
        if not text.endswith("\n"):
            handle.truncate(len(text.encode("utf-8")) - len(lines[-1].encode("utf-8")))
            lines[-1] = ""
        done: Set[ScanKey] = set()
        for number, line in enumerate(lines[1:], start=2):
            if not line:
                continue
            try:
                entry = json.loads(line)
                done.add((int(entry["sol"]), entry["target"], entry["scan"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                raise JournalError(f"Journal line {number} is unreadable: {path}")
        handle.seek(0, os.SEEK_END)
        return done

    def record(self, sol: int, target: str, scan: str) -> None:
        entry = {
            "sol": int(sol),
            "target": target,
            "scan": scan,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        self._handle.write(json.dumps(entry, sort_keys=True) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.done.add((int(sol), target, scan))

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "BatchJournal":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def pending(scans: Iterable[ScanKey], done: Set[ScanKey]) -> Tuple[List[ScanKey], List[ScanKey]]:
    """Split scans into (to run, already recorded)."""
    run, skipped = [], []
    for key in scans:
        (skipped if (int(key[0]), key[1] or "", key[2]) in done else run).append(key)
    return run, skipped
