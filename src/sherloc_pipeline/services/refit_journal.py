"""Batch journal for ``sherloc refit``.

A journal is a JSON Lines file. The first line identifies the batch: the
database, data and results roots, scan selection, effective-config digest
and code digest. Each later line names one scan that finished with every
required stage run and no stage error. ``--resume`` continues a batch only
when the identity is unchanged, and skips only the scans the journal lists.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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
    }


def run_failures(metadata: Dict[str, Any]) -> List[str]:
    """Reasons a ``run_full_pipeline`` result is not a complete run; empty means complete.

    A run is complete when every required stage was recorded, no stage holds a
    caught error (``exception`` or ``<name>_error`` in its extra), and Raman
    persistence ran for all three domains. Stage warnings alone do not count:
    a scan with zero-peak points warns and is still complete.
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
            if not isinstance(persisted.get(domain), dict) and f"{domain}_error" not in persisted:
                failures.append(f"raman_persistence: {domain} did not run")
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
        path.parent.mkdir(parents=True, exist_ok=True)
        header = dict(identity, started_at=datetime.now(timezone.utc).isoformat())
        tmp = path.with_name(f".{path.name}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(header, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)  # fails if a concurrent create won
        except FileExistsError:
            raise JournalError(f"Journal already exists: {path}. Use --resume to continue it.")
        finally:
            tmp.unlink()
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
        handle = open(path, "r+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise JournalError(f"Journal is in use by another process: {path}")
        handle.seek(0, os.SEEK_END)
        return handle

    @staticmethod
    def _read(handle, path: Path, identity: Dict[str, Any]) -> Set[ScanKey]:
        handle.seek(0)
        text = handle.read()
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
