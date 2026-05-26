from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class ManifestResolutionError(RuntimeError):
    """Raised when Loupe manifest discovery cannot uniquely resolve a working directory."""


@dataclass(frozen=True)
class ManifestCandidate:
    workspace: str
    working_dir: Path
    source: str
    metadata: Dict[str, str]


def _iter_working_directories(sol_dir: Path) -> Iterable[Path]:
    if not sol_dir.exists():
        return

    for scan_dir in sol_dir.iterdir():
        if not scan_dir.is_dir():
            continue
        if scan_dir.name.startswith(".") or "archive" in scan_dir.name.lower():
            continue
        for candidate in scan_dir.iterdir():
            if candidate.is_dir() and candidate.name.endswith("_Loupe_working"):
                yield candidate


def _read_loupe_manifest(loupe_path: Path) -> Optional[Dict[str, str]]:
    try:
        with loupe_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            manifest: Dict[str, str] = {}
            for row in reader:
                if not row:
                    continue
                key = str(row[0]).strip()
                if not key:
                    continue
                value = ""
                if len(row) > 1:
                    value = str(row[1]).strip()
                manifest[key] = value
    except Exception as exc:
        raise ManifestResolutionError(f"Failed to parse Loupe manifest at {loupe_path}: {exc}") from exc

    return manifest or None


def _build_manifest_candidates(sol_dir: Path) -> List[ManifestCandidate]:
    candidates: List[ManifestCandidate] = []
    for working_dir in _iter_working_directories(sol_dir):
        loupe_path = working_dir / "loupe.csv"
        if not loupe_path.exists():
            continue
        manifest = _read_loupe_manifest(loupe_path)
        if not manifest:
            continue
        workspace = (
            manifest.get("human_readable_workspace")
            or manifest.get("workspace")
            or manifest.get("scan")
        )
        if not workspace:
            continue
        candidates.append(
            ManifestCandidate(
                workspace=str(workspace).strip(),
                working_dir=working_dir,
                source=str(loupe_path),
                metadata=manifest,
            )
        )
    return candidates


def resolve_manifest_working_directory(
    base_data_dir: Path,
    sol: str,
    scan: str,
) -> Optional[Path]:
    sol_dir = base_data_dir / f"sol_{sol}"
    manifest_candidates = _build_manifest_candidates(sol_dir)
    if not manifest_candidates:
        return None

    normalized_scan = scan.strip().casefold()
    matches: List[ManifestCandidate] = [
        candidate
        for candidate in manifest_candidates
        if candidate.workspace.strip().casefold() == normalized_scan
    ]

    if not matches:
        return None

    if len(matches) > 1:
        paths = ", ".join(str(candidate.working_dir) for candidate in matches)
        raise ManifestResolutionError(
            f"Multiple Loupe manifests match sol {sol} scan {scan}: {paths}. "
            "Prune archives or verify loupe.csv entries."
        )

    working_dir = matches[0].working_dir
    required_files = [
        "loupe.csv",
        "spatial.csv",
        "darkSubSpectra.csv",
        "photodiodeRaw.csv",
    ]
    missing = [name for name in required_files if not (working_dir / name).exists()]
    if missing:
        raise ManifestResolutionError(
            f"Manifest match {working_dir} missing required files: {', '.join(missing)}"
        )

    return working_dir.resolve()


__all__ = ["ManifestResolutionError", "resolve_manifest_working_directory"]
