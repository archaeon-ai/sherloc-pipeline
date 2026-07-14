"""Neutral, atomic delivery handoff for completed pipeline runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image

from .base import ServiceResult
from .errors import HandoffError


_PRODUCT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,191}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SOL_SEGMENT_RE = re.compile(r"^sol_\d+$")


def _timestamp(value: float | None = None) -> str:
    instant = (
        datetime.now(timezone.utc)
        if value is None
        else datetime.fromtimestamp(value, timezone.utc)
    )
    return instant.isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _portable_locator(path: Path, source_root: Path) -> str:
    try:
        locator = path.resolve(strict=True).relative_to(
            source_root.resolve(strict=True)
        ).as_posix()
    except (OSError, ValueError) as exc:
        raise HandoffError("handoff source escapes the configured data root") from exc
    pure = PurePosixPath(locator)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
        or "\\" in locator
    ):
        raise HandoffError("handoff source locator is not portable")
    return locator


def _product_id(path: Path) -> str:
    product_id = path.stem
    if path.suffix.lower() != ".png" or not _PRODUCT_ID_RE.fullmatch(product_id):
        raise HandoffError("handoff image has an invalid product identity")
    return product_id


def _image_evidence(
    path: Path, source_root: Path
) -> tuple[dict, tuple[int, int], bool]:
    evidence, fingerprint = _stable_file_evidence(path, source_root)
    try:
        with Image.open(path) as image:
            dimensions = image.size
            single_channel = len(image.getbands()) == 1
            image.verify()
        if _fingerprint(path.stat()) != fingerprint:
            raise HandoffError("handoff source changed during inventory")
    except OSError as exc:
        raise HandoffError("handoff image is not a valid PNG") from exc
    return evidence, dimensions, single_channel


def _stable_file_evidence(
    path: Path, source_root: Path
) -> tuple[dict, tuple[int, int, int, int]]:
    try:
        before = path.stat()
    except OSError as exc:
        raise HandoffError("handoff source cannot be read") from exc
    if not path.is_file() or before.st_size <= 0:
        raise HandoffError("handoff source is missing or empty")
    try:
        digest = _sha256(path)
    except OSError as exc:
        raise HandoffError("handoff source cannot be read") from exc
    try:
        after = path.stat()
    except OSError as exc:
        raise HandoffError("handoff source cannot be read") from exc
    if _fingerprint(before) != _fingerprint(after):
        raise HandoffError("handoff source changed during inventory")
    return {
        "source_rel_locator": _portable_locator(path, source_root),
        "byte_size": after.st_size,
        "sha256": digest,
        "mtime": _timestamp(after.st_mtime),
    }, _fingerprint(after)


def _file_evidence(path: Path, source_root: Path) -> dict:
    evidence, _fingerprint_value = _stable_file_evidence(path, source_root)
    return evidence


def _pngs(working_dir: Path) -> list[Path]:
    image_dir = working_dir / "img"
    if not image_dir.is_dir():
        raise HandoffError("handoff workspace has no image directory")
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )


def _product_pngs(working_dir: Path) -> list[Path]:
    return [
        path
        for path in _pngs(working_dir)
        if path.with_suffix(".CSV").is_file() or path.with_suffix(".csv").is_file()
    ]


def _colorized_working_dir(working_dir: Path, source_root: Path) -> Path:
    locator = PurePosixPath(_portable_locator(working_dir, source_root))
    parts = list(locator.parts)
    for index, part in enumerate(parts):
        if _SOL_SEGMENT_RE.fullmatch(part):
            parts[index] = f"{part}_colorized"
            return source_root.joinpath(*parts)
    raise HandoffError("handoff workspace has no sol segment")


def _raw_entries(working_dir: Path, source_root: Path) -> tuple[list[dict], set[str]]:
    entries: list[dict] = []
    products: set[str] = set()
    images = _product_pngs(working_dir)
    if not images:
        raise HandoffError("handoff workspace has no product PNG images")
    for path in images:
        product_id = _product_id(path)
        if product_id in products:
            raise HandoffError("handoff workspace has duplicate product identities")
        evidence, dimensions, single_channel = _image_evidence(path, source_root)
        if dimensions != (1648, 1200) or not single_channel:
            raise HandoffError("raw ACI image is not in the full frame")
        products.add(product_id)
        entries.append({
            "product_id": product_id,
            "role": "raw_grayscale",
            "provenance": "sherloc_delivery",
            "rendition_key": "product",
            **evidence,
            "coordinate_frame": "aci_full_frame",
            "width_px": dimensions[0],
            "height_px": dimensions[1],
            "sidecars": [],
        })
    return entries, products


def _colorized_entries(
    working_dir: Path,
    source_root: Path,
    raw_products: set[str],
) -> list[dict]:
    colorized_dir = _colorized_working_dir(working_dir, source_root)
    if not colorized_dir.exists():
        return []
    images = _product_pngs(colorized_dir)
    if not images:
        raise HandoffError("colorized handoff workspace has no product PNG images")
    sidecars = [
        {"role": role, **_file_evidence(colorized_dir / filename, source_root)}
        for role, filename in (("spatial", "spatial.csv"), ("loupe", "loupe.csv"))
    ]
    entries: list[dict] = []
    seen: set[str] = set()
    for path in images:
        product_id = _product_id(path)
        if product_id not in raw_products:
            raise HandoffError("colorized rendition has no raw product")
        if product_id in seen:
            raise HandoffError("colorized workspace has duplicate product identities")
        seen.add(product_id)
        evidence, dimensions, _single_channel = _image_evidence(path, source_root)
        locator = evidence["source_rel_locator"]
        entries.append({
            "product_id": product_id,
            "role": "colorized",
            "provenance": "sherloc_delivery",
            "rendition_key": locator,
            **evidence,
            "coordinate_frame": (
                "aci_full_frame"
                if dimensions == (1648, 1200)
                else "workspace_crop"
            ),
            "width_px": dimensions[0],
            "height_px": dimensions[1],
            "sidecars": sidecars,
        })
    return entries


def build_handoff_manifest(
    *,
    run_id: str,
    source_root: Path,
    working_dir: Path,
    completed_at: str | None = None,
) -> dict:
    """Build a closed handoff document from one completed scan workspace."""
    if not _RUN_ID_RE.fullmatch(run_id):
        raise HandoffError("handoff run ID is invalid")
    raw, products = _raw_entries(working_dir, source_root)
    colorized = _colorized_entries(working_dir, source_root, products)
    stamp = completed_at or _timestamp()
    return {
        "schema_version": "aci-handoff-manifest.v1",
        "producer": "sherloc-pipeline",
        "run_id": run_id,
        "created_at": stamp,
        "epoch": stamp,
        "selector": {"product_ids": sorted(products)},
        "entries": sorted(
            raw + colorized,
            key=lambda entry: (
                entry["product_id"], entry["role"], entry["rendition_key"]
            ),
        ),
    }


def _publish_atomic(output_dir: Path, run_id: str, document: dict) -> Path:
    temporary = output_dir / f"{run_id}.tmp"
    ready = output_dir / f"{run_id}.ready"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if temporary.exists() or ready.exists():
            raise FileExistsError("handoff run already exists")
        payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Linking exposes the complete file atomically and fails if another
        # publisher won the run ID, closing the check-then-rename race.
        os.link(temporary, ready)
        temporary.unlink()
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise HandoffError("configured handoff could not be published") from exc
    return ready


class HandoffService:
    """Publish a handoff only when a destination directory is configured."""

    def publish_if_configured(
        self,
        *,
        output_dir: str | Path | None,
        run_id: str,
        source_root: Path,
        working_dir: Path | None,
    ) -> ServiceResult:
        if not output_dir:
            return ServiceResult(
                summary="Delivery handoff skipped",
                metadata={"status": "skipped", "reason": "not_configured"},
            )
        if working_dir is None:
            raise HandoffError("configured handoff workspace was not found")
        document = build_handoff_manifest(
            run_id=run_id,
            source_root=source_root,
            working_dir=working_dir,
        )
        ready = _publish_atomic(Path(output_dir), run_id, document)
        return ServiceResult(
            summary="Delivery handoff published",
            artifacts=[ready],
            metadata={
                "status": "ready",
                "schema_version": document["schema_version"],
                "run_id": run_id,
                "products": len(document["selector"]["product_ids"]),
                "entries": len(document["entries"]),
                "path": str(ready),
            },
        )
