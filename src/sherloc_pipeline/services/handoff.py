"""Neutral, atomic delivery handoff for completed pipeline runs."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image

from sherloc_pipeline.vision.img_reader import get_raw_vicar_label, read_aci_image

from .base import ServiceResult
from .errors import HandoffError


_PRODUCT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,191}$")
_ACI_PRODUCT_PREFIX_RE = re.compile(r"^SC[0-3]_")
_ANGLE_RANGE_RENDER_RE = re.compile(r"_\d{1,3}-\d{1,3}$")
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


def _resolved_portable_source(path: Path, source_root: Path) -> tuple[Path, str]:
    try:
        resolved = path.resolve(strict=True)
        root = source_root.resolve()
    except OSError as exc:
        raise HandoffError("handoff source cannot be read") from exc
    try:
        locator = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise HandoffError("handoff source escapes the configured data root") from exc
    pure = PurePosixPath(locator)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
        or "\\" in locator
    ):
        raise HandoffError("handoff source locator is not portable")
    return resolved, locator


def _portable_locator(path: Path, source_root: Path) -> str:
    return _resolved_portable_source(path, source_root)[1]


def _product_id(path: Path) -> str:
    product_id = path.stem
    if (
        path.suffix not in (".PNG", ".IMG")
        or not _PRODUCT_ID_RE.fullmatch(product_id)
        or not _ACI_PRODUCT_PREFIX_RE.match(product_id)
    ):
        raise HandoffError("handoff image has an invalid product identity")
    return product_id


def _image_evidence(
    path: Path, source_root: Path
) -> tuple[dict, tuple[int, int], bool]:
    evidence, fingerprint, resolved = _stable_file_evidence(path, source_root)
    try:
        with Image.open(resolved) as image:
            dimensions = image.size
            single_channel = len(image.getbands()) == 1
            if image.format != "PNG":
                raise HandoffError("handoff image is not a valid PNG")
            image.verify()
        if _fingerprint(resolved.stat()) != fingerprint:
            raise HandoffError("handoff source changed during inventory")
    except OSError as exc:
        raise HandoffError("handoff image is not a valid PNG") from exc
    return {
        "source_rel_locator": evidence["source_rel_locator"],
        "source_format": "png",
        "source_byte_size": evidence["byte_size"],
        "source_sha256": evidence["sha256"],
        "byte_size": evidence["byte_size"],
        "sha256": evidence["sha256"],
        "mtime": evidence["mtime"],
    }, dimensions, single_channel


def _canonical_png_bytes(path: Path) -> bytes:
    label = get_raw_vicar_label(path)
    band_count = label.get("NB", label.get("BANDS", 1))
    if band_count != 1:
        raise HandoffError("raw ACI image is not single-band")
    image, _metadata = read_aci_image(path)
    if image.shape != (1200, 1648) or image.dtype.name != "uint8":
        raise HandoffError("raw ACI image is not in the full frame")
    output = io.BytesIO()
    Image.fromarray(image, mode="L").save(
        output,
        format="PNG",
        optimize=False,
        compress_level=6,
    )
    return output.getvalue()


def _vicar_evidence(path: Path, source_root: Path) -> dict:
    evidence, fingerprint, resolved = _stable_file_evidence(path, source_root)
    try:
        canonical = _canonical_png_bytes(resolved)
        if _fingerprint(resolved.stat()) != fingerprint:
            raise HandoffError("handoff source changed during inventory")
    except (OSError, ValueError) as exc:
        raise HandoffError("handoff VICAR source cannot be decoded") from exc
    return {
        "source_rel_locator": evidence["source_rel_locator"],
        "source_format": "vicar_img",
        "source_byte_size": evidence["byte_size"],
        "source_sha256": evidence["sha256"],
        "byte_size": len(canonical),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "mtime": evidence["mtime"],
    }


def _stable_file_evidence(
    path: Path, source_root: Path
) -> tuple[dict, tuple[int, int, int, int], Path]:
    resolved, locator = _resolved_portable_source(path, source_root)
    try:
        before = resolved.stat()
    except OSError as exc:
        raise HandoffError("handoff source cannot be read") from exc
    if not resolved.is_file() or before.st_size <= 0:
        raise HandoffError("handoff source is missing or empty")
    try:
        digest = _sha256(resolved)
    except OSError as exc:
        raise HandoffError("handoff source cannot be read") from exc
    try:
        after = resolved.stat()
    except OSError as exc:
        raise HandoffError("handoff source cannot be read") from exc
    if _fingerprint(before) != _fingerprint(after):
        raise HandoffError("handoff source changed during inventory")
    return {
        "source_rel_locator": locator,
        "byte_size": after.st_size,
        "sha256": digest,
        "mtime": _timestamp(after.st_mtime),
    }, _fingerprint(after), resolved


def _file_evidence(path: Path, source_root: Path) -> dict:
    evidence, _fingerprint_value, _resolved = _stable_file_evidence(
        path, source_root
    )
    return evidence


def _pngs(working_dir: Path, source_root: Path) -> list[Path]:
    image_dir, _locator = _resolved_portable_source(
        working_dir / "img", source_root
    )
    if not image_dir.is_dir():
        raise HandoffError("handoff workspace has no image directory")
    return sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() == ".png"
    )


def _has_valid_image_metadata(path: Path, source_root: Path) -> bool:
    for candidate in (path.with_suffix(".CSV"), path.with_suffix(".csv")):
        try:
            candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise HandoffError("handoff source cannot be read") from exc
        resolved, _locator = _resolved_portable_source(candidate, source_root)
        if resolved.is_file():
            return True
    return False


def _product_pngs(working_dir: Path, source_root: Path) -> list[Path]:
    candidates = [
        path
        for path in _pngs(working_dir, source_root)
        if _ACI_PRODUCT_PREFIX_RE.match(path.stem)
        and not _ANGLE_RANGE_RENDER_RE.search(path.stem)
    ]
    products = [
        path for path in candidates
        if _has_valid_image_metadata(path, source_root)
    ]
    if len(products) != len(candidates):
        raise HandoffError("handoff ACI product is missing same-stem metadata")
    return products


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
    images = _product_pngs(working_dir, source_root)
    if not images:
        acquisition_dir = working_dir.parent
        images = sorted(
            path for path in acquisition_dir.iterdir()
            if path.suffix.upper() == ".IMG"
            and _ACI_PRODUCT_PREFIX_RE.match(path.stem)
            and not _ANGLE_RANGE_RENDER_RE.search(path.stem)
        )
    if not images:
        raise HandoffError("handoff workspace has no raw ACI source")
    for path in images:
        product_id = _product_id(path)
        if product_id in products:
            raise HandoffError("handoff workspace has duplicate product identities")
        if path.suffix.upper() == ".PNG":
            evidence, dimensions, single_channel = _image_evidence(path, source_root)
            if dimensions != (1648, 1200) or not single_channel:
                fallback = working_dir.parent / f"{product_id}.IMG"
                if not fallback.is_file():
                    raise HandoffError("raw ACI image is not in the full frame")
                evidence = _vicar_evidence(fallback, source_root)
                dimensions = (1648, 1200)
        else:
            evidence = _vicar_evidence(path, source_root)
            dimensions = (1648, 1200)
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


def build_raw_handoff_entry(path: Path, source_root: Path) -> dict:
    """Build one verified v2 raw entry for bounded migration preparation."""
    product_id = _product_id(path)
    if path.suffix.upper() == ".IMG":
        evidence = _vicar_evidence(path, source_root)
    else:
        evidence, dimensions, single_channel = _image_evidence(path, source_root)
        if dimensions != (1648, 1200) or not single_channel:
            raise HandoffError("raw ACI image is not in the full frame")
    return {
        "product_id": product_id,
        "role": "raw_grayscale",
        "provenance": "sherloc_delivery",
        "rendition_key": "product",
        **evidence,
        "coordinate_frame": "aci_full_frame",
        "width_px": 1648,
        "height_px": 1200,
        "sidecars": [],
    }


def _colorized_entries(
    working_dir: Path,
    source_root: Path,
    raw_products: set[str],
) -> list[dict]:
    colorized_dir = _colorized_working_dir(working_dir, source_root)
    if not colorized_dir.exists():
        return []
    images = _product_pngs(colorized_dir, source_root)
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
        "schema_version": "aci-handoff-manifest.v2",
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
    directory_fd: int | None = None
    published = False
    temporary_created = False
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if temporary.exists() or ready.exists():
            raise FileExistsError("handoff run already exists")
        payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        temporary_created = True
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(output_dir, os.O_RDONLY)
        # Probe directory durability before exposing a ready name. Filesystems
        # that cannot fsync directories fail while the manifest is still hidden.
        os.fsync(directory_fd)
        # Linking exposes the complete file atomically and fails if another
        # publisher won the run ID, closing the check-then-rename race.
        os.link(temporary, ready)
        published = True
        temporary.unlink()
        temporary_created = False
        os.fsync(directory_fd)
    except OSError as exc:
        if published:
            try:
                ready.unlink()
                if directory_fd is not None:
                    os.fsync(directory_fd)
            except OSError:
                pass
        if temporary_created:
            try:
                temporary.unlink()
            except OSError:
                pass
        raise HandoffError("configured handoff could not be published") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
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
