"""Trusted local release-index ingestion; no discovery or remote lookups."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import false, func, tuple_
from sqlalchemy.orm import Session

from sherloc_pipeline.database.models import ScanORM


class ReleaseProduct(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    product_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    released: bool


class ReleaseIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["public-release-1"]
    release_id: str = Field(min_length=1)
    products: list[ReleaseProduct]


class PublicReleaseIndex:
    def __init__(self, path: Path, expected_sha256: str | None = None):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("public release index digest mismatch")
        index = ReleaseIndex.model_validate_json(raw)
        self.sha256 = digest
        self.release_id = index.release_id
        self.products = {(p.product_id, p.version_id): p for p in index.products}
        if len(self.products) != len(index.products):
            raise ValueError("duplicate release product/version")

    def apply(self, product_id: str, metadata: dict) -> dict:
        metadata = {k: v for k, v in metadata.items() if k != "public_release"}
        product = self.products.get((product_id, metadata.get("version")))
        if product is not None and product.released:
            metadata["public_release"] = {
                "schema_version": "public-release-1",
                "eligible": True,
                "product_id": product.product_id,
                "version_id": product.version_id,
                "source_sha256": product.sha256,
                "release_id": self.release_id,
                "index_sha256": self.sha256,
            }
        return metadata


def public_release_expression(index: PublicReleaseIndex | None):
    """Match stored evidence against the pinned startup snapshot."""
    if index is None:
        return false()
    trusted = [
        (p.product_id, p.version_id, p.sha256, index.release_id, index.sha256)
        for p in index.products.values() if p.released
    ]
    if not trusted:
        return false()
    meta = ScanORM.pds4_metadata
    evidence = meta["public_release"]
    return (
        (ScanORM.data_source == "pds4")
        & (func.json_type(meta, "$.public_release.eligible") == "true")
        & (evidence["schema_version"].as_string() == "public-release-1")
        & (evidence["product_id"].as_string() == ScanORM.scan_id)
        & (evidence["version_id"].as_string() == meta["version"].as_string())
        & (func.json_type(meta, "$.version") == "text")
        & (func.json_type(meta, "$.public_release.version_id") == "text")
        & tuple_(
            evidence["product_id"].as_string(),
            evidence["version_id"].as_string(),
            evidence["source_sha256"].as_string(),
            evidence["release_id"].as_string(),
            evidence["index_sha256"].as_string(),
        ).in_(trusted)
    )


def has_public_release(scan, index: PublicReleaseIndex | None) -> bool:
    if index is None or getattr(scan, "data_source", None) != "pds4":
        return False
    meta = getattr(scan, "pds4_metadata", None)
    if not isinstance(meta, dict) or not isinstance(meta.get("version"), str):
        return False
    evidence = meta.get("public_release")
    if not isinstance(evidence, dict) or evidence.get("eligible") is not True:
        return False
    product = index.products.get((scan.scan_id, meta.get("version")))
    return (
        product is not None and product.released
        and evidence.get("schema_version") == "public-release-1"
        and evidence.get("product_id") == product.product_id
        and evidence.get("version_id") == product.version_id
        and evidence.get("source_sha256") == product.sha256
        and evidence.get("release_id") == index.release_id
        and evidence.get("index_sha256") == index.sha256
    )


def backfill_public_release(session: Session, index: PublicReleaseIndex) -> int:
    scans = session.query(ScanORM).filter(ScanORM.data_source == "pds4").all()
    for scan in scans:
        scan.pds4_metadata = index.apply(scan.scan_id, scan.pds4_metadata or {})
    return len(scans)
