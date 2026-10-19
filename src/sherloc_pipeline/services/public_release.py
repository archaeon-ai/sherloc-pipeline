"""Trusted local release-index ingestion; no discovery or remote lookups."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func
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
    def __init__(self, path: Path):
        raw = path.read_bytes()
        index = ReleaseIndex.model_validate_json(raw)
        self.sha256 = hashlib.sha256(raw).hexdigest()
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


def public_release_expression():
    """Use the same SQL predicate for list filtering and direct access."""
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
        & (func.json_type(meta, "$.public_release.release_id") == "text")
        & (func.length(evidence["version_id"].as_string()) > 0)
        & (func.length(evidence["release_id"].as_string()) > 0)
        & (func.length(evidence["index_sha256"].as_string()) == 64)
        & (~evidence["index_sha256"].as_string().op("GLOB")("*[^0-9a-f]*"))
        & (func.length(evidence["source_sha256"].as_string()) == 64)
        & (~evidence["source_sha256"].as_string().op("GLOB")("*[^0-9a-f]*"))
    )


def has_public_release(scan) -> bool:
    meta = getattr(scan, "pds4_metadata", None)
    if not isinstance(meta, dict):
        return False
    evidence = meta.get("public_release")
    if not isinstance(evidence, dict) or evidence.get("eligible") is not True:
        return False
    try:
        product = ReleaseProduct.model_validate({
            "product_id": evidence.get("product_id"),
            "version_id": evidence.get("version_id"),
            "sha256": evidence.get("source_sha256"),
            "released": True,
        })
        return (
            evidence.get("schema_version") == "public-release-1"
            and product.product_id == scan.scan_id
            and product.version_id == meta.get("version")
            and isinstance(evidence.get("release_id"), str)
            and bool(evidence["release_id"])
            and isinstance(evidence.get("index_sha256"), str)
            and len(evidence["index_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in evidence["index_sha256"])
        )
    except ValueError:
        return False


def backfill_public_release(session: Session, index: PublicReleaseIndex) -> int:
    scans = session.query(ScanORM).filter(ScanORM.data_source == "pds4").all()
    for scan in scans:
        scan.pds4_metadata = index.apply(scan.scan_id, scan.pds4_metadata or {})
    return len(scans)
