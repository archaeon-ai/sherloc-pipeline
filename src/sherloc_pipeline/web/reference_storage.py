"""Tier-aware reference read adapter; no HTTP reference routes."""

from collections.abc import Callable
import hashlib
import io
from typing import BinaryIO

from sherloc_pipeline.services.reference_store import MAX_OBJECT_BYTES, object_locator
# Serving checks live in the services layer; re-exported here for web callers.
from sherloc_pipeline.services.reference_serving import can_serve_template, can_serve_trace

__all__ = ["R2ReferenceStore", "can_serve_template", "can_serve_trace"]


class R2ReferenceStore:
    def __init__(self, tier: str, reader: Callable[[str], bytes | BinaryIO] | None = None):
        if tier != "team":
            raise ValueError("public instances cannot read reference objects")
        self.tier = tier
        self.reader = reader

    def read_verified(self, locator: str, sha256: str) -> bytes:
        if locator != object_locator(sha256):
            raise ValueError("invalid reference locator")
        if self.reader is not None:
            body = self.reader(locator)
            stream = io.BytesIO(body) if isinstance(body, bytes) else body
        else:
            from sherloc_pipeline.web.r2_reader import get_r2_client_and_config

            client, config = get_r2_client_and_config()
            if config["tier"] != self.tier:
                raise ValueError("reference storage tier mismatch")
            stream = client.get_object(Bucket=config["bucket"], Key=locator)["Body"]
        chunks = []
        size = 0
        checksum = hashlib.sha256()
        try:
            while size <= MAX_OBJECT_BYTES:
                chunk = stream.read(min(64 * 1024, MAX_OBJECT_BYTES + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_OBJECT_BYTES:
                    raise ValueError("reference object size or SHA-256 mismatch")
                checksum.update(chunk)
                chunks.append(chunk)
            if checksum.hexdigest() != sha256:
                raise ValueError("reference object size or SHA-256 mismatch")
            return b"".join(chunks)
        finally:
            stream.close()
