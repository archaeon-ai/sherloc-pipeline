"""Immutable content-addressed local objects and verified read interface."""

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

MAX_OBJECT_BYTES = 64 * 1024 * 1024


class ReferenceStore(Protocol):
    def read_verified(self, locator: str, sha256: str) -> bytes: ...


def object_locator(sha256: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("invalid reference SHA-256")
    return f"sherloc-references/sha256/{sha256}"


def verify_bytes(data: bytes, sha256: str) -> bytes:
    object_locator(sha256)
    if len(data) > MAX_OBJECT_BYTES or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("reference object size or SHA-256 mismatch")
    return data


class LocalReferenceStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def _path(self, locator: str, sha256: str) -> Path:
        if locator != object_locator(sha256):
            raise ValueError("invalid reference locator")
        path = (self.root / locator).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("reference path escapes object root")
        return path

    def read_verified(self, locator: str, sha256: str) -> bytes:
        path = self._path(locator, sha256)
        with path.open("rb") as stream:
            return verify_bytes(stream.read(MAX_OBJECT_BYTES + 1), sha256)

    def put_verified(self, data: bytes, sha256: str) -> str:
        verify_bytes(data, sha256)
        locator = object_locator(sha256)
        path = self._path(locator, sha256)
        if path.exists():
            self.read_verified(locator, sha256)
            return locator
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    self.read_verified(locator, sha256)
                if hasattr(os, "O_DIRECTORY"):
                    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            finally:
                temporary.unlink()
        return locator
