"""Verify that an existing Loupe extraction still matches its source archive."""

import hashlib
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from zipfile import ZipFile

from sherloc_pipeline.services.ingestion import IngestionError


def _digest(stream: BinaryIO) -> bytes:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.digest()


def verify_existing_extraction(archive_path: Path, sol_dir: Path) -> None:
    """Refuse reuse unless the complete file inventory and bytes match.

    This reads source files only; it neither overwrites an existing extraction
    nor creates a second persisted record of the archive's identity.
    """
    message = (
        f"Existing directory {sol_dir} differs from archive {archive_path.name}. "
        "Extract the archive into a new data directory and rerun with --data-dir."
    )
    if sol_dir.is_symlink() or not sol_dir.is_dir():
        raise IngestionError(message)

    files = {}
    for path in sol_dir.rglob("*"):
        if path.is_symlink():
            raise IngestionError(message)
        if path.is_file():
            files[path.relative_to(sol_dir).as_posix()] = path

    with ZipFile(archive_path) as archive:
        members = {}
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if (not parts or parts[0] != sol_dir.name or ".." in parts
                    or member.filename.startswith("/") or "\\" in member.filename):
                raise IngestionError(message)
            if member.is_dir():
                continue
            relative = "/".join(parts[1:])
            if not relative or relative in members:
                raise IngestionError(message)
            members[relative] = member

        if members.keys() != files.keys():
            raise IngestionError(message)
        for name, member in members.items():
            if files[name].stat().st_size != member.file_size:
                raise IngestionError(message)
            with files[name].open("rb") as extracted, archive.open(member) as source:
                if _digest(extracted) != _digest(source):
                    raise IngestionError(message)
