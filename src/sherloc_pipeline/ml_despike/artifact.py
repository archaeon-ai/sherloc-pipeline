"""sha256-pinned fetch-and-cache of the ML CR detector ONNX artifact.

The fetch path crosses a trust boundary: it downloads executable-adjacent
content (an ONNX graph) over the network. Controls (MLD-DET-003,
MLD-SEC-002):

- the expected digest is pinned in tracked code (``ModelManifest.sha256``);
- every artifact — explicit path, cache hit, or fresh download — is
  streaming-hashed before use, with no bypass;
- downloads go to a temporary file and are atomically renamed into the
  cache only after the digest matches, so no partially written or
  unverified file is ever loadable;
- a mismatched cache file is quarantined under a non-loadable name
  (``<name>.corrupt-<timestamp>``) before re-fetching;
- failures raise typed errors carrying the manual remedy (download URL +
  cache path).

The fetcher uses stdlib ``urllib`` over HTTPS only — no new runtime
dependency. Errors surface as ``PreprocessingError`` at the service
boundary (wrapped by the preprocessing service, not here).
"""

import hashlib
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST, ModelManifest

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 1024 * 1024

#: Finite fetch timeout (connect and per-read) so an unreachable or
#: stalled host fails actionably (MLD-IFC-008 AC2) instead of hanging on
#: the stdlib default socket timeout.
_FETCH_TIMEOUT_SECONDS = 60.0


class ArtifactDigestError(RuntimeError):
    """A model artifact's sha256 digest does not match the pinned digest."""


class ArtifactFetchError(RuntimeError):
    """The model artifact could not be obtained (network/path failure)."""


def default_cache_dir() -> Path:
    """Cache directory for fetched ML despike artifacts.

    Follows the SAM checkpoint precedent
    (``~/.cache/sherloc-pipeline/sam_checkpoints``), namespaced for this
    package.
    """
    return Path.home() / ".cache" / "sherloc-pipeline" / "ml_despike"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _remedy_text(manifest: ModelManifest, cache_dir: Path) -> str:
    return (
        f"manually download {manifest.download_url} and place it at "
        f"{cache_dir / manifest.artifact_filename}, or pass an explicit "
        f"artifact path"
    )


def _fetch_to_cache(manifest: ModelManifest, cache_dir: Path, target: Path) -> Path:
    """Download, verify, and atomically install the artifact into the cache."""
    scheme = urlparse(manifest.download_url).scheme
    if scheme != "https":
        raise ArtifactFetchError(
            f"refusing to fetch model artifact over {scheme!r} "
            f"(https required): {manifest.download_url}"
        )

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f"{manifest.artifact_filename}.fetch-", dir=cache_dir
    )
    tmp_path = Path(tmp_name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(tmp_fd, "wb") as tmp_fh:
            with urlopen(manifest.download_url, timeout=_FETCH_TIMEOUT_SECONDS) as response:
                while True:
                    chunk = response.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    digest.update(chunk)
                    tmp_fh.write(chunk)
    except (URLError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise ArtifactFetchError(
            f"failed to fetch model artifact from {manifest.download_url}: "
            f"{exc}. Remedy: {_remedy_text(manifest, cache_dir)}"
        ) from exc

    actual = digest.hexdigest()
    if actual != manifest.sha256:
        tmp_path.unlink(missing_ok=True)
        raise ArtifactDigestError(
            f"fetched model artifact failed sha256 verification: expected "
            f"{manifest.sha256}, got {actual} (from {manifest.download_url}). "
            f"The artifact was discarded and not cached."
        )

    os.replace(tmp_path, target)
    logger.info("fetched and cached ML despike artifact at %s", target)
    return target


def resolve_artifact(
    manifest: ModelManifest = DEFAULT_MANIFEST,
    artifact_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Return a digest-verified local path to the model artifact.

    Resolution order:

    1. An explicit ``artifact_path`` (tests, parity harness) is verified
       against the pinned digest and returned. Verification is never
       bypassed; the supplied file is never modified or quarantined.
    2. A cached copy is re-verified by streaming hash on every load. On
       match it is returned; on mismatch it is quarantined as
       ``<name>.corrupt-<timestamp>`` and resolution proceeds to fetch.
    3. The artifact is fetched from ``manifest.download_url`` (https
       only), verified, and atomically installed into the cache.

    Args:
        manifest: Frozen artifact identity (digest, filename, URL).
        artifact_path: Optional explicit artifact location.
        cache_dir: Cache directory override (defaults to
            ``~/.cache/sherloc-pipeline/ml_despike``). Used by tests; the
            pipeline always uses the default.

    Raises:
        ArtifactDigestError: A candidate artifact's digest does not match
            the pinned digest (the candidate is never loadable afterward).
        ArtifactFetchError: The artifact is absent and could not be
            fetched; the message names the manual remedy.
    """
    if artifact_path is not None:
        artifact_path = Path(artifact_path)
        if not artifact_path.is_file():
            raise ArtifactFetchError(
                f"explicit artifact path does not exist: {artifact_path}"
            )
        actual = _sha256_file(artifact_path)
        if actual != manifest.sha256:
            raise ArtifactDigestError(
                f"artifact at {artifact_path} failed sha256 verification: "
                f"expected {manifest.sha256}, got {actual}"
            )
        return artifact_path

    cache_dir = default_cache_dir() if cache_dir is None else Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / manifest.artifact_filename

    if target.is_file():
        actual = _sha256_file(target)
        if actual == manifest.sha256:
            return target
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        quarantine = target.with_name(f"{target.name}.corrupt-{timestamp}")
        os.replace(target, quarantine)
        logger.warning(
            "cached ML despike artifact failed sha256 verification "
            "(expected %s, got %s); quarantined to %s and re-fetching",
            manifest.sha256,
            actual,
            quarantine,
        )

    return _fetch_to_cache(manifest, cache_dir, target)
