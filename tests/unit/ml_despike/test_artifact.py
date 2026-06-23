"""Mocked-network tests for artifact fetch-and-cache integrity mechanics.

Covers MLD-DET-003 AC1–AC4 (decomposing MLD-IFC-006 and MLD-SEC-002):
fetch-verify-cache, cache re-verification with quarantine, fetched-bytes
tamper rejection, explicit-path verification with no bypass, and
actionable failure remedies. No test touches the network or the real
user cache.
"""

import dataclasses
import hashlib
import io
from urllib.error import URLError

import pytest

from sherloc_pipeline.ml_despike import artifact as artifact_mod
from sherloc_pipeline.ml_despike.artifact import (
    ArtifactDigestError,
    ArtifactFetchError,
    default_cache_dir,
    resolve_artifact,
)
from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST

PAYLOAD = b"stub-onnx-artifact-bytes-" * 64
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def manifest():
    return dataclasses.replace(
        DEFAULT_MANIFEST,
        artifact_filename="stub-artifact.onnx",
        sha256=PAYLOAD_SHA256,
        download_url="https://example.com/models/stub-artifact.onnx",
    )


class _FetchRecorder:
    """Stand-in for ``urlopen`` serving fixed bytes and counting calls."""

    def __init__(self, payload=PAYLOAD, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0
        self.timeouts = []

    def __call__(self, url, timeout=None):
        self.calls += 1
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return io.BytesIO(self.payload)


@pytest.fixture
def fetcher(monkeypatch):
    recorder = _FetchRecorder()
    monkeypatch.setattr(artifact_mod, "urlopen", recorder)
    return recorder


class TestExplicitPath:
    def test_good_digest_returned_without_network(self, tmp_path, manifest, fetcher):
        path = tmp_path / "supplied.onnx"
        path.write_bytes(PAYLOAD)
        resolved = resolve_artifact(manifest, artifact_path=path)
        assert resolved == path
        assert fetcher.calls == 0

    def test_bad_digest_rejected_no_bypass(self, tmp_path, manifest, fetcher):
        path = tmp_path / "supplied.onnx"
        tampered = PAYLOAD + b"!"
        path.write_bytes(tampered)
        with pytest.raises(ArtifactDigestError) as excinfo:
            resolve_artifact(manifest, artifact_path=path)
        message = str(excinfo.value)
        assert manifest.sha256 in message
        assert hashlib.sha256(tampered).hexdigest() in message
        # The user's file is rejected, not modified or quarantined.
        assert path.read_bytes() == tampered
        assert fetcher.calls == 0

    def test_missing_path_raises_fetch_error(self, tmp_path, manifest, fetcher):
        missing = tmp_path / "nope.onnx"
        with pytest.raises(ArtifactFetchError, match="does not exist"):
            resolve_artifact(manifest, artifact_path=missing)
        assert fetcher.calls == 0


class TestFetchAndCache:
    def test_first_use_fetches_second_hits_cache(self, tmp_path, manifest, fetcher):
        cache = tmp_path / "cache"
        first = resolve_artifact(manifest, cache_dir=cache)
        assert first == cache / manifest.artifact_filename
        assert first.read_bytes() == PAYLOAD
        assert fetcher.calls == 1

        second = resolve_artifact(manifest, cache_dir=cache)
        assert second == first
        assert fetcher.calls == 1  # no network on cache hit

    def test_cache_dir_created_if_absent(self, tmp_path, manifest, fetcher):
        cache = tmp_path / "deep" / "nested" / "cache"
        resolved = resolve_artifact(manifest, cache_dir=cache)
        assert resolved.is_file()

    def test_no_temp_residue_after_success(self, tmp_path, manifest, fetcher):
        cache = tmp_path / "cache"
        resolve_artifact(manifest, cache_dir=cache)
        leftovers = [p for p in cache.iterdir() if ".fetch-" in p.name]
        assert leftovers == []

    def test_fetch_uses_finite_timeout(self, tmp_path, manifest, fetcher):
        """A stalled host must fail, not hang (MLD-IFC-008 AC2)."""
        resolve_artifact(manifest, cache_dir=tmp_path / "cache")
        assert fetcher.timeouts == [pytest.approx(60.0)]


class TestCacheTamper:
    def test_corrupt_cache_quarantined_and_refetched(self, tmp_path, manifest, fetcher):
        cache = tmp_path / "cache"
        cache.mkdir()
        target = cache / manifest.artifact_filename
        target.write_bytes(b"corrupted-bytes")

        resolved = resolve_artifact(manifest, cache_dir=cache)

        assert resolved == target
        assert resolved.read_bytes() == PAYLOAD
        assert fetcher.calls == 1
        quarantined = list(cache.glob(f"{manifest.artifact_filename}.corrupt-*"))
        assert len(quarantined) == 1
        assert quarantined[0].read_bytes() == b"corrupted-bytes"


class TestFetchTamper:
    def test_tampered_download_rejected_and_discarded(self, tmp_path, manifest, monkeypatch):
        recorder = _FetchRecorder(payload=PAYLOAD + b"tampered")
        monkeypatch.setattr(artifact_mod, "urlopen", recorder)
        cache = tmp_path / "cache"

        with pytest.raises(ArtifactDigestError) as excinfo:
            resolve_artifact(manifest, cache_dir=cache)

        message = str(excinfo.value)
        assert manifest.sha256 in message
        assert hashlib.sha256(PAYLOAD + b"tampered").hexdigest() in message
        # Nothing loadable was left behind: no cache target, no temp file.
        assert not (cache / manifest.artifact_filename).exists()
        assert list(cache.iterdir()) == []


class TestFetchFailure:
    def test_network_failure_names_remedy(self, tmp_path, manifest, monkeypatch):
        recorder = _FetchRecorder(error=URLError("connection refused"))
        monkeypatch.setattr(artifact_mod, "urlopen", recorder)
        cache = tmp_path / "cache"

        with pytest.raises(ArtifactFetchError) as excinfo:
            resolve_artifact(manifest, cache_dir=cache)

        message = str(excinfo.value)
        assert manifest.download_url in message
        assert str(cache / manifest.artifact_filename) in message
        assert list(cache.iterdir()) == []

    def test_timeout_failure_names_remedy(self, tmp_path, manifest, monkeypatch):
        """Socket timeouts surface as the actionable fetch error."""
        recorder = _FetchRecorder(error=TimeoutError("timed out"))
        monkeypatch.setattr(artifact_mod, "urlopen", recorder)
        cache = tmp_path / "cache"

        with pytest.raises(ArtifactFetchError) as excinfo:
            resolve_artifact(manifest, cache_dir=cache)

        message = str(excinfo.value)
        assert manifest.download_url in message
        assert str(cache / manifest.artifact_filename) in message
        assert list(cache.iterdir()) == []

    def test_non_https_url_refused_before_network(self, tmp_path, manifest, fetcher):
        http_manifest = dataclasses.replace(
            manifest, download_url="http://example.com/models/stub-artifact.onnx"
        )
        with pytest.raises(ArtifactFetchError, match="https required"):
            resolve_artifact(http_manifest, cache_dir=tmp_path / "cache")
        assert fetcher.calls == 0


class TestDefaultCacheDir:
    def test_under_user_cache_namespace(self):
        path = default_cache_dir()
        assert path.parts[-2:] == ("sherloc-pipeline", "ml_despike")
        assert ".cache" in path.parts
