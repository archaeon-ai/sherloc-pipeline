"""
Unit tests for R1Dataset extraction and NPZ caching.

Tests cover:
    - Wavenumber calibration (polynomial, NOT linspace)
    - R1 mask produces exactly 523 channels
    - Known peak positions (sulfate, carbonate, olivine)
    - BLOB decompression
    - Cache save/load round-trip with checksum verification
    - Cache verification against database
    - Error handling (missing cache, checksum mismatch)
"""

import hashlib
import json
import sqlite3
import tempfile
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sherloc_pipeline.core.r1_extraction import (
    DEFAULT_CACHE_PATH,
    DEFAULT_DB_PATH,
    R1Dataset,
    _compute_checksum,
    _decompress_spectrum,
    _get_git_sha,
)
from sherloc_pipeline.core.calibration import calculate_loupe_wavelength_wavenumber


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

class TestWavenumberCalibration:
    """Verify polynomial calibration produces correct R1 parameters."""

    def test_full_calibration_shape(self):
        """calculate_loupe_wavelength_wavenumber returns 2148-element arrays."""
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=2148)
        assert wavelength.shape == (2148,)
        assert wavenumber.shape == (2148,)

    def test_r1_mask_produces_523_channels(self):
        """R1 wavelength mask (250-282 nm) selects exactly 523 channels."""
        wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels=2148)
        r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)
        assert r1_mask.sum() == 523

    def test_r1_channel_range(self):
        """R1 channels span indices 52 to 574."""
        wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels=2148)
        r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)
        indices = np.where(r1_mask)[0]
        assert indices[0] == 52
        assert indices[-1] == 574

    def test_r1_wavenumber_range(self):
        """R1 wavenumber range is approximately 238-4765 cm-1."""
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=2148)
        r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)
        r1_wn = wavenumber[r1_mask]
        assert r1_wn.min() == pytest.approx(238, abs=5)
        assert r1_wn.max() == pytest.approx(4765, abs=5)

    def test_known_peak_sulfate_1018(self):
        """Sulfate nu1 peak at ~1018 cm-1 is within resolution of calibrated axis."""
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=2148)
        r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)
        r1_wn = wavenumber[r1_mask]
        closest = r1_wn[np.argmin(np.abs(r1_wn - 1018))]
        assert abs(closest - 1018) < 10  # Within 10 cm-1

    def test_known_peak_carbonate_1085(self):
        """Carbonate nu1 peak at ~1085 cm-1 is within resolution."""
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=2148)
        r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)
        r1_wn = wavenumber[r1_mask]
        closest = r1_wn[np.argmin(np.abs(r1_wn - 1085))]
        assert abs(closest - 1085) < 10

    def test_known_peak_olivine_820(self):
        """Olivine peak at ~820 cm-1 is within resolution."""
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=2148)
        r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)
        r1_wn = wavenumber[r1_mask]
        closest = r1_wn[np.argmin(np.abs(r1_wn - 820))]
        assert abs(closest - 820) < 10

    def test_known_peak_olivine_850(self):
        """Olivine peak at ~850 cm-1 is within resolution."""
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=2148)
        r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)
        r1_wn = wavenumber[r1_mask]
        closest = r1_wn[np.argmin(np.abs(r1_wn - 850))]
        assert abs(closest - 850) < 10

    def test_wavenumber_is_not_linspace(self):
        """Verify wavenumber is NOT linearly spaced (polynomial calibration)."""
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=2148)
        r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)
        r1_wn = wavenumber[r1_mask]
        diffs = np.diff(r1_wn)
        # Polynomial produces non-uniform spacing; linspace would be constant
        assert diffs.std() > 0.01, "Wavenumber diffs should not be constant (not linspace)"


# ---------------------------------------------------------------------------
# BLOB decompression tests
# ---------------------------------------------------------------------------

class TestDecompression:
    """Test zlib BLOB decompression."""

    def test_decompress_roundtrip(self):
        """Compress then decompress float32 data."""
        original = np.random.randn(2148).astype(np.float32)
        blob = zlib.compress(original.tobytes())
        result = _decompress_spectrum(blob)
        np.testing.assert_array_equal(result, original)

    def test_decompress_dtype(self):
        """Decompressed data is float32."""
        original = np.ones(2148, dtype=np.float32)
        blob = zlib.compress(original.tobytes())
        result = _decompress_spectrum(blob)
        assert result.dtype == np.float32

    def test_decompress_shape(self):
        """Decompressed data has correct length."""
        original = np.zeros(2148, dtype=np.float32)
        blob = zlib.compress(original.tobytes())
        result = _decompress_spectrum(blob)
        assert result.shape == (2148,)


# ---------------------------------------------------------------------------
# Checksum tests
# ---------------------------------------------------------------------------

class TestChecksum:
    """Test SHA256 checksum computation."""

    def test_checksum_deterministic(self):
        """Same array produces same checksum."""
        arr = np.ones((10, 523), dtype=np.float32)
        c1 = _compute_checksum(arr)
        c2 = _compute_checksum(arr)
        assert c1 == c2

    def test_checksum_differs_for_different_data(self):
        """Different arrays produce different checksums."""
        a1 = np.zeros((10, 523), dtype=np.float32)
        a2 = np.ones((10, 523), dtype=np.float32)
        assert _compute_checksum(a1) != _compute_checksum(a2)

    def test_checksum_format(self):
        """Checksum is a 64-char hex string (SHA256)."""
        arr = np.array([1.0, 2.0], dtype=np.float32)
        cs = _compute_checksum(arr)
        assert len(cs) == 64
        assert all(c in "0123456789abcdef" for c in cs)


# ---------------------------------------------------------------------------
# R1Dataset construction tests
# ---------------------------------------------------------------------------

class TestR1DatasetInit:
    """Test R1Dataset initialization."""

    def test_default_parameters(self):
        """Default init uses expected parameters."""
        ds = R1Dataset()
        assert ds.db_path == DEFAULT_DB_PATH
        assert ds.spectrum_type == "dark_subtracted"
        assert ds.mars_only is False
        assert ds.spectra is None
        assert ds.wavenumber is None
        assert ds.metadata is None

    def test_custom_parameters(self):
        """Custom parameters are stored."""
        ds = R1Dataset(db_path="/tmp/test.db", spectrum_type="active", mars_only=True)
        assert ds.db_path == "/tmp/test.db"
        assert ds.spectrum_type == "active"
        assert ds.mars_only is True

    def test_n_channels_property(self):
        """n_channels returns 523."""
        ds = R1Dataset()
        assert ds.n_channels == 523

    def test_r1_mask_computed_on_init(self):
        """R1 mask is computed during __init__."""
        ds = R1Dataset()
        assert ds._r1_mask.sum() == 523
        assert ds._r1_wavenumber.shape == (523,)


# ---------------------------------------------------------------------------
# Cache round-trip tests (in-memory, no real DB)
# ---------------------------------------------------------------------------

class TestCacheRoundTrip:
    """Test NPZ cache save and load."""

    def _make_dataset(self, n_spectra=100):
        """Create a synthetic R1Dataset for testing."""
        ds = R1Dataset(db_path="/tmp/fake.db")
        ds.spectra = np.random.randn(n_spectra, 523).astype(np.float32)
        ds.wavenumber = ds._r1_wavenumber.copy()
        ds.metadata = pd.DataFrame(
            {
                "scan_point_id": [f"sp-{i}" for i in range(n_spectra)],
                "scan_id": [f"sc-{i // 10}" for i in range(n_spectra)],
                "target": [f"target-{i % 5}" for i in range(n_spectra)],
                "sol_number": [100 + i % 20 for i in range(n_spectra)],
            }
        )
        return ds

    def test_save_and_load(self, tmp_path):
        """Save then load produces identical data."""
        ds = self._make_dataset()
        cache_file = str(tmp_path / "test_cache.npz")
        ds.save_cache(cache_file)

        ds2 = R1Dataset.from_cache(cache_file)
        np.testing.assert_array_equal(ds2.spectra, ds.spectra)
        np.testing.assert_array_equal(ds2.wavenumber, ds.wavenumber)
        assert list(ds2.metadata.columns) == ["scan_point_id", "scan_id", "target", "sol_number"]
        assert len(ds2.metadata) == len(ds.metadata)

    def test_save_creates_parent_dirs(self, tmp_path):
        """save_cache creates parent directories."""
        ds = self._make_dataset(10)
        cache_file = str(tmp_path / "nested" / "dir" / "cache.npz")
        ds.save_cache(cache_file)
        assert Path(cache_file).exists()

    def test_save_returns_checksum(self, tmp_path):
        """save_cache returns the SHA256 checksum string."""
        ds = self._make_dataset(10)
        cache_file = str(tmp_path / "cache.npz")
        cs = ds.save_cache(cache_file)
        assert len(cs) == 64  # SHA256 hex
        assert cs == _compute_checksum(ds.spectra)

    def test_save_without_load_raises(self):
        """save_cache raises ValueError if load() not called."""
        ds = R1Dataset()
        with pytest.raises(ValueError, match="No data loaded"):
            ds.save_cache("/tmp/nope.npz")

    def test_load_nonexistent_cache_raises(self):
        """from_cache raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Cache file not found"):
            R1Dataset.from_cache("/tmp/nonexistent_file.npz")

    def test_checksum_verification_on_load(self, tmp_path):
        """from_cache detects corrupted data via checksum."""
        ds = self._make_dataset(10)
        cache_file = str(tmp_path / "cache.npz")
        ds.save_cache(cache_file)

        # Corrupt the file by modifying the spectra array
        data = dict(np.load(cache_file, allow_pickle=True))
        data["spectra"] = np.zeros_like(data["spectra"])  # Replace with zeros
        np.savez_compressed(tmp_path / "corrupted.npz", **data)

        with pytest.raises(ValueError, match="checksum mismatch"):
            R1Dataset.from_cache(str(tmp_path / "corrupted.npz"), verify_checksum=True)

    def test_skip_checksum_verification(self, tmp_path):
        """from_cache with verify_checksum=False skips check."""
        ds = self._make_dataset(10)
        cache_file = str(tmp_path / "cache.npz")
        ds.save_cache(cache_file)

        # Corrupt but skip verification
        data = dict(np.load(cache_file, allow_pickle=True))
        data["spectra"] = np.zeros_like(data["spectra"])
        np.savez_compressed(tmp_path / "corrupted.npz", **data)

        # Should not raise
        ds2 = R1Dataset.from_cache(str(tmp_path / "corrupted.npz"), verify_checksum=False)
        assert ds2.spectra is not None

    def test_metadata_preserved(self, tmp_path):
        """Metadata DataFrame is correctly round-tripped through cache."""
        ds = self._make_dataset(5)
        cache_file = str(tmp_path / "cache.npz")
        ds.save_cache(cache_file)

        ds2 = R1Dataset.from_cache(cache_file)
        pd.testing.assert_frame_equal(
            ds2.metadata.reset_index(drop=True),
            ds.metadata.reset_index(drop=True),
            check_dtype=False,  # numpy may return different string types
        )

    def test_wavenumber_preserved(self, tmp_path):
        """Wavenumber array is identical after round-trip."""
        ds = self._make_dataset(5)
        cache_file = str(tmp_path / "cache.npz")
        ds.save_cache(cache_file)

        ds2 = R1Dataset.from_cache(cache_file)
        np.testing.assert_allclose(ds2.wavenumber, ds.wavenumber, rtol=1e-10)

    def test_query_params_preserved(self, tmp_path):
        """Query parameters are stored and restored from cache."""
        ds = R1Dataset(db_path="/tmp/custom.db", spectrum_type="active", mars_only=True)
        ds.spectra = np.random.randn(5, 523).astype(np.float32)
        ds.wavenumber = ds._r1_wavenumber.copy()
        ds.metadata = pd.DataFrame(
            {
                "scan_point_id": ["a", "b", "c", "d", "e"],
                "scan_id": ["s1"] * 5,
                "target": ["t1"] * 5,
                "sol_number": [100] * 5,
            }
        )
        cache_file = str(tmp_path / "cache.npz")
        ds.save_cache(cache_file)

        ds2 = R1Dataset.from_cache(cache_file)
        assert ds2.spectrum_type == "active"
        assert ds2.mars_only is True


# ---------------------------------------------------------------------------
# Cache verification tests
# ---------------------------------------------------------------------------

class TestCacheVerification:
    """Test verify_cache method."""

    def _make_dataset_with_cache(self, tmp_path, n_spectra=10):
        """Create a dataset and save to cache."""
        ds = R1Dataset(db_path="/tmp/fake.db")
        ds.spectra = np.random.randn(n_spectra, 523).astype(np.float32)
        ds.wavenumber = ds._r1_wavenumber.copy()
        ds.metadata = pd.DataFrame(
            {
                "scan_point_id": [f"sp-{i}" for i in range(n_spectra)],
                "scan_id": [f"sc-{i}" for i in range(n_spectra)],
                "target": [f"t-{i}" for i in range(n_spectra)],
                "sol_number": [100 + i for i in range(n_spectra)],
            }
        )
        cache_file = str(tmp_path / "cache.npz")
        ds.save_cache(cache_file)
        return ds, cache_file

    def test_verify_missing_cache(self, tmp_path):
        """verify_cache returns invalid for missing file."""
        ds = R1Dataset(db_path="/tmp/fake.db")
        result = ds.verify_cache(str(tmp_path / "nonexistent.npz"))
        assert result["valid"] is False
        assert "not found" in result["reason"]

    def test_verify_valid_cache(self, tmp_path):
        """verify_cache returns valid when checksum and count match."""
        ds, cache_file = self._make_dataset_with_cache(tmp_path, n_spectra=10)

        # Mock the database count to match
        with patch("sherloc_pipeline.core.r1_extraction.sqlite3") as mock_sqlite3:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = (10,)
            mock_sqlite3.connect.return_value = mock_conn

            result = ds.verify_cache(cache_file)

        assert result["valid"] is True
        assert result["details"]["checksum_valid"] is True

    def test_verify_stale_cache(self, tmp_path):
        """verify_cache detects row count mismatch."""
        ds, cache_file = self._make_dataset_with_cache(tmp_path, n_spectra=10)

        # Mock DB returning different count
        with patch("sherloc_pipeline.core.r1_extraction.sqlite3") as mock_sqlite3:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = (20,)  # Different!
            mock_sqlite3.connect.return_value = mock_conn

            result = ds.verify_cache(cache_file)

        assert result["valid"] is False
        assert "Row count mismatch" in result["reason"]


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------

class TestSummary:
    """Test summary method."""

    def test_summary_not_loaded(self):
        """Summary returns status=not loaded when data not loaded."""
        ds = R1Dataset()
        s = ds.summary()
        assert s["status"] == "not loaded"

    def test_summary_after_data(self):
        """Summary returns expected keys with data."""
        ds = R1Dataset()
        ds.spectra = np.random.randn(100, 523).astype(np.float32)
        ds.wavenumber = ds._r1_wavenumber.copy()
        ds.metadata = pd.DataFrame(
            {
                "scan_point_id": [f"sp-{i}" for i in range(100)],
                "scan_id": [f"sc-{i // 10}" for i in range(100)],
                "target": [f"target-{i % 5}" for i in range(100)],
                "sol_number": [100 + i for i in range(100)],
            }
        )

        s = ds.summary()
        assert s["n_spectra"] == 100
        assert s["n_channels"] == 523
        assert s["spectrum_type"] == "dark_subtracted"
        assert s["mars_only"] is False
        assert s["unique_targets"] == 5
        assert s["unique_scans"] == 10
        assert "wavenumber_min" in s
        assert "wavenumber_max" in s


# ---------------------------------------------------------------------------
# Database load test with mock
# ---------------------------------------------------------------------------

class TestDatabaseLoad:
    """Test database loading with mocked SQLite."""

    def test_load_returns_correct_shapes(self):
        """load() produces (N, 523) spectra, (523,) wavenumber, N-row metadata."""
        n_spectra = 5
        # Create fake compressed 2148-channel spectra
        fake_blobs = []
        for i in range(n_spectra):
            arr = np.random.randn(2148).astype(np.float32)
            fake_blobs.append(zlib.compress(arr.tobytes()))

        fake_rows = [
            (fake_blobs[i], f"sp-{i}", f"sc-{i}", f"target-{i}", 100 + i)
            for i in range(n_spectra)
        ]

        ds = R1Dataset(db_path="/tmp/fake.db")

        with patch("sherloc_pipeline.core.r1_extraction.sqlite3") as mock_sqlite3:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            # fetchmany returns all rows first, then empty
            mock_cursor.fetchmany.side_effect = [fake_rows, []]
            mock_conn.execute.return_value = mock_cursor
            mock_sqlite3.connect.return_value = mock_conn

            spectra, wavenumber, metadata = ds.load()

        assert spectra.shape == (n_spectra, 523)
        assert spectra.dtype == np.float32
        assert wavenumber.shape == (523,)
        assert len(metadata) == n_spectra
        assert list(metadata.columns) == ["scan_point_id", "scan_id", "target", "sol_number"]

    def test_load_no_data_raises(self):
        """load() raises ValueError if no spectra found."""
        ds = R1Dataset(db_path="/tmp/fake.db")

        with patch("sherloc_pipeline.core.r1_extraction.sqlite3") as mock_sqlite3:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchmany.return_value = []
            mock_conn.execute.return_value = mock_cursor
            mock_sqlite3.connect.return_value = mock_conn

            with pytest.raises(ValueError, match="No spectra found"):
                ds.load()

    def test_load_mars_only_adds_filter(self):
        """load() with mars_only=True adds target IS NOT NULL filter."""
        n_spectra = 2
        fake_blobs = []
        for i in range(n_spectra):
            arr = np.random.randn(2148).astype(np.float32)
            fake_blobs.append(zlib.compress(arr.tobytes()))

        fake_rows = [
            (fake_blobs[i], f"sp-{i}", f"sc-{i}", f"target-{i}", 100 + i)
            for i in range(n_spectra)
        ]

        ds = R1Dataset(db_path="/tmp/fake.db", mars_only=True)

        with patch("sherloc_pipeline.core.r1_extraction.sqlite3") as mock_sqlite3:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchmany.side_effect = [fake_rows, []]
            mock_conn.execute.return_value = mock_cursor
            mock_sqlite3.connect.return_value = mock_conn

            ds.load()

            # Verify the query includes the mars_only filter
            call_args = mock_conn.execute.call_args
            query = call_args[0][0]
            assert "target IS NOT NULL" in query

    def test_load_r1_masking_correct(self):
        """load() correctly applies R1 mask to 2148-channel data."""
        # Create a spectrum with known pattern: all zeros except channels 52-574
        full_spectrum = np.zeros(2148, dtype=np.float32)
        full_spectrum[52:575] = 1.0  # R1 region (channels 52-574 inclusive)
        blob = zlib.compress(full_spectrum.tobytes())

        ds = R1Dataset(db_path="/tmp/fake.db")

        with patch("sherloc_pipeline.core.r1_extraction.sqlite3") as mock_sqlite3:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchmany.side_effect = [
                [(blob, "sp-0", "sc-0", "target-0", 100)],
                [],
            ]
            mock_conn.execute.return_value = mock_cursor
            mock_sqlite3.connect.return_value = mock_conn

            spectra, _, _ = ds.load()

        # All 523 R1 channels should be 1.0 (they fall within 52-574)
        assert spectra.shape == (1, 523)
        assert np.all(spectra[0] == 1.0)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestGitSha:
    """Test git SHA retrieval."""

    def test_git_sha_returns_string(self):
        """_get_git_sha returns a string (even if git not available)."""
        sha = _get_git_sha()
        assert isinstance(sha, str)
        assert len(sha) > 0

    def test_git_sha_handles_failure(self):
        """_get_git_sha returns 'unknown' if git fails."""
        with patch("sherloc_pipeline.core.r1_extraction.subprocess.run", side_effect=Exception("no git")):
            sha = _get_git_sha()
        assert sha == "unknown"
