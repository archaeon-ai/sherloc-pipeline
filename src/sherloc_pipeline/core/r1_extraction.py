"""
R1 Raman dataset extraction and caching for SHERLOC spectral analysis.

This module provides the R1Dataset class that extracts R1 Raman spectra from
the PHASE database with correct polynomial calibration, and caches to NPZ
for fast reuse by downstream ML agents.

Key design decisions:
    - Wavenumber axis uses Loupe polynomial calibration (NEVER np.linspace)
    - R1 region defined by wavelength mask (250-282 nm), yielding 523 channels
    - BLOBs are zlib-compressed float32 arrays of 2148 channels each
    - NPZ cache includes checksum and row count for staleness detection

Example:
    >>> from sherloc_pipeline.core.r1_extraction import R1Dataset
    >>> ds = R1Dataset(db_path='./phase.db')
    >>> spectra, wavenumber, metadata = ds.load()
    >>> ds.save_cache('outputs/r1_dataset_cache.npz')
    >>> # Later:
    >>> ds2 = R1Dataset.from_cache('outputs/r1_dataset_cache.npz')
"""

import hashlib
import json
import logging
import sqlite3
import subprocess
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sherloc_pipeline.core.calibration import calculate_loupe_wavelength_wavenumber
from sherloc_pipeline.core.utils import require_file

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = "./phase.db"
DEFAULT_CACHE_PATH = "outputs/r1_dataset_cache.npz"


def _get_git_sha() -> str:
    """Get current git SHA for reproducibility metadata."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _compute_checksum(arr: np.ndarray) -> str:
    """Compute SHA256 checksum of a numpy array."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _decompress_spectrum(blob: bytes) -> np.ndarray:
    """Decompress a zlib-compressed float32 spectrum blob.

    Args:
        blob: zlib-compressed bytes containing float32 array.

    Returns:
        1-D numpy array of float32 intensities (2148 channels).
    """
    return np.frombuffer(zlib.decompress(blob), dtype=np.float32)


class R1Dataset:
    """R1 Raman dataset with polynomial-calibrated wavenumber axis.

    Extracts the R1 (Raman) region from SHERLOC PHASE database spectra
    using the Loupe polynomial calibration. Each spectrum is 2148 channels
    on the CCD; the R1 region is the 523 channels at wavelengths 250-282 nm.

    Attributes:
        spectra: np.ndarray of shape (N, 523) -- R1 region intensities
        wavenumber: np.ndarray of shape (523,) -- polynomial-calibrated wavenumber axis
        metadata: pd.DataFrame with columns scan_point_id, scan_id, target, sol_number

    Args:
        db_path: Path to the PHASE SQLite database.
        spectrum_type: Filter on spectrum_type column (default: 'dark_subtracted').
        mars_only: If True, filter to scans with target IS NOT NULL.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        spectrum_type: str = "dark_subtracted",
        mars_only: bool = False,
    ):
        self.db_path = db_path
        self.spectrum_type = spectrum_type
        self.mars_only = mars_only

        # These are populated by load() or from_cache()
        self.spectra: Optional[np.ndarray] = None
        self.wavenumber: Optional[np.ndarray] = None
        self.metadata: Optional[pd.DataFrame] = None

        # Compute R1 calibration once
        self._wavelength, self._full_wavenumber = calculate_loupe_wavelength_wavenumber(
            n_channels=2148
        )
        self._r1_mask = (self._wavelength >= 250.0) & (self._wavelength <= 282.0)
        self._r1_wavenumber = self._full_wavenumber[self._r1_mask]

    @property
    def n_channels(self) -> int:
        """Number of R1 channels (523)."""
        return int(self._r1_mask.sum())

    def load(self) -> tuple:
        """Load R1 spectra from database with correct polynomial calibration.

        Executes a JOIN query across spectra -> scan_points -> scans to fetch
        dark-subtracted intensity BLOBs along with metadata. Each 2148-channel
        BLOB is decompressed and masked to the R1 region (523 channels).

        Returns:
            Tuple of (spectra, wavenumber, metadata):
                - spectra: np.ndarray (N, 523) float32
                - wavenumber: np.ndarray (523,) float64
                - metadata: pd.DataFrame with scan_point_id, scan_id, target, sol_number
        """
        logger.info(
            "Loading R1 dataset from %s (spectrum_type=%s, mars_only=%s)",
            self.db_path,
            self.spectrum_type,
            self.mars_only,
        )

        conn = sqlite3.connect(self.db_path)
        try:
            query = """
                SELECT s.intensities, sp.id as scan_point_id, sc.id as scan_id,
                       sc.target, sc.sol_number
                FROM spectra s
                JOIN scan_points sp ON s.scan_point_id = sp.id
                JOIN scans sc ON sp.scan_id = sc.id
                WHERE s.spectrum_type = ?
            """
            params = [self.spectrum_type]

            if self.mars_only:
                query += " AND sc.target IS NOT NULL"

            cursor = conn.execute(query, params)

            # Process results in batches for memory efficiency
            blobs = []
            meta_rows = []
            batch_size = 10000
            total = 0

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                for row in rows:
                    blob, sp_id, sc_id, target, sol_number = row
                    full_spectrum = _decompress_spectrum(blob)
                    r1_spectrum = full_spectrum[self._r1_mask]
                    blobs.append(r1_spectrum)
                    meta_rows.append(
                        {
                            "scan_point_id": sp_id,
                            "scan_id": sc_id,
                            "target": target,
                            "sol_number": sol_number,
                        }
                    )

                total += len(rows)
                if total % 50000 == 0:
                    logger.info("  Loaded %d spectra so far...", total)

        finally:
            conn.close()

        logger.info("Loaded %d spectra total", total)

        if not blobs:
            raise ValueError(
                f"No spectra found for spectrum_type={self.spectrum_type}, "
                f"mars_only={self.mars_only}"
            )

        self.spectra = np.array(blobs, dtype=np.float32)
        self.wavenumber = self._r1_wavenumber.copy()
        self.metadata = pd.DataFrame(meta_rows)

        logger.info(
            "R1Dataset: spectra shape=%s, wavenumber shape=%s, metadata rows=%d",
            self.spectra.shape,
            self.wavenumber.shape,
            len(self.metadata),
        )

        return self.spectra, self.wavenumber, self.metadata

    def save_cache(self, path: str = DEFAULT_CACHE_PATH) -> str:
        """Save loaded dataset to NPZ cache with metadata for verification.

        The cache includes:
            - spectra: (N, 523) float32 array
            - wavenumber: (523,) float64 array
            - db_row_count: number of spectra rows
            - git_sha: current git commit for reproducibility
            - query_params: JSON string of extraction parameters
            - checksum: SHA256 of spectra array bytes

        Args:
            path: Output path for the .npz file.

        Returns:
            The checksum string written to the cache.

        Raises:
            ValueError: If load() has not been called yet.
        """
        if self.spectra is None:
            raise ValueError("No data loaded. Call load() first.")

        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        checksum = _compute_checksum(self.spectra)
        git_sha = _get_git_sha()
        query_params = json.dumps(
            {
                "db_path": self.db_path,
                "spectrum_type": self.spectrum_type,
                "mars_only": self.mars_only,
            }
        )

        # Save metadata DataFrame columns as separate arrays
        np.savez_compressed(
            cache_path,
            spectra=self.spectra,
            wavenumber=self.wavenumber,
            meta_scan_point_id=self.metadata["scan_point_id"].values,
            meta_scan_id=self.metadata["scan_id"].values,
            meta_target=self.metadata["target"].values.astype(str),
            meta_sol_number=self.metadata["sol_number"].values,
            db_row_count=np.array([len(self.spectra)]),
            git_sha=np.array([git_sha]),
            query_params=np.array([query_params]),
            checksum=np.array([checksum]),
        )

        logger.info(
            "Saved R1 cache to %s (%d spectra, checksum=%s)",
            cache_path,
            len(self.spectra),
            checksum[:16] + "...",
        )
        return checksum

    @classmethod
    def from_cache(
        cls,
        path: str = DEFAULT_CACHE_PATH,
        verify_checksum: bool = True,
        db_path: str = DEFAULT_DB_PATH,
    ) -> "R1Dataset":
        """Load R1Dataset from a previously saved NPZ cache.

        Args:
            path: Path to the .npz cache file.
            verify_checksum: If True, verify SHA256 checksum on load.
            db_path: Database path (used for staleness checks, not for loading).

        Returns:
            R1Dataset instance with spectra, wavenumber, and metadata populated.

        Raises:
            FileNotFoundError: If cache file does not exist.
            ValueError: If checksum verification fails.
        """
        cache_path = Path(path)
        require_file(cache_path, "Cache file not found")

        logger.info("Loading R1 cache from %s", cache_path)
        data = np.load(cache_path, allow_pickle=True)

        spectra = data["spectra"]
        wavenumber = data["wavenumber"]
        checksum_stored = str(data["checksum"][0])

        if verify_checksum:
            checksum_actual = _compute_checksum(spectra)
            if checksum_actual != checksum_stored:
                raise ValueError(
                    f"Cache checksum mismatch: stored={checksum_stored[:16]}... "
                    f"actual={checksum_actual[:16]}..."
                )
            logger.info("Cache checksum verified OK")

        # Reconstruct query params from cache
        query_params = json.loads(str(data["query_params"][0]))

        instance = cls(
            db_path=query_params.get("db_path", db_path),
            spectrum_type=query_params.get("spectrum_type", "dark_subtracted"),
            mars_only=query_params.get("mars_only", False),
        )
        instance.spectra = spectra
        instance.wavenumber = wavenumber

        # Reconstruct metadata DataFrame
        instance.metadata = pd.DataFrame(
            {
                "scan_point_id": data["meta_scan_point_id"],
                "scan_id": data["meta_scan_id"],
                "target": data["meta_target"],
                "sol_number": data["meta_sol_number"],
            }
        )

        logger.info(
            "Loaded R1 cache: %d spectra, %d channels, checksum OK",
            spectra.shape[0],
            spectra.shape[1],
        )
        return instance

    def verify_cache(self, path: str = DEFAULT_CACHE_PATH) -> dict:
        """Verify cache integrity and staleness against the database.

        Checks:
            1. Cache file exists
            2. Checksum matches stored value
            3. Row count matches current database count

        Args:
            path: Path to the .npz cache file.

        Returns:
            Dict with keys: valid (bool), reason (str), details (dict).
        """
        cache_path = Path(path)
        result = {
            "valid": False,
            "reason": "",
            "details": {},
        }

        # Check file exists
        if not cache_path.exists():
            result["reason"] = f"Cache file not found: {cache_path}"
            return result

        # Load cache metadata
        data = np.load(cache_path, allow_pickle=True)
        spectra = data["spectra"]
        checksum_stored = str(data["checksum"][0])
        db_row_count_cached = int(data["db_row_count"][0])
        git_sha = str(data["git_sha"][0])
        query_params = json.loads(str(data["query_params"][0]))

        result["details"]["cached_row_count"] = db_row_count_cached
        result["details"]["cached_git_sha"] = git_sha
        result["details"]["cached_query_params"] = query_params
        result["details"]["spectra_shape"] = list(spectra.shape)

        # Verify checksum
        checksum_actual = _compute_checksum(spectra)
        if checksum_actual != checksum_stored:
            result["reason"] = (
                f"Checksum mismatch: stored={checksum_stored[:16]}... "
                f"actual={checksum_actual[:16]}..."
            )
            return result

        result["details"]["checksum_valid"] = True

        # Check row count against current database
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT COUNT(*) FROM spectra s
                JOIN scan_points sp ON s.scan_point_id = sp.id
                JOIN scans sc ON sp.scan_id = sc.id
                WHERE s.spectrum_type = ?
            """
            params = [query_params.get("spectrum_type", "dark_subtracted")]
            if query_params.get("mars_only", False):
                query += " AND sc.target IS NOT NULL"

            db_count = conn.execute(query, params).fetchone()[0]
            conn.close()

            result["details"]["current_db_count"] = db_count
            if db_count != db_row_count_cached:
                result["reason"] = (
                    f"Row count mismatch: cache={db_row_count_cached}, "
                    f"database={db_count}"
                )
                return result
        except Exception as e:
            result["reason"] = f"Database check failed: {e}"
            return result

        result["valid"] = True
        result["reason"] = "Cache is valid and current"
        return result

    def summary(self) -> dict:
        """Return a summary of the loaded dataset for logging/reporting.

        Returns:
            Dict with dataset statistics.
        """
        if self.spectra is None:
            return {"status": "not loaded"}

        return {
            "n_spectra": self.spectra.shape[0],
            "n_channels": self.spectra.shape[1],
            "wavenumber_min": float(self.wavenumber.min()),
            "wavenumber_max": float(self.wavenumber.max()),
            "spectrum_type": self.spectrum_type,
            "mars_only": self.mars_only,
            "unique_targets": int(self.metadata["target"].nunique()),
            "unique_scans": int(self.metadata["scan_id"].nunique()),
            "sol_range": (
                int(self.metadata["sol_number"].min()),
                int(self.metadata["sol_number"].max()),
            ),
            "intensity_range": (
                float(self.spectra.min()),
                float(self.spectra.max()),
            ),
            "intensity_mean": float(self.spectra.mean()),
        }
