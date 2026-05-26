"""
PDS archive HTTP client for downloading SHERLOC data.

This module provides the PDSDownloader class for discovering and downloading
SHERLOC processed spectral products from the PDS Geosciences Node (WUSTL).
It supports cache-aware downloading, version selection, and retry with
exponential backoff.

The client downloads CSV + XML product pairs organized by sol into a local
cache directory with the structure:
    cache_dir/sol_SSSS/data_processed/*.csv
    cache_dir/sol_SSSS/data_processed/*.xml

See PDS_INGESTION_SPEC.md §2 (Archive Structure) and §6 (Download Strategy).

Example:
    >>> from sherloc_pipeline.core.pds_client import PDSDownloader
    >>>
    >>> with PDSDownloader() as client:
    ...     sols = client.discover_available_sols()
    ...     result = client.download_sol(921)
    ...     print(f"Downloaded {result.n_downloaded}, skipped {result.n_skipped}")
"""

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from sherloc_pipeline.config import PDSConfig
from sherloc_pipeline.core.pds_constants import (
    PDS_BASE_URL,
    PDS_DEFAULT_CACHE_DIR,
    PDS_SEARCH_API_BASE,
)
from sherloc_pipeline.models.pds import PDSProductId

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

__all__ = [
    "PDS_BASE_URL",
    "PDS_DEFAULT_CACHE_DIR",
    "PDS_SEARCH_API_BASE",
    "PDSDownloader",
    "PDSDownloadError",
    "SolDownloadResult",
]


def _require_httpx() -> Any:
    """Import ``httpx`` lazily, with a friendly hint if the extra is missing."""
    try:
        import httpx  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PDS extras not installed. Install with: "
            "pip install 'sherloc-pipeline[pds]'"
        ) from exc
    return httpx


class PDSDownloadError(RuntimeError):
    """Error during PDS data download."""

    pass


@dataclass
class SolDownloadResult:
    """Result of downloading products for a single sol.

    Attributes:
        sol: Mars sol number.
        downloaded: Paths of newly downloaded files.
        skipped: Paths of files skipped (already cached).
        errors: Error messages for failed downloads.
    """

    sol: int
    downloaded: List[Path] = field(default_factory=list)
    skipped: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def n_downloaded(self) -> int:
        """Number of newly downloaded files."""
        return len(self.downloaded)

    @property
    def n_skipped(self) -> int:
        """Number of cached files skipped."""
        return len(self.skipped)


class PDSDownloader:
    """HTTP client for downloading SHERLOC data from PDS Geosciences Node.

    Provides discovery of available sols and per-sol download of processed
    spectral products (CSV + XML pairs). Supports caching (skip already-
    downloaded files), version selection (keep highest version), and
    retry with exponential backoff on HTTP errors.

    Args:
        base_url: PDS Geosciences Node base URL for the SHERLOC bundle.
            Defaults to the WUSTL archive URL.
        cache_dir: Local directory for downloaded files. Sol data is stored
            as ``cache_dir/sol_SSSS/data_processed/``.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum retry attempts on transient HTTP errors
            (5xx, timeouts, connection errors).
        backoff_factor: Base multiplier for exponential backoff between
            retries. Delay = backoff_factor ** attempt.

    Example:
        >>> downloader = PDSDownloader(cache_dir=Path("./pds"))
        >>> sols = downloader.discover_available_sols()
        >>> result = downloader.download_sol(921)
        >>> print(f"Downloaded: {result.n_downloaded}")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        self.base_url = (base_url or PDS_BASE_URL).rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else PDS_DEFAULT_CACHE_DIR
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._client: Optional["httpx.Client"] = None
        self._inventory_cache: Optional[List[str]] = None

    @classmethod
    def from_config(cls, config: Optional[PDSConfig] = None) -> "PDSDownloader":
        """Create a PDSDownloader from pipeline configuration.

        Loads PDS settings from the given PDSConfig (or the global pipeline
        config if none is provided).

        Args:
            config: PDSConfig instance. If None, loads from the global
                pipeline config via ``get_config().pds``.

        Returns:
            Configured PDSDownloader instance.
        """
        if config is None:
            from sherloc_pipeline.config import get_config
            config = get_config().pds
        return cls(
            base_url=config.base_url,
            cache_dir=Path(config.cache_dir),
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            backoff_factor=config.backoff_factor,
        )

    @property
    def client(self) -> "httpx.Client":
        """Lazy-initialized HTTP client. Requires the ``[pds]`` extra."""
        if self._client is None or self._client.is_closed:
            httpx = _require_httpx()
            self._client = httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": "sherloc-pipeline/1.0"},
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None and not self._client.is_closed:
            self._client.close()

    def __enter__(self) -> "PDSDownloader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_available_sols(self) -> List[int]:
        """Query collection inventory to discover available sols.

        Fetches the ``collection_data_processed_inventory.csv`` from the PDS
        Geosciences Node and parses LIDVID entries to extract unique sol
        numbers from product filenames.

        Returns:
            Sorted list of sol numbers that have processed data available.

        Raises:
            PDSDownloadError: If the inventory cannot be fetched or parsed.
        """
        inventory_lines = self._fetch_inventory()

        sols: Set[int] = set()
        for line in inventory_lines:
            # Inventory lines: P,LIDVID or S,LID
            # Product IDs contain sol: ss__SSSS_...
            match = re.search(r"ss__(\d{4})_", line)
            if match:
                sols.add(int(match.group(1)))

        result = sorted(sols)
        if result:
            logger.info(
                "Discovered %d sols with processed data (range: %d-%d)",
                len(result),
                result[0],
                result[-1],
            )
        else:
            logger.warning("No sols found in collection inventory")
        return result

    def download_sol(
        self,
        sol: int,
        *,
        force: bool = False,
        collections: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SolDownloadResult:
        """Download all processed products for a sol.

        Downloads CSV + XML pairs for all products of the specified sol.
        Already-downloaded files are skipped unless ``force=True``. When
        multiple versions of the same product exist, only the highest
        version is downloaded.

        Args:
            sol: Mars sol number to download.
            force: If True, re-download even if files exist locally.
            collections: PDS collections to download from. Defaults to
                ``["data_processed"]``.
            progress_callback: Optional callback invoked after each file
                is downloaded or skipped, receiving ``(done_files, total_files)``
                accumulated across all collections.

        Returns:
            SolDownloadResult with counts of downloaded, skipped, and
            failed files.

        Raises:
            PDSDownloadError: On unrecoverable download errors.
        """
        if collections is None:
            collections = ["data_processed"]

        result = SolDownloadResult(sol=sol)

        # Mutable progress counter shared across collections
        _progress: Optional[Dict[str, int]] = (
            {"done": 0, "total": 0} if progress_callback is not None else None
        )

        for collection in collections:
            self._download_sol_collection(
                sol,
                collection,
                result,
                force=force,
                progress_callback=progress_callback,
                _progress=_progress,
            )

        logger.info(
            "Sol %d: downloaded %d files, skipped %d, errors %d",
            sol,
            result.n_downloaded,
            result.n_skipped,
            len(result.errors),
        )
        return result

    def list_local_sols(self) -> List[int]:
        """List sols that already exist in the local cache directory.

        Returns:
            Sorted list of sol numbers with local data.
        """
        sols: List[int] = []
        if not self.cache_dir.exists():
            return sols

        for entry in self.cache_dir.iterdir():
            if entry.is_dir():
                match = re.match(r"sol_(\d+)$", entry.name)
                if match:
                    sols.append(int(match.group(1)))

        return sorted(sols)

    def resolve_aci_urls(self, lidvids: List[str]) -> Dict[str, str]:
        """Resolve ACI LIDVIDs to download URLs via the PDS Search API.

        For each LIDVID, extracts the LID (stripping the ``::version``
        suffix) and queries the PDS Search API to obtain the file
        download URL from the imaging archive.

        Args:
            lidvids: List of PDS LIDVIDs to resolve.

        Returns:
            Mapping of ``{lidvid: download_url}`` for successful lookups.
            LIDVIDs that cannot be resolved are omitted (with a warning).
        """
        url_map: Dict[str, str] = {}

        for lidvid in lidvids:
            # Extract LID: everything before "::" version suffix
            lid = lidvid.split("::")[0] if "::" in lidvid else lidvid

            search_url = (
                f"{PDS_SEARCH_API_BASE}/products"
                f"?q=lid eq \"{lid}\""
                f"&fields=ops:Data_File_Info.ops:file_ref"
                f"&limit=1"
            )

            try:
                response = self.client.get(
                    search_url,
                    headers={"Accept": "application/json"},
                    timeout=30.0,
                )
                if response.status_code != 200:
                    logger.warning(
                        "PDS Search API returned %d for LID %s",
                        response.status_code,
                        lid,
                    )
                    continue

                data = response.json().get("data", [])
                if not data:
                    logger.warning("No PDS Search results for LID %s", lid)
                    continue

                file_refs = (
                    data[0]
                    .get("properties", {})
                    .get("ops:Data_File_Info.ops:file_ref", [])
                )
                if not file_refs:
                    logger.warning(
                        "No file_ref in PDS Search result for LID %s", lid
                    )
                    continue

                url_map[lidvid] = file_refs[0]
                logger.debug("Resolved %s → %s", lid, file_refs[0])

            except Exception as exc:
                logger.warning(
                    "Failed to resolve ACI LIDVID %s: %s", lidvid, exc
                )

        logger.info(
            "Resolved %d/%d ACI LIDVIDs to download URLs",
            len(url_map),
            len(lidvids),
        )
        return url_map

    def download_aci_image(self, url: str, dest: Path) -> bool:
        """Download a single ACI image file.

        Skips the download if ``dest`` already exists and has size > 0.
        Downloads to a temporary file first, then renames atomically.

        Args:
            url: Remote URL of the ACI image (typically a VICAR .IMG).
            dest: Local file path to write to.

        Returns:
            True if the file was downloaded (or already existed),
            False on failure.
        """
        if dest.exists() and dest.stat().st_size > 0:
            logger.debug("ACI image already cached: %s", dest.name)
            return True

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".IMG.tmp")

        try:
            response = self.client.get(url, timeout=60.0)
            if response.status_code != 200:
                logger.warning(
                    "Failed to download ACI image %s: HTTP %d",
                    url,
                    response.status_code,
                )
                return False

            tmp.write_bytes(response.content)
            tmp.rename(dest)
            logger.debug(
                "Downloaded ACI image %s (%d bytes)",
                dest.name,
                len(response.content),
            )
            return True

        except Exception as exc:
            logger.warning("Failed to download ACI image %s: %s", url, exc)
            tmp.unlink(missing_ok=True)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_sol_collection(
        self,
        sol: int,
        collection: str,
        result: SolDownloadResult,
        *,
        force: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        _progress: Optional[Dict[str, int]] = None,
    ) -> None:
        """Download products for a sol from a specific collection."""
        product_ids = self._find_sol_products(sol, collection)

        if not product_ids:
            logger.warning("No products found for sol %d in %s", sol, collection)
            return

        # Keep only the highest version of each product
        selected = self._select_highest_versions(product_ids)

        logger.info(
            "Sol %d/%s: %d products (%d after version selection)",
            sol,
            collection,
            len(product_ids),
            len(selected),
        )

        # Update total file count for progress tracking (CSV + XML per product)
        if _progress is not None:
            _progress["total"] += 2 * len(selected)

        # Prepare local directory
        sol_dir = self.cache_dir / f"sol_{sol:04d}" / collection
        sol_dir.mkdir(parents=True, exist_ok=True)

        # Download each product's CSV and XML files
        for pid in selected:
            for filename in (pid.csv_filename, pid.xml_filename):
                dest = sol_dir / filename

                if dest.exists() and not force:
                    result.skipped.append(dest)
                else:
                    # PDS server uses 5-digit zero-padded sol directories
                    url = (
                        f"{self.base_url}/{collection}/"
                        f"sol_{sol:05d}/{filename}"
                    )

                    try:
                        self._download_file(url, dest)
                        result.downloaded.append(dest)
                    except PDSDownloadError as e:
                        error_msg = f"Failed to download {filename}: {e}"
                        logger.error(error_msg)
                        result.errors.append(error_msg)

                # Report progress after each file (downloaded, skipped, or failed)
                if _progress is not None:
                    _progress["done"] += 1
                    if progress_callback is not None:
                        progress_callback(_progress["done"], _progress["total"])

    def _fetch_inventory(self) -> List[str]:
        """Fetch and cache the collection inventory CSV.

        Returns:
            List of non-empty inventory lines.

        Raises:
            PDSDownloadError: If the inventory cannot be fetched.
        """
        if self._inventory_cache is not None:
            return self._inventory_cache

        url = (
            f"{self.base_url}/data_processed/"
            "collection_data_processed_inventory.csv"
        )

        logger.info("Fetching collection inventory from %s", url)
        response = self._request_with_retry(url)

        # Filter out blank lines and comment lines
        lines = [
            line
            for line in response.text.strip().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        self._inventory_cache = lines
        logger.info("Inventory contains %d entries", len(lines))
        return lines

    def _find_sol_products(
        self, sol: int, collection: str
    ) -> List[PDSProductId]:
        """Find all product IDs for a sol in the collection inventory.

        Parses the inventory to extract product identifiers for the
        specified sol and attempts to parse each as a PDSProductId.

        Args:
            sol: Sol number to search for.
            collection: Collection name (used for filtering context).

        Returns:
            List of parsed PDSProductId instances for the sol.
        """
        inventory_lines = self._fetch_inventory()
        sol_pattern = f"ss__{sol:04d}_"
        products: List[PDSProductId] = []

        for line in inventory_lines:
            if sol_pattern not in line:
                continue

            # PDS4 inventory format: member_status,LIDVID_LID
            # e.g.: P,urn:nasa:pds:...:data_processed:product_id::1.0
            parts = line.split(",", 1)
            if len(parts) < 2:
                continue

            lidvid = parts[1].strip()

            # Strip version suffix (::X.Y) if present to get the LID
            lid = lidvid.split("::")[0] if "::" in lidvid else lidvid

            # Extract product_id: last component after the last colon
            product_id = lid.rsplit(":", 1)[-1]

            # Skip non-product entries (collection-level references)
            if not product_id.startswith("ss__"):
                continue

            # Parse as PDS filename (product_id IS filename without extension)
            try:
                pid = PDSProductId.from_filename(f"{product_id}.csv")
                products.append(pid)
            except (ValueError, Exception) as e:
                logger.debug(
                    "Could not parse product ID '%s': %s", product_id, e
                )

        return products

    @staticmethod
    def _select_highest_versions(
        products: List[PDSProductId],
    ) -> List[PDSProductId]:
        """Select highest version for each unique product.

        Groups products by their base identity (sol, sclk, obs_id,
        product_type, processing variant) and keeps only the one with
        the highest version number.

        Args:
            products: List of parsed product IDs, possibly with
                multiple versions of the same product.

        Returns:
            Filtered list retaining only the highest version of each
            unique product.
        """
        groups: Dict[str, List[PDSProductId]] = defaultdict(list)
        for pid in products:
            # Key captures everything that identifies a product except version
            key = (
                f"{pid.sol}_{pid.sclk}_{pid.obs_id}_"
                f"{pid.product_type}_{pid.middle}"
            )
            groups[key].append(pid)

        selected: List[PDSProductId] = []
        for key, versions in groups.items():
            best = max(versions, key=lambda p: p.version)
            if len(versions) > 1:
                logger.info(
                    "Version selection: %s has %d versions, selected v%02d",
                    key,
                    len(versions),
                    best.version,
                )
            selected.append(best)

        return selected

    def _download_file(self, url: str, dest: Path) -> None:
        """Download a single file with atomic write.

        Downloads to a temporary file first, then renames to the final
        destination to prevent partial files on interruption.

        Args:
            url: Remote URL to download.
            dest: Local file path to write to.

        Raises:
            PDSDownloadError: If download or write fails.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        response = self._request_with_retry(url)

        # Write atomically via temp file to avoid partial downloads
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            tmp.write_bytes(response.content)
            tmp.rename(dest)
            logger.debug(
                "Downloaded %s (%d bytes)", dest.name, len(response.content)
            )
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise PDSDownloadError(f"Failed to write {dest}: {e}") from e

    def _request_with_retry(self, url: str) -> "httpx.Response":
        """Make HTTP GET request with exponential backoff retry.

        Retries on transient errors: server errors (5xx), timeouts, and
        connection errors. Non-retryable errors (4xx except 429) raise
        immediately.

        Args:
            url: URL to request.

        Returns:
            Successful HTTP response (status 200).

        Raises:
            PDSDownloadError: If all retry attempts fail or a
                non-retryable error occurs.
        """
        httpx = _require_httpx()
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.get(url)

                if response.status_code == 200:
                    return response

                if response.status_code == 404:
                    raise PDSDownloadError(f"Not found (404): {url}")

                # Retry on server errors (5xx) and rate limiting (429)
                if response.status_code >= 500 or response.status_code == 429:
                    last_error = PDSDownloadError(
                        f"Server error {response.status_code}: {url}"
                    )
                    if attempt < self.max_retries:
                        delay = self.backoff_factor ** attempt
                        logger.warning(
                            "HTTP %d for %s, retry %d/%d in %.1fs",
                            response.status_code,
                            url,
                            attempt + 1,
                            self.max_retries,
                            delay,
                        )
                        time.sleep(delay)
                        continue

                # Non-retryable client error (4xx)
                raise PDSDownloadError(
                    f"HTTP {response.status_code}: {url}"
                )

            except httpx.TimeoutException as e:
                last_error = PDSDownloadError(
                    f"Timeout downloading {url}: {e}"
                )
                if attempt < self.max_retries:
                    delay = self.backoff_factor ** attempt
                    logger.warning(
                        "Timeout for %s, retry %d/%d in %.1fs",
                        url,
                        attempt + 1,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)

            except httpx.ConnectError as e:
                last_error = PDSDownloadError(
                    f"Connection error for {url}: {e}"
                )
                if attempt < self.max_retries:
                    delay = self.backoff_factor ** attempt
                    logger.warning(
                        "Connection error for %s, retry %d/%d in %.1fs",
                        url,
                        attempt + 1,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)

        raise last_error or PDSDownloadError(f"Failed to download {url}")
