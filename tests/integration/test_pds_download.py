"""Integration tests for PDS download client (Sol 921).

Verifies PDSDownloader against existing cached Sol 921 data:
- File counts and sizes
- Filename parsing for all products
- Cache-skip behavior (already-downloaded files are skipped)
- Force re-download to a fresh directory
- Version selection (highest version kept)
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sherloc_pipeline.core.pds_client import PDSDownloader, SolDownloadResult
from sherloc_pipeline.models.pds import PDSProductId

from tests.integration.conftest import (
    PDS_CACHE_DIR,
    SOL_921_DIR,
    requires_sol921_data,
)

pytestmark = requires_sol921_data

# Expected counts for Sol 921 data_processed collection
EXPECTED_CSV_COUNT = 52
EXPECTED_XML_COUNT = 52
EXPECTED_TOTAL_FILES = EXPECTED_CSV_COUNT + EXPECTED_XML_COUNT

# Unique SCLK groups in Sol 921 (5 observations + 1 calibration)
EXPECTED_SCLK_GROUPS = {
    "0748731011",  # calibration (obs 645)
    "0748731413",  # detail (obs 045)
    "0748732975",  # detail (obs 435)
    "0748735042",  # detail (obs 800)
    "0748735903",  # detail (obs 665)
    "0748736149",  # survey (obs 380)
}


def _build_mock_inventory(filenames: list[str]) -> str:
    """Build a mock PDS4 collection inventory CSV from filenames."""
    lines = []
    for fname in filenames:
        base = fname.rsplit(".", 1)[0]  # strip .csv/.xml
        # Only include CSV entries (XML is derived from the same product ID)
        if fname.endswith(".csv"):
            lid = f"urn:nasa:pds:mars2020_sherloc:data_processed:{base}"
            # Extract version from filename (last 2 digits before extension)
            version = int(base[-2:])
            lines.append(f"P,{lid}::{version}.0")
    return "\n".join(lines)


@pytest.fixture
def sol_921_csv_files() -> list[Path]:
    """Return sorted list of CSV files in Sol 921 data_processed."""
    assert SOL_921_DIR.exists(), f"Sol 921 data not found at {SOL_921_DIR}"
    return sorted(SOL_921_DIR.glob("*.csv"))


@pytest.fixture
def sol_921_xml_files() -> list[Path]:
    """Return sorted list of XML files in Sol 921 data_processed."""
    assert SOL_921_DIR.exists(), f"Sol 921 data not found at {SOL_921_DIR}"
    return sorted(SOL_921_DIR.glob("*.xml"))


@pytest.fixture
def mock_inventory_text(sol_921_csv_files) -> str:
    """Build mock inventory text from actual Sol 921 filenames."""
    return _build_mock_inventory([f.name for f in sol_921_csv_files])


class TestSol921FileIntegrity:
    """Verify existing Sol 921 cached data is intact and parseable."""

    def test_csv_count(self, sol_921_csv_files):
        """Sol 921 should have exactly 52 CSV files."""
        assert len(sol_921_csv_files) == EXPECTED_CSV_COUNT

    def test_xml_count(self, sol_921_xml_files):
        """Sol 921 should have exactly 52 XML files."""
        assert len(sol_921_xml_files) == EXPECTED_XML_COUNT

    def test_csv_xml_pairs_match(self, sol_921_csv_files, sol_921_xml_files):
        """Every CSV should have a matching XML label and vice versa."""
        csv_bases = {f.stem for f in sol_921_csv_files}
        xml_bases = {f.stem for f in sol_921_xml_files}
        assert csv_bases == xml_bases, (
            f"CSV/XML mismatch: "
            f"CSV only: {csv_bases - xml_bases}, "
            f"XML only: {xml_bases - csv_bases}"
        )

    def test_all_files_nonzero(self, sol_921_csv_files, sol_921_xml_files):
        """No files should be empty (would indicate download corruption)."""
        for f in sol_921_csv_files + sol_921_xml_files:
            assert f.stat().st_size > 0, f"Empty file: {f.name}"

    def test_all_filenames_parse_as_product_ids(self, sol_921_csv_files):
        """Every CSV filename should parse as a valid PDSProductId."""
        for csv_file in sol_921_csv_files:
            pid = PDSProductId.from_filename(csv_file.name)
            assert pid.sol == 921
            assert pid.csv_filename == csv_file.name
            assert pid.xml_filename == csv_file.stem + ".xml"

    def test_sclk_groups(self, sol_921_csv_files):
        """Sol 921 should contain expected SCLK observation groups."""
        sclks = set()
        for csv_file in sol_921_csv_files:
            pid = PDSProductId.from_filename(csv_file.name)
            sclks.add(str(pid.sclk).zfill(10))
        assert sclks == EXPECTED_SCLK_GROUPS

    def test_product_type_distribution(self, sol_921_csv_files):
        """Sol 921 should have the expected product type distribution."""
        types: dict[str, int] = {}
        for csv_file in sol_921_csv_files:
            pid = PDSProductId.from_filename(csv_file.name)
            # use_enum_values=True stores as string at runtime
            ptype = str(pid.product_type)
            types[ptype] = types.get(ptype, 0) + 1

        # All 6 SCLK groups have RM1-RM6 (6 types × 6 groups = 36, minus
        # calibration group which has no RRS/RCS but has RM1-6)
        # RMO: 5 groups (not calibration which has it too actually)
        # Check that core spectral types are present
        assert "rrs" in types, "Missing RRS (Raman spectral) products"
        assert "rmo" in types, "Missing RMO (position/motor) products"
        assert "rm1" in types, "Missing RM1 products"
        assert "rli" in types, "Missing RLI products"
        assert "rls" in types, "Missing RLS products"


class TestPDSDownloaderLocalOps:
    """Test PDSDownloader operations against real local cache."""

    def test_list_local_sols_includes_921(self):
        """list_local_sols should detect Sol 921 in ./pds."""
        downloader = PDSDownloader(cache_dir=PDS_CACHE_DIR)
        sols = downloader.list_local_sols()
        assert 921 in sols

    def test_list_local_sols_returns_sorted(self):
        """list_local_sols should return sorted sol numbers."""
        downloader = PDSDownloader(cache_dir=PDS_CACHE_DIR)
        sols = downloader.list_local_sols()
        assert sols == sorted(sols)

    def test_list_local_sols_empty_dir(self, tmp_path):
        """list_local_sols on empty directory returns empty list."""
        downloader = PDSDownloader(cache_dir=tmp_path)
        assert downloader.list_local_sols() == []

    def test_list_local_sols_nonexistent_dir(self, tmp_path):
        """list_local_sols on nonexistent directory returns empty list."""
        downloader = PDSDownloader(cache_dir=tmp_path / "nonexistent")
        assert downloader.list_local_sols() == []


class TestDownloadSolCacheBehavior:
    """Test download_sol cache-skip and force-download behavior."""

    def test_download_sol_skips_all_cached(self, mock_inventory_text):
        """download_sol should skip all files when Sol 921 is already cached."""
        downloader = PDSDownloader(cache_dir=PDS_CACHE_DIR)

        # Mock HTTP: inventory fetch returns our mock, file downloads not called
        mock_response = MagicMock()
        mock_response.text = mock_inventory_text
        mock_response.status_code = 200

        with patch.object(
            downloader, "_request_with_retry", return_value=mock_response
        ):
            result = downloader.download_sol(921)

        assert result.sol == 921
        assert result.n_downloaded == 0, "Should not download any files"
        assert result.n_skipped == EXPECTED_TOTAL_FILES, (
            f"Should skip all {EXPECTED_TOTAL_FILES} files, "
            f"got {result.n_skipped}"
        )
        assert len(result.errors) == 0, f"Unexpected errors: {result.errors}"

    def test_download_sol_force_redownloads(
        self, tmp_path, sol_921_csv_files, mock_inventory_text
    ):
        """download_sol(force=True) to a fresh dir downloads all files."""
        # Set up a fresh cache with pre-existing files (copy a small subset)
        sol_dir = tmp_path / "sol_0921" / "data_processed"
        sol_dir.mkdir(parents=True)

        # Copy first 2 CSV+XML pairs as pre-existing cached files
        test_files = sol_921_csv_files[:2]
        for csv_file in test_files:
            shutil.copy2(csv_file, sol_dir / csv_file.name)
            xml_file = csv_file.with_suffix(".xml")
            shutil.copy2(xml_file, sol_dir / xml_file.name)

        downloader = PDSDownloader(cache_dir=tmp_path)

        # Track which URLs get "downloaded" via _download_file
        downloaded_urls = []
        original_content = {}

        def mock_request(url):
            """Return mock inventory or mock file content."""
            resp = MagicMock()
            resp.status_code = 200
            if "inventory" in url:
                resp.text = mock_inventory_text
            else:
                # For file downloads, return small test content
                fname = url.rsplit("/", 1)[-1]
                source = SOL_921_DIR / fname
                if source.exists():
                    resp.content = source.read_bytes()
                else:
                    resp.content = b"mock-content"
                downloaded_urls.append(url)
            return resp

        with patch.object(
            downloader, "_request_with_retry", side_effect=mock_request
        ):
            result = downloader.download_sol(921, force=True)

        # With force=True, all files should be downloaded (none skipped)
        assert result.n_skipped == 0, "force=True should not skip any files"
        assert result.n_downloaded == EXPECTED_TOTAL_FILES
        assert len(result.errors) == 0

    def test_download_sol_without_force_skips_existing(
        self, tmp_path, sol_921_csv_files, mock_inventory_text
    ):
        """download_sol without force skips existing, downloads missing."""
        sol_dir = tmp_path / "sol_0921" / "data_processed"
        sol_dir.mkdir(parents=True)

        # Copy only first CSV+XML pair
        first_csv = sol_921_csv_files[0]
        shutil.copy2(first_csv, sol_dir / first_csv.name)
        xml_file = first_csv.with_suffix(".xml")
        shutil.copy2(xml_file, sol_dir / xml_file.name)

        downloader = PDSDownloader(cache_dir=tmp_path)

        def mock_request(url):
            resp = MagicMock()
            resp.status_code = 200
            if "inventory" in url:
                resp.text = mock_inventory_text
            else:
                fname = url.rsplit("/", 1)[-1]
                source = SOL_921_DIR / fname
                resp.content = source.read_bytes() if source.exists() else b"x"
            return resp

        with patch.object(
            downloader, "_request_with_retry", side_effect=mock_request
        ):
            result = downloader.download_sol(921)

        # 2 files pre-existed (1 CSV + 1 XML), rest should be downloaded
        assert result.n_skipped == 2
        assert result.n_downloaded == EXPECTED_TOTAL_FILES - 2
        assert len(result.errors) == 0

        # Verify all files now exist
        csv_count = len(list(sol_dir.glob("*.csv")))
        xml_count = len(list(sol_dir.glob("*.xml")))
        assert csv_count == EXPECTED_CSV_COUNT
        assert xml_count == EXPECTED_XML_COUNT


class TestVersionSelection:
    """Test version selection logic with Sol 921 product IDs."""

    def test_select_highest_versions(self, sol_921_csv_files):
        """Version selection should keep highest version per product."""
        # Parse all Sol 921 products
        products = [
            PDSProductId.from_filename(f.name) for f in sol_921_csv_files
        ]

        # Add a fake lower-version duplicate of a v02 product
        v02_products = [p for p in products if p.version >= 2]
        assert len(v02_products) > 0, "Sol 921 should have some v02+ products"

        # Create a v01 duplicate of the first v02 product
        v02 = v02_products[0]
        v01_filename = v02.filename[:-6] + "01.csv"  # change version to 01
        v01 = PDSProductId.from_filename(v01_filename)

        # Add the lower version to the list
        products_with_dup = products + [v01]
        assert len(products_with_dup) == len(products) + 1

        # Version selection should eliminate the duplicate
        selected = PDSDownloader._select_highest_versions(products_with_dup)
        assert len(selected) == len(products), (
            f"Version selection should keep {len(products)} products, "
            f"got {len(selected)}"
        )

        # The selected v02 product should be present, not the v01
        selected_filenames = {p.filename for p in selected}
        assert v02.filename in selected_filenames
        assert v01.filename not in selected_filenames

    def test_all_sol_921_products_are_highest_version(self, sol_921_csv_files):
        """Cached Sol 921 files should already be highest versions."""
        products = [
            PDSProductId.from_filename(f.name) for f in sol_921_csv_files
        ]
        selected = PDSDownloader._select_highest_versions(products)
        # No duplicates in cached data, so selection should return same count
        assert len(selected) == len(products)


class TestSolDownloadResult:
    """Test SolDownloadResult dataclass."""

    def test_empty_result(self):
        result = SolDownloadResult(sol=921)
        assert result.n_downloaded == 0
        assert result.n_skipped == 0
        assert result.sol == 921

    def test_result_with_data(self, tmp_path):
        result = SolDownloadResult(
            sol=921,
            downloaded=[tmp_path / "a.csv", tmp_path / "b.csv"],
            skipped=[tmp_path / "c.csv"],
            errors=["failed: d.csv"],
        )
        assert result.n_downloaded == 2
        assert result.n_skipped == 1
        assert len(result.errors) == 1
