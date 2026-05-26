"""Unit tests for models/ingestion.py:extract_sol_from_path().

Regression guards capturing current behavior before structural refactor.
Also verifies that the services/image_ingestion.py version exists (will be consolidated in R-003).
"""

from pathlib import Path

import pytest

from sherloc_pipeline.models.ingestion import extract_sol_from_path


# ---------------------------------------------------------------------------
# extract_sol_from_path (models.ingestion -- canonical version)
# ---------------------------------------------------------------------------


class TestExtractSolFromPath:

    def test_loupe_style_path(self):
        """Standard Loupe path: ./sol_0921/detail_1/..."""
        path = Path("./sol_0921/detail_1/SrlcSpec_Loupe_working")
        assert extract_sol_from_path(path) == 921

    def test_sol_with_suffix(self):
        """Path with sol directory having a suffix: sol_1242_a."""
        path = Path("./sol_1242_a/detail_1")
        assert extract_sol_from_path(path) == 1242

    def test_four_digit_sol(self):
        """Four-digit sol number."""
        path = Path("/data/loupe/sol_1500/survey_1")
        assert extract_sol_from_path(path) == 1500

    def test_zero_padded_sol(self):
        """Zero-padded sol: sol_0001 → 1."""
        path = Path("./sol_0001/detail_1")
        assert extract_sol_from_path(path) == 1

    def test_sol_0(self):
        """sol_0000 → 0."""
        path = Path("./sol_0000/detail_1")
        assert extract_sol_from_path(path) == 0

    def test_no_sol_directory(self):
        """Path without any sol directory → None."""
        path = Path("./calibration/some_file.csv")
        assert extract_sol_from_path(path) is None

    def test_multiple_sol_dirs_returns_first(self):
        """If path has multiple sol_XXXX components, returns the first match."""
        path = Path("/data/sol_0100/backup/sol_0200/detail_1")
        result = extract_sol_from_path(path)
        assert result == 100

    def test_sol_in_filename_not_matched(self):
        """The function iterates over path.parts, so sol_ must be a directory name."""
        # A filename like sol_0921.csv is a valid path part
        path = Path("./sol_0921.csv")
        # sol_0921.csv is the filename part -- re.match("sol_(\d+)", "sol_0921.csv") WILL match
        # because re.match only checks the start of the string
        result = extract_sol_from_path(path)
        assert result == 921

    def test_deep_nested_path(self):
        """Sol extracted from deeply nested path."""
        path = Path("/mnt/data/mars/sherloc/sol_0500/detail_2/SrlcSpec_working/spectra")
        assert extract_sol_from_path(path) == 500

    def test_relative_path(self):
        """Relative path with sol directory."""
        path = Path("sol_0300/detail_1")
        assert extract_sol_from_path(path) == 300


# ---------------------------------------------------------------------------
# Verify services/image_ingestion version exists
# ---------------------------------------------------------------------------


class TestServicesVersionExists:

    def test_services_extract_sol_importable(self):
        """The services.image_ingestion version should be importable (will be consolidated in R-003)."""
        from sherloc_pipeline.services.image_ingestion import extract_sol_from_path as services_extract_sol
        assert callable(services_extract_sol)

    def test_services_version_same_result_standard_path(self):
        """Both versions should return the same result for standard paths."""
        from sherloc_pipeline.services.image_ingestion import extract_sol_from_path as services_extract_sol

        test_paths = [
            Path("./sol_0921/detail_1"),
            Path("./sol_1500/survey_1"),
            Path("/data/calibration/no_sol_here"),
        ]
        for path in test_paths:
            models_result = extract_sol_from_path(path)
            services_result = services_extract_sol(path)
            assert models_result == services_result, (
                f"Mismatch for {path}: models={models_result}, services={services_result}"
            )
