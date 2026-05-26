"""Unit tests for `sherloc pds-download` (§11.1).

The HTTP layer is mocked via ``pytest-httpx``; the CLI is exercised
end-to-end through ``typer.testing.CliRunner`` against a real (but
isolated) cache directory under ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app
from sherloc_pipeline.core.pds_constants import PDS_BASE_URL


# Restrict pytest-httpx to PDS hosts so unrelated httpx calls pass through.
pytestmark = pytest.mark.httpx_mock(
    should_mock=lambda request: "pds-geosciences.wustl.edu" in request.url.host
    or "pds.mcp.nasa.gov" in request.url.host,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# A minimal sol_0921 inventory: one CSV product (rrs) at version 01.
# Filename grammar (PDSProductId._PDS_FILENAME_RE):
#   ss__SSSS_CCCCCCCCCC_NNNxxx__DDDDDDDsrlcQQQQQ<middle>VV.ext
SOL_921_PRODUCT_BASE = "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj"
SOL_921_INVENTORY = (
    f"P,urn:nasa:pds:mars2020_sherloc:data_processed:{SOL_921_PRODUCT_BASE}01::1.0\n"
)


def _add_pds_responses(httpx_mock, sol: int, base_name: str) -> None:
    """Wire mocks for one inventory fetch + one CSV/XML pair download."""
    inventory_url = (
        f"{PDS_BASE_URL}/data_processed/"
        "collection_data_processed_inventory.csv"
    )
    httpx_mock.add_response(url=inventory_url, text=SOL_921_INVENTORY)

    sol_dir_5 = f"sol_{sol:05d}"
    csv_url = (
        f"{PDS_BASE_URL}/data_processed/{sol_dir_5}/{base_name}01.csv"
    )
    xml_url = (
        f"{PDS_BASE_URL}/data_processed/{sol_dir_5}/{base_name}01.xml"
    )
    httpx_mock.add_response(url=csv_url, content=b"wavelength,intensity\n100,0.5\n")
    httpx_mock.add_response(url=xml_url, content=b"<Product/>")


class TestSherlocPdsDownload:
    def test_requires_one_mode(self, runner, tmp_path):
        result = runner.invoke(
            app,
            ["pds-download", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code != 0
        assert "Specify one of" in result.output

    def test_mutually_exclusive_modes(self, runner, tmp_path):
        result = runner.invoke(
            app,
            [
                "pds-download",
                "--sol", "921",
                "--auto",
                "--output-dir", str(tmp_path),
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_inverted_range_rejected(self, runner, tmp_path):
        result = runner.invoke(
            app,
            [
                "pds-download",
                "--sol-range", "1000", "100",
                "--output-dir", str(tmp_path),
            ],
        )
        assert result.exit_code != 0
        assert "Invalid range" in result.output

    def test_single_sol_download(self, runner, tmp_path, httpx_mock):
        _add_pds_responses(httpx_mock, 921, SOL_921_PRODUCT_BASE)

        result = runner.invoke(
            app,
            [
                "pds-download",
                "--sol", "921",
                "--output-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output

        sol_dir = tmp_path / "sol_0921" / "data_processed"
        assert (sol_dir / f"{SOL_921_PRODUCT_BASE}01.csv").exists()
        assert (sol_dir / f"{SOL_921_PRODUCT_BASE}01.xml").exists()

    def test_single_sol_download_json_report(
        self, runner, tmp_path, httpx_mock
    ):
        import json

        _add_pds_responses(httpx_mock, 921, SOL_921_PRODUCT_BASE)

        result = runner.invoke(
            app,
            [
                "--json",
                "pds-download",
                "--sol", "921",
                "--output-dir", str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output

        # Last line of stdout is the JSON payload.
        payload = json.loads(result.output.strip().splitlines()[-1])
        assert payload["command"] == "pds-download"
        assert payload["result"]["sols"] == [921]
        assert payload["result"]["downloaded"] == 2
        assert payload["result"]["errors"] == 0
