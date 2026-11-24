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


class TestPdsCacheDirFromConfig:
    """Without a directory flag, pds-download and pds-ingest use the
    configured ``pds.cache_dir`` (env ``SHERLOC_PDS_CACHE_DIR``), the same
    directory the image lookups read."""

    @pytest.fixture(autouse=True)
    def _fresh_config(self):
        from sherloc_pipeline.config import reset_config

        reset_config()
        yield
        reset_config()

    def test_default_matches_shipped_config(self, monkeypatch):
        from sherloc_pipeline.config import PDSConfig, get_config
        from sherloc_pipeline.core.pds_constants import PDS_DEFAULT_CACHE_DIR

        monkeypatch.delenv("SHERLOC_PDS_CACHE_DIR", raising=False)
        assert Path(get_config().pds.cache_dir) == PDS_DEFAULT_CACHE_DIR
        assert Path(PDSConfig().cache_dir) == PDS_DEFAULT_CACHE_DIR

    def test_download_writes_to_configured_cache(
        self, runner, tmp_path, httpx_mock, monkeypatch
    ):
        cache = tmp_path / "configured"
        monkeypatch.setenv("SHERLOC_PDS_CACHE_DIR", str(cache))
        _add_pds_responses(httpx_mock, 921, SOL_921_PRODUCT_BASE)

        result = runner.invoke(app, ["pds-download", "--sol", "921"])
        assert result.exit_code == 0, result.output

        sol_dir = cache / "sol_0921" / "data_processed"
        assert (sol_dir / f"{SOL_921_PRODUCT_BASE}01.csv").exists()

    def test_ingest_reads_configured_cache(self, runner, tmp_path, monkeypatch):
        cache = tmp_path / "configured-missing"
        monkeypatch.setenv("SHERLOC_PDS_CACHE_DIR", str(cache))

        result = runner.invoke(app, ["pds-ingest", "--sol", "921"])
        assert result.exit_code != 0
        assert "configured-missing" in result.output.replace("\n", "")

    @pytest.mark.parametrize("command, flag", [
        ("pds-download", "--output-dir"),
        ("pds-ingest", "--pds-dir"),
    ])
    def test_help_names_the_configured_default(self, runner, command, flag):
        result = runner.invoke(app, [command, "--help"], env={"COLUMNS": "200"})
        assert result.exit_code == 0, result.output
        text = " ".join(result.output.split())
        assert flag in text
        assert "default: config pds.cache_dir, ./pds" in text
