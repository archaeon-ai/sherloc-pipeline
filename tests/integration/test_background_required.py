"""Background subtraction on averages must not be skipped silently (issue #124).

Runs run_scan on the committed sol_0921 fixture workspace from a working
directory with no data/background/. Without SHERLOC_BACKGROUND_DIR the
shipped default_file stays relative to the working directory, which is where
it used to fail to resolve (tests/conftest.py points the variable at the
fixtures, so these tests set the relative form explicitly).
"""

from pathlib import Path

import pytest

from sherloc_pipeline.config import get_config
from sherloc_pipeline.services.errors import PreprocessingError
from sherloc_pipeline.services.preprocessing import PreprocessingService, resolve_default_background

WORKSPACE_SOL, WORKSPACE_SCAN = "0921", "detail_1"
TARGET = "Background_Required"
DEFAULT_NAME = "Arm_Stowed_post-anomaly_900ppp_trimmed_mean_1266.csv"
PACKAGED = Path(__file__).resolve().parents[2] / "src/sherloc_pipeline/data/background" / DEFAULT_NAME


@pytest.fixture
def clean_cwd(tmp_path, monkeypatch):
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    assert not (work / "data/background").exists()
    return work


@pytest.fixture
def background_cfg(monkeypatch):
    bs = get_config().preprocessing.setdefault("background_subtraction", {})
    monkeypatch.setitem(bs, "allow_missing", False)
    return bs


def run(tmp_path, fixtures_path, **kwargs):
    return PreprocessingService().run_scan(
        sol=WORKSPACE_SOL, target=TARGET, scan=WORKSPACE_SCAN,
        data_dir=fixtures_path / "loupe", results_dir=tmp_path / "results",
        generate_plots=False, despike_method="none", **kwargs,
    )


def test_the_shipped_default_resolves_to_the_packaged_file_from_any_directory(clean_cwd):
    resolved = resolve_default_background(Path("./data/background") / DEFAULT_NAME)
    assert resolved == PACKAGED
    # An explicit absolute path is never replaced by the packaged copy.
    missing = clean_cwd / "elsewhere" / DEFAULT_NAME
    assert resolve_default_background(missing) == missing


def test_averages_are_background_subtracted_with_the_default(tmp_path, fixtures_path, clean_cwd, background_cfg, monkeypatch):
    # The default as config.yaml ships it when SHERLOC_BACKGROUND_DIR is unset.
    monkeypatch.setitem(background_cfg, "default_file", f"./data/background/{DEFAULT_NAME}")
    result = run(tmp_path, fixtures_path)

    record = result.metadata["background_subtraction"]
    assert record["applied"] is True
    assert Path(record["file"]) == PACKAGED
    assert record["n_averages"] >= 1
    assert not any("Background subtraction" in w for w in result.warnings)
    assert list((tmp_path / "results").rglob("*_bkgsub.csv"))
    # The background plot goes with the scan's plots, never next to the packaged file.
    assert not PACKAGED.with_suffix(".png").exists()


def test_a_missing_background_stops_the_run(tmp_path, fixtures_path, clean_cwd, background_cfg, monkeypatch):
    monkeypatch.setitem(background_cfg, "default_file", "./data/background/synthetic-missing.csv")
    with pytest.raises(PreprocessingError, match="needs a background spectrum"):
        run(tmp_path, fixtures_path)


def test_no_default_and_no_file_stops_the_run(tmp_path, fixtures_path, clean_cwd, background_cfg, monkeypatch):
    monkeypatch.setitem(background_cfg, "default_file", None)
    with pytest.raises(PreprocessingError, match="default_file is not set"):
        run(tmp_path, fixtures_path)


def test_an_unreadable_background_stops_the_run(tmp_path, fixtures_path, clean_cwd, background_cfg):
    bad = clean_cwd / "synthetic-unreadable.csv"
    bad.write_text("not,a\nbackground\n")
    with pytest.raises(PreprocessingError, match="Background subtraction failed"):
        run(tmp_path, fixtures_path, background_file=bad)


@pytest.mark.parametrize("via", ["config", "argument"])
def test_the_opt_out_runs_without_background_and_records_it(tmp_path, fixtures_path, clean_cwd, background_cfg, monkeypatch, via):
    monkeypatch.setitem(background_cfg, "default_file", "./data/background/synthetic-missing.csv")
    if via == "config":
        monkeypatch.setitem(background_cfg, "allow_missing", True)
        result = run(tmp_path, fixtures_path)
    else:
        result = run(tmp_path, fixtures_path, allow_missing_background=True)

    record = result.metadata["background_subtraction"]
    assert record["applied"] is False
    assert record["allow_missing"] is True
    assert "background file not found" in record["reason"]
    assert any(w.startswith("Background subtraction not applied (allow_missing)") for w in result.warnings)
    assert not list((tmp_path / "results").rglob("*_bkgsub.csv"))
