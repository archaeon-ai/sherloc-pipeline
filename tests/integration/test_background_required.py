"""Background subtraction on averages must not be skipped silently (issue #124).

Runs run_scan on the committed sol_0921 fixture workspace from a working
directory with no data/background/. Without SHERLOC_BACKGROUND_DIR the
shipped default_file stays relative to the working directory, which is where
it used to fail to resolve (tests/conftest.py points the variable at the
fixtures, so these tests set the relative form explicitly).
"""

import importlib.resources
import shutil
import tomllib
from fnmatch import fnmatch
from pathlib import Path

import pytest

from sherloc_pipeline.config import get_config
from sherloc_pipeline.core.data_ingestion import DataIngestion
from sherloc_pipeline.services.errors import PreprocessingError
from sherloc_pipeline.services.preprocessing import PreprocessingService, resolve_default_background
from sherloc_pipeline.visualization import preprocessing_plots

WORKSPACE_SOL, WORKSPACE_SCAN = "0921", "detail_1"
TARGET = "Background_Required"
DEFAULT_NAME = "Arm_Stowed_post-anomaly_900ppp_trimmed_mean_1266.csv"
REPO = Path(__file__).resolve().parents[2]
PACKAGED = REPO / "src/sherloc_pipeline/data/background" / DEFAULT_NAME


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
    assert record["n_averages"] == 2  # mean and trimmed mean
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


def _unreadable(directory):
    bad = directory / "synthetic-unreadable.csv"
    bad.write_text("not,a\nbackground\n")
    return bad


def test_an_unreadable_background_stops_the_run(tmp_path, fixtures_path, clean_cwd, background_cfg):
    with pytest.raises(PreprocessingError, match="background file unreadable"):
        run(tmp_path, fixtures_path, background_file=_unreadable(clean_cwd))


def test_the_opt_out_covers_an_unreadable_background(tmp_path, fixtures_path, clean_cwd, background_cfg):
    result = run(tmp_path, fixtures_path, background_file=_unreadable(clean_cwd), allow_missing_background=True)
    record = result.metadata["background_subtraction"]
    assert record["applied"] is False
    assert "background file unreadable" in record["reason"]
    assert not list((tmp_path / "results").rglob("*_bkgsub*"))


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


def test_the_wheel_ships_the_default_background():
    package_data = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]
    shipped = f"data/background/{DEFAULT_NAME}"
    assert any(fnmatch(shipped, pattern) for pattern in package_data["sherloc_pipeline"])


def test_a_read_only_installed_default_is_used_and_left_untouched(tmp_path, fixtures_path, clean_cwd, background_cfg, monkeypatch):
    # Stand-in for an installed package: importlib.resources resolves the
    # data package to a read-only directory outside the source tree.
    installed = tmp_path / "site-packages/sherloc_pipeline/data"
    (installed / "background").mkdir(parents=True)
    shutil.copy(PACKAGED, installed / "background" / DEFAULT_NAME)
    monkeypatch.setattr(importlib.resources, "files", lambda package: installed)
    monkeypatch.setitem(background_cfg, "default_file", f"./data/background/{DEFAULT_NAME}")
    for d in (installed / "background", installed):
        d.chmod(0o555)
    try:
        result = run(tmp_path, fixtures_path)
        record = result.metadata["background_subtraction"]
        assert record["applied"] is True
        assert Path(record["file"]) == installed / "background" / DEFAULT_NAME
        assert sorted(p.name for p in (installed / "background").iterdir()) == [DEFAULT_NAME]
        assert list((tmp_path / "results").rglob(f"test_plots/{Path(DEFAULT_NAME).stem}.png"))
    finally:
        for d in (installed, installed / "background"):
            d.chmod(0o755)


@pytest.mark.parametrize("allow_missing", [False, True])
def test_a_missing_average_stops_the_run(tmp_path, fixtures_path, clean_cwd, background_cfg, monkeypatch, allow_missing):
    real = DataIngestion.generate_statistical_analysis

    def without_trimmed_mean(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        for path in (tmp_path / "results").rglob("*_R1_raw-n_*p_trim_mean.csv"):
            path.unlink()
        return out

    monkeypatch.setattr(DataIngestion, "generate_statistical_analysis", without_trimmed_mean)
    monkeypatch.setitem(background_cfg, "default_file", f"./data/background/{DEFAULT_NAME}")
    # The opt-out covers a missing background, not missing averages.
    with pytest.raises(PreprocessingError, match="needs the R1 averages"):
        run(tmp_path, fixtures_path, allow_missing_background=allow_missing)
    assert not list((tmp_path / "results").rglob("*_bkgsub*"))


@pytest.mark.parametrize("allow_missing", [False, True])
def test_a_failure_after_the_first_average_leaves_no_bkgsub_output(tmp_path, fixtures_path, clean_cwd, background_cfg, monkeypatch, allow_missing):
    real = preprocessing_plots.plot_average_bkgsub_baseline
    calls = []

    def fail_on_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic plot failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(preprocessing_plots, "plot_average_bkgsub_baseline", fail_on_second)
    monkeypatch.setitem(background_cfg, "default_file", f"./data/background/{DEFAULT_NAME}")
    with pytest.raises(PreprocessingError, match="synthetic plot failure"):
        run(tmp_path, fixtures_path, allow_missing_background=allow_missing)
    assert len(calls) == 2  # The first average was fully written before the failure.
    assert not list((tmp_path / "results").rglob("*_bkgsub*"))
    assert not list((tmp_path / "results").rglob(f"{Path(DEFAULT_NAME).stem}.png"))
