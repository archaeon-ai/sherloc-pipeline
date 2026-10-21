"""Background subtraction on averages must not be skipped silently (issue #124).

Runs run_scan on the committed sol_0921 fixture workspace from a working
directory with no data/background/. Without SHERLOC_BACKGROUND_DIR the
shipped default_file stays relative to the working directory, which is where
it used to fail to resolve (tests/conftest.py points the variable at the
fixtures, so these tests set the relative form explicitly).
"""

import hashlib
import importlib.resources
import re
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
    assert record["sha256"] == hashlib.sha256(PACKAGED.read_bytes()).hexdigest()
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
        assert list((tmp_path / "results").rglob("test_plots/*_R1_background.png"))
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
    assert len(calls) == 2  # The first average was staged before the failure.
    assert not list((tmp_path / "results").rglob("*_bkgsub*"))
    assert not list((tmp_path / "results").rglob(".staging-*"))
    assert not list((tmp_path / "results").rglob("*_R1_background.png"))


def _outputs(results):
    return {p: p.read_bytes() for p in sorted(results.rglob("*_bkgsub*"))}


def test_a_missing_average_stops_the_run_even_without_a_background(tmp_path, fixtures_path, clean_cwd, background_cfg, monkeypatch):
    real = DataIngestion.generate_statistical_analysis

    def without_trimmed_mean(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        for path in (tmp_path / "results").rglob("*_R1_raw-n_*p_trim_mean.csv"):
            path.unlink()
        return out

    monkeypatch.setattr(DataIngestion, "generate_statistical_analysis", without_trimmed_mean)
    monkeypatch.setitem(background_cfg, "default_file", "./data/background/synthetic-missing.csv")
    with pytest.raises(PreprocessingError, match="needs the R1 averages"):
        run(tmp_path, fixtures_path, allow_missing_background=True)


def test_a_comparison_plot_failure_stops_the_run(tmp_path, fixtures_path, clean_cwd, background_cfg, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic comparison failure")

    monkeypatch.setattr(preprocessing_plots, "plot_average_bkgsub_comparison", fail)
    monkeypatch.setitem(background_cfg, "default_file", f"./data/background/{DEFAULT_NAME}")
    with pytest.raises(PreprocessingError, match="synthetic comparison failure"):
        run(tmp_path, fixtures_path)
    assert not list((tmp_path / "results").rglob("*_bkgsub*"))
    assert not list((tmp_path / "results").rglob(".staging-*"))


def test_a_failed_rerun_leaves_the_previous_outputs_unchanged(tmp_path, fixtures_path, clean_cwd, background_cfg, monkeypatch):
    monkeypatch.setitem(background_cfg, "default_file", f"./data/background/{DEFAULT_NAME}")
    run(tmp_path, fixtures_path)
    before = _outputs(tmp_path / "results")
    assert before

    real = preprocessing_plots.plot_average_bkgsub_baseline
    calls = []

    def fail_on_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic plot failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(preprocessing_plots, "plot_average_bkgsub_baseline", fail_on_second)
    with pytest.raises(PreprocessingError, match="synthetic plot failure"):
        run(tmp_path, fixtures_path)
    assert _outputs(tmp_path / "results") == before
    assert not list((tmp_path / "results").rglob(".staging-*"))


def test_an_opt_out_rerun_removes_the_previous_bkgsub_outputs(tmp_path, fixtures_path, clean_cwd, background_cfg, monkeypatch):
    monkeypatch.setitem(background_cfg, "default_file", f"./data/background/{DEFAULT_NAME}")
    run(tmp_path, fixtures_path)
    before = _outputs(tmp_path / "results")
    assert before
    record_plot = list((tmp_path / "results").rglob("test_plots/*_R1_background.png"))
    assert len(record_plot) == 1

    monkeypatch.setitem(background_cfg, "default_file", "./data/background/synthetic-missing.csv")
    result = run(tmp_path, fixtures_path, allow_missing_background=True)
    record = result.metadata["background_subtraction"]
    assert record["applied"] is False
    background_plot = next((tmp_path / "results").rglob("test_plots/*_R1_background.png"), None)
    assert background_plot is None
    assert sorted(record["removed_stale"]) == sorted(
        [str(p) for p in before] + [str(p) for p in record_plot]
    )
    assert not list((tmp_path / "results").rglob("*_bkgsub*"))
    assert not any("_bkgsub" in str(a) for a in result.artifacts)


def test_a_failed_move_into_place_restores_the_previous_outputs(tmp_path, fixtures_path, clean_cwd, background_cfg, monkeypatch):
    monkeypatch.setitem(background_cfg, "default_file", f"./data/background/{DEFAULT_NAME}")
    run(tmp_path, fixtures_path)
    before = _outputs(tmp_path / "results")
    assert before

    real_replace = Path.replace
    moves = []

    def fail_on_third_move(self, target):
        if self.name.startswith(".staging-"):
            moves.append(self)
            if len(moves) == 3:
                raise OSError("synthetic replace failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_on_third_move)
    with pytest.raises(PreprocessingError, match="synthetic replace failure"):
        run(tmp_path, fixtures_path)
    assert _outputs(tmp_path / "results") == before
    assert not list((tmp_path / "results").rglob(".staging-*"))
    assert not list((tmp_path / "results").rglob(".backup-*"))


def test_an_opt_out_rerun_removes_outputs_of_every_trim_and_the_background_plot(tmp_path, fixtures_path, clean_cwd, background_cfg, monkeypatch):
    monkeypatch.setitem(background_cfg, "default_file", f"./data/background/{DEFAULT_NAME}")
    run(tmp_path, fixtures_path)
    results = tmp_path / "results"
    background_plot = next(results.rglob("test_plots/*_R1_background.png"))
    # Outputs of an earlier run with another trim percentage.
    current = next(results.rglob("*_R1_raw-n_*p_trim_mean_bkgsub.csv"))
    other_trim = current.with_name(re.sub(r"\d+(\.\d+)?p_trim_mean", "8p_trim_mean", current.name))
    assert other_trim != current
    shutil.copy(current, other_trim)

    monkeypatch.setitem(background_cfg, "default_file", "./data/background/synthetic-missing.csv")
    result = run(tmp_path, fixtures_path, allow_missing_background=True)
    removed = result.metadata["background_subtraction"]["removed_stale"]
    assert str(other_trim) in removed and str(background_plot) in removed
    assert not other_trim.exists() and not background_plot.exists()


def test_stale_output_cleanup_treats_names_literally(tmp_path):
    class Ingestion:
        def get_results_path(self, target, sol, scan):
            return tmp_path

    (tmp_path / "test_plots").mkdir()
    prefix = "0921_Calibration[1]_detail_1_R1_"
    stale = [
        tmp_path / f"{prefix}raw-n_mean_bkgsub.csv",
        tmp_path / "test_plots" / f"{prefix}raw-n_8p_trim_mean_bkgsub_comparison.png",
        tmp_path / "test_plots" / f"{prefix}background.png",
    ]
    other_scan = tmp_path / "0921_Calibration1_detail_1_R1_raw-n_mean_bkgsub.csv"
    for path in stale + [other_scan]:
        path.write_text("synthetic")
    found = PreprocessingService._bkgsub_output_paths(Ingestion(), "Calibration[1]", "0921", "detail_1")
    assert found == sorted(stale)


def test_a_rerun_at_another_trim_uses_this_runs_trimmed_mean(tmp_path, fixtures_path, clean_cwd, background_cfg, monkeypatch):
    monkeypatch.setitem(background_cfg, "default_file", f"./data/background/{DEFAULT_NAME}")
    results = tmp_path / "results"

    def trim_tokens(pattern):
        return {re.search(r"_(\d+(?:\.\d+)?p_trim_mean)[_.]", p.name).group(1) for p in results.rglob(pattern)}

    # An earlier run at a 10% trim leaves its averages in the scan directory;
    # its token sorts before the default 2% one.
    monkeypatch.setitem(get_config().preprocessing, "trim_mean_baseline_pct", 0.10)
    run(tmp_path, fixtures_path)
    (earlier,) = trim_tokens("*_R1_raw-n_*p_trim_mean_bkgsub.csv")

    monkeypatch.setitem(get_config().preprocessing, "trim_mean_baseline_pct", 0.02)
    result = run(tmp_path, fixtures_path)
    current = trim_tokens("*_R1_raw-n_*p_trim_mean.csv") - {earlier}
    assert len(current) == 1 and sorted([earlier, *current])[0] == earlier
    (current,) = current
    assert result.metadata["background_subtraction"]["applied"] is True
    assert trim_tokens("*_R1_raw-n_*p_trim_mean_bkgsub.csv") == {earlier, current}
