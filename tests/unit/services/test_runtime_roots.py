"""#173: RuntimeContext.bootstrap() and resolve_sol_paths() default to the
configured paths.data_root / paths.results_root (env-overridable)."""
from pathlib import Path

import pytest

from sherloc_pipeline.config import load_config
from sherloc_pipeline.services.runtime import RuntimeContext


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("SHERLOC_DATA_DIR", raising=False)
    monkeypatch.delenv("SHERLOC_RESULTS_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-empty"))
    return monkeypatch


def test_bootstrap_uses_configured_roots(clean_env):
    cfg = load_config()
    ctx = RuntimeContext.bootstrap(config=cfg)
    assert ctx.data_root == Path(cfg.paths["data_root"]).resolve()
    assert ctx.results_root == Path(cfg.paths["results_root"]).resolve()
    assert ctx.data_root.name == "data" and ctx.results_root.name == "results"


def test_environment_variables_override_configured_roots(clean_env, tmp_path):
    clean_env.setenv("SHERLOC_DATA_DIR", str(tmp_path / "d"))
    clean_env.setenv("SHERLOC_RESULTS_DIR", str(tmp_path / "r"))
    ctx = RuntimeContext.bootstrap(config=load_config())
    assert ctx.data_root == (tmp_path / "d").resolve()
    assert ctx.results_root == (tmp_path / "r").resolve()


def test_explicit_arguments_still_win(clean_env, tmp_path):
    clean_env.setenv("SHERLOC_DATA_DIR", str(tmp_path / "d"))
    ctx = RuntimeContext.bootstrap(tmp_path / "x", tmp_path / "y", config=load_config())
    assert ctx.data_root == (tmp_path / "x").resolve()
    assert ctx.results_root == (tmp_path / "y").resolve()


def test_legacy_keys_still_read(clean_env, tmp_path):
    cfg = load_config()
    cfg.paths = {"data_dir": str(tmp_path / "old-d"), "results_dir": str(tmp_path / "old-r")}
    ctx = RuntimeContext.bootstrap(config=cfg)
    assert ctx.data_root == (tmp_path / "old-d").resolve()
    assert ctx.results_root == (tmp_path / "old-r").resolve()


def test_resolve_scan_context_reads_configured_data_root(clean_env, tmp_path, monkeypatch):
    from sherloc_pipeline.services import paths as paths_mod
    clean_env.setenv("SHERLOC_DATA_DIR", str(tmp_path / "configured-root"))
    cfg = load_config()
    monkeypatch.setattr(paths_mod, "get_runtime_config", lambda: cfg)
    with pytest.raises(FileNotFoundError, match="configured-root"):
        paths_mod.resolve_scan_context("0001", "T", "detail_1")
