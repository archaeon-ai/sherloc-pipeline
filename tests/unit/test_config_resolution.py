"""Tests for XDG-compliant path resolution in config.py (§3.2.1)."""

import os
from pathlib import Path

import pytest

from sherloc_pipeline.config import resolve_path, resolve_paths, reset_config


# ---------------------------------------------------------------------------
# resolve_path()
# ---------------------------------------------------------------------------


class TestResolvePath:
    """Unit tests for the resolve_path() helper."""

    def test_env_var_wins(self, monkeypatch, tmp_path):
        """Environment variable takes priority over config value and XDG."""
        env_target = str(tmp_path / "env_db.sqlite")
        monkeypatch.setenv("SHERLOC_DB_PATH", env_target)
        result = resolve_path("./sherloc.db", "SHERLOC_DB_PATH", "phase.db")
        assert result == Path(env_target)

    def test_env_var_wins_over_existing_xdg(self, monkeypatch, tmp_path):
        """Env var wins even when the XDG path already exists on disk."""
        # Create a real XDG path
        xdg_home = tmp_path / "xdg"
        xdg_path = xdg_home / "sherloc" / "phase.db"
        xdg_path.mkdir(parents=True)
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg_home))

        env_target = str(tmp_path / "env_override.db")
        monkeypatch.setenv("SHERLOC_DB_PATH", env_target)

        result = resolve_path("./sherloc.db", "SHERLOC_DB_PATH", "phase.db")
        assert result == Path(env_target)

    def test_xdg_fallback_when_path_exists(self, monkeypatch, tmp_path):
        """XDG path is used when it exists and no env var is set."""
        xdg_home = tmp_path / "xdg"
        xdg_subdir = "phase.db"
        xdg_path = xdg_home / "sherloc" / xdg_subdir
        xdg_path.mkdir(parents=True)

        monkeypatch.setenv("XDG_DATA_HOME", str(xdg_home))
        monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)

        result = resolve_path("./sherloc.db", "SHERLOC_DB_PATH", xdg_subdir)
        assert result == xdg_path

    def test_xdg_skipped_when_path_missing(self, monkeypatch, tmp_path):
        """XDG path that does not exist on disk is silently skipped."""
        xdg_home = tmp_path / "xdg_empty"
        xdg_home.mkdir()
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg_home))
        monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)

        result = resolve_path("./sherloc.db", "SHERLOC_DB_PATH", "phase.db")
        # XDG path doesn't exist → fall through to config value
        assert result == Path("./sherloc.db")

    def test_config_value_fallback_no_env_no_xdg(self, monkeypatch):
        """Config value is used when no env var is set and xdg_subdir is None."""
        monkeypatch.delenv("SHERLOC_DATA_DIR", raising=False)
        result = resolve_path("./data", "SHERLOC_DATA_DIR")
        assert result == Path("./data")

    def test_xdg_skipped_when_subdir_is_none(self, monkeypatch, tmp_path):
        """XDG step is entirely bypassed when xdg_subdir=None."""
        xdg_home = tmp_path / "xdg"
        # Even if we set XDG_DATA_HOME, with subdir=None it should not matter
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg_home))
        monkeypatch.delenv("SHERLOC_BACKGROUND_DIR", raising=False)

        result = resolve_path("./data/background", "SHERLOC_BACKGROUND_DIR", None)
        assert result == Path("./data/background")

    def test_default_xdg_home(self, monkeypatch, tmp_path):
        """Default XDG_DATA_HOME (~/.local/share) is used when env not set."""
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)

        # The default XDG path almost certainly doesn't exist in test env,
        # so we should fall back to the config value.
        result = resolve_path("./fallback.db", "SHERLOC_DB_PATH", "phase.db")
        assert result == Path("./fallback.db")

    def test_returns_path_object(self, monkeypatch):
        """resolve_path always returns a Path, not a str."""
        monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)
        result = resolve_path("./sherloc.db", "SHERLOC_DB_PATH")
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# resolve_paths()
# ---------------------------------------------------------------------------


class TestResolvePaths:
    """Unit tests for the resolve_paths() dict transformation."""

    def _base_config(self):
        """Minimal config dict that mirrors config.yaml structure."""
        return {
            "paths": {
                "data_root": "./data",
                "results_root": "./results",
                "background_dir": "./data/background",
            },
            "database": {
                "path": "./sherloc.db",
                "pds_path": "./sherloc_pds.db",
            },
            "pds": {
                "cache_dir": "./data/pds",
            },
            "wavelength": {"raman_coefficients": [1, 2, 3]},
            "fitting": {"max_peaks": 5},
            "preprocessing": {"trim_mean_baseline_pct": 0.02},
        }

    def test_env_var_overrides_data_root(self, monkeypatch, tmp_path):
        """SHERLOC_DATA_DIR env var overrides paths.data_root."""
        env_val = str(tmp_path / "custom_data")
        monkeypatch.setenv("SHERLOC_DATA_DIR", env_val)
        monkeypatch.delenv("SHERLOC_RESULTS_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_BACKGROUND_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)
        monkeypatch.delenv("SHERLOC_PDS_DB_PATH", raising=False)
        monkeypatch.delenv("SHERLOC_PDS_CACHE_DIR", raising=False)

        cfg = self._base_config()
        resolved = resolve_paths(cfg)

        assert resolved["paths"]["data_root"] == env_val

    def test_env_var_overrides_db_path(self, monkeypatch, tmp_path):
        """SHERLOC_DB_PATH env var overrides database.path."""
        env_val = str(tmp_path / "custom.db")
        monkeypatch.setenv("SHERLOC_DB_PATH", env_val)
        monkeypatch.delenv("SHERLOC_DATA_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_RESULTS_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_BACKGROUND_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_PDS_DB_PATH", raising=False)
        monkeypatch.delenv("SHERLOC_PDS_CACHE_DIR", raising=False)

        cfg = self._base_config()
        resolved = resolve_paths(cfg)

        assert resolved["database"]["path"] == env_val

    def test_env_var_overrides_pds_db_path(self, monkeypatch, tmp_path):
        """SHERLOC_PDS_DB_PATH env var overrides database.pds_path."""
        env_val = str(tmp_path / "custom_pds.db")
        monkeypatch.setenv("SHERLOC_PDS_DB_PATH", env_val)
        monkeypatch.delenv("SHERLOC_DATA_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_RESULTS_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_BACKGROUND_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)
        monkeypatch.delenv("SHERLOC_PDS_CACHE_DIR", raising=False)

        cfg = self._base_config()
        resolved = resolve_paths(cfg)

        assert resolved["database"]["pds_path"] == env_val

    def test_default_fallback_no_env_vars(self, monkeypatch):
        """When no env vars are set and XDG dirs don't exist, config values are used."""
        for var in [
            "SHERLOC_DATA_DIR", "SHERLOC_RESULTS_DIR", "SHERLOC_BACKGROUND_DIR",
            "SHERLOC_DB_PATH", "SHERLOC_PDS_DB_PATH", "SHERLOC_PDS_CACHE_DIR",
            "XDG_DATA_HOME",
        ]:
            monkeypatch.delenv(var, raising=False)

        cfg = self._base_config()
        resolved = resolve_paths(cfg)

        # Compare as Path objects to normalize ./ prefixes
        assert Path(resolved["paths"]["data_root"]) == Path("./data")
        assert Path(resolved["paths"]["results_root"]) == Path("./results")
        assert Path(resolved["paths"]["background_dir"]) == Path("./data/background")
        assert Path(resolved["database"]["path"]) == Path("./sherloc.db")
        assert Path(resolved["database"]["pds_path"]) == Path("./sherloc_pds.db")
        assert Path(resolved["pds"]["cache_dir"]) == Path("./data/pds")

    def test_non_path_keys_unchanged(self, monkeypatch):
        """Non-path config sections are not mutated."""
        for var in [
            "SHERLOC_DATA_DIR", "SHERLOC_RESULTS_DIR", "SHERLOC_BACKGROUND_DIR",
            "SHERLOC_DB_PATH", "SHERLOC_PDS_DB_PATH", "SHERLOC_PDS_CACHE_DIR",
            "XDG_DATA_HOME",
        ]:
            monkeypatch.delenv(var, raising=False)

        cfg = self._base_config()
        resolved = resolve_paths(cfg)

        assert resolved["wavelength"] == {"raman_coefficients": [1, 2, 3]}
        assert resolved["fitting"] == {"max_peaks": 5}
        assert resolved["preprocessing"] == {"trim_mean_baseline_pct": 0.02}

    def test_input_dict_not_mutated(self, monkeypatch):
        """resolve_paths does not modify the input dict."""
        monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)
        monkeypatch.delenv("SHERLOC_DATA_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_RESULTS_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_BACKGROUND_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_PDS_DB_PATH", raising=False)
        monkeypatch.delenv("SHERLOC_PDS_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)

        cfg = self._base_config()
        original_data_root = cfg["paths"]["data_root"]

        resolve_paths(cfg)

        assert cfg["paths"]["data_root"] == original_data_root

    def test_missing_database_section(self, monkeypatch):
        """resolve_paths handles configs without a database section."""
        monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)
        monkeypatch.delenv("SHERLOC_PDS_DB_PATH", raising=False)
        monkeypatch.delenv("SHERLOC_DATA_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_RESULTS_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_BACKGROUND_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_PDS_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)

        cfg = self._base_config()
        del cfg["database"]

        resolved = resolve_paths(cfg)
        # Should not raise, database section should be present (empty or default)
        assert "database" in resolved

    def test_xdg_path_resolution_for_results(self, monkeypatch, tmp_path):
        """XDG fallback works for results_root when the directory exists."""
        xdg_home = tmp_path / "xdg"
        xdg_results = xdg_home / "sherloc" / "results"
        xdg_results.mkdir(parents=True)

        monkeypatch.setenv("XDG_DATA_HOME", str(xdg_home))
        monkeypatch.delenv("SHERLOC_RESULTS_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_DATA_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_BACKGROUND_DIR", raising=False)
        monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)
        monkeypatch.delenv("SHERLOC_PDS_DB_PATH", raising=False)
        monkeypatch.delenv("SHERLOC_PDS_CACHE_DIR", raising=False)

        cfg = self._base_config()
        resolved = resolve_paths(cfg)

        assert resolved["paths"]["results_root"] == str(xdg_results)
