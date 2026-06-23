from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Union, Any

from ..config import Config
from .config import compute_config_hash, get_runtime_config

PathLike = Union[str, Path]


def _normalize_path(path: Optional[PathLike]) -> Optional[Path]:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    try:
        return resolved if resolved.is_absolute() else resolved.resolve()
    except FileNotFoundError:
        # Path.resolve(strict=False) is only available in Python 3.12+; emulate here.
        return resolved if resolved.is_absolute() else resolved.absolute()


def _extract_path(config: Config, key: str, fallback: str) -> Path:
    candidate: Optional[Any] = None
    try:
        paths_section = getattr(config, "paths", None)
        if isinstance(paths_section, dict):
            candidate = paths_section.get(key)
        elif paths_section is not None:
            candidate = getattr(paths_section, key, None)
    except Exception:
        candidate = None

    raw = candidate if candidate is not None else fallback
    normalized = _normalize_path(raw)
    if normalized is None:
        raise ValueError(f"Unable to resolve runtime path for '{key}'")
    return normalized


@dataclass(frozen=True)
class RuntimeOverrides:
    data_root: Optional[Path] = field(default=None)
    results_root: Optional[Path] = field(default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", _normalize_path(self.data_root))
        object.__setattr__(self, "results_root", _normalize_path(self.results_root))

    @property
    def has_overrides(self) -> bool:
        return self.data_root is not None or self.results_root is not None


@dataclass(frozen=True)
class RuntimeContext:
    config: Config
    data_root: Path
    results_root: Path
    config_hash: str = field(init=False)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    overrides: RuntimeOverrides = field(default_factory=RuntimeOverrides)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", _normalize_path(self.data_root))
        object.__setattr__(self, "results_root", _normalize_path(self.results_root))
        object.__setattr__(self, "overrides", RuntimeOverrides(
            data_root=self.overrides.data_root,
            results_root=self.overrides.results_root,
        ) if self.overrides is not None else RuntimeOverrides())
        object.__setattr__(self, "config_hash", compute_config_hash(self.config))

    @classmethod
    def bootstrap(
        cls,
        data_dir: Optional[PathLike] = None,
        results_dir: Optional[PathLike] = None,
        *,
        config: Optional[Config] = None,
        run_id: Optional[str] = None,
    ) -> RuntimeContext:
        cfg = config if config is not None else get_runtime_config()
        default_data = _extract_path(cfg, "data_dir", "../data/loupe")
        default_results = _extract_path(cfg, "results_dir", "../results")
        overrides = RuntimeOverrides(data_root=data_dir, results_root=results_dir)

        data_root = overrides.data_root or default_data
        results_root = overrides.results_root or default_results

        return cls(
            config=cfg,
            data_root=data_root,
            results_root=results_root,
            run_id=run_id or uuid.uuid4().hex,
            overrides=overrides if overrides.has_overrides else RuntimeOverrides(),
        )

    def with_overrides(
        self,
        *,
        data_dir: Optional[PathLike] = None,
        results_dir: Optional[PathLike] = None,
        refresh_run_id: bool = True,
    ) -> RuntimeContext:
        overrides = RuntimeOverrides(data_root=data_dir, results_root=results_dir)
        if not overrides.has_overrides and not refresh_run_id:
            return self

        if not overrides.has_overrides:
            return replace(
                self,
                run_id=self.run_id if not refresh_run_id else uuid.uuid4().hex,
            )

        return RuntimeContext(
            config=self.config,
            data_root=overrides.data_root or self.data_root,
            results_root=overrides.results_root or self.results_root,
            run_id=uuid.uuid4().hex if refresh_run_id else self.run_id,
            overrides=overrides,
        )

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "data_root": str(self.data_root),
            "results_root": str(self.results_root),
            "config_hash": self.config_hash,
            "overrides": {
                "data_root": str(self.overrides.data_root) if self.overrides.data_root else None,
                "results_root": str(self.overrides.results_root) if self.overrides.results_root else None,
            },
        }


__all__ = ["RuntimeContext", "RuntimeOverrides"]


