from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageMetadata:
    name: str
    start_time: str
    end_time: str
    duration_s: float
    artifacts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": self.duration_s,
            "artifacts": list(self.artifacts),
            "warnings": list(self.warnings),
            "extra": dict(self.extra),
        }


@dataclass
class RunMetadata:
    run_id: str
    config_hash: str
    sol: str
    target: str
    scan: str
    results_path: str
    data_root: Optional[str] = None
    results_root: Optional[str] = None
    archived_results_path: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_s: Optional[float] = None
    stages: List[StageMetadata] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def add_stage(self, stage: StageMetadata) -> None:
        self.stages.append(stage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "sol": self.sol,
            "target": self.target,
            "scan": self.scan,
            "results_path": self.results_path,
            "data_root": self.data_root,
            "results_root": self.results_root,
            "archived_results_path": self.archived_results_path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": self.duration_s,
            "stages": [stage.to_dict() for stage in self.stages],
            "artifacts": list(self.artifacts),
            "warnings": list(self.warnings),
            "extra": dict(self.extra),
        }


__all__ = ["RunMetadata", "StageMetadata"]
