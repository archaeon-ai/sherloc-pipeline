"""Versioned schema models for CLI structured JSON output (§3.2.9).

All CLI commands that support --json output produce one of these models.
The schema_version field allows consumers to detect breaking changes.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class CLIResult(BaseModel):
    """Successful CLI command result."""

    schema_version: str = "1.0.0"
    pipeline_version: str
    command: str
    result: Dict[str, Any]
    metadata: Dict[str, Any] = {}


class CLIError(BaseModel):
    """CLI command error."""

    schema_version: str = "1.0.0"
    pipeline_version: str
    error_type: str
    message: str
    context: Dict[str, Any] = {}
    exit_code: int


class PipelineManifest(BaseModel):
    """Per-scan pipeline run manifest."""

    schema_version: str = "1.0.0"
    pipeline_version: str
    scan_id: int
    sol: int
    target: str
    scan_name: str
    processed_at: str
    config_hash: str
    artifacts: Dict[str, Any] = {}
    findings: Optional[Dict[str, Any]] = None
