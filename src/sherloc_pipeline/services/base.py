"""
Base service utilities for SHERLOC pipeline.

This module provides shared result structures and utilities for the services layer.
Services return lightweight structured results that can be consumed by both CLI
and programmatic interfaces.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ServiceResult:
    """Standardized result structure for service operations.
    
    This dataclass provides a consistent return format for all service methods,
    making it easy for CLI commands and programmatic consumers to handle results
    uniformly.
    
    Attributes:
        summary: Human-readable summary of the operation
        artifacts: List of output file paths created by the service
        warnings: List of warning messages (non-fatal issues)
        metadata: Optional dictionary for additional structured data
        
    Example:
        >>> result = ServiceResult(
        ...     summary="Processed 5 scans successfully",
        ...     artifacts=[Path("results/scan1.csv"), Path("results/scan2.csv")],
        ...     warnings=["Low SNR detected in scan 3"],
        ...     metadata={"points_processed": 42, "processing_time": 12.5}
        ... )
        >>> result.to_dict()
        {
            'summary': 'Processed 5 scans successfully',
            'artifacts': ['results/scan1.csv', 'results/scan2.csv'],
            'warnings': ['Low SNR detected in scan 3'],
            'metadata': {'points_processed': 42, 'processing_time': 12.5}
        }
    """
    
    summary: str
    artifacts: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization or API consumption.
        
        Returns:
            Dictionary representation with all fields, converting Path objects
            to strings for JSON compatibility.
        """
        return {
            "summary": self.summary,
            "artifacts": [str(artifact) for artifact in self.artifacts],
            "warnings": self.warnings,
            "metadata": self.metadata or {},
        }


