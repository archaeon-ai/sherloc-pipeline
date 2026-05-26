"""
Service-layer error hierarchy for SHERLOC pipeline.

This module defines typed error classes for the services layer, providing
structured error handling with context information and exit codes suitable
for CLI consumption.

Usage:
    from sherloc_pipeline.services.errors import PreprocessingError, enrich
    
    # Raise a basic error
    raise PreprocessingError("Failed to load spectrum", exit_code=1)
    
    # Enrich with scan context
    error = PreprocessingError("Processing failed", exit_code=1, context={"file": "data.csv"})
    enriched = enrich(error, sol="0921", target="Amherst_Point", scan="detail_1")
    raise enriched
"""

from typing import Dict, Any, Optional


class SherlocServiceError(RuntimeError):
    """Base exception for all service-layer errors.
    
    This exception provides structured error information with:
    - A human-readable message
    - An exit code for CLI consumption
    - Optional context dictionary for additional metadata
    
    Attributes:
        message: Human-readable error message
        exit_code: Exit code for CLI (default: 1)
        context: Optional dictionary with additional error context
        
    Example:
        >>> error = SherlocServiceError("Operation failed", exit_code=2)
        >>> error.message
        'Operation failed'
        >>> error.exit_code
        2
    """
    
    def __init__(
        self,
        message: str,
        exit_code: int = 1,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize service error.
        
        Args:
            message: Human-readable error message
            exit_code: Exit code for CLI (default: 1)
            context: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code
        # Copy context to prevent callers from mutating internal dict
        self.context = dict(context) if context else {}
    
    def __str__(self) -> str:
        """Return error message."""
        return self.message


class PipelineRunError(SherlocServiceError):
    """Error during full pipeline execution.
    
    This error indicates a failure during the orchestration of multiple
    pipeline stages (preprocessing, fitting, spatial, review).
    
    Example:
        >>> raise PipelineRunError("Pipeline failed at fitting stage", exit_code=2)
    """
    pass


class PreprocessingError(SherlocServiceError):
    """Error during data preprocessing operations.
    
    This error indicates a failure during data ingestion, despiking,
    baseline correction, or background subtraction.
    
    Example:
        >>> raise PreprocessingError("Failed to load spectrum file", exit_code=1)
    """
    pass


class FittingError(SherlocServiceError):
    """Error during peak fitting operations.
    
    This error indicates a failure during mineral, organic, hydration,
    or average fitting operations.
    
    Example:
        >>> raise FittingError("No valid peaks found for fitting", exit_code=1)
    """
    pass


class SpatialError(SherlocServiceError):
    """Error during spatial overlay operations.
    
    This error indicates a failure during spatial visualization,
    overlay generation, or merged label operations.
    
    Example:
        >>> raise SpatialError("Failed to generate spatial overlay", exit_code=1)
    """
    pass


class ReviewError(SherlocServiceError):
    """Error during review and aggregation operations.
    
    This error indicates a failure during user review processing,
    accepted peaks aggregation, or unified table generation.
    
    Example:
        >>> raise ReviewError("Failed to apply review flags", exit_code=1)
    """
    pass


def enrich(
    error: SherlocServiceError,
    *,
    sol: str,
    target: str,
    scan: str,
) -> SherlocServiceError:
    """Enrich an error with scan context metadata.
    
    This function creates a new error instance with merged context,
    adding scan identifiers (sol, target, scan) to the existing context.
    The original error is not modified (immutable operation).
    
    Args:
        error: The original error to enrich
        sol: Sol number (e.g., "0921")
        target: Target name (e.g., "Amherst_Point")
        scan: Scan type (e.g., "detail_1")
        
    Returns:
        New error instance with merged context containing scan metadata
        
    Example:
        >>> original = PreprocessingError("Failed", exit_code=1, context={"file": "data.csv"})
        >>> enriched = enrich(original, sol="0921", target="Amherst_Point", scan="detail_1")
        >>> enriched.context
        {'file': 'data.csv', 'sol': '0921', 'target': 'Amherst_Point', 'scan': 'detail_1'}
        >>> original.context  # Original unchanged
        {'file': 'data.csv'}
    """
    # Create a new context dict by merging (avoiding mutation)
    new_context = dict(error.context)
    new_context.update({
        "sol": sol,
        "target": target,
        "scan": scan,
    })
    
    # Create a new error instance of the same type
    return type(error)(
        message=error.message,
        exit_code=error.exit_code,
        context=new_context,
    )

