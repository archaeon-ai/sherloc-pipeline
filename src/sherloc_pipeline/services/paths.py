"""
Path resolution utilities for services layer.

This module provides utilities for resolving scan contexts and normalizing paths,
ensuring paths stay within configured roots and providing a consistent interface
for scan-based operations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..core.data_ingestion import DataIngestion
from ..core.manifest import ManifestResolutionError, resolve_manifest_working_directory
from .config import get_runtime_config
from .runtime import RuntimeContext


@dataclass
class ScanContext:
    """Normalized scan context with validated paths.
    
    This dataclass provides a structured representation of a scan's context,
    including all necessary paths and metadata. Paths are normalized to
    absolute paths and validated to ensure they stay within configured roots.
    
    Attributes:
        sol: Sol number (e.g., "0921")
        target: Target name (e.g., "Amherst_Point")
        scan: Scan type (e.g., "detail_1", "line", "detail")
        base_data_dir: Absolute path to base data directory (sol_XXXX folders)
        results_dir: Absolute path to results directory
        working_dir: Absolute path to scan's working directory (contains Loupe data)
        results_path: Absolute path to scan's results directory
        
    Example:
        >>> context = resolve_scan_context("0921", "Amherst_Point", "detail_1")
        >>> print(context.working_dir)
        /path/to/data/loupe/sol_0921/detail_1/SrlcSpecSpecSohRaw_XXXXX-XXXXX-X_Loupe_working
        >>> print(context.results_path)
        /path/to/results/Amherst_Point/0921_Amherst_Point_detail_1
    """
    
    sol: str
    target: str
    scan: str
    base_data_dir: Path
    results_dir: Path
    working_dir: Path
    results_path: Path


def resolve_scan_context(
    sol: str,
    target: str,
    scan: str,
    data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    *,
    context: Optional[RuntimeContext] = None,
) -> ScanContext:
    """Resolve scan context with normalized paths.
    
    This function creates a ScanContext by using DataIngestion to discover
    the working directory and results path. It ensures all paths are absolute
    and validates that paths stay within configured roots.
    
    Args:
        sol: Sol number (e.g., "0921")
        target: Target name (e.g., "Amherst_Point")
        scan: Scan type (e.g., "detail_1", "line", "detail")
        data_dir: Optional base data directory. If None, uses config default (unless context provided).
        results_dir: Optional results directory. If None, uses config default (unless context provided).
        context: Optional RuntimeContext providing resolved paths and configuration.
    
    Returns:
        ScanContext with all normalized paths
    
    Raises:
        FileNotFoundError: If sol directory or scan working directory not found
        ValueError: If paths would traverse outside configured roots
    
    Example:
        >>> context = resolve_scan_context("0921", "Amherst_Point", "detail_1")
        >>> ingestion = DataIngestion(
        ...     base_data_dir=context.base_data_dir,
        ...     results_dir=context.results_dir,
        ...     sol=context.sol,
        ...     target=context.target,
        ...     scan=context.scan
        ... )
        >>> # Use context.working_dir and context.results_path for operations
    """
    if context is not None:
        base_data_dir = Path(context.data_root)
        results_dir_path = Path(context.results_root)
    else:
        # Get runtime configuration
        config = get_runtime_config()
    
        # Resolve data directory from config if not provided
        if data_dir is None:
            data_dir_str = config.paths.get('data_dir', '../data/loupe') if hasattr(config, 'paths') else '../data/loupe'
            data_dir = Path(data_dir_str)
    
        # Resolve results directory from config if not provided
        if results_dir is None:
            results_dir_str = config.paths.get('results_dir', '../results') if hasattr(config, 'paths') else '../results'
            results_dir = Path(results_dir_str)
    
        # Normalize to absolute paths
        base_data_dir = Path(data_dir).resolve()
        results_dir_path = Path(results_dir).resolve()
    
    # Validate base_data_dir exists
    if not base_data_dir.exists():
        raise FileNotFoundError(f"Base data directory not found: {base_data_dir}")
    
    # Validate sol directory exists
    sol_dir = base_data_dir / f"sol_{sol}"
    if not sol_dir.exists():
        raise FileNotFoundError(f"Sol directory not found: {sol_dir}")
    
    # Create DataIngestion instance
    ingestion = DataIngestion(
        base_data_dir=base_data_dir,
        results_dir=results_dir_path,
        sol=sol,
        target=target,
        scan=scan,
    )

    try:
        manifest_dir = resolve_manifest_working_directory(
            base_data_dir=base_data_dir,
            sol=sol,
            scan=scan,
        )
    except ManifestResolutionError as exc:
        raise RuntimeError(
            f"Loupe manifest resolution failed for sol {sol} scan {scan}: {exc}"
        ) from exc

    if manifest_dir is not None:
        working_dir = manifest_dir
    else:
        try:
            working_dir = ingestion.find_working_directory(sol, scan)
        except RuntimeError as error:
            scan_dir = sol_dir / scan
            fallback_dirs = []
            if scan_dir.exists():
                try:
                    fallback_dirs = ingestion._discover_working_directories(scan_dir)  # type: ignore[attr-defined]
                except Exception:
                    fallback_dirs = []
            if fallback_dirs:
                paths = ", ".join(str(path.resolve()) for path in fallback_dirs)
                raise RuntimeError(
                    "Multiple working directories discovered without manifest metadata. "
                    f"Resolve Loupe manifests or prune archives for sol {sol} scan {scan}: {paths}"
                ) from error
            raise
        if working_dir is None:
            raise FileNotFoundError(
                f"No working directory found for sol {sol}, scan {scan}. "
                f"Expected directory structure: {sol_dir}/{scan}/SrlcSpecSpecSohRaw_*_Loupe_working"
            )
    
    # Normalize working directory to absolute path
    working_dir = Path(working_dir).resolve()
    
    # Validate working_dir is within base_data_dir (prevent path traversal)
    try:
        working_dir.relative_to(base_data_dir)
    except ValueError:
        raise ValueError(
            f"Working directory {working_dir} is outside base data directory {base_data_dir}"
        )
    
    # Get results path
    results_path = ingestion.get_results_path(target=target, sol=sol, scan=scan)
    results_path = Path(results_path).resolve()
    
    # Validate results_path is within results_dir (prevent path traversal)
    try:
        results_path.relative_to(results_dir_path)
    except ValueError as exc:
        raise ValueError(
            f"Results path {results_path} is outside configured results directory {results_dir_path}"
        ) from exc
    
    return ScanContext(
        sol=sol,
        target=target,
        scan=scan,
        base_data_dir=base_data_dir,
        results_dir=results_dir_path,
        working_dir=working_dir,
        results_path=results_path,
    )
