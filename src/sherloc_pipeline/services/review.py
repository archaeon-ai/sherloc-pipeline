"""
Review service for SHERLOC pipeline.

This module provides orchestration services for applying manual review at scan-level
to target and project tables, and optionally regenerating spatial overlays.

The ReviewService encapsulates the orchestration logic that was previously in the CLI
module, providing a clean service interface that can be used by both CLI commands
and programmatic consumers.

Orchestration order:
1. Read reviewed scan-level CSV
2. Update target-level table (drop-and-append)
3. Update project-level table (drop-and-append)
4. Optionally regenerate spatial overlays using SpatialService

Usage:
    from sherloc_pipeline.services.review import ReviewService
    from rich.console import Console
    
    service = ReviewService(console=Console())
    result = service.apply_review(
        sol="1613",
        target="Nordoya",
        scan="detail",
        regenerate_overlays=True
    )
    print(result.summary)
    for artifact in result.artifacts:
        print(f"  {artifact}")
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console

from .base import ServiceResult
from .errors import ReviewError, enrich
from .paths import resolve_scan_context
from .runtime import RuntimeContext
from sherloc_pipeline.core.accepted_assembler import (
    ACCEPTED_PEAKS_COLUMNS,
    normalize_accepted_peaks_df,
    write_accepted_table,
)
from sherloc_pipeline.services.errors import SherlocServiceError

logger = logging.getLogger(__name__)


class ReviewService:
    """Service for orchestrating review and aggregation operations.
    
    This service coordinates:
    1. Reading reviewed scan-level accepted peaks CSV
    2. Updating target-level accepted peaks table (drop-and-append)
    3. Updating project-level accepted peaks table (drop-and-append)
    4. Optionally regenerating spatial overlays
    
    The service maintains console output consistency by accepting an optional
    Console instance, allowing CLI commands to use their existing console while
    programmatic consumers can provide their own or use a default.
    
    The service enforces the `user_keep` contract:
    - Does not mutate source CSVs in-place
    - Validates required columns are present
    - Preserves user_keep semantics when propagating to target/project tables
    
    Attributes:
        console: Rich Console instance for progress/output (defaults to new Console)
        
    Example:
        >>> service = ReviewService()
        >>> result = service.apply_review("1613", "Nordoya", "detail")
        >>> print(result.summary)
        'Applied review for scan 1613/Nordoya/detail successfully'
        >>> print(f"Updated {len(result.artifacts)} tables")
    """
    
    def __init__(self, console: Optional[Console] = None, *, context: Optional[RuntimeContext] = None):
        """Initialize review service.
        
        Args:
            console: Optional Rich Console instance. If None, creates a new Console.
            context: Optional RuntimeContext providing resolved configuration and paths. If None,
                a new context is bootstrapped.
        """
        self.console = console if console is not None else Console()
        self.context = context if context is not None else RuntimeContext.bootstrap()
    
    def apply_review(
        self,
        sol: str,
        target: str,
        scan: str,
        regenerate_overlays: bool = True,
        upscale: Optional[int] = None,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ) -> ServiceResult:
        """Apply manual review at scan-level to target & project tables.
        
        This method reads the reviewed scan-level accepted peaks CSV and propagates
        it to target and project-level tables using drop-and-append semantics.
        Optionally regenerates spatial overlays using SpatialService.
        
        Args:
            sol: Sol number (e.g., "1613")
            target: Target name (e.g., "Nordoya")
            scan: Scan type (e.g., "detail", "line")
            regenerate_overlays: If True, regenerate spatial overlays using --use-reviewed
            upscale: Override upscaling factor for overlay regeneration (defaults to config)
            data_dir: Base data directory. If None, uses config default.
            results_dir: Results directory. If None, uses config default.
        
        Returns:
            ServiceResult with summary, artifacts (target/project tables, overlay outputs if regen), warnings, and metadata
            
        Raises:
            ReviewError: If review application fails (missing CSV, invalid format, etc.)
            
        Example:
            >>> service = ReviewService()
            >>> result = service.apply_review(
            ...     sol="1613",
            ...     target="Nordoya",
            ...     scan="detail",
            ...     regenerate_overlays=True
            ... )
            >>> print(result.summary)
            'Applied review for scan 1613/Nordoya/detail successfully'
        """
        run_context = self.context.with_overrides(
            data_dir=data_dir,
            results_dir=results_dir,
        )
        cfg = run_context.config
        
        try:
            # Resolve scan context
            context = resolve_scan_context(
                sol=sol,
                target=target,
                scan=scan,
                data_dir=data_dir,
                results_dir=results_dir,
                context=run_context,
            )
        except (FileNotFoundError, ValueError) as e:
            error = ReviewError(
                f"Failed to resolve scan context: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan},
            )
            raise enrich(error, sol=sol, target=target, scan=scan) from e
        
        # Locate scan-level accepted peaks CSV
        scan_csv = context.results_path / f"{sol}_{target}_{scan}_accepted_peaks.csv"
        if not scan_csv.exists():
            error = ReviewError(
                f"Scan-level accepted_peaks file not found: {scan_csv}",
                exit_code=1,
                context={"scan_csv": str(scan_csv)},
            )
            raise enrich(error, sol=sol, target=target, scan=scan)
        
        artifacts = []
        warnings = []
        
        try:
            # Read reviewed scan-level CSV
            df = pd.read_csv(scan_csv)

            missing_cols = [col for col in ACCEPTED_PEAKS_COLUMNS if col not in df.columns]
            if missing_cols:
                error = ReviewError(
                    f"Scan CSV missing schema columns: {missing_cols}",
                    exit_code=1,
                    context={"scan_csv": str(scan_csv), "missing_cols": missing_cols},
                )
                raise enrich(error, sol=sol, target=target, scan=scan)

            try:
                df = normalize_accepted_peaks_df(df)
            except ValueError as exc:
                error = ReviewError(
                    f"Failed to normalize scan CSV against accepted peaks schema: {exc}",
                    exit_code=1,
                    context={"scan_csv": str(scan_csv)},
                )
                raise enrich(error, sol=sol, target=target, scan=scan) from exc
            
            # Enforce user_keep contract: don't mutate source CSV in-place
            # The CSV should already have user_keep flags set by the user
            # We just propagate it to target/project tables
            
            # Update target-level table (drop-and-append)
            tgt_out = context.results_path.parent / f"{target}_accepted_peaks.csv"
            
            if tgt_out.exists():
                try:
                    old = pd.read_csv(tgt_out)
                except Exception as exc:
                    error = ReviewError(
                        f"Failed to read target-level table: {exc}",
                        exit_code=1,
                        context={"target_table": str(tgt_out)},
                    )
                    raise enrich(error, sol=sol, target=target, scan=scan) from exc
                old = old[~((old['sol'].astype(str) == str(sol)) &
                            (old['target'] == target) &
                            (old['scan'] == scan))]
                df_t = pd.concat([old, df], ignore_index=True)
            else:
                df_t = df

            write_accepted_table(tgt_out, df_t)
            artifacts.append(tgt_out)

            # Update project-level table (drop-and-append)
            proj_out = context.results_dir / "SHERLOC_accepted_peaks_master.csv"

            if proj_out.exists():
                try:
                    oldp = pd.read_csv(proj_out)
                except Exception as exc:
                    error = ReviewError(
                        f"Failed to read project-level table: {exc}",
                        exit_code=1,
                        context={"project_table": str(proj_out)},
                    )
                    raise enrich(error, sol=sol, target=target, scan=scan) from exc
                oldp = oldp[~((oldp['sol'].astype(str) == str(sol)) &
                              (oldp['target'] == target) &
                              (oldp['scan'] == scan))]
                df_p = pd.concat([oldp, df], ignore_index=True)
            else:
                df_p = df

            write_accepted_table(proj_out, df_p)
            artifacts.append(proj_out)
            
            try:
                self.console.print(f"[green]Updated target-level: {tgt_out} and project-level: {proj_out}[/green]")
            except Exception:
                pass  # Console might not support Rich formatting
            
        except ReviewError:
            raise  # Re-raise ReviewError as-is
        except Exception as e:
            error = ReviewError(
                f"Failed to update aggregate tables: {e}",
                exit_code=1,
                context={"scan_csv": str(scan_csv)},
            )
            raise enrich(error, sol=sol, target=target, scan=scan) from e
        
        # Optionally regenerate overlays
        if regenerate_overlays:
            try:
                from .spatial import SpatialService
                
                # Determine default upscale from config if not provided
                if upscale is None:
                    try:
                        image_cfg = getattr(cfg, 'image', None)
                        if isinstance(image_cfg, dict):
                            upscale = int(image_cfg.get('default_upscale_factor', 3))
                        else:
                            upscale = int(getattr(image_cfg, 'default_upscale_factor', 3))
                    except Exception:
                        upscale = 3
                
                # Delegate to SpatialService
                spatial_service = SpatialService(console=self.console, context=run_context)
                spatial_result = spatial_service.render_overlay(
                    sol=sol,
                    target=target,
                    scan=scan,
                    layers='minerals,organics,hydration',
                    upscale=int(upscale),
                    save_debug=True,
                    use_reviewed=True,
                    data_dir=run_context.data_root,
                    results_dir=run_context.results_root,
                )
                
                # Merge artifacts and warnings from spatial service
                artifacts.extend(spatial_result.artifacts)
                warnings.extend(spatial_result.warnings)
                
            except Exception as e:
                # Log warning but don't fail the review operation
                warning_msg = f"Overlay regeneration failed: {e}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
                try:
                    self.console.print(f"[yellow]{warning_msg}[/yellow]")
                except Exception:
                    pass
        
        summary = f"Applied review for scan {sol}/{target}/{scan} successfully"
        if regenerate_overlays:
            summary += " (overlays regenerated)"
        
        metadata = {
            "scan_csv": str(scan_csv),
            "target_table": str(tgt_out),
            "project_table": str(proj_out),
            "regenerate_overlays": regenerate_overlays,
        }
        
        return ServiceResult(
            summary=summary,
            artifacts=artifacts,
            warnings=warnings,
            metadata=metadata,
        )


    def write_unified_tables(
        self,
        sol: str,
        target: str,
        scan: str,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ) -> ServiceResult:
        """Build per-point accepted peaks across modalities and write scan/target/project tables.
        
        This method aggregates accepted peaks from per-modality CSV files (minerals_fit/,
        organics_fit/, hydration_fit/) and creates unified scan/target/project-level tables.
        This is different from `apply_review` which reads an already-reviewed scan CSV.
        
        Schema: sol,target,scan,modality,point,mean,amplitude,fwhm,snr,id_label,peak_ID,keep,user_keep,reviewed,reject_reason,r_squared
        Defaults: user_keep=True, reviewed=False, reject_reason=""
        
        Args:
            sol: Sol number (e.g., "1613")
            target: Target name (e.g., "Nordoya")
            scan: Scan type (e.g., "detail", "line")
            data_dir: Base data directory. If None, uses config default.
            results_dir: Results directory. If None, uses config default.
        
        Returns:
            ServiceResult with summary, artifacts (scan/target/project tables), warnings, and metadata
            
        Raises:
            ReviewError: If table generation fails
            
        Example:
            >>> service = ReviewService()
            >>> result = service.write_unified_tables("1613", "Nordoya", "detail")
            >>> print(result.summary)
            'Scan-level table built successfully'
        """
        run_context = self.context.with_overrides(
            data_dir=data_dir,
            results_dir=results_dir,
        )
        cfg = run_context.config
        
        try:
            # Resolve scan context
            context = resolve_scan_context(
                sol=sol,
                target=target,
                scan=scan,
                data_dir=data_dir,
                results_dir=results_dir,
                context=run_context,
            )
        except (FileNotFoundError, ValueError) as e:
            error = ReviewError(
                f"Failed to resolve scan context: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan},
            )
            raise enrich(error, sol=sol, target=target, scan=scan) from e
        
        artifacts = []
        warnings = []
        
        try:
            from sherloc_pipeline.core.accepted_assembler import build_scan_df, write_scan_target_project
            from sherloc_pipeline.core.data_ingestion import DataIngestion
            
            ingestion = DataIngestion(
                base_data_dir=context.base_data_dir,
                results_dir=context.results_dir,
                sol=sol,
                target=target,
                scan=scan,
            )
            base = context.results_path
            
            # Determine thresholds from config
            try:
                f_lo = 25.0
                fitting_cfg = getattr(cfg, 'fitting', {})
                if isinstance(fitting_cfg, dict):
                    f_hi_keep = float(fitting_cfg.get('filter_fwhm_min_cm1', 30.0))
                else:
                    f_hi_keep = float(getattr(fitting_cfg, 'filter_fwhm_min_cm1', 30.0))
            except Exception:
                f_lo, f_hi_keep = 25.0, 30.0
            
            df = build_scan_df(base=base, sol=sol, target=target, scan=scan, f_lo=f_lo, f_hi_keep=f_hi_keep)
            if df.empty:
                warning_msg = "No peaks discovered for scan-level table."
                warnings.append(warning_msg)
                try:
                    self.console.print(f"[yellow]{warning_msg}[/yellow]")
                except Exception:
                    pass
            else:
                # Write scan/target/project
                write_scan_target_project(
                    base=base,
                    results_root=ingestion.results_dir,
                    sol=sol,
                    target=target,
                    scan=scan,
                    df=df
                )
                # Track artifacts
                scan_csv = base / f"{sol}_{target}_{scan}_accepted_peaks.csv"
                target_csv = base.parent / f"{target}_accepted_peaks.csv"
                project_csv = ingestion.results_dir / "SHERLOC_accepted_peaks_master.csv"
                artifacts.extend([scan_csv, target_csv, project_csv])
                
                try:
                    self.console.print(f"[green]Scan-level table built: {len(df)} rows (minerals+organics+hydration).[/green]")
                except Exception:
                    pass
            
            summary = f"Scan-level table built successfully for {sol}/{target}/{scan}"
            metadata = {
                "scan_csv": str(scan_csv) if not df.empty else None,
                "target_csv": str(target_csv) if not df.empty else None,
                "project_csv": str(project_csv) if not df.empty else None,
                "rows": len(df),
            }
            
            return ServiceResult(
                summary=summary,
                artifacts=artifacts,
                warnings=warnings,
                metadata=metadata,
            )
            
        except ReviewError:
            raise  # Re-raise ReviewError as-is
        except Exception as e:
            error = ReviewError(
                f"Failed to build unified accepted tables: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan},
            )
            raise enrich(error, sol=sol, target=target, scan=scan) from e

