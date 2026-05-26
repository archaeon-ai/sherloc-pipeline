"""
Preprocessing service for SHERLOC pipeline.

This module provides orchestration services for preprocessing SHERLOC scan data,
including data ingestion, laser normalization, despiking, baseline correction,
statistical analysis, and background subtraction.

The PreprocessingService encapsulates the orchestration logic that was previously
in the CLI module, providing a clean service interface that can be used by both
CLI commands and programmatic consumers.

Usage:
    from sherloc_pipeline.services.preprocessing import PreprocessingService
    from rich.console import Console
    
    service = PreprocessingService(console=Console())
    result = service.run_scan(
        sol="0921",
        target="Amherst_Point",
        scan="detail_1",
        data_dir=Path("../data/loupe"),
        results_dir=Path("../results"),
        generate_plots=True
    )
    print(result.summary)
    for artifact in result.artifacts:
        print(f"  {artifact}")
"""

import logging
import re
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .base import ServiceResult
from .errors import PreprocessingError, enrich
from .paths import resolve_scan_context
from .runtime import RuntimeContext

logger = logging.getLogger(__name__)


class PreprocessingService:
    """Service for orchestrating SHERLOC scan preprocessing operations.
    
    This service coordinates multiple preprocessing steps:
    1. Data ingestion (raw spectra loading and restructuring)
    2. Laser normalization
    3. Optional R1 despiking
    4. Optional R1 baseline correction
    5. Statistical analysis (mean, trimmed mean)
    6. Optional background subtraction on averages
    
    The service maintains console output consistency by accepting an optional
    Console instance, allowing CLI commands to use their existing console while
    programmatic consumers can provide their own or use a default.
    
    Attributes:
        console: Rich Console instance for progress/output (defaults to new Console)
        
    Example:
        >>> service = PreprocessingService()
        >>> result = service.run_scan("0921", "Amherst_Point", "detail_1")
        >>> print(result.summary)
        'Processed scan 0921/Amherst_Point/detail_1 successfully'
        >>> print(f"Generated {len(result.artifacts)} artifacts")
    """
    
    def __init__(self, console: Optional[Console] = None, *, context: Optional[RuntimeContext] = None):
        """Initialize preprocessing service.
        
        Args:
            console: Optional Rich Console instance. If None, creates a new Console.
            context: Optional RuntimeContext providing resolved configuration and paths. If None,
                a new context is bootstrapped.
        """
        self.console = console if console is not None else Console()
        self.context = context if context is not None else RuntimeContext.bootstrap()
    
    def run_scan(
        self,
        sol: str,
        target: str,
        scan: str,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        generate_plots: bool = True,
        verbose: bool = False,
        despike_r1: bool = True,
        despike_window: Optional[int] = None,
        despike_threshold: Optional[float] = None,
        despike_max_iter: Optional[int] = None,
        despike_example_point: int = 50,
        baseline_r1: bool = True,
        baseline_lam: Optional[float] = None,
        baseline_asym: Optional[float] = None,
        baseline_iters: Optional[int] = None,
        background_file: Optional[Path] = None,
        bkg_scale_method: str = 'ppp',
        bkg_laser_roi: Optional[str] = None,
    ) -> ServiceResult:
        """Process a single SHERLOC scan through the complete preprocessing pipeline.
        
        This method orchestrates all preprocessing steps for a single scan:
        - Data ingestion and restructuring
        - Laser normalization
        - Optional R1 despiking
        - Optional R1 baseline correction
        - Statistical analysis (mean, trimmed mean)
        - Optional background subtraction on averages
        
        Args:
            sol: Sol number (e.g., "0921")
            target: Target name (e.g., "Amherst_Point")
            scan: Scan type (e.g., "detail_1", "line")
            data_dir: Base data directory containing sol_XXXX folders. If None, uses config default.
            results_dir: Results directory. If None, uses config default.
            generate_plots: Whether to generate PNG plots (CSV files always written)
            verbose: Enable verbose logging
            despike_r1: Apply R1 despiking
            despike_window: Despike window size (odd >=3). If None, uses config default.
            despike_threshold: Despike robust z-score threshold. If None, uses config default.
            despike_max_iter: Maximum despike iterations. If None, uses config default.
            despike_example_point: Point index for despike test plot
            baseline_r1: Apply asPLS baseline to R1 variants
            baseline_lam: asPLS lambda. If None, uses config default.
            baseline_asym: asPLS asymmetric coefficient. If None, uses config default.
            baseline_iters: asPLS max iterations. If None, uses config default.
            background_file: Background spectrum CSV for background subtraction.
                If None, uses config default.
            bkg_scale_method: Background scaling method: "ppp" (default) or "laser"
            bkg_laser_roi: Override laser ROI as 'lo,hi' (e.g., '600,700')
        
        Returns:
            ServiceResult with summary, artifacts, warnings, and metadata
            
        Raises:
            PreprocessingError: If any preprocessing step fails
            
        Example:
            >>> service = PreprocessingService()
            >>> result = service.run_scan(
            ...     sol="0921",
            ...     target="Amherst_Point",
            ...     scan="detail_1",
            ...     generate_plots=True
            ... )
            >>> print(result.summary)
            'Processed scan 0921/Amherst_Point/detail_1 successfully'
        """
        from sherloc_pipeline.core.data_ingestion import normalize_target_name
        target = normalize_target_name(target)

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        run_context = self.context.with_overrides(
            data_dir=data_dir,
            results_dir=results_dir,
        )
        cfg = run_context.config
        
        try:
            # Resolve scan context (validates paths, finds working directory)
            context = resolve_scan_context(
                sol=sol,
                target=target,
                scan=scan,
                data_dir=data_dir,
                results_dir=results_dir,
                context=run_context,
            )
        except (FileNotFoundError, ValueError) as e:
            error = PreprocessingError(
                f"Failed to resolve scan context: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan}
            )
            raise enrich(error, sol=sol, target=target, scan=scan)
        
        # Display header
        self.console.print(f"[bold blue]SHERLOC Pipeline - Single Scan Processing[/bold blue]")
        self.console.print(f"[yellow]Sol: {sol}, Target: {target}, Scan: {scan}[/yellow]")
        self.console.print(f"[blue]Data directory: {context.base_data_dir}[/blue]")
        
        try:
            # Import here to avoid circular imports
            from sherloc_pipeline.core.data_ingestion import DataIngestion
            from sherloc_pipeline.core.laser_normalization import process_laser_normalization
            
            # Initialize data ingestion
            ingestion = DataIngestion(
                base_data_dir=context.base_data_dir,
                results_dir=context.results_dir,
                sol=sol,
                target=target,
                scan=scan
            )
            
            working_dir = context.working_dir
            self.console.print(f"[green]Found working directory: {working_dir.name}[/green]")
            
            # Track artifacts and warnings
            artifacts: List[Path] = []
            warnings: List[str] = []
            metadata: Dict[str, Any] = {
                "sol": sol,
                "target": target,
                "scan": scan,
                "working_dir": str(working_dir),
                "results_path": str(context.results_path),
            }
            
            # Process the scan with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                
                # Step 1: Data Ingestion - Raw Spectra
                task1 = progress.add_task("Loading and restructuring raw spectra...", total=None)
                try:
                    raw_spectra = ingestion.process_raw_spectra(working_dir)
                    progress.update(task1, description="✅ Raw spectra processed")
                except Exception as e:
                    progress.update(task1, description="❌ Raw spectra processing failed")
                    error = PreprocessingError(
                        f"Failed to process raw spectra: {e}",
                        exit_code=1,
                        context={"working_dir": str(working_dir)}
                    )
                    raise enrich(error, sol=sol, target=target, scan=scan)
                
                # Step 2: Laser Normalization
                task2 = progress.add_task("Applying laser normalization...", total=None)
                try:
                    process_laser_normalization(
                        working_dir=working_dir,
                        sol=sol,
                        target=target,
                        scan=scan,
                        generate_plots=generate_plots,
                        results_dir=ingestion.results_dir
                    )
                    progress.update(task2, description="✅ Laser normalization completed")
                except Exception as e:
                    progress.update(task2, description="❌ Laser normalization failed")
                    error = PreprocessingError(
                        f"Failed to apply laser normalization: {e}",
                        exit_code=1,
                        context={"working_dir": str(working_dir)}
                    )
                    raise enrich(error, sol=sol, target=target, scan=scan)
                
                # Step 3: Data Ingestion - Normalized Spectra
                task3 = progress.add_task("Processing normalized spectra...", total=None)
                try:
                    normalized_spectra = ingestion.process_normalized_spectra(working_dir)
                    progress.update(task3, description="✅ Normalized spectra processed")
                except Exception as e:
                    progress.update(task3, description="❌ Normalized spectra processing failed")
                    error = PreprocessingError(
                        f"Failed to process normalized spectra: {e}",
                        exit_code=1,
                        context={"working_dir": str(working_dir)}
                    )
                    raise enrich(error, sol=sol, target=target, scan=scan)
                
                # Step 4: Optional R1 Despiking
                if despike_r1:
                    task3a = progress.add_task("Applying R1 despiking...", total=None)
                    try:
                        from sherloc_pipeline.core.preprocessing import (
                            DespikeParams, despike_r1_dataframe,
                        )
                        from sherloc_pipeline.visualization.preprocessing_plots import plot_r1_despike_verification_single
                        
                        # Extract despike config
                        pre = getattr(cfg, 'preprocessing', None)
                        dsp = getattr(pre, 'despike', None) if pre is not None else None
                        if isinstance(dsp, dict):
                            cfg_params = dsp
                        elif dsp is not None:
                            cfg_params = getattr(dsp, '__dict__', {})
                        else:
                            cfg_params = {}
                        
                        params = DespikeParams(
                            window_size=despike_window or cfg_params.get('window_size', 7),
                            zscore_threshold=despike_threshold or cfg_params.get('zscore_threshold', 6.0),
                            max_iterations=despike_max_iter or cfg_params.get('max_iterations', 1),
                            interpolation_method=cfg_params.get('interpolation_method', 'linear'),
                        )
                        
                        if 'R1' in normalized_spectra:
                            r1_df = normalized_spectra['R1']
                            # Export restructured raw normalized R1
                            ingestion.save_spectral_data(
                                spectra_df=r1_df,
                                target=target,
                                sol=sol,
                                scan=scan,
                                spectral_region="R1",
                                processing_level="normalized",
                            )
                            r1_despiked, spike_mask = despike_r1_dataframe(r1_df, params)
                            normalized_spectra['R1_despiked'] = r1_despiked
                            
                            # Generate verification plots
                            if generate_plots:
                                test_plots_dir = ingestion.get_results_path(target=target, sol=sol, scan=scan) / "test_plots"
                                test_plots_dir.mkdir(parents=True, exist_ok=True)
                                try:
                                    cols = [c for c in r1_df.columns if isinstance(c, (int, np.integer))]
                                    chosen_point = despike_example_point if despike_example_point in r1_df.columns else (cols[len(cols)//2] if cols else 0)
                                except Exception:
                                    chosen_point = 0
                                plot_r1_despike_verification_single(
                                    r1_df=r1_df,
                                    r1_despiked_df=r1_despiked,
                                    spike_mask_df=spike_mask,
                                    output_dir=test_plots_dir,
                                    sol=sol,
                                    target=target,
                                    scan=scan,
                                    point=chosen_point,
                                    threshold=params.zscore_threshold,
                                )
                            
                            # Export despiked spectra CSV
                            ingestion.save_spectral_data(
                                spectra_df=r1_despiked,
                                target=target,
                                sol=sol,
                                scan=scan,
                                spectral_region="R1",
                                processing_level="normalized_despiked",
                            )
                        progress.update(task3a, description="✅ R1 despiking applied")
                    except Exception as e:
                        progress.update(task3a, description="❌ R1 despiking failed")
                        error = PreprocessingError(
                            f"Failed to apply R1 despiking: {e}",
                            exit_code=1,
                            context={"working_dir": str(working_dir)}
                        )
                        raise enrich(error, sol=sol, target=target, scan=scan)
                
                # Step 5: Optional R1 Baseline Correction
                if baseline_r1:
                    task4 = progress.add_task("Applying R1 baseline correction...", total=None)
                    try:
                        from sherloc_pipeline.core.preprocessing import (
                            BaselineParams, baseline_r1_dataframe,
                            build_weight_vector_from_windows,
                        )
                        from sherloc_pipeline.visualization.preprocessing_plots import plot_r1_baseline_verification_single
                        
                        # Extract baseline config
                        pre = getattr(cfg, 'preprocessing', None)
                        bl = pre.get('baseline', None) if isinstance(pre, dict) else getattr(pre, 'baseline', None) if pre is not None else None
                        if isinstance(bl, dict):
                            lam_def = bl.get('lam', 1e6)
                            asym_def = bl.get('asymmetric_coef', 0.01)
                            iters_def = bl.get('iters', 10)
                            diff_def = bl.get('diff_order', 2)
                            tol_def = bl.get('tol', 1e-3)
                        else:
                            lam_def = getattr(bl, 'lam', 1e6) if bl else 1e6
                            asym_def = getattr(bl, 'asymmetric_coef', 0.01) if bl else 0.01
                            iters_def = getattr(bl, 'iters', 10) if bl else 10
                            diff_def = getattr(bl, 'diff_order', 2) if bl else 2
                            tol_def = getattr(bl, 'tol', 1e-3) if bl else 1e-3
                        
                        bl_params = BaselineParams(
                            lam=baseline_lam or lam_def,
                            asymmetric_coef=baseline_asym or asym_def,
                            iters=baseline_iters or iters_def,
                            diff_order=diff_def,
                            tol=tol_def,
                        )
                        
                        # Get keep windows from config
                        try:
                            pre_cfg = getattr(ingestion.config, 'preprocessing', None)
                            bl_cfg = pre_cfg.get('baseline', {}) if isinstance(pre_cfg, dict) else getattr(pre_cfg, 'baseline', {}) or {}
                            if not isinstance(bl_cfg, dict):
                                bl_cfg = {}
                            kw = bl_cfg.get('keep_windows', [])
                            keep_windows = [tuple(map(float, w)) for w in kw] if kw else [(600.0, 1130.0), (1300.0, 1720.0), (3000.0, 3800.0)]
                            keep_weight = float(bl_cfg.get('keep_weight', 0.01))
                        except Exception:
                            keep_windows = [(600.0, 1130.0), (1300.0, 1720.0), (3000.0, 3800.0)]
                            keep_weight = 0.01
                        
                        # R1 normalized baseline
                        if 'R1' in normalized_spectra:
                            r1_norm = normalized_spectra['R1']
                            weights = build_weight_vector_from_windows(r1_norm['raman_shift'].values, keep_windows, 1.0, keep_weight)
                            r1_corr, r1_base = baseline_r1_dataframe(r1_norm, bl_params, weights=weights)
                            ingestion.save_spectral_data(
                                spectra_df=r1_corr,
                                target=target, sol=sol, scan=scan, spectral_region="R1", processing_level="normalized_baselined",
                            )
                            
                            # Baseline verification plots
                            if generate_plots:
                                test_plots_dir = ingestion.get_results_path(target=target, sol=sol, scan=scan) / "test_plots"
                                plot_r1_baseline_verification_single(
                                    r1_df=r1_norm,
                                    r1_corrected_df=r1_corr,
                                    r1_baseline_df=r1_base,
                                    output_dir=test_plots_dir,
                                    sol=sol, target=target, scan=scan,
                                    point=despike_example_point,
                                    variant="raw",
                                )
                        
                        # R1 despiked baseline (if present)
                        if 'R1_despiked' in normalized_spectra:
                            r1_dsp = normalized_spectra['R1_despiked']
                            weights = build_weight_vector_from_windows(r1_dsp['raman_shift'].values, keep_windows, 1.0, keep_weight)
                            r1d_corr, r1d_base = baseline_r1_dataframe(r1_dsp, bl_params, weights=weights)
                            ingestion.save_spectral_data(
                                spectra_df=r1d_corr,
                                target=target, sol=sol, scan=scan, spectral_region="R1", processing_level="normalized_despiked_baselined",
                            )
                            
                            # Baseline verification plots
                            if generate_plots:
                                test_plots_dir = ingestion.get_results_path(target=target, sol=sol, scan=scan) / "test_plots"
                                plot_r1_baseline_verification_single(
                                    r1_df=r1_dsp,
                                    r1_corrected_df=r1d_corr,
                                    r1_baseline_df=r1d_base,
                                    output_dir=test_plots_dir,
                                    sol=sol, target=target, scan=scan,
                                    point=despike_example_point,
                                    variant="despiked",
                                )
                        
                        progress.update(task4, description="✅ R1 baseline correction completed")
                    except Exception as e:
                        progress.update(task4, description="❌ R1 baseline correction failed")
                        error = PreprocessingError(
                            f"Failed to apply R1 baseline correction: {e}",
                            exit_code=1,
                            context={"working_dir": str(working_dir)}
                        )
                        raise enrich(error, sol=sol, target=target, scan=scan)
                
                # Step 6: Statistical Analysis
                task5 = progress.add_task("Generating statistical analysis...", total=None)
                try:
                    results = ingestion.generate_statistical_analysis(sol, target, scan, working_dir=working_dir)
                    progress.update(task5, description="✅ Statistical analysis completed")
                    metadata["statistical_analysis"] = results
                except Exception as e:
                    progress.update(task5, description="❌ Statistical analysis failed")
                    error = PreprocessingError(
                        f"Failed to generate statistical analysis: {e}",
                        exit_code=1,
                        context={"working_dir": str(working_dir)}
                    )
                    raise enrich(error, sol=sol, target=target, scan=scan)
                
                # Step 7: Background Subtraction on Averages
                task6 = progress.add_task("Applying background subtraction on averages...", total=None)
                try:
                    self._process_background_subtraction(
                        ingestion=ingestion,
                        working_dir=working_dir,
                        sol=sol,
                        target=target,
                        scan=scan,
                        background_file=background_file,
                        bkg_scale_method=bkg_scale_method,
                        bkg_laser_roi=bkg_laser_roi,
                        generate_plots=generate_plots,
                        despike_example_point=despike_example_point,
                        config=cfg,
                    )
                    progress.update(task6, description="✅ Background subtraction completed")
                except Exception as e:
                    logger.exception("Background subtraction failed")
                    progress.update(task6, description=f"⚠️ Background subtraction skipped: {e}")
                    warnings.append(f"Background subtraction skipped: {e}")
            
            # Collect artifacts from results directory
            results_path = context.results_path
            if results_path.exists():
                for csv_file in results_path.rglob("*.csv"):
                    artifacts.append(csv_file)
                if generate_plots:
                    for png_file in results_path.rglob("*.png"):
                        artifacts.append(png_file)
            
            # Build summary
            summary = f"Processed scan {sol}/{target}/{scan} successfully"
            if results:
                file_summary = ", ".join([f"{k}: {v} files" for k, v in results.items()])
                summary += f" ({file_summary})"
            
            return ServiceResult(
                summary=summary,
                artifacts=artifacts,
                warnings=warnings,
                metadata=metadata,
            )
            
        except PreprocessingError:
            # Re-raise PreprocessingError as-is (already enriched)
            raise
        except Exception as e:
            error = PreprocessingError(
                f"Unexpected error during preprocessing: {e}",
                exit_code=1,
                context={"working_dir": str(working_dir) if 'working_dir' in locals() else "unknown"}
            )
            raise enrich(error, sol=sol, target=target, scan=scan)
    
    def run_batch(
        self,
        data_dir: Path,
        results_dir: Optional[Path] = None,
        generate_plots: bool = True,
        verbose: bool = False,
    ) -> ServiceResult:
        """Process multiple SHERLOC scans in batch mode.
        
        This method automatically discovers all available scans across multiple sols
        and processes them sequentially using run_scan().
        
        Args:
            data_dir: Base data directory containing sol_XXXX folders
            results_dir: Results directory. If None, uses config default.
            generate_plots: Whether to generate PNG plots
            verbose: Enable verbose logging
        
        Returns:
            ServiceResult with summary, artifacts, warnings, and metadata
            
        Raises:
            PreprocessingError: If batch processing fails
            
        Example:
            >>> service = PreprocessingService()
            >>> result = service.run_batch(Path("../data/loupe"))
            >>> print(f"Processed {result.metadata['scans_processed']} scans")
        """
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        run_context = self.context.with_overrides(
            data_dir=data_dir,
            results_dir=results_dir,
        )
        resolved_data_dir = run_context.data_root
        resolved_results_dir = run_context.results_root
        
        self.console.print(f"[bold blue]SHERLOC Pipeline - Batch Processing[/bold blue]")
        self.console.print(f"[blue]Data directory: {resolved_data_dir}[/blue]")
        
        try:
            from sherloc_pipeline.core.data_ingestion import DataIngestion
            
            # Initialize data ingestion for batch processing
            ingestion = DataIngestion(
                base_data_dir=resolved_data_dir,
                results_dir=resolved_results_dir
            )
            
            # Discover all available scans
            all_scans = ingestion.discover_all_scans()
            if not all_scans:
                error = PreprocessingError(
                    f"No scans found in {resolved_data_dir}",
                    exit_code=1,
                    context={"data_dir": str(resolved_data_dir)}
                )
                raise error
            
            self.console.print(f"[green]Found {len(all_scans)} scans to process[/green]")
            
            # Track batch results
            all_artifacts: List[Path] = []
            all_warnings: List[str] = []
            successful_scans: List[Dict[str, str]] = []
            failed_scans: List[Dict[str, str]] = []
            
            # Process each scan
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                
                for i, (sol, target, scan, working_dir_path) in enumerate(all_scans, 1):
                    task = progress.add_task(f"Processing {sol}/{target}/{scan} ({i}/{len(all_scans)})...", total=None)
                    
                    try:
                        # Process this scan using run_scan
                        scan_result = self.run_scan(
                            sol=sol,
                            target=target,
                            scan=scan,
                            data_dir=resolved_data_dir,
                            results_dir=resolved_results_dir,
                            generate_plots=generate_plots,
                            verbose=verbose,
                        )
                        
                        all_artifacts.extend(scan_result.artifacts)
                        all_warnings.extend(scan_result.warnings)
                        successful_scans.append({
                            "sol": sol,
                            "target": target,
                            "scan": scan,
                        })
                        progress.update(task, description=f"✅ {sol}/{target}/{scan} completed")
                    except PreprocessingError as e:
                        failed_scans.append({
                            "sol": sol,
                            "target": target,
                            "scan": scan,
                            "error": str(e),
                        })
                        all_warnings.append(f"{sol}/{target}/{scan} failed: {e.message}")
                        progress.update(task, description=f"❌ {sol}/{target}/{scan} failed: {e.message}")
                        if verbose:
                            self.console.print(f"[red]Error processing {sol}/{target}/{scan}: {e.message}[/red]")
            
            # Build summary
            summary = f"Batch processing completed: {len(successful_scans)} successful"
            if failed_scans:
                summary += f", {len(failed_scans)} failed"
            
            metadata = {
                "scans_processed": len(successful_scans),
                "scans_failed": len(failed_scans),
                "successful_scans": successful_scans,
                "failed_scans": failed_scans,
            }
            
            return ServiceResult(
                summary=summary,
                artifacts=all_artifacts,
                warnings=all_warnings,
                metadata=metadata,
            )
            
        except PreprocessingError:
            raise
        except Exception as e:
            error = PreprocessingError(
                f"Unexpected error during batch processing: {e}",
                exit_code=1,
                context={"data_dir": str(data_dir)}
            )
            raise error
    
    def _process_background_subtraction(
        self,
        ingestion,
        working_dir: Path,
        sol: str,
        target: str,
        scan: str,
        background_file: Optional[Path],
        bkg_scale_method: str,
        bkg_laser_roi: Optional[str],
        generate_plots: bool,
        despike_example_point: int,
        *,
        config: Optional[Any] = None,
    ) -> None:
        """Process background subtraction on averaged spectra.
        
        This is a private helper method that handles the complex background
        subtraction logic, including baseline-only processing and optional
        background subtraction with scaling.
        """
        config = config or self.context.config
        cfg = config
        
        # Resolve background path
        background = None
        bg_path = None
        if background_file is not None:
            background = background_file
            bg_path = str(background_file)
        else:
            # Resolve default background path from config
            try:
                pre = getattr(cfg, 'preprocessing', None)
                if isinstance(pre, dict):
                    bs = pre.get('background_subtraction', {})
                    bg_path = bs.get('default_file', None)
                else:
                    bs = getattr(pre, 'background_subtraction', None)
                    if isinstance(bs, dict):
                        bg_path = bs.get('default_file', None)
                    else:
                        bg_path = getattr(bs, 'default_file', None) if bs else None
            except Exception:
                bg_path = None
            
            if bg_path:
                p = Path(str(bg_path))
                try:
                    if not p.is_absolute():
                        if not p.exists():
                            p_cwd = (Path.cwd() / p).resolve()
                            if p_cwd.exists():
                                p = p_cwd
                            else:
                                try:
                                    repo_root = Path(__file__).resolve().parents[4]
                                    p_repo = (repo_root / p).resolve()
                                    if p_repo.exists():
                                        p = p_repo
                                except Exception:
                                    pass
                except Exception:
                    pass
                background = p
        
        # Report background path
        try:
            exists = (background is not None) and Path(background).exists()
            self.console.print(f"[blue]Background file:[/blue] {background if background is not None else 'None'} (exists={exists})")
        except Exception:
            pass
        
        # Always baseline the R1 averaged spectra
        try:
            target_dir = ingestion.get_results_path(target=target, sol=sol, scan=scan)
            test_plots_dir = ingestion.get_results_path(target=target, sol=sol, scan=scan) / "test_plots"
            
            # Load baselining params from config
            pre2 = getattr(ingestion.config, 'preprocessing', None)
            bl2 = pre2.get('baseline', None) if isinstance(pre2, dict) else getattr(pre2, 'baseline', None) if pre2 is not None else None
            if isinstance(bl2, dict):
                lam2 = bl2.get('lam', 1e6)
                asym2 = bl2.get('asymmetric_coef', 0.01)
                it2 = bl2.get('iters', 10)
                diff2 = bl2.get('diff_order', 2)
                tol2 = bl2.get('tol', 1e-3)
            else:
                lam2 = getattr(bl2, 'lam', 1e6) if bl2 else 1e6
                asym2 = getattr(bl2, 'asymmetric_coef', 0.01) if bl2 else 0.01
                it2 = getattr(bl2, 'iters', 10) if bl2 else 10
                diff2 = getattr(bl2, 'diff_order', 2) if bl2 else 2
                tol2 = getattr(bl2, 'tol', 1e-3) if bl2 else 1e-3
            
            from sherloc_pipeline.core.preprocessing import (
                baseline_aspls,
                _baseline_aspls_with_weights,
                build_weight_vector_from_windows,
            )
            from sherloc_pipeline.visualization.preprocessing_plots import (
                plot_average_bkgsub_baseline, plot_average_baseline,
                plot_average_bkgsub_comparison, plot_average_single_series,
            )
            from sherloc_pipeline.core.baseline import BaselineParams
            
            blp = BaselineParams(lam=lam2, asymmetric_coef=asym2, iters=it2, diff_order=diff2, tol=tol2)

            # Resolve the actual trim-mean filename token for this scan
            # (may differ from configured 2% due to dynamic trim on small scans)
            import re as _re_mod
            _scan_dir = ingestion.get_results_path(target=target, sol=sol, scan=scan)
            _tm_matches = sorted(_scan_dir.glob(
                f"{sol}_{target}_{scan}_R1_raw-n_*p_trim_mean.csv"
            ))
            if _tm_matches:
                _m = _re_mod.search(r'(\d+(?:\.\d+)?p_trim_mean)', _tm_matches[0].name)
                trim_token = _m.group(1) if _m else '2p_trim_mean'
            else:
                trim_token = '2p_trim_mean'

            for avg_kind in ["mean", "trimmed_mean"]:
                avg_path = ingestion.get_results_path(
                    target=target,
                    sol=sol,
                    scan=scan,
                    spectral_region="R1",
                    processing_level=f"raw-n_{trim_token if avg_kind == 'trimmed_mean' else 'mean'}",
                    file_extension=".csv",
                )
                if not avg_path.exists():
                    continue
                
                avg = pd.read_csv(avg_path)
                x = avg.iloc[:, 0].values
                y = avg["intensity"].values
                
                # Get keep windows from config
                try:
                    pre_cfg = getattr(ingestion.config, 'preprocessing', None)
                    if isinstance(pre_cfg, dict):
                        bl_cfg = pre_cfg.get('baseline', {}) or {}
                    else:
                        bl_cfg = getattr(pre_cfg, 'baseline', {}) or {}
                        if not isinstance(bl_cfg, dict):
                            bl_cfg = {}
                    kw = bl_cfg.get('keep_windows', [])
                    keep_windows = [tuple(map(float, w)) for w in kw] if kw else [(600.0, 1130.0), (1300.0, 1720.0), (3000.0, 3800.0)]
                    keep_weight = float(bl_cfg.get('keep_weight', 0.01))
                except Exception:
                    keep_windows = [(600.0, 1130.0), (1300.0, 1720.0), (3000.0, 3800.0)]
                    keep_weight = 0.01
                
                weights = build_weight_vector_from_windows(
                    x, keep_windows=keep_windows, default_weight=1.0, keep_weight=keep_weight
                )
                avg_corr, avg_base = _baseline_aspls_with_weights(pd.Series(y), blp, weights)
                
                # Save baseline-only CSV
                bl_only_out = pd.DataFrame({avg.columns[0]: x, 'intensity': avg_corr.values})
                bl_only_out_path = ingestion.get_results_path(
                    target=target,
                    sol=sol,
                    scan=scan,
                    spectral_region="R1",
                    processing_level=f"raw-n_{trim_token if avg_kind == 'trimmed_mean' else 'mean'}_baselined",
                    file_extension=".csv",
                )
                bl_only_out.to_csv(bl_only_out_path, index=False)
                
                # Plot baseline-only overlay
                bl_only_png_path = test_plots_dir / f"{sol}_{target}_{scan}_R1_raw-n_{trim_token if avg_kind == 'trimmed_mean' else 'mean'}_baseline_only_overlay.png"
                plot_average_baseline(
                    x_axis=x,
                    avg_raw=y,
                    baseline=avg_base.values,
                    avg_baselined=avg_corr.values,
                    output_path=bl_only_png_path,
                    title=f"sol {sol} {target} {scan} R1 raw-n {trim_token if avg_kind == 'trimmed_mean' else 'mean'} baseline-only",
                )
                
                # Write companion PNG next to CSV
                bl_only_main_png = ingestion.get_results_path(
                    target=target,
                    sol=sol,
                    scan=scan,
                    spectral_region="R1",
                    processing_level=f"raw-n_{trim_token if avg_kind == 'trimmed_mean' else 'mean'}_baselined",
                    file_extension=".png",
                )
                plot_average_single_series(
                    x_axis=x,
                    y_series=avg_corr.values,
                    output_path=bl_only_main_png,
                    title=f"sol {sol} {target} {scan} R1 raw-n {trim_token if avg_kind == 'trimmed_mean' else 'mean'} baseline-only",
                    color="#2ca02c",
                )
        except Exception:
            # Non-fatal; continue to optional background subtraction
            pass
        
        # Optional background subtraction
        if background is not None:
            try:
                bg_path_coerced = Path(str(background))
                if not bg_path_coerced.exists():
                    raise FileNotFoundError(f"Background path not found: {bg_path_coerced}")
                bg_df = pd.read_csv(str(bg_path_coerced))
            except Exception as e:
                logger.exception("Failed to open background file")
                raise
            
            # Expect columns: raman_shift/intensity
            x_bg = bg_df.iloc[:, 0].values
            y_bg = bg_df.iloc[:, 1].values
            y_bg_scaled = y_bg
            
            # Scale background by pulses-per-point ratio
            try:
                shots_per_spec = self._read_shots_per_spec(working_dir / "loupe.csv")
                cfg_bg_ppp = self._config_background_ppp(cfg)
                ppp_scale = self._compute_scale(shots_per_spec, cfg_bg_ppp)
                y_bg_scaled = y_bg * ppp_scale
                bg_title_suffix = f" (PPP scaled {ppp_scale:.3f}x)" if ppp_scale != 1.0 else ""
            except Exception:
                ppp_scale = 1.0
                bg_title_suffix = ""
            
            # Save a plot of the background
            from sherloc_pipeline.visualization.preprocessing_plots import plot_average_single_series
            import pathlib
            bg_plot_path = pathlib.Path(background).with_suffix('.png')
            plot_average_single_series(
                x_axis=x_bg,
                y_series=y_bg_scaled,
                output_path=bg_plot_path,
                title=f"Background spectrum: {pathlib.Path(background).name}{bg_title_suffix}",
            )
            
            # Process each average kind
            target_dir = ingestion.get_results_path(target=target, sol=sol, scan=scan)
            test_plots_dir = ingestion.get_results_path(target=target, sol=sol, scan=scan) / "test_plots"
            
            for avg_kind in ["mean", "trimmed_mean"]:
                avg_path = ingestion.get_results_path(
                    target=target,
                    sol=sol,
                    scan=scan,
                    spectral_region="R1",
                    processing_level=f"raw-n_{trim_token if avg_kind == 'trimmed_mean' else 'mean'}",
                    file_extension=".csv",
                )
                if not avg_path.exists():
                    continue
                
                avg = pd.read_csv(avg_path)
                x = avg.iloc[:, 0].values
                y = avg["intensity"].values
                
                if x_bg is not None and y_bg is not None:
                    # Interpolate background onto x
                    roi = (600.0, 700.0)
                    clip = (0.1, 5.0)
                    try:
                        pre_cfg = getattr(ingestion.config, 'preprocessing', None)
                        bs_cfg = None
                        if isinstance(pre_cfg, dict):
                            bs_cfg = pre_cfg.get('background_subtraction', {}) or {}
                        else:
                            bs_cfg = getattr(pre_cfg, 'background_subtraction', {}) or {}
                            if not isinstance(bs_cfg, dict):
                                bs_cfg = {}
                        cfg_ls = bs_cfg.get('laser_scale', {}) if isinstance(bs_cfg, dict) else {}
                        if isinstance(cfg_ls, dict):
                            r = cfg_ls.get('roi')
                            if isinstance(r, (list, tuple)) and len(r) == 2:
                                roi = (float(r[0]), float(r[1]))
                            c = cfg_ls.get('clip')
                            if isinstance(c, (list, tuple)) and len(c) == 2:
                                clip = (float(c[0]), float(c[1]))
                    except Exception:
                        pass
                    
                    if bkg_laser_roi:
                        try:
                            lo, hi = [float(v) for v in str(bkg_laser_roi).split(',')]
                            roi = (lo, hi)
                        except Exception:
                            logger.warning("Invalid --bkg-laser-roi, expected 'lo,hi'")
                    
                    amp_scan = self._compute_laser_peak_amplitude(x, y, roi=roi)
                    amp_bg = self._compute_laser_peak_amplitude(x_bg, y_bg, roi=roi)
                    
                    # Method selection
                    scale_method = (bkg_scale_method or 'ppp').lower()
                    if scale_method not in ("laser", "ppp"):
                        scale_method = "ppp"
                    scale_used = ppp_scale
                    method_used = "ppp"
                    if scale_method == "laser":
                        if amp_scan is not None and amp_bg is not None and amp_bg > 0:
                            scale_used = float(amp_scan) / float(amp_bg)
                            scale_used = max(clip[0], min(clip[1], scale_used))
                            method_used = "laser"
                        else:
                            logger.warning("Laser scaling requested but amplitudes unavailable; using PPP")
                    
                    logger.info(f"Background scaling method={method_used} roi={roi} scale={scale_used:.4f} (amp_scan={amp_scan}, amp_bg={amp_bg}, ppp_scale={ppp_scale})")
                    
                    y_bg_interp = np.interp(x, x_bg, y_bg * scale_used)
                    y_bkgsub = y - y_bg_interp
                    
                    # Comparative overlay
                    try:
                        y_bg_unscaled_interp = np.interp(x, x_bg, y_bg)
                        cmp_path = test_plots_dir / f"{sol}_{target}_{scan}_R1_raw-n_{trim_token if avg_kind == 'trimmed_mean' else 'mean'}_bkgsub_comparison.png"
                        plot_average_bkgsub_comparison(
                            x_axis=x,
                            avg_raw=y,
                            bg_unscaled_interp=y_bg_unscaled_interp,
                            scale_used=scale_used,
                            output_path=cmp_path,
                            title=f"bkgsub comparison: scaled (s={scale_used:.3f}, {method_used}) vs unscaled",
                        )
                    except Exception as e:
                        logger.warning(f"failed to write bkgsub comparison overlay: {e}")
                    
                    # Save background-subtracted (pre-baseline) average
                    bkgsub_out_path = ingestion.get_results_path(
                        target=target,
                        sol=sol,
                        scan=scan,
                        spectral_region="R1",
                        processing_level=f"raw-n_{trim_token if avg_kind == 'trimmed_mean' else 'mean'}_bkgsub",
                        file_extension=".csv",
                    )
                    pd.DataFrame({avg.columns[0]: x, 'intensity': y_bkgsub}).to_csv(bkgsub_out_path, index=False)
                    
                    try:
                        self.console.print(f"[green]Wrote:[/green] {bkgsub_out_path}")
                    except Exception:
                        pass
                    
                    # Baseline after background subtraction
                    try:
                        pre_cfg2 = getattr(ingestion.config, 'preprocessing', None)
                        if isinstance(pre_cfg2, dict):
                            bl_cfg2 = pre_cfg2.get('baseline', {}) or {}
                        else:
                            bl_cfg2 = getattr(pre_cfg2, 'baseline', {}) or {}
                            if not isinstance(bl_cfg2, dict):
                                bl_cfg2 = {}
                        kw2 = bl_cfg2.get('keep_windows', [])
                        keep_windows = [tuple(map(float, w)) for w in kw2] if kw2 else [(600.0, 1130.0), (1300.0, 1720.0), (3000.0, 3800.0)]
                        keep_weight = float(bl_cfg2.get('keep_weight', 0.01))
                    except Exception:
                        keep_windows = [(600.0, 1130.0), (1300.0, 1720.0), (3000.0, 3800.0)]
                        keep_weight = 0.01
                    
                    weights = build_weight_vector_from_windows(
                        x, keep_windows=keep_windows, default_weight=1.0, keep_weight=keep_weight
                    )
                    corr, base = _baseline_aspls_with_weights(pd.Series(y_bkgsub), blp, weights)
                    out = pd.DataFrame({avg.columns[0]: x, 'intensity': corr.values})
                    out_path = ingestion.get_results_path(
                        target=target,
                        sol=sol,
                        scan=scan,
                        spectral_region="R1",
                        processing_level=f"raw-n_{trim_token if avg_kind == 'trimmed_mean' else 'mean'}_bkgsub_baselined",
                        file_extension=".csv",
                    )
                    out.to_csv(out_path, index=False)
                    
                    # Individual plots
                    from matplotlib import pyplot as plt
                    
                    # bkgsub only
                    bkgsub_png_path = ingestion.get_results_path(
                        target=target,
                        sol=sol,
                        scan=scan,
                        spectral_region="R1",
                        processing_level=f"raw-n_{trim_token if avg_kind == 'trimmed_mean' else 'mean'}_bkgsub",
                        file_extension=".png",
                    )
                    plt.figure(figsize=(12, 6))
                    plt.plot(x, y_bkgsub, color="#7f7f7f", linewidth=1.5)
                    plt.xlabel("Raman Shift (cm⁻¹)")
                    plt.ylabel("Intensity (counts)")
                    plt.title(f"sol {sol} {target} {scan} R1 raw-n {trim_token if avg_kind == 'trimmed_mean' else 'mean'} bkgsub")
                    plt.grid(True, alpha=0.3)
                    try:
                        x0 = float(x[0])
                    except Exception:
                        x0 = 0.0
                    plt.xlim([x0, 4000])
                    bkgsub_png_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(bkgsub_png_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    
                    # bkgsub + baseline
                    bkgsub_bl_png_path = ingestion.get_results_path(
                        target=target,
                        sol=sol,
                        scan=scan,
                        spectral_region="R1",
                        processing_level=f"raw-n_{trim_token if avg_kind == 'trimmed_mean' else 'mean'}_bkgsub_baselined",
                        file_extension=".png",
                    )
                    plt.figure(figsize=(12, 6))
                    plt.plot(x, corr.values, color="#2ca02c", linewidth=1.5)
                    plt.xlabel("Raman Shift (cm⁻¹)")
                    plt.ylabel("Intensity (counts)")
                    plt.title(f"sol {sol} {target} {scan} R1 raw-n {trim_token if avg_kind == 'trimmed_mean' else 'mean'} bkgsub + baseline")
                    plt.grid(True, alpha=0.3)
                    try:
                        x0 = float(x[0])
                    except Exception:
                        x0 = 0.0
                    plt.xlim([x0, 4000])
                    bkgsub_bl_png_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(bkgsub_bl_png_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    
                    # Composite verification plot
                    plot_dir = ingestion.get_results_path(target=target, sol=sol, scan=scan) / "test_plots"
                    plot_path = plot_dir / f"{sol}_{target}_{scan}_R1_raw-n_{trim_token if avg_kind == 'trimmed_mean' else 'mean'}_bkgsub_baseline_test.png"
                    plot_average_bkgsub_baseline(
                        x_axis=x,
                        avg_raw=avg['intensity'].values,
                        bg_interp=y_bg_interp,
                        avg_bkgsub=y_bkgsub,
                        avg_bkgsub_baselined=corr.values,
                        output_path=plot_path,
                        title=f"sol {sol} {target} {scan} R1 raw-n {trim_token if avg_kind == 'trimmed_mean' else 'mean'} bkgsub + baseline",
                    )
    
    def _read_shots_per_spec(self, loupe_csv: Path) -> Optional[float]:
        """Read shots_per_spec from a loupe.csv file. Returns None if unavailable."""
        try:
            if loupe_csv.exists():
                import csv
                with open(loupe_csv, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2 and str(row[0]).strip() == 'shots_per_spec':
                            try:
                                return float(str(row[1]).split()[0])
                            except Exception:
                                return None
        except Exception:
            return None
        return None
    
    def _config_background_ppp(self, cfg) -> Optional[float]:
        """Return pulses-per-point for background from config if set, else None."""
        try:
            pre = getattr(cfg, 'preprocessing', {})
            if isinstance(pre, dict):
                bs = pre.get('background_subtraction', {})
                ppp = bs.get('pulses_per_point', None)
            else:
                bs = getattr(pre, 'background_subtraction', {})
                ppp = getattr(bs, 'pulses_per_point', None)
            if ppp is None:
                return None
            return float(ppp)
        except Exception:
            return None
    
    def _compute_scale(self, shots_per_spec: Optional[float], bg_ppp: Optional[float]) -> float:
        """Compute scaling factor shots/bg_ppp when both present; otherwise 1.0."""
        try:
            if shots_per_spec and bg_ppp and bg_ppp > 0:
                return float(shots_per_spec) / float(bg_ppp)
        except Exception:
            pass
        return 1.0
    
    def _compute_laser_peak_amplitude(self, x_vals, y_vals, roi=(600.0, 700.0)) -> Optional[float]:
        """Estimate laser-line amplitude in ROI using a robust baseline (median) and peak height.
        
        Returns None if ROI invalid or no positive peak over baseline.
        """
        try:
            x_arr = np.asarray(x_vals)
            y_arr = np.asarray(y_vals)
            mask = (x_arr >= float(roi[0])) & (x_arr <= float(roi[1]))
            if mask.sum() < 3:
                return None
            y_roi = y_arr[mask]
            baseline = float(np.median(y_roi))
            amp = float(np.max(y_roi) - baseline)
            if not np.isfinite(amp) or amp <= 0:
                return None
            return amp
        except Exception:
            return None

