"""CLI application for SHERLOC pipeline.

Provides commands:
    - init: Bootstrap SHERLOC_HOME and run database migrations
    - full-pipeline: Run complete processing workflow
    - apply-review: Apply manual review decisions and regenerate overlays
    - plot: Generate spectral plots from Loupe data or pipeline outputs
    - fit-fluor: Fit fluorescence spectra and persist peaks
    - persist-peaks: Persist Raman peak CSVs to database by domain
    - ingest: Ingest Loupe data into PHASE SQLite database
    - process-new: Ingest and process a new sol (unzip -> ingest -> pipeline)
    - pds-download: Download SHERLOC products from the PDS Geosciences Node
    - pds-ingest: Ingest PDS4 data into phase_pds.db
    - db-stats: Show database statistics
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
import sys
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from sherloc_pipeline.services.pipeline import PipelineService
from sherloc_pipeline.services.review import ReviewService
from sherloc_pipeline.services.spectral import SpectralService, SpectralPlotRequest
from sherloc_pipeline.services.runtime import RuntimeContext
from sherloc_pipeline.services.errors import SherlocServiceError
from sherloc_pipeline.services.ingestion import IngestionService, IngestionError
from sherloc_pipeline.services.pds_ingestion import PDSIngestionService, PDSIngestionError
from sherloc_pipeline.services.pixl_ingestion import PixliseIngestionService, PixliseIngestionError
from sherloc_pipeline.services.pds_ingestion import PDS_DEFAULT_CACHE_DIR
from sherloc_pipeline.config import get_config
from sherloc_pipeline.models.schemas.cli import CLIResult, CLIError
from sherloc_pipeline import __version__ as _pipeline_version

app = typer.Typer(
    name="sherloc",
    help="SHERLOC Mars 2020 Raman/Fluorescence Data Processing Pipeline",
    add_completion=False,
)

console = Console()


@app.callback()
def _main_callback(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output structured JSON to stdout"),
) -> None:
    """SHERLOC pipeline CLI."""
    ctx.ensure_object(dict)
    ctx.obj["json"] = json_output


def _apply_trim_pct_override(trim_pct: Optional[float]) -> None:
    """Override config trim_mean_baseline_pct from CLI --trim-pct flag."""
    if trim_pct is None:
        return
    if trim_pct < 0 or trim_pct > 50:
        console.print(f"[red]--trim-pct must be between 0 and 50, got {trim_pct}[/red]")
        raise typer.Exit(code=1)
    get_config().preprocessing['trim_mean_baseline_pct'] = trim_pct / 100.0
    console.print(f"[cyan]Trim override: {trim_pct}% per tail[/cyan]")


def _apply_model_selection_override(model_selection: Optional[str]) -> None:
    """Override config parsimony.model_selection from CLI --model-selection flag.

    Default (config.yaml): "aicc". Override to "ftest" for sequential
    F-test peak-count selection. See PUBLIC_TOOLKIT_ARCHITECTURE_SPEC and
    fitting.py for the trade-off.
    """
    if model_selection is None:
        return
    accepted = {"aicc", "ftest"}
    if model_selection not in accepted:
        console.print(
            f"[red]--model-selection must be one of {sorted(accepted)}, "
            f"got {model_selection!r}[/red]"
        )
        raise typer.Exit(code=1)
    fitting_cfg = get_config().fitting
    parsimony = fitting_cfg.setdefault("parsimony", {})
    parsimony["model_selection"] = model_selection
    console.print(f"[cyan]Model selection override: {model_selection}[/cyan]")


@app.command("full-pipeline")
def full_pipeline(
    ctx: typer.Context,
    sol: str = typer.Argument(..., help="Sol number (e.g., 0921)"),
    target: str = typer.Argument(..., help="Target name (e.g., Amherst_Point)"),
    scan: str = typer.Argument(..., help="Scan type (e.g., detail_1)"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Data directory (defaults to config)"),
    results_dir: Optional[Path] = typer.Option(None, "--results-dir", help="Results directory (defaults to config)"),
    trim_pct: Optional[float] = typer.Option(
        None, "--trim-pct",
        help="Override trim %% per tail (e.g., 8 for 8%%). Default: config.yaml (2%%).",
    ),
    model_selection: Optional[str] = typer.Option(
        None, "--model-selection",
        help="Peak-count model selection: 'aicc' or 'ftest'. Default: config.yaml (aicc).",
    ),
):
    """Run the complete SHERLOC processing pipeline.

    Workflow:
        1. Preprocessing (despike → baseline → background)
        2. Fitting (minerals, organics, hydration)
        3. Generate review tables
        4. Fit label averages
        5. Render spatial overlays

    Examples:
        sherloc full-pipeline 0921 Amherst_Point detail_1
        sherloc full-pipeline 0921 Amherst_Point detail_1 --trim-pct 8
        sherloc full-pipeline 0921 Amherst_Point detail_1 --model-selection ftest
    """
    json_mode = (ctx.obj or {}).get("json", False)
    if json_mode:
        import logging as _logging
        _logging.basicConfig(stream=sys.stderr)

    try:
        _apply_trim_pct_override(trim_pct)
        _apply_model_selection_override(model_selection)
        service = PipelineService(console=console)
        result = service.run_full_pipeline(
            sol=sol,
            target=target,
            scan=scan,
            data_dir=data_dir,
            results_dir=results_dir,
        )

        if json_mode:
            import json as json_mod
            output = CLIResult(
                pipeline_version=_pipeline_version,
                command="full-pipeline",
                result={
                    "summary": result.summary,
                    "warnings": result.warnings or [],
                },
                metadata=result.metadata or {},
            )
            print(json_mod.dumps(output.model_dump(), default=str))
        else:
            console.print(f"\n[green]{result.summary}[/green]")
            if result.warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in result.warnings:
                    console.print(f"  - {warning}")

        sys.exit(0)

    except SherlocServiceError as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=e.message,
                exit_code=e.exit_code,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]✗ Pipeline failed: {e.message}[/red]")
        sys.exit(e.exit_code)
    except Exception as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]✗ Pipeline failed: {e}[/red]")
        sys.exit(1)


@app.command("apply-review")
def apply_review_cmd(
    ctx: typer.Context,
    sol: str = typer.Argument(..., help="Sol number (e.g., 0921)"),
    target: str = typer.Argument(..., help="Target name (e.g., Amherst_Point)"),
    scan: str = typer.Argument(..., help="Scan type (e.g., detail_1)"),
    regen: bool = typer.Option(False, "--regen", help="Regenerate spatial overlays with reviewed peaks"),
    upscale: Optional[int] = typer.Option(None, "--upscale", help="Upscale factor for overlays (defaults to config)"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Data directory (defaults to config)"),
    results_dir: Optional[Path] = typer.Option(None, "--results-dir", help="Results directory (defaults to config)"),
):
    """Apply manual review decisions and optionally regenerate overlays.

    This command:
        1. Reads the reviewed scan-level CSV (with user_keep flags)
        2. Updates target-level table
        3. Updates project-level master table
        4. If --regen, regenerates overlays with only user_keep=True peaks

    Workflow:
        1. Edit the scan CSV: results/<target>/<sol>_<scan>/<sol>_<target>_<scan>_accepted_peaks.csv
        2. Toggle user_keep column (True/False) for each peak
        3. Run: sherloc apply-review <sol> <target> <scan> --regen

    Example:
        sherloc apply-review 0921 Amherst_Point detail_1 --regen
    """
    json_mode = (ctx.obj or {}).get("json", False)
    if json_mode:
        import logging as _logging
        _logging.basicConfig(stream=sys.stderr)

    try:
        service = ReviewService(console=console)
        result = service.apply_review(
            sol=sol,
            target=target,
            scan=scan,
            regenerate_overlays=regen,
            upscale=upscale,
            data_dir=data_dir,
            results_dir=results_dir,
        )

        if json_mode:
            import json as json_mod
            output = CLIResult(
                pipeline_version=_pipeline_version,
                command="apply-review",
                result={
                    "summary": result.summary,
                    "warnings": result.warnings or [],
                },
                metadata=result.metadata or {},
            )
            print(json_mod.dumps(output.model_dump(), default=str))
        else:
            console.print(f"\n[green]{result.summary}[/green]")
            if result.warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in result.warnings:
                    console.print(f"  - {warning}")

        sys.exit(0)

    except SherlocServiceError as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=e.message,
                exit_code=e.exit_code,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]✗ Review failed: {e.message}[/red]")
        sys.exit(e.exit_code)
    except FileNotFoundError as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type="FileNotFoundError",
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]✗ File not found: {e}[/red]")
        sys.exit(1)
    except ValueError as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type="ValueError",
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]✗ Invalid data: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]✗ Review failed: {e}[/red]")
        sys.exit(1)


def _parse_range(value: str) -> Tuple[float, float]:
    """Parse comma-separated range string into tuple.
    
    Args:
        value: String in format "min,max" (e.g., "700,1200")
    
    Returns:
        Tuple of (min, max) as floats
        
    Raises:
        typer.BadParameter: If format is invalid
    """
    try:
        parts = value.split(",")
        if len(parts) != 2:
            raise typer.BadParameter(f"Expected format 'min,max', got '{value}'")
        return (float(parts[0].strip()), float(parts[1].strip()))
    except ValueError as e:
        raise typer.BadParameter(f"Could not parse range '{value}': {e}")


@app.command("plot")
def plot_cmd(
    ctx: typer.Context,
    # Scan identification (required)
    sol: str = typer.Option(..., "--sol", help="Sol number (e.g., 921 or 0921)"),
    target: str = typer.Option(..., "--target", help="Target name (e.g., Amherst_Point)"),
    scan: str = typer.Option(..., "--scan", help="Scan identifier (e.g., detail_1, line_1)"),
    
    # Mode selection (point mode options)
    point: Optional[int] = typer.Option(None, "--point", 
        help="Point index for single-point mode. Without --level: process from Loupe. With --level: load from results."),
    level: Optional[str] = typer.Option(None, "--level", 
        help="Load from pipeline outputs (legacy). Options: normalized, normalized_baselined, normalized_despiked_baselined"),
    
    # Subset mode option
    points: Optional[str] = typer.Option(None, "--points",
        help="Comma-separated point indices for subset averaging (e.g., 21,41,49,71,86)"),
    
    # Averaging options (averaged mode)
    avg: str = typer.Option("trim-mean", "--avg", 
        help="Averaging method: mean, median, or trim-mean"),
    trim_pct: float = typer.Option(2.0, "--trim-pct",
        help="Trim percentage for trim-mean (e.g., 2 for 2%%)"),
    
    # Processing options (averaged mode)
    background: Optional[str] = typer.Option(None, "--background", "-bg",
        help="Background type: 'as' (arm stowed) or 'fs' (fused silica)"),
    bgscale: str = typer.Option("auto", "--bgscale",
        help="Background scale: 'auto' (PPP-based) or explicit float (e.g., 0.7)"),
    baseline: bool = typer.Option(False, "--baseline", help="Apply baseline correction"),
    fit: bool = typer.Option(False, "--fit", help="Apply Gaussian fitting"),
    fit_range: Optional[str] = typer.Option(None, "--fit-range", 
        help="Fit range: min,max (e.g., 700,1200)"),
    single_peak: Optional[float] = typer.Option(None, "--single-peak",
        help="Fit single peak at specified center position (cm^-1)"),
    n_peaks: Optional[int] = typer.Option(None, "--n-peaks",
        help="Maximum number of peaks to fit (1-10)"),
    min_snr: Optional[float] = typer.Option(None, "--min-snr",
        help="Minimum SNR threshold for peak acceptance (default: 3.0 from config)"),
    fwhm_min: Optional[float] = typer.Option(None, "--fwhm-min",
        help="Minimum FWHM in cm^-1 for peak acceptance (default: 30 from config)"),
    fwhm_max: Optional[float] = typer.Option(None, "--fwhm-max",
        help="Maximum FWHM in cm^-1 for peak acceptance (default: 90 from config)"),
    
    # Axis controls
    xlim: Optional[str] = typer.Option(None, "--xlim", help="X-axis limits: min,max"),
    ylim: Optional[str] = typer.Option(None, "--ylim", help="Y-axis limits: min,max"),
    
    # Domain selection
    domain: Optional[str] = typer.Option(None, "--domain",
        help="Plot domain: raman (default), fluor, or both"),

    # Export
    export: str = typer.Option("both", "--export",
        help="Export format: csv, png, or both"),
    no_metadata: bool = typer.Option(False, "--no-metadata",
        help="Skip JSON metadata file export"),

    # Path overrides
    data_dir: Optional[Path] = typer.Option(None, "--data-dir",
        help="Data directory (defaults to config)"),
    results_dir: Optional[Path] = typer.Option(None, "--results-dir",
        help="Results directory (defaults to config)"),
):
    """Generate spectral plots from Loupe data or pipeline outputs.
    
    AVERAGED MODE (default):
        Process Loupe data → averaged spectrum with optional background subtraction,
        baseline correction, and Gaussian fitting.
        
        Example:
            sherloc plot --sol 921 --target Amherst_Point --scan detail_1 \\
                --background fs --baseline --fit --export both
    
    SUBSET MODE:
        Average specific points. Triggered by --points option.
        
        Example:
            sherloc plot --sol 921 --target Amherst_Point --scan detail_1 \\
                --points 21,41,49,71,86,87,88,90,91,92,98 \\
                --avg trim-mean --background fs --baseline --fit --export both
    
    FITTING OPTIONS (requires --fit):
        --single-peak: Fit exactly one Gaussian near specified center (cm^-1).
        --n-peaks: Limit automatic peak detection to at most N peaks.
        --min-snr: Override minimum SNR threshold for peak acceptance (default: 3.0).
        --fwhm-min: Override minimum FWHM in cm^-1 (default: 30).
        --fwhm-max: Override maximum FWHM in cm^-1 (default: 90).
        
        Examples:
            # Fit single carbonate peak at ~1090 cm^-1
            sherloc plot --sol 921 --target Amherst_Point --scan detail_1 \\
                --background fs --baseline --fit --single-peak 1090 \\
                --fit-range 1000,1200 --export both
            
            # Limit to 2 peaks maximum
            sherloc plot --sol 921 --target Amherst_Point --scan detail_1 \\
                --background fs --baseline --fit --n-peaks 2 --export both
            
            # Relax thresholds for weak peaks
            sherloc plot --sol 921 --target Amherst_Point --scan detail_1 \\
                --background fs --baseline --fit --min-snr 2.0 --fwhm-max 120 --export both
    
    POINT MODE:
        Visualize a single point spectrum.

        Without --level: Process from raw Loupe data (supports --background, --baseline, --fit)
        With --level: Load from pre-processed pipeline outputs

        Examples:
            # Process single point from Loupe data (new default)
            sherloc plot --sol 921 --target Amherst_Point --scan detail_1 \\
                --point 91 --background fs --baseline --fit --export both

            # Load from existing pipeline output (legacy)
            sherloc plot --sol 921 --target Amherst_Point --scan detail_1 \\
                --point 91 --level normalized_despiked_baselined \\
                --xlim 700,1300 --export png

    DOMAIN FLAG (--domain):
        Control which spectral domain to plot. Default is "raman".

        --domain raman   Raman peaks only (default)
        --domain fluor   Fluorescence peaks only
        --domain both    Combined Raman + fluorescence overlay

        Examples:
            sherloc plot --sol 293 --target Quartier --scan HDR_1 --fit --domain fluor
            sherloc plot --sol 293 --target Quartier --scan HDR_1 --fit --domain both

    Output files are saved to: results/<target>/plots/
    """
    json_mode = (ctx.obj or {}).get("json", False)
    if json_mode:
        import logging as _logging
        _logging.basicConfig(stream=sys.stderr)

    try:
        # Parse --points if provided
        parsed_points: Optional[list[int]] = None
        if points is not None:
            try:
                parsed_points = [int(p.strip()) for p in points.split(",")]
            except ValueError:
                console.print(f"[red]✗ Invalid --points format: {points}[/red]")
                console.print("  Expected comma-separated integers (e.g., 21,41,49,71)")
                raise typer.Exit(code=1)
        
        # Determine mode (--point for single point, --points for subset, else averaged)
        if point is not None:
            mode = "point"
        elif parsed_points is not None:
            mode = "subset"
        else:
            mode = "averaged"
        
        # Validate --point and --points are mutually exclusive
        if point is not None and parsed_points is not None:
            console.print("[red]✗ --point and --points are mutually exclusive[/red]")
            console.print("  Use --point for single-point visualization")
            console.print("  Use --points for subset averaging")
            raise typer.Exit(code=1)
        
        # Validate point mode requirements
        if mode == "point":
            # level is optional - determines data source:
            # - With level: load from pre-processed pipeline outputs (legacy)
            # - Without level: process from raw Loupe data (new behavior)
            
            # If level is provided, validate it
            if level is not None:
                valid_levels = {"normalized", "normalized_baselined", "normalized_despiked_baselined"}
                if level not in valid_levels:
                    console.print(f"[red]✗ Invalid level: {level}[/red]")
                    console.print(f"  Valid options: {', '.join(sorted(valid_levels))}")
                    raise typer.Exit(code=1)
                
                # Warn if processing flags are set with --level (they'll be ignored)
                if background or baseline or fit:
                    console.print("[yellow]⚠ Processing flags (--background, --baseline, --fit) are ignored when --level is specified.[/yellow]")
                    console.print("  Remove --level to process from Loupe data with these options.")
        
        # Validate subset mode requirements
        if mode == "subset":
            assert parsed_points is not None
            if len(parsed_points) < 2:
                console.print("[red]✗ --points requires at least 2 points to average[/red]")
                raise typer.Exit(code=1)
        
        # Validate domain
        valid_domains = ("raman", "fluor", "both")
        parsed_domain = domain or "raman"
        if parsed_domain not in valid_domains:
            console.print(f"[red]✗ Invalid domain: {parsed_domain}[/red]")
            console.print(f"  Valid options: {', '.join(valid_domains)}")
            raise typer.Exit(code=1)

        # Validate averaging method (averaged mode only)
        if avg not in ("mean", "median", "trim-mean"):
            console.print(f"[red]✗ Invalid averaging method: {avg}[/red]")
            console.print("  Valid options: mean, median, trim-mean")
            raise typer.Exit(code=1)
        
        # Validate background
        if background is not None and background not in ("as", "fs"):
            console.print(f"[red]✗ Invalid background: {background}[/red]")
            console.print("  Valid options: as (arm stowed), fs (fused silica)")
            raise typer.Exit(code=1)
        
        # Parse bgscale
        parsed_bgscale: float | str
        if bgscale == "auto":
            parsed_bgscale = "auto"
        else:
            try:
                parsed_bgscale = float(bgscale)
            except ValueError:
                console.print(f"[red]✗ Invalid bgscale: {bgscale}[/red]")
                console.print("  Must be 'auto' or a number (e.g., 0.7)")
                raise typer.Exit(code=1)
        
        # Validate export format
        if export not in ("csv", "png", "both"):
            console.print(f"[red]✗ Invalid export format: {export}[/red]")
            console.print("  Valid options: csv, png, both")
            raise typer.Exit(code=1)
        
        # Validate single_peak and n_peaks mutual exclusivity
        if single_peak is not None and n_peaks is not None:
            console.print("[red]✗ --single-peak and --n-peaks are mutually exclusive[/red]")
            raise typer.Exit(code=1)
        
        # Validate that single_peak requires --fit
        if single_peak is not None and not fit:
            console.print("[red]✗ --single-peak requires --fit to be enabled[/red]")
            raise typer.Exit(code=1)
        
        # Validate that n_peaks requires --fit
        if n_peaks is not None and not fit:
            console.print("[red]✗ --n-peaks requires --fit to be enabled[/red]")
            raise typer.Exit(code=1)
        
        # Validate n_peaks range
        if n_peaks is not None and (n_peaks < 1 or n_peaks > 10):
            console.print(f"[red]✗ --n-peaks must be between 1 and 10, got {n_peaks}[/red]")
            raise typer.Exit(code=1)
        
        # Parse optional range arguments
        parsed_fit_range: Optional[Tuple[float, float]] = None
        if fit_range is not None:
            parsed_fit_range = _parse_range(fit_range)
        
        parsed_xlim: Optional[Tuple[float, float]] = None
        if xlim is not None:
            parsed_xlim = _parse_range(xlim)
        
        parsed_ylim: Optional[Tuple[float, float]] = None
        if ylim is not None:
            parsed_ylim = _parse_range(ylim)
        
        # Build request based on mode
        if mode == "point":
            request = SpectralPlotRequest(
                sol=sol,
                target=target,
                scan=scan,
                mode="point",
                point=point,
                level=level,
                # Processing flags (only used when level=None, i.e. Loupe processing)
                background=background,  # type: ignore
                bgscale=parsed_bgscale,
                baseline=baseline,
                fit=fit,
                fit_range=parsed_fit_range,
                single_peak_center=single_peak,
                n_peaks=n_peaks,
                min_snr=min_snr,
                fwhm_min=fwhm_min,
                fwhm_max=fwhm_max,
                domain=parsed_domain,  # type: ignore
                xlim=parsed_xlim,
                ylim=parsed_ylim,
                export=export,  # type: ignore
                no_metadata=no_metadata,
            )
        elif mode == "subset":
            request = SpectralPlotRequest(
                sol=sol,
                target=target,
                scan=scan,
                mode="subset",
                points=parsed_points,
                avg_method=avg,  # type: ignore
                trim_pct=trim_pct,
                background=background,  # type: ignore
                bgscale=parsed_bgscale,
                baseline=baseline,
                fit=fit,
                fit_range=parsed_fit_range,
                single_peak_center=single_peak,
                n_peaks=n_peaks,
                min_snr=min_snr,
                fwhm_min=fwhm_min,
                fwhm_max=fwhm_max,
                domain=parsed_domain,  # type: ignore
                xlim=parsed_xlim,
                ylim=parsed_ylim,
                export=export,  # type: ignore
                no_metadata=no_metadata,
            )
        else:
            request = SpectralPlotRequest(
                sol=sol,
                target=target,
                scan=scan,
                mode="averaged",
                avg_method=avg,  # type: ignore
                trim_pct=trim_pct,
                background=background,  # type: ignore
                bgscale=parsed_bgscale,
                baseline=baseline,
                fit=fit,
                fit_range=parsed_fit_range,
                single_peak_center=single_peak,
                n_peaks=n_peaks,
                min_snr=min_snr,
                fwhm_min=fwhm_min,
                fwhm_max=fwhm_max,
                domain=parsed_domain,  # type: ignore
                xlim=parsed_xlim,
                ylim=parsed_ylim,
                export=export,  # type: ignore
                no_metadata=no_metadata,
            )
        
        # Process
        # Create RuntimeContext with any path overrides
        context = RuntimeContext.bootstrap(
            data_dir=data_dir,
            results_dir=results_dir,
        )
        service = SpectralService(console=console, context=context)
        result = service.process(request)
        
        if json_mode:
            import json as json_mod
            output = CLIResult(
                pipeline_version=_pipeline_version,
                command="plot",
                result={
                    "summary": result.summary,
                    "artifacts": [str(a) for a in (result.artifacts or [])],
                    "warnings": result.warnings or [],
                },
                metadata=result.metadata or {},
            )
            print(json_mod.dumps(output.model_dump(), default=str))
        else:
            # Print summary
            console.print(f"\n[green]✓ {result.summary}[/green]")

            # Print artifacts
            if result.artifacts:
                console.print("\nGenerated files:")
                for artifact in result.artifacts:
                    console.print(f"  • {artifact}")

            # Print warnings if any
            if result.warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in result.warnings:
                    console.print(f"  - {warning}")

        sys.exit(0)

    except SherlocServiceError as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=e.message,
                exit_code=e.exit_code,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]✗ Plot failed: {e.message}[/red]")
        sys.exit(e.exit_code)
    except typer.Exit:
        raise  # Re-raise typer exits
    except Exception as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]✗ Plot failed: {e}[/red]")
        sys.exit(1)


@app.command("ingest")
def ingest_cmd(
    ctx: typer.Context,
    path: Path = typer.Argument(..., help="Path to ingest (sol dir, workspace, or loupe root)"),
    database: Optional[Path] = typer.Option(
        None, "--database", "-d",
        help="Database path (defaults to ./phase.db)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-ingest even if data already exists"
    ),
    no_spectra: bool = typer.Option(
        False, "--no-spectra",
        help="Skip spectra data (faster ingestion for metadata-only)"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n",
        help="Maximum number of sols to process (for testing)"
    ),
    stats: bool = typer.Option(
        False, "--stats",
        help="Show database statistics after ingestion"
    ),
):
    """Ingest Loupe data into the PHASE SQLite database.

    Supports three ingestion modes based on the path provided:

    DIRECTORY MODE (Loupe root):
        Ingest all sol directories in a Loupe data root.
        Path should contain sol_XXXX directories.

        Example:
            sherloc ingest ./data/loupe

    SOL MODE:
        Ingest a single sol directory.
        Path should be a sol_XXXX directory.

        Example:
            sherloc ingest ./data/loupe/sol_0921

    WORKSPACE MODE:
        Ingest a single Loupe workspace.
        Path should be a *_Loupe_working directory.

        Example:
            sherloc ingest ./data/loupe/sol_0921/detail_1/SrlcSpec..._Loupe_working

    IDEMPOTENCY:
        Ingestion is idempotent by default. Re-ingesting the same data is a no-op.
        Use --force to re-ingest and overwrite existing data.

    DATABASE:
        Data is stored in SQLite at ./phase.db by default.
        Use --database to specify a different location.
    """
    json_mode = (ctx.obj or {}).get("json", False)
    if json_mode:
        import logging as _logging
        _logging.basicConfig(stream=sys.stderr)

    try:
        path = Path(path).absolute()

        if not path.exists():
            console.print(f"[red]Path not found: {path}[/red]")
            raise typer.Exit(code=1)

        # Create service
        service = IngestionService(
            console=console,
            database_path=database,
            include_spectra=not no_spectra,
        )

        # Determine mode based on path
        if (path / "loupe.csv").exists():
            # Workspace mode
            result = service.ingest_workspace(path, force=force)
        elif path.name.startswith("sol_"):
            # Sol mode
            result = service.ingest_sol(path, force=force)
        elif any(path.glob("sol_*")):
            # Directory mode
            result = service.ingest_directory(path, force=force, limit=limit)
        else:
            console.print(f"[red]Cannot determine ingestion mode for path: {path}[/red]")
            console.print("  Path should be a Loupe directory, sol directory, or workspace.")
            raise typer.Exit(code=1)

        # Print result
        success = result.metadata.get("success", True)
        db_stats_data = service.get_database_stats() if stats else {}

        if json_mode:
            import json as json_mod
            output = CLIResult(
                pipeline_version=_pipeline_version,
                command="ingest",
                result={
                    "summary": result.summary,
                    "success": success,
                    "warnings": result.warnings or [],
                    "errors": (result.metadata or {}).get("errors", []),
                    "db_stats": db_stats_data,
                },
                metadata=result.metadata or {},
            )
            print(json_mod.dumps(output.model_dump(), default=str))
        else:
            if success:
                console.print(f"\n[green]{result.summary}[/green]")
            else:
                console.print(f"\n[yellow]{result.summary}[/yellow]")

            # Print warnings
            if result.warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in result.warnings:
                    console.print(f"  - {warning}")

            # Print errors from metadata
            if result.metadata and result.metadata.get("errors"):
                console.print("\n[red]Errors:[/red]")
                for error in result.metadata["errors"]:
                    console.print(f"  - {error}")

            # Show stats if requested
            if stats:
                console.print("\n[bold]Database Statistics:[/bold]")
                console.print(f"  Sols: {db_stats_data['sols']}")
                console.print(f"  Scans: {db_stats_data['scans']}")
                console.print(f"  Scan Points: {db_stats_data['scan_points']}")
                console.print(f"  Spectra: {db_stats_data['spectra']}")
                console.print(f"  Instrument States: {db_stats_data['instrument_states']}")
                console.print(f"  Context Images: {db_stats_data['context_images']}")

        sys.exit(0 if success else 1)

    except IngestionError as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type="IngestionError",
                message=e.message,
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Ingestion failed: {e.message}[/red]")
        sys.exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Ingestion failed: {e}[/red]")
        sys.exit(1)


@app.command("process-new")
def process_new_cmd(
    ctx: typer.Context,
    path: Path = typer.Argument(..., help="Path to sol zip file or extracted sol directory"),
    database: Optional[Path] = typer.Option(
        None, "--database", "-d",
        help="Database path (defaults to ./phase.db)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show what would be processed without running"
    ),
    skip_ingest: bool = typer.Option(
        False, "--skip-ingest",
        help="Skip ingest step (already ingested, just run pipeline)"
    ),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Data directory (defaults to config)"),
    results_dir: Optional[Path] = typer.Option(None, "--results-dir", help="Results directory (defaults to config)"),
    trim_pct: Optional[float] = typer.Option(
        None, "--trim-pct",
        help="Override trim %% per tail (e.g., 8 for 8%%). Default: config.yaml (2%%).",
    ),
    model_selection: Optional[str] = typer.Option(
        None, "--model-selection",
        help="Peak-count model selection: 'aicc' or 'ftest'. Default: config.yaml (aicc).",
    ),
):
    """Ingest and process a new sol: unzip -> ingest -> pipeline.

    Chains the steps needed to process a freshly downloaded sol:
    1. If path is a .zip, extract to the Loupe data directory
    2. Run ingestion (auto-sets target from .lpe filename)
    3. Run full-pipeline on each science (mars_target) scan

    Examples:
        sherloc process-new /path/to/data/loupe/sol_1771.zip
        sherloc process-new ./data/loupe/sol_1771
        sherloc process-new sol_1771.zip --dry-run
        sherloc process-new ./data/loupe/sol_1771 --skip-ingest
        sherloc process-new sol_1771.zip --trim-pct 8
        sherloc process-new ./data/loupe/sol_1771 --model-selection ftest
    """
    import zipfile

    json_mode = (ctx.obj or {}).get("json", False)
    if json_mode:
        import logging as _logging
        _logging.basicConfig(stream=sys.stderr)

    try:
        _apply_trim_pct_override(trim_pct)
        _apply_model_selection_override(model_selection)
        path = Path(path).absolute()
        t0 = time.monotonic()

        if not path.exists():
            console.print(f"[red]Path not found: {path}[/red]")
            raise typer.Exit(code=1)

        db_path = database or Path("./phase.db")

        # Resolve data root from config
        from sherloc_pipeline.services.runtime import RuntimeContext
        context = RuntimeContext.bootstrap(data_dir=data_dir, results_dir=results_dir)
        loupe_data_root = context.data_root

        # Step 1: Unzip if needed
        if path.suffix.lower() == ".zip":
            # Derive sol directory name from zip filename (e.g., sol_1771.zip -> sol_1771)
            sol_dir_name = path.stem
            sol_dir = loupe_data_root / sol_dir_name

            if sol_dir.exists():
                console.print(f"[yellow]Directory already exists: {sol_dir}[/yellow]")
                console.print("  Skipping extraction, using existing directory.")
            else:
                if dry_run:
                    console.print(f"[cyan]Would extract:[/cyan] {path} -> {loupe_data_root}/")
                else:
                    console.print(f"Extracting {path.name} to {loupe_data_root}/...")
                    with zipfile.ZipFile(path, "r") as zf:
                        zf.extractall(loupe_data_root)
                    if not sol_dir.exists():
                        # Check if zip extracted to a differently named directory
                        console.print(f"[red]Expected directory {sol_dir} not found after extraction[/red]")
                        raise typer.Exit(code=1)
                    console.print(f"  Extracted to {sol_dir}")
        else:
            sol_dir = path

        # For dry-run with a zip that hasn't been extracted, peek inside the zip
        if dry_run and not sol_dir.is_dir() and path.suffix.lower() == ".zip":
            import re as _re
            console.print(f"\n[cyan]Zip contents preview:[/cyan]")
            with zipfile.ZipFile(path, "r") as zf:
                # Find .lpe file for target name
                lpe_names = [n for n in zf.namelist() if n.endswith(".lpe")]
                if lpe_names:
                    lpe_stem = Path(lpe_names[0]).stem
                    m = _re.match(r"Sol_\d+_\s*(.*)", lpe_stem)
                    lpe_target = m.group(1).replace("_", " ").strip() if m else "(unknown)"
                    console.print(f"  Target from .lpe: {lpe_target}")
                # Find workspace directories
                ws_dirs = sorted(set(
                    n.split("/")[1] for n in zf.namelist()
                    if "/" in n and "Loupe_working" in n.split("/")[1]
                ))
                console.print(f"  Workspaces: {len(ws_dirs)}")
                for ws in ws_dirs:
                    console.print(f"    - {ws}")
            console.print(f"\n[yellow]DRY RUN — no changes made[/yellow]")
            raise typer.Exit(code=0)

        if not sol_dir.is_dir():
            console.print(f"[red]Not a directory: {sol_dir}[/red]")
            raise typer.Exit(code=1)

        # Extract sol number
        from sherloc_pipeline.models.ingestion import extract_sol_from_path
        sol_number = extract_sol_from_path(sol_dir)
        if sol_number is None:
            console.print(f"[red]Cannot extract sol number from: {sol_dir}[/red]")
            raise typer.Exit(code=1)

        # Step 2: Ingest
        if skip_ingest:
            console.print(f"[yellow]Skipping ingest (--skip-ingest)[/yellow]")
        elif dry_run:
            from sherloc_pipeline.models.ingestion import discover_workspaces, extract_target_from_lpe
            workspaces = discover_workspaces(sol_dir)
            lpe_target = extract_target_from_lpe(sol_dir)
            console.print(f"\n[cyan]Would ingest:[/cyan] sol {sol_number}")
            console.print(f"  Target from .lpe: {lpe_target or '(none)'}")
            console.print(f"  Workspaces: {len(workspaces)}")
            for ws in workspaces:
                console.print(f"    - {ws.parent.name}/{ws.name}")
        else:
            console.print(f"\n[bold]Step 1: Ingesting sol {sol_number}...[/bold]")
            service = IngestionService(
                console=console,
                database_path=db_path,
                include_spectra=True,
                ingestion_mode="all_regions",
            )
            result = service.ingest_sol(sol_dir, force=False)
            console.print(f"  [green]{result.summary}[/green]")

        # Step 3: Run pipeline on mars_target scans
        from sherloc_pipeline.database.connection import get_engine, get_session
        from sherloc_pipeline.database.models import ScanORM

        engine = get_engine(db_path)
        with get_session(engine) as session:
            scans = (
                session.query(ScanORM.sol_number, ScanORM.target, ScanORM.scan_name)
                .filter(ScanORM.sol_number == sol_number)
                .filter(ScanORM.target_type == "mars_target")
                .order_by(ScanORM.scan_name)
                .all()
            )

        if not scans:
            console.print(f"\n[yellow]No mars_target scans found for sol {sol_number}[/yellow]")
            # Show all scans for debugging
            with get_session(engine) as session:
                all_scans = (
                    session.query(ScanORM.scan_name, ScanORM.target, ScanORM.target_type)
                    .filter(ScanORM.sol_number == sol_number)
                    .all()
                )
            if all_scans:
                console.print("  All scans for this sol:")
                for scn, tgt, ttype in all_scans:
                    console.print(f"    {scn}: target={tgt}, type={ttype}")
            raise typer.Exit(code=0)

        if dry_run:
            console.print(f"\n[cyan]Would run pipeline on {len(scans)} science scan(s):[/cyan]")
            for sol_num, tgt, scn in scans:
                console.print(f"  - {scn} (target={tgt})")
            console.print(f"\n[yellow]DRY RUN — no changes made[/yellow]")
            raise typer.Exit(code=0)

        console.print(f"\n[bold]Step 2: Running pipeline on {len(scans)} science scan(s)...[/bold]")
        pipeline_service = PipelineService(console=console)
        total_errors: List[str] = []

        for sol_num, tgt, scn in scans:
            console.print(f"\n  Processing {scn} (target={tgt})...")
            try:
                result = pipeline_service.run_full_pipeline(
                    sol=f"{sol_num:04d}",
                    target=tgt or "",
                    scan=scn,
                    data_dir=data_dir,
                    results_dir=results_dir,
                )
                console.print(f"    [green]{result.summary}[/green]")
            except Exception as e:
                total_errors.append(f"{scn}: {e}")
                console.print(f"    [red]Failed: {e}[/red]")

        # Summary
        elapsed = time.monotonic() - t0
        if json_mode:
            import json as json_mod
            output = CLIResult(
                pipeline_version=_pipeline_version,
                command="process-new",
                result={
                    "sol": sol_number,
                    "scans_processed": len(scans),
                    "elapsed_seconds": round(elapsed, 1),
                    "errors": total_errors,
                },
            )
            print(json_mod.dumps(output.model_dump(), default=str))
        else:
            console.print(f"\n[bold]Done:[/bold] sol {sol_number}, {len(scans)} scan(s) processed in {elapsed:.1f}s")
            if total_errors:
                console.print(f"\n[red]{len(total_errors)} error(s):[/red]")
                for err in total_errors:
                    console.print(f"  - {err}")
        sys.exit(1 if total_errors else 0)

    except IngestionError as e:
        if json_mode:
            import json as json_mod
            err_out = CLIError(
                pipeline_version=_pipeline_version,
                error_type="IngestionError",
                message=e.message,
                exit_code=1,
            )
            print(json_mod.dumps(err_out.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Process-new failed: {e.message}[/red]")
        sys.exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        if json_mode:
            import json as json_mod
            err_out = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err_out.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Process-new failed: {e}[/red]")
        sys.exit(1)


@app.command("db-stats")
def db_stats_cmd(
    ctx: typer.Context,
    database: Optional[Path] = typer.Option(
        None, "--database", "-d",
        help="Database path (defaults to ./phase.db)"
    ),
):
    """Show PHASE database statistics.

    Displays counts for all tables in the database.

    Example:
        sherloc db-stats
        sherloc db-stats --database /path/to/custom.db
    """
    json_mode = (ctx.obj or {}).get("json", False)
    if json_mode:
        import logging as _logging
        _logging.basicConfig(stream=sys.stderr)

    try:
        service = IngestionService(
            console=console,
            database_path=database,
            include_spectra=False,
        )

        db_stats = service.get_database_stats()
        sols = service.list_sols()

        if json_mode:
            import json as json_mod
            output = CLIResult(
                pipeline_version=_pipeline_version,
                command="db-stats",
                result={
                    "database": str(service.database_path),
                    "stats": db_stats,
                    "sols": sols,
                },
            )
            print(json_mod.dumps(output.model_dump(), default=str))
        else:
            console.print("[bold]PHASE Database Statistics[/bold]")
            console.print(f"  Database: {service.database_path}")
            console.print()
            console.print(f"  Sols: {db_stats['sols']}")
            console.print(f"  Scans: {db_stats['scans']}")
            console.print(f"  Scan Points: {db_stats['scan_points']}")
            console.print(f"  Spectra: {db_stats['spectra']}")
            console.print(f"  Instrument States: {db_stats['instrument_states']}")
            console.print(f"  CCD Configurations: {db_stats['ccd_configurations']}")
            console.print(f"  Scanner Calibrations: {db_stats['scanner_calibrations']}")
            console.print(f"  Context Images: {db_stats['context_images']}")
            console.print(f"  Regions of Interest: {db_stats['regions_of_interest']}")

            if sols:
                console.print()
                if len(sols) <= 20:
                    console.print(f"  Ingested sols: {', '.join(str(s) for s in sols)}")
                else:
                    console.print(f"  Ingested sols: {sols[0]}...{sols[-1]} ({len(sols)} total)")

        sys.exit(0)

    except Exception as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Failed to get stats: {e}[/red]")
        sys.exit(1)


def _list_local_sols(cache_dir: Path) -> List[int]:
    """List sol numbers from sol_NNNN directories in a PDS cache directory."""
    sols: List[int] = []
    if not cache_dir.exists():
        return sols
    for entry in cache_dir.iterdir():
        if entry.is_dir():
            match = re.match(r"sol_(\d+)$", entry.name)
            if match:
                sols.append(int(match.group(1)))
    return sorted(sols)


def _write_json_report(report: dict, dest: str, console: Console) -> None:
    """Write a JSON ingestion report to file or stdout.

    Args:
        report: Report dict to serialize.
        dest: File path or '-' for stdout.
        console: Rich console for status messages.
    """
    report_str = json.dumps(report, indent=2, default=str)
    if dest == "-":
        console.print(report_str, highlight=False)
    else:
        out_path = Path(dest)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_str + "\n")
        console.print(f"\n[bold]Report written to:[/bold] {out_path}")


@app.command("pds-ingest")
def pds_ingest_cmd(
    ctx: typer.Context,
    sol: Optional[int] = typer.Option(
        None, "--sol",
        help="Single sol number to ingest"
    ),
    sol_range: Optional[Tuple[int, int]] = typer.Option(
        None, "--sol-range",
        help="Sol range START END (inclusive)"
    ),
    auto: bool = typer.Option(
        False, "--auto",
        help="Ingest all locally cached sols"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Parse and validate without writing to database"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-ingest even if data already exists"
    ),
    pds_dir: Optional[Path] = typer.Option(
        None, "--pds-dir",
        help="PDS cache directory (default: ./pds)"
    ),
    pds_database: Optional[Path] = typer.Option(
        None, "--pds-database", "-d",
        help="PDS database path (default: ./phase_pds.db)"
    ),
    loupe_database: Optional[Path] = typer.Option(
        None, "--loupe-database",
        help="Loupe DB for target name cross-reference (default: ./phase.db)"
    ),
    check_updates: bool = typer.Option(
        False, "--check-updates",
        help="Check for version updates without ingesting"
    ),
    report_json: Optional[str] = typer.Option(
        None, "--report-json",
        help="Write JSON ingestion report to file (use '-' for stdout)"
    ),
    stats: bool = typer.Option(
        False, "--stats",
        help="Show database statistics after ingestion"
    ),
):
    """Ingest PDS4 SHERLOC data into phase_pds.db.

    Modes (mutually exclusive, exactly one required):

        --sol N            Ingest a single sol
        --sol-range S E    Ingest all cached sols in range [S, E]
        --auto             Ingest all locally cached sols

    Data is stored in phase_pds.db, separate from Loupe's phase.db.
    Target names are resolved via SCLK cross-reference to Loupe DB.
    Ingestion is idempotent; use --force to re-ingest.
    Version updates are auto-detected and re-ingested (numeric tuple
    comparison per spec s6). Use --check-updates for read-only check.

    Examples:
        sherloc pds-ingest --sol 921
        sherloc pds-ingest --sol-range 100 1000
        sherloc pds-ingest --auto --force --stats
        sherloc pds-ingest --sol 921 --dry-run
        sherloc pds-ingest --sol 921 --check-updates
    """
    json_mode = (ctx.obj or {}).get("json", False)
    if json_mode:
        import logging as _logging
        _logging.basicConfig(stream=sys.stderr)

    try:
        # --- Validate mode selection ---
        modes = sum([sol is not None, sol_range is not None, auto])
        if modes == 0:
            console.print("[red]Specify one of --sol, --sol-range, or --auto[/red]")
            raise typer.Exit(code=1)
        if modes > 1:
            console.print("[red]--sol, --sol-range, and --auto are mutually exclusive[/red]")
            raise typer.Exit(code=1)

        # --- Resolve PDS cache directory ---
        cache_dir = Path(pds_dir) if pds_dir else PDS_DEFAULT_CACHE_DIR
        if not cache_dir.exists():
            console.print(f"[red]PDS cache directory not found: {cache_dir}[/red]")
            raise typer.Exit(code=1)

        # --- Build list of sol numbers ---
        if sol is not None:
            sol_numbers = [sol]
        elif sol_range is not None:
            start, end = sol_range
            if start > end:
                console.print(f"[red]Invalid range: start ({start}) > end ({end})[/red]")
                raise typer.Exit(code=1)
            all_local = _list_local_sols(cache_dir)
            sol_numbers = [s for s in all_local if start <= s <= end]
            if not sol_numbers:
                console.print(
                    f"[yellow]No locally cached sols in range {start}-{end}[/yellow]"
                )
                raise typer.Exit(code=0)
        else:  # auto
            sol_numbers = _list_local_sols(cache_dir)
            if not sol_numbers:
                console.print("[yellow]No locally cached sols found[/yellow]")
                raise typer.Exit(code=0)

        t0 = time.monotonic()

        console.print(
            f"[bold]PDS Ingestion[/bold] — {len(sol_numbers)} sol(s) to process"
        )
        if dry_run:
            console.print("[yellow]DRY RUN — no database writes[/yellow]")
        if force:
            console.print("[yellow]FORCE — re-ingesting existing data[/yellow]")
        console.print()

        # --- Dry-run mode: parse and report only ---
        if dry_run:
            from sherloc_pipeline.services.pds_ingestion import (
                PDSObservationGrouper,
                PDSLabelParser,
            )

            grouper = PDSObservationGrouper()
            label_parser = PDSLabelParser()

            table = Table(title="Dry Run Summary")
            table.add_column("Sol", style="cyan", justify="right")
            table.add_column("Observations", justify="right")
            table.add_column("Types", style="green")
            table.add_column("Products", justify="right")
            table.add_column("Status")

            total_obs = 0
            for s in sol_numbers:
                sol_dir = cache_dir / f"sol_{s:04d}" / "data_processed"
                if not sol_dir.exists():
                    table.add_row(
                        str(s), "-", "-", "-", "[red]directory missing[/red]"
                    )
                    continue
                try:
                    groups = grouper.group_sol_directory(
                        sol_dir, label_parser=label_parser
                    )
                    types = ", ".join(
                        sorted(set(g.scan_type or "unknown" for g in groups))
                    )
                    n_products = sum(len(g.products) for g in groups)
                    total_obs += len(groups)
                    table.add_row(
                        str(s), str(len(groups)), types,
                        str(n_products), "[green]ready[/green]"
                    )
                except Exception as e:
                    table.add_row(str(s), "-", "-", "-", f"[red]{e}[/red]")

            console.print(table)
            console.print(
                f"\n[bold]Total:[/bold] {total_obs} observations "
                f"across {len(sol_numbers)} sols"
            )

            if report_json is not None:
                elapsed = time.monotonic() - t0
                report = {
                    "command": "pds-ingest",
                    "dry_run": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "sols_processed": len(sol_numbers),
                    "observations_discovered": total_obs,
                    "observations_ingested": 0,
                    "observations_skipped": 0,
                    "observations_failed": 0,
                    "spectra_created": 0,
                    "context_images_created": 0,
                    "errors": [],
                    "warnings": [],
                    "version_updates": [],
                    "elapsed_time_seconds": round(elapsed, 3),
                }
                _write_json_report(report, report_json, console)

            sys.exit(0)

        # --- Check-updates mode: read-only version check ---
        if check_updates:
            service = PDSIngestionService(
                pds_db_path=pds_database,
                loupe_db_path=None,  # No Loupe needed for version check
            )

            table = Table(title="Version Update Check")
            table.add_column("Sol", style="cyan", justify="right")
            table.add_column("Observation", style="white")
            table.add_column("Status", style="green")
            table.add_column("Version", justify="center")

            total_new = 0
            total_updated = 0
            total_current = 0

            for s in sol_numbers:
                sol_dir = cache_dir / f"sol_{s:04d}" / "data_processed"
                if not sol_dir.exists():
                    table.add_row(
                        str(s), "-", "[red]directory missing[/red]", "-"
                    )
                    continue

                try:
                    report = service.check_for_updates(sol_dir)

                    for obs in report["new"]:
                        table.add_row(
                            str(s), obs["observation_key"],
                            "[blue]new[/blue]",
                            obs["version"],
                        )
                        total_new += 1

                    for obs in report["updated"]:
                        table.add_row(
                            str(s), obs["observation_key"],
                            "[yellow]update available[/yellow]",
                            f"{obs['old_version']} → {obs['new_version']}",
                        )
                        total_updated += 1

                    for obs in report["current"]:
                        table.add_row(
                            str(s), obs["observation_key"],
                            "[green]current[/green]",
                            obs["version"],
                        )
                        total_current += 1

                    for err in report["errors"]:
                        table.add_row(str(s), "-", f"[red]{err}[/red]", "-")
                except Exception as e:
                    table.add_row(str(s), "-", f"[red]{e}[/red]", "-")

            console.print(table)
            console.print(
                f"\n[bold]Summary:[/bold] "
                f"{total_new} new, "
                f"{total_updated} updates available, "
                f"{total_current} current"
            )

            if total_updated > 0:
                console.print(
                    "\n[yellow]Run without --check-updates to apply "
                    "version updates automatically.[/yellow]"
                )

            sys.exit(0)

        # --- Normal ingestion ---
        if loupe_database is None:
            default_loupe = Path("./phase.db")
            if default_loupe.exists():
                loupe_database = default_loupe

        service = PDSIngestionService(
            pds_db_path=pds_database,
            loupe_db_path=loupe_database,
        )

        total_obs_ingested = 0
        total_obs_skipped = 0
        total_obs_updated = 0
        total_points = 0
        total_spectra = 0
        total_context = 0
        total_sols = 0
        all_errors: List[str] = []
        all_warnings: List[str] = []
        all_version_updates: List[dict] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Ingesting sols...", total=len(sol_numbers)
            )

            for s in sol_numbers:
                sol_dir = cache_dir / f"sol_{s:04d}" / "data_processed"
                progress.update(task, description=f"Sol {s}...")

                if not sol_dir.exists():
                    all_errors.append(
                        f"Sol {s}: directory not found ({sol_dir})"
                    )
                    progress.advance(task)
                    continue

                try:
                    result = service.ingest_sol(sol_dir, force=force)
                    meta = result.metadata or {}
                    total_obs_ingested += meta.get(
                        "observations_ingested", 0
                    )
                    total_obs_skipped += meta.get(
                        "observations_skipped", 0
                    )
                    total_points += meta.get("points_ingested", 0)
                    total_spectra += meta.get("spectra_ingested", 0)
                    total_context += meta.get(
                        "context_images_ingested", 0
                    )
                    total_obs_updated += meta.get(
                        "observations_updated", 0
                    )
                    total_sols += 1
                    if meta.get("version_updates"):
                        all_version_updates.extend(meta["version_updates"])
                    if meta.get("errors"):
                        all_errors.extend(meta["errors"])
                    if result.warnings:
                        all_warnings.extend(result.warnings)
                except PDSIngestionError as e:
                    all_errors.append(f"Sol {s}: {e.message}")
                except Exception as e:
                    all_errors.append(f"Sol {s}: {e}")

                progress.advance(task)

        # --- Print summary ---
        console.print()
        has_errors = bool(all_errors)

        summary = (
            f"Processed {total_sols} sol(s): "
            f"{total_obs_ingested} ingested, "
            f"{total_obs_skipped} skipped, "
            f"{total_points} points, "
            f"{total_spectra} spectra, "
            f"{total_context} context images"
        )
        if total_obs_updated:
            summary += f", {total_obs_updated} version-updated"

        if has_errors:
            console.print(f"[yellow]{summary}[/yellow]")
        else:
            console.print(f"[green]{summary}[/green]")

        if all_version_updates:
            console.print("\n[bold]Version Updates:[/bold]")
            for vu in all_version_updates:
                console.print(
                    f"  {vu['observation_key']}: "
                    f"{vu['old_version']} → {vu['new_version']}"
                )

        if all_warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for w in all_warnings:
                console.print(f"  - {w}")

        if all_errors:
            console.print("\n[red]Errors:[/red]")
            for e in all_errors:
                console.print(f"  - {e}")

        # --- Stats ---
        if stats:
            db_stats = service.get_database_stats()
            console.print("\n[bold]PDS Database Statistics:[/bold]")
            console.print(f"  Database: {service.pds_db_path}")
            console.print(f"  Sols: {db_stats['sols']}")
            console.print(f"  Scans: {db_stats['scans']}")
            console.print(f"  Scan Points: {db_stats['scan_points']}")
            console.print(f"  Spectra: {db_stats['spectra']}")
            console.print(f"  Context Images: {db_stats['context_images']}")

        # --- JSON report ---
        if report_json is not None:
            elapsed = time.monotonic() - t0
            report = {
                "command": "pds-ingest",
                "dry_run": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sols_processed": total_sols,
                "observations_ingested": total_obs_ingested,
                "observations_skipped": total_obs_skipped,
                "observations_failed": len(all_errors),
                "spectra_created": total_spectra,
                "points_created": total_points,
                "context_images_created": total_context,
                "version_updates": all_version_updates,
                "errors": all_errors,
                "warnings": all_warnings,
                "elapsed_time_seconds": round(elapsed, 3),
            }
            if force:
                report["force"] = True
            _write_json_report(report, report_json, console)

        if json_mode:
            import json as json_mod
            elapsed = time.monotonic() - t0
            output = CLIResult(
                pipeline_version=_pipeline_version,
                command="pds-ingest",
                result={
                    "sols_processed": total_sols,
                    "observations_ingested": total_obs_ingested,
                    "observations_skipped": total_obs_skipped,
                    "errors": all_errors,
                    "warnings": all_warnings,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
            print(json_mod.dumps(output.model_dump(), default=str))

        sys.exit(1 if has_errors else 0)

    except PDSIngestionError as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type="PDSIngestionError",
                message=e.message,
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]PDS ingestion failed: {e.message}[/red]")
        sys.exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]PDS ingestion failed: {e}[/red]")
        sys.exit(1)


@app.command("pixl-ingest")
def pixl_ingest_cmd(
    ctx: typer.Context,
    source: Path = typer.Argument(
        ...,
        help="Path to directory containing Pixlise export zip files (e.g., /nas/000_pixl)",
    ),
    database: Optional[Path] = typer.Option(
        None, "--database", "-d",
        help="Database path (defaults to /data/pixl/pixlise.db)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-ingest even if data already exists"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n",
        help="Maximum number of zip files to process (for testing)"
    ),
    stats: bool = typer.Option(
        False, "--stats",
        help="Show database statistics after ingestion"
    ),
):
    """Ingest PIXL Pixlise data into SQLite database.

    Parses Pixlise Data Export zip files and stores the data in
    /data/pixl/pixlise.db (or specified database path).

    DATA SOURCE:
        The source directory should contain Pixlise export zip files
        named "Pixlise Data Export YYYY-MM-DD (N).zip".

        Default location: /nas/000_pixl (NAS mount point)

    WHAT'S INGESTED:
        - Target metadata (name, RTT, PIQUANT version)
        - AutoQuant-PDS quantification (16 oxides with errors and intensities)
        - Beam location coordinates (per-image pixel locations)
        - Context images (PCW, DTU_MSA, SIF, CSC_ACI)

    IDEMPOTENCY:
        Ingestion is idempotent by default (based on RTT uniqueness).
        Re-ingesting the same data is a no-op. Use --force to re-ingest.

    Examples:
        # Ingest all exports from NAS
        sherloc pixl-ingest /nas/000_pixl

        # Ingest with custom database
        sherloc pixl-ingest /nas/000_pixl --database /data/pixl/test.db

        # Ingest first 5 for testing
        sherloc pixl-ingest /nas/000_pixl --limit 5 --stats

        # Force re-ingest and show stats
        sherloc pixl-ingest /nas/000_pixl --force --stats
    """
    json_mode = (ctx.obj or {}).get("json", False)
    if json_mode:
        import logging as _logging
        _logging.basicConfig(stream=sys.stderr)

    try:
        source = Path(source).absolute()

        if not source.exists():
            console.print(f"[red]Source directory not found: {source}[/red]")
            raise typer.Exit(code=1)

        # Create service
        service = PixliseIngestionService(
            console=console,
            database_path=database,
        )

        # Run ingestion
        result = service.ingest_directory(source, force=force, limit=limit)

        # Print result
        success = result.metadata.get("success", True)
        if success:
            console.print(f"\n[green]{result.summary}[/green]")
        else:
            console.print(f"\n[yellow]{result.summary}[/yellow]")

        # Print warnings
        if result.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in result.warnings:
                console.print(f"  - {warning}")

        # Print errors from metadata
        if result.metadata and result.metadata.get("errors"):
            console.print("\n[red]Errors:[/red]")
            for error in result.metadata["errors"]:
                console.print(f"  - {error}")

        # Show stats if requested
        if stats:
            db_stats = service.get_database_stats()
            console.print("\n[bold]Database Statistics:[/bold]")
            console.print(f"  Database: {service.database_path}")
            console.print(f"  Targets: {db_stats['targets']}")
            console.print(f"  Quant Points: {db_stats['quant_points']:,}")
            console.print(f"  Images: {db_stats['images']}")
            console.print(f"  Beam Locations: {db_stats['beam_locations']:,}")

            # List targets if not too many
            targets = service.list_targets()
            if targets and len(targets) <= 20:
                console.print("\n[bold]Targets:[/bold]")
                for t in targets:
                    console.print(
                        f"  {t['name']}: {t['n_points']:,} points "
                        f"(RTT={t['rtt']})"
                    )
            elif targets:
                console.print(f"\n  {len(targets)} targets total")

        if json_mode:
            import json as json_mod
            output = CLIResult(
                pipeline_version=_pipeline_version,
                command="pixl-ingest",
                result={
                    "summary": result.summary,
                    "success": success,
                    "warnings": result.warnings or [],
                    "errors": (result.metadata or {}).get("errors", []),
                },
                metadata=result.metadata or {},
            )
            print(json_mod.dumps(output.model_dump(), default=str))

        sys.exit(0 if success else 1)

    except PixliseIngestionError as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type="PixliseIngestionError",
                message=e.message,
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Pixlise ingestion failed: {e.message}[/red]")
        sys.exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Pixlise ingestion failed: {e}[/red]")
        sys.exit(1)


def _iter_all_scans(
    database_path: Path,
    console: Console,
    target_type: Optional[str] = None,
    exclude_type: Optional[str] = None,
    science_only: bool = False,
):
    """Yield (sol_number, target, scan_name) for all scans in the database.

    Args:
        database_path: Path to SQLite database.
        console: Rich console for output.
        target_type: Optional filter — one of 'mars_target', 'cal_target', 'engineering'.
            If None, returns all scans (backward compatible).
        exclude_type: Optional exclusion — e.g. 'engineering' to skip engineering scans.
        science_only: If True, return mars_target scans + cal meteorite (SAU-008) scans.
            Overrides target_type and exclude_type.
    """
    from sherloc_pipeline.database.connection import get_engine, get_session
    from sherloc_pipeline.database.models import ScanORM
    from sqlalchemy import or_

    engine = get_engine(database_path)
    with get_session(engine) as session:
        query = (
            session.query(
                ScanORM.sol_number, ScanORM.target, ScanORM.scan_name
            )
            .order_by(ScanORM.sol_number, ScanORM.target, ScanORM.scan_name)
        )
        if science_only:
            query = query.filter(or_(
                ScanORM.target_type == "mars_target",
                ScanORM.scan_name.contains("meteorite"),
                ScanORM.scan_name.contains("MarsMeteorite"),
                ScanORM.target.contains("meteorite"),
            ))
        else:
            if target_type is not None:
                query = query.filter(ScanORM.target_type == target_type)
            if exclude_type is not None:
                query = query.filter(ScanORM.target_type != exclude_type)
        scans = query.all()
    # Normalize target names: DB stores spaces, filesystem/CLI uses underscores
    return [(sol, (tgt or "").replace(" ", "_"), scn) for sol, tgt, scn in scans]


@app.command("fit-fluor")
def fit_fluor_cmd(
    ctx: typer.Context,
    sol: Optional[str] = typer.Option(None, "--sol", help="Sol number (e.g., 0293)"),
    target: Optional[str] = typer.Option(None, "--target", help="Target name (e.g., Quartier)"),
    scan: Optional[str] = typer.Option(None, "--scan", help="Scan identifier (e.g., HDR_1)"),
    all_scans: bool = typer.Option(False, "--all", help="Fit fluorescence for all scans in the database"),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip plot generation (faster for parameter tuning)"),
    database: Optional[Path] = typer.Option(
        None, "--database", "-d",
        help="Database path (defaults to ./phase.db)"
    ),
):
    """Fit fluorescence spectra and persist peaks to the database.

    Queries R2/R3 spectra from the database, fits multi-Gaussian fluorescence
    models, assigns group labels, and persists peaks to fitted_peaks table.

    SINGLE SCAN:
        sherloc fit-fluor --sol 293 --target Quartier --scan HDR_1

    ALL SCANS:
        sherloc fit-fluor --all

    Fluorescence fitting parameters are read from config.yaml
    (fluorescence_fitting section).
    """
    json_mode = (ctx.obj or {}).get("json", False)
    if json_mode:
        import logging as _logging
        _logging.basicConfig(stream=sys.stderr)

    try:
        # Validate mode
        if all_scans and any([sol, target, scan]):
            console.print("[red]--all is mutually exclusive with --sol/--target/--scan[/red]")
            raise typer.Exit(code=1)
        if not all_scans and not all([sol, target, scan]):
            console.print("[red]Provide --sol, --target, and --scan, or use --all[/red]")
            raise typer.Exit(code=1)

        db_path = database or Path("./phase.db")
        if not db_path.exists():
            console.print(f"[red]Database not found: {db_path}[/red]")
            raise typer.Exit(code=1)

        from sherloc_pipeline.services.fitting import FittingService

        service = FittingService(console=console, database_path=db_path)

        if all_scans:
            scans = _iter_all_scans(db_path, console)
            console.print(f"[bold]Fluorescence fitting[/bold] — {len(scans)} scan(s)")

            total_peaks = 0
            errors: List[str] = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Fitting fluorescence...", total=len(scans))
                for sol_num, tgt, scn in scans:
                    progress.update(task, description=f"Sol {sol_num} {tgt} {scn}")
                    try:
                        result = service.fit_fluorescence(
                            sol=f"{sol_num:04d}", target=tgt or "", scan=scn,
                            generate_plots=not no_plots,
                        )
                        total_peaks += (result.metadata or {}).get("peaks_inserted", 0)
                    except Exception as e:
                        errors.append(f"Sol {sol_num} {tgt} {scn}: {e}")
                    progress.advance(task)

            if json_mode:
                import json as json_mod
                output = CLIResult(
                    pipeline_version=_pipeline_version,
                    command="fit-fluor",
                    result={
                        "total_peaks": total_peaks,
                        "scans_processed": len(scans),
                        "errors": errors,
                    },
                )
                print(json_mod.dumps(output.model_dump(), default=str))
            else:
                console.print(f"\n[green]Fitted {total_peaks} fluorescence peaks across {len(scans)} scan(s)[/green]")
                if errors:
                    console.print(f"\n[yellow]{len(errors)} error(s):[/yellow]")
                    for err in errors[:20]:
                        console.print(f"  - {err}")
                    if len(errors) > 20:
                        console.print(f"  ... and {len(errors) - 20} more")
            sys.exit(1 if errors else 0)
        else:
            result = service.fit_fluorescence(
                sol=sol, target=target, scan=scan,
                generate_plots=not no_plots,
            )
            if json_mode:
                import json as json_mod
                output = CLIResult(
                    pipeline_version=_pipeline_version,
                    command="fit-fluor",
                    result={
                        "summary": result.summary,
                        "warnings": result.warnings or [],
                    },
                    metadata=result.metadata or {},
                )
                print(json_mod.dumps(output.model_dump(), default=str))
            else:
                console.print(f"\n[green]{result.summary}[/green]")
                if result.warnings:
                    console.print("\n[yellow]Warnings:[/yellow]")
                    for w in result.warnings:
                        console.print(f"  - {w}")
            sys.exit(0)

    except SherlocServiceError as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=e.message,
                exit_code=e.exit_code,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Fluorescence fitting failed: {e.message}[/red]")
        sys.exit(e.exit_code)
    except typer.Exit:
        raise
    except Exception as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Fluorescence fitting failed: {e}[/red]")
        sys.exit(1)


@app.command("persist-peaks")
def persist_peaks_cmd(
    ctx: typer.Context,
    domain: str = typer.Option(..., "--domain", help="Raman domain: minerals, organics, or hydration"),
    sol: Optional[str] = typer.Option(None, "--sol", help="Sol number (e.g., 0921)"),
    target: Optional[str] = typer.Option(None, "--target", help="Target name (e.g., Amherst_Point)"),
    scan: Optional[str] = typer.Option(None, "--scan", help="Scan identifier (e.g., detail_1)"),
    all_scans: bool = typer.Option(False, "--all", help="Persist peaks for all scans in the database"),
    database: Optional[Path] = typer.Option(
        None, "--database", "-d",
        help="Database path (defaults to ./phase.db)"
    ),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Data directory (defaults to config)"),
    results_dir: Optional[Path] = typer.Option(None, "--results-dir", help="Results directory (defaults to config)"),
):
    """Persist Raman peak CSVs to the database for a specific domain.

    Discovers per-point peak CSV artifacts, applies quality filters, assigns
    feature labels (mineral names, D/G bands, OH/H2O bands), and writes
    peaks to the fitted_peaks table.

    SINGLE SCAN:
        sherloc persist-peaks --domain minerals --sol 921 --target Amherst_Point --scan detail_1

    ALL SCANS (backfill):
        sherloc persist-peaks --domain minerals --all
        sherloc persist-peaks --domain organics --all
        sherloc persist-peaks --domain hydration --all

    Idempotent: re-running replaces only the specified domain's peaks.
    """
    json_mode = (ctx.obj or {}).get("json", False)
    if json_mode:
        import logging as _logging
        _logging.basicConfig(stream=sys.stderr)

    try:
        # Validate domain
        valid_domains = ("minerals", "organics", "hydration")
        if domain not in valid_domains:
            console.print(f"[red]Invalid domain: {domain}[/red]")
            console.print(f"  Valid options: {', '.join(valid_domains)}")
            console.print("  For fluorescence, use: sherloc fit-fluor")
            raise typer.Exit(code=1)

        # Validate mode
        if all_scans and any([sol, target, scan]):
            console.print("[red]--all is mutually exclusive with --sol/--target/--scan[/red]")
            raise typer.Exit(code=1)
        if not all_scans and not all([sol, target, scan]):
            console.print("[red]Provide --sol, --target, and --scan, or use --all[/red]")
            raise typer.Exit(code=1)

        db_path = database or Path("./phase.db")
        if not db_path.exists():
            console.print(f"[red]Database not found: {db_path}[/red]")
            raise typer.Exit(code=1)

        from sherloc_pipeline.services.fitting import FittingService

        service = FittingService(console=console, database_path=db_path)

        if all_scans:
            scans = _iter_all_scans(db_path, console)
            console.print(f"[bold]Persisting {domain} peaks[/bold] — {len(scans)} scan(s)")

            total_peaks = 0
            errors: List[str] = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Persisting {domain}...", total=len(scans))
                for sol_num, tgt, scn in scans:
                    progress.update(task, description=f"Sol {sol_num} {tgt} {scn}")
                    try:
                        result = service.persist_raman_peaks(
                            sol=f"{sol_num:04d}", target=tgt or "", scan=scn,
                            domain=domain,
                            data_dir=data_dir,
                            results_dir=results_dir,
                        )
                        total_peaks += (result.metadata or {}).get("peaks_inserted", 0)
                    except Exception as e:
                        errors.append(f"Sol {sol_num} {tgt} {scn}: {e}")
                    progress.advance(task)

            if json_mode:
                import json as json_mod
                output = CLIResult(
                    pipeline_version=_pipeline_version,
                    command="persist-peaks",
                    result={
                        "domain": domain,
                        "total_peaks": total_peaks,
                        "scans_processed": len(scans),
                        "errors": errors,
                    },
                )
                print(json_mod.dumps(output.model_dump(), default=str))
            else:
                console.print(f"\n[green]Persisted {total_peaks} {domain} peaks across {len(scans)} scan(s)[/green]")
                if errors:
                    console.print(f"\n[yellow]{len(errors)} error(s):[/yellow]")
                    for err in errors[:20]:
                        console.print(f"  - {err}")
                    if len(errors) > 20:
                        console.print(f"  ... and {len(errors) - 20} more")
            sys.exit(1 if errors else 0)
        else:
            result = service.persist_raman_peaks(
                sol=sol, target=target, scan=scan, domain=domain,
                data_dir=data_dir, results_dir=results_dir,
            )
            if json_mode:
                import json as json_mod
                output = CLIResult(
                    pipeline_version=_pipeline_version,
                    command="persist-peaks",
                    result={
                        "domain": domain,
                        "summary": result.summary,
                        "warnings": result.warnings or [],
                    },
                    metadata=result.metadata or {},
                )
                print(json_mod.dumps(output.model_dump(), default=str))
            else:
                console.print(f"\n[green]{result.summary}[/green]")
                if result.warnings:
                    console.print("\n[yellow]Warnings:[/yellow]")
                    for w in result.warnings:
                        console.print(f"  - {w}")
            sys.exit(0)

    except SherlocServiceError as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=e.message,
                exit_code=e.exit_code,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Persist peaks failed: {e.message}[/red]")
        sys.exit(e.exit_code)
    except typer.Exit:
        raise
    except Exception as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Persist peaks failed: {e}[/red]")
        sys.exit(1)


@app.command("backfill")
def backfill_cmd(
    ctx: typer.Context,
    database: Optional[Path] = typer.Option(
        None, "--database", "-d",
        help="Database path (defaults to ./phase.db)"
    ),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Data directory (defaults to config)"),
    results_dir: Optional[Path] = typer.Option(None, "--results-dir", help="Results directory (defaults to config)"),
    domains: Optional[str] = typer.Option(
        None, "--domains",
        help="Comma-separated domains to backfill (default: all). Options: minerals,organics,hydration,fluorescence"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show scan count without running backfill"),
    no_engineering: bool = typer.Option(False, "--no-engineering", help="Exclude engineering scans"),
    science: bool = typer.Option(False, "--science", help="Mars targets + cal meteorite (SAU-008) only"),
):
    """Backfill all peak domains across the full mission.

    Runs all 4 domains in sequence:
        1. persist-peaks --domain minerals --all
        2. persist-peaks --domain organics --all
        3. persist-peaks --domain hydration --all
        4. fit-fluor --all (fluorescence)

    Each domain is idempotent: re-running replaces only that domain's peaks.

    Examples:
        # Backfill all domains
        sherloc backfill

        # Backfill only minerals and organics
        sherloc backfill --domains minerals,organics

        # Dry run to see scan count
        sherloc backfill --dry-run
    """
    json_mode = (ctx.obj or {}).get("json", False)
    if json_mode:
        import logging as _logging
        _logging.basicConfig(stream=sys.stderr)

    try:
        db_path = database or Path("./phase.db")
        if not db_path.exists():
            console.print(f"[red]Database not found: {db_path}[/red]")
            raise typer.Exit(code=1)

        # Parse domains
        raman_domains = ["minerals", "organics", "hydration"]
        all_domains = raman_domains + ["fluorescence"]

        if domains:
            selected = [d.strip() for d in domains.split(",")]
            for d in selected:
                if d not in all_domains:
                    console.print(f"[red]Invalid domain: {d}[/red]")
                    console.print(f"  Valid options: {', '.join(all_domains)}")
                    raise typer.Exit(code=1)
        else:
            selected = all_domains

        # Discover scans
        scans = _iter_all_scans(
            db_path, console,
            exclude_type="engineering" if no_engineering else None,
            science_only=science,
        )
        console.print(f"[bold]Backfill[/bold] — {len(scans)} scan(s), {len(selected)} domain(s): {', '.join(selected)}")

        if dry_run:
            console.print("\n[yellow]DRY RUN — no changes made[/yellow]")
            raise typer.Exit(code=0)

        from sherloc_pipeline.services.fitting import FittingService

        service = FittingService(console=console, database_path=db_path)

        # Track totals per domain
        domain_stats: dict = {}
        all_errors: List[str] = []

        for domain_name in selected:
            is_fluor = domain_name == "fluorescence"
            label = "Fitting fluorescence" if is_fluor else f"Persisting {domain_name}"

            domain_peaks = 0
            domain_errors: List[str] = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"{label}...", total=len(scans))
                for sol_num, tgt, scn in scans:
                    progress.update(task, description=f"{label}: Sol {sol_num} {tgt} {scn}")
                    try:
                        if is_fluor:
                            result = service.fit_fluorescence(
                                sol=f"{sol_num:04d}", target=tgt or "", scan=scn,
                            )
                            domain_peaks += (result.metadata or {}).get("peaks_inserted", 0)
                        else:
                            result = service.persist_raman_peaks(
                                sol=f"{sol_num:04d}", target=tgt or "", scan=scn,
                                domain=domain_name,
                                data_dir=data_dir,
                                results_dir=results_dir,
                            )
                            domain_peaks += (result.metadata or {}).get("peaks_inserted", 0)
                    except Exception as e:
                        domain_errors.append(f"Sol {sol_num} {tgt} {scn}: {e}")
                    progress.advance(task)

            domain_stats[domain_name] = {"peaks": domain_peaks, "errors": len(domain_errors)}
            all_errors.extend([f"[{domain_name}] {e}" for e in domain_errors])

            if domain_errors:
                console.print(f"  [yellow]{domain_name}: {domain_peaks} peaks, {len(domain_errors)} error(s)[/yellow]")
            else:
                console.print(f"  [green]{domain_name}: {domain_peaks} peaks[/green]")

        # Summary
        total_peaks = sum(s["peaks"] for s in domain_stats.values())
        total_errors = sum(s["errors"] for s in domain_stats.values())

        if json_mode:
            import json as json_mod
            output = CLIResult(
                pipeline_version=_pipeline_version,
                command="backfill",
                result={
                    "scans_processed": len(scans),
                    "total_peaks": total_peaks,
                    "domain_stats": domain_stats,
                    "errors": all_errors,
                },
            )
            print(json_mod.dumps(output.model_dump(), default=str))
        else:
            console.print(f"\n[bold]Backfill complete:[/bold]")
            console.print(f"  Scans: {len(scans)}")
            console.print(f"  Total peaks: {total_peaks}")
            for d, s in domain_stats.items():
                console.print(f"    {d}: {s['peaks']} peaks")
            if total_errors:
                console.print(f"  Errors: {total_errors}")

            if all_errors:
                console.print(f"\n[yellow]Errors ({len(all_errors)}):[/yellow]")
                for err in all_errors[:30]:
                    console.print(f"  - {err}")
                if len(all_errors) > 30:
                    console.print(f"  ... and {len(all_errors) - 30} more")

        sys.exit(1 if all_errors else 0)

    except SherlocServiceError as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=e.message,
                exit_code=e.exit_code,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Backfill failed: {e.message}[/red]")
        sys.exit(e.exit_code)
    except typer.Exit:
        raise
    except Exception as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Backfill failed: {e}[/red]")
        sys.exit(1)


@app.command("extract-training")
def extract_training_cmd(
    ctx: typer.Context,
    output: Path = typer.Option(
        "outputs/training_data.jsonl", "--output", "-o",
        help="Output JSONL file path"
    ),
    database: Optional[Path] = typer.Option(
        None, "--database", "-d",
        help="Database path (defaults to ./phase.db)"
    ),
    snr_threshold: float = typer.Option(
        2.0, "--snr", help="Minimum SNR threshold for peak inclusion"
    ),
):
    """Extract unified training data as JSONL from fitted peaks.

    Queries all persisted peaks (SNR >= threshold), groups by scan point,
    and produces cross-modal feature records with Raman + fluorescence peaks,
    doublet ratios, and auto-generated phase labels.

    Format per spec section 9.3: each line is a JSON object with "input"
    (textual description of peaks at a scan point) and "output" (phase label).

    Examples:
        sherloc extract-training
        sherloc extract-training --output ./train.jsonl --snr 3.0
    """
    json_mode = (ctx.obj or {}).get("json", False)
    if json_mode:
        import logging as _logging
        _logging.basicConfig(stream=sys.stderr)

    try:
        db_path = database or Path("./phase.db")
        if not db_path.exists():
            console.print(f"[red]Database not found: {db_path}[/red]")
            raise typer.Exit(code=1)

        from sherloc_pipeline.services.fitting import FittingService

        service = FittingService(console=console, database_path=db_path)
        result = service.extract_training_jsonl(
            output_path=output,
            snr_threshold=snr_threshold,
        )

        if json_mode:
            import json as json_mod
            output_obj = CLIResult(
                pipeline_version=_pipeline_version,
                command="extract-training",
                result={
                    "summary": result.summary,
                    "output_path": str(output),
                },
                metadata=result.metadata or {},
            )
            print(json_mod.dumps(output_obj.model_dump(), default=str))
        else:
            console.print(f"\n[green]{result.summary}[/green]")
            if result.metadata:
                console.print(f"  Peaks queried: {result.metadata.get('total_peaks_queried', 0)}")
                console.print(f"  Records written: {result.metadata.get('total_records', 0)}")
                console.print(f"  Doublets detected: {result.metadata.get('total_doublets', 0)}")

        sys.exit(0)

    except SherlocServiceError as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=e.message,
                exit_code=e.exit_code,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Training data extraction failed: {e.message}[/red]")
        sys.exit(e.exit_code)
    except typer.Exit:
        raise
    except Exception as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Training data extraction failed: {e}[/red]")
        sys.exit(1)


@app.command("reclassify-targets")
def reclassify_targets_cmd(
    ctx: typer.Context,
    database: Optional[Path] = typer.Option(
        None, "--database", "-d",
        help="Database path (defaults to ./phase.db)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show changes without writing"),
):
    """Re-run target classification on all scans.

    Updates the target_type column for every scan by re-applying
    classify_target_type(target, scan_name). Use this after updating
    classification rules in models/spectra.py.

    Examples:
        sherloc reclassify-targets
        sherloc reclassify-targets --dry-run
    """
    json_mode = (ctx.obj or {}).get("json", False)
    if json_mode:
        import logging as _logging
        _logging.basicConfig(stream=sys.stderr)

    try:
        from sherloc_pipeline.database.connection import get_engine, get_session
        from sherloc_pipeline.database.models import ScanORM
        from sherloc_pipeline.models.spectra import classify_target_type

        db_path = database or Path("./phase.db")
        if not db_path.exists():
            console.print(f"[red]Database not found: {db_path}[/red]")
            raise typer.Exit(code=1)

        engine = get_engine(db_path)
        changed = 0
        total = 0

        with get_session(engine) as session:
            scans = session.query(ScanORM).all()
            total = len(scans)

            for scan_orm in scans:
                new_type = classify_target_type(scan_orm.target, scan_orm.scan_name)
                if scan_orm.target_type != new_type:
                    if not dry_run:
                        scan_orm.target_type = new_type
                    changed += 1

        if json_mode:
            import json as json_mod
            output = CLIResult(
                pipeline_version=_pipeline_version,
                command="reclassify-targets",
                result={
                    "changed": changed,
                    "total": total,
                    "dry_run": dry_run,
                },
            )
            print(json_mod.dumps(output.model_dump(), default=str))
        elif dry_run:
            console.print(f"[yellow]DRY RUN — {changed}/{total} scans would change[/yellow]")
        else:
            console.print(f"[green]Reclassified {changed}/{total} scans[/green]")

        sys.exit(0)

    except typer.Exit:
        raise
    except Exception as e:
        if json_mode:
            import json as json_mod
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json_mod.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"\n[red]Reclassify failed: {e}[/red]")
        sys.exit(1)


@app.command("init")
def init_cmd(
    ctx: typer.Context,
    database: Optional[Path] = typer.Option(
        None, "--database", "-d",
        help="Database path (default: $SHERLOC_DB or ./phase.db)",
    ),
    mode: str = typer.Option(
        "empty", "--mode",
        help="Workflow hint shown in next-step guidance: pds | loupe | empty",
    ),
    sherloc_home: Optional[Path] = typer.Option(
        None, "--sherloc-home",
        help="Base directory to scaffold (default: $SHERLOC_HOME or .)",
    ),
) -> None:
    """Bootstrap a SHERLOC working directory and database.

    Creates the SHERLOC_HOME directory tree (data/, outputs/,
    .cache/sherloc/), runs Alembic migrations against the target
    database (creating it if missing), validates the home directory
    is writable, and prints next-step guidance.

    Examples:
        sherloc init
        sherloc init --mode pds
        sherloc init --database ./phase.db --mode loupe
    """
    import os
    from sherloc_pipeline.database import init_database

    json_mode = (ctx.obj or {}).get("json", False)

    accepted_modes = {"pds", "loupe", "empty"}
    if mode not in accepted_modes:
        console.print(
            f"[red]--mode must be one of {sorted(accepted_modes)}, got '{mode}'[/red]"
        )
        raise typer.Exit(code=1)

    home_root = Path(
        sherloc_home or os.environ.get("SHERLOC_HOME") or "."
    ).resolve()

    db_path = Path(
        database or os.environ.get("SHERLOC_DB") or (home_root / "phase.db")
    )

    subdirs = [
        home_root / "data",
        home_root / "outputs",
        home_root / ".cache" / "sherloc",
    ]

    try:
        home_root.mkdir(parents=True, exist_ok=True)
        for sub in subdirs:
            sub.mkdir(parents=True, exist_ok=True)

        # Writability probe — Alembic itself would fail later, but a
        # dedicated check produces a clearer error.
        probe = home_root / ".cache" / "sherloc" / ".sherloc-init-probe"
        try:
            probe.write_text("ok")
            probe.unlink()
        except OSError as exc:
            console.print(
                f"[red]SHERLOC_HOME is not writable: {home_root} ({exc})[/red]"
            )
            raise typer.Exit(code=1)

        init_database(db_path)

    except typer.Exit:
        raise
    except Exception as e:
        if json_mode:
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"[red]sherloc init failed: {e}[/red]")
        sys.exit(1)

    next_step_hints = {
        "pds": "sherloc pds-download --sol 921",
        "loupe": "sherloc ingest <path/to/loupe-export.zip>",
        "empty": "sherloc --help",
    }
    next_step = next_step_hints[mode]

    if json_mode:
        output = CLIResult(
            pipeline_version=_pipeline_version,
            command="init",
            result={
                "sherloc_home": str(home_root),
                "database": str(db_path),
                "mode": mode,
                "created": [str(p) for p in subdirs],
                "next_step": next_step,
            },
        )
        print(json.dumps(output.model_dump(), default=str))
    else:
        console.print(f"[green]Initialized SHERLOC_HOME at {home_root}[/green]")
        console.print(f"[green]Database ready at {db_path}[/green]")
        console.print(f"\nNext: [cyan]{next_step}[/cyan]")


@app.command("pds-download")
def pds_download_cmd(
    ctx: typer.Context,
    sol: Optional[int] = typer.Option(
        None, "--sol",
        help="Single sol number to download",
    ),
    sol_range: Optional[Tuple[int, int]] = typer.Option(
        None, "--sol-range",
        help="Sol range START END (inclusive)",
    ),
    auto: bool = typer.Option(
        False, "--auto",
        help="Download all sols available in the PDS inventory",
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="PDS cache directory (default: ./pds)",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-download even if files exist locally",
    ),
) -> None:
    """Download SHERLOC processed products from the PDS Geosciences Node.

    Wraps the PDSDownloader. Requires the 'pds' install extra.

    Modes (mutually exclusive, exactly one required):

        --sol N            Download a single sol
        --sol-range S E    Download all sols in [S, E] (intersected with PDS inventory)
        --auto             Download all sols listed in the PDS inventory

    Examples:
        sherloc pds-download --sol 921
        sherloc pds-download --sol-range 100 1000
        sherloc pds-download --auto --output-dir ./data/pds
    """
    json_mode = (ctx.obj or {}).get("json", False)

    modes = sum([sol is not None, sol_range is not None, auto])
    if modes == 0:
        console.print("[red]Specify one of --sol, --sol-range, or --auto[/red]")
        raise typer.Exit(code=1)
    if modes > 1:
        console.print("[red]--sol, --sol-range, and --auto are mutually exclusive[/red]")
        raise typer.Exit(code=1)

    cache_dir = Path(output_dir) if output_dir else PDS_DEFAULT_CACHE_DIR

    try:
        from sherloc_pipeline.services.pds_ingestion import PDSDownloader
    except ImportError as exc:
        console.print(
            r"[red]PDS extras not installed. "
            r"Install with: pip install 'sherloc-pipeline\[pds]'[/red]"
        )
        if json_mode:
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type="ImportError",
                message=str(exc),
                exit_code=1,
            )
            print(json.dumps(err.model_dump(), default=str), file=sys.stderr)
        raise typer.Exit(code=1)

    try:
        with PDSDownloader(cache_dir=cache_dir) as downloader:
            # Probe httpx early so a missing extra surfaces before we
            # spin a Progress UI.
            try:
                _ = downloader.client
            except ImportError as exc:
                from rich.markup import escape
                console.print(f"[red]{escape(str(exc))}[/red]")
                raise typer.Exit(code=1)

            if sol is not None:
                sol_numbers: List[int] = [sol]
            elif sol_range is not None:
                start, end = sol_range
                if start > end:
                    console.print(
                        f"[red]Invalid range: start ({start}) > end ({end})[/red]"
                    )
                    raise typer.Exit(code=1)
                available = downloader.discover_available_sols()
                sol_numbers = [s for s in available if start <= s <= end]
                if not sol_numbers:
                    console.print(
                        f"[yellow]No sols in PDS inventory match range {start}-{end}[/yellow]"
                    )
                    raise typer.Exit(code=0)
            else:  # auto
                sol_numbers = downloader.discover_available_sols()
                if not sol_numbers:
                    console.print("[yellow]No sols found in PDS inventory[/yellow]")
                    raise typer.Exit(code=0)

            console.print(
                f"[bold]PDS Download[/bold] — {len(sol_numbers)} sol(s) → {cache_dir}"
            )
            if force:
                console.print("[yellow]FORCE — re-downloading existing files[/yellow]")

            t0 = time.monotonic()
            total_downloaded = 0
            total_skipped = 0
            total_errors = 0

            for s in sol_numbers:
                result = downloader.download_sol(s, force=force)
                total_downloaded += result.n_downloaded
                total_skipped += result.n_skipped
                total_errors += len(result.errors)
                console.print(
                    f"  sol {s}: downloaded={result.n_downloaded} "
                    f"skipped={result.n_skipped} errors={len(result.errors)}"
                )

            elapsed = time.monotonic() - t0

    except typer.Exit:
        raise
    except Exception as e:
        if json_mode:
            err = CLIError(
                pipeline_version=_pipeline_version,
                error_type=type(e).__name__,
                message=str(e),
                exit_code=1,
            )
            print(json.dumps(err.model_dump(), default=str), file=sys.stderr)
        else:
            console.print(f"[red]sherloc pds-download failed: {e}[/red]")
        sys.exit(1)

    if json_mode:
        output = CLIResult(
            pipeline_version=_pipeline_version,
            command="pds-download",
            result={
                "sols": sol_numbers,
                "output_dir": str(cache_dir),
                "downloaded": total_downloaded,
                "skipped": total_skipped,
                "errors": total_errors,
                "elapsed_seconds": round(elapsed, 3),
            },
        )
        print(json.dumps(output.model_dump(), default=str))
    else:
        console.print(
            f"\n[bold]Total:[/bold] {total_downloaded} downloaded, "
            f"{total_skipped} skipped, {total_errors} errors "
            f"in {elapsed:.1f}s"
        )

    if total_errors > 0:
        sys.exit(1)


@app.command("serve")
def serve(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8002, help="Bind port"),
    database: Optional[Path] = typer.Option(None, help="Database path"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes"),
) -> None:
    """Start the SHERLOC web interface."""
    try:
        import uvicorn
        from sherloc_pipeline.web.app import create_app
    except ImportError:
        console.print(
            "[red]Web extras not installed. Install with: pip install sherloc-pipeline[web][/red]"
        )
        sys.exit(1)

    console.print(f"Starting SHERLOC Web API on {host}:{port}")
    if database:
        console.print(f"Database: {database}")

    web_app = create_app(database_path=database)
    uvicorn.run(web_app, host=host, port=port, reload=reload)


def main():
    """Entry point for the sherloc CLI."""
    app()


if __name__ == "__main__":
    main()
