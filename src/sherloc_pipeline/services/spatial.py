"""
Spatial service for SHERLOC pipeline.

This module provides orchestration services for generating spatial overlays,
rendering accepted-peak locations on ACI images, and merging labels across scans.

The SpatialService encapsulates the orchestration logic that was previously in the CLI
module, providing a clean service interface that can be used by both CLI commands
and programmatic consumers.

Usage:
    from sherloc_pipeline.services.spatial import SpatialService
    from rich.console import Console
    
    service = SpatialService(console=Console())
    result = service.render_overlay(
        sol="0921",
        target="Amherst_Point",
        scan="detail_1",
        layers="minerals,organics,hydration",
        upscale=3
    )
    print(result.summary)
    for artifact in result.artifacts:
        print(f"  {artifact}")
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .base import ServiceResult
from .errors import SpatialError, enrich
from .paths import resolve_scan_context
from .runtime import RuntimeContext

logger = logging.getLogger(__name__)

# Alias for numpy to match CLI code
_np = np


class SpatialService:
    """Service for orchestrating spatial overlay operations.
    
    This service coordinates:
    1. Loading ACI images and spatial tables
    2. Rendering overlays for minerals, organics, and hydration layers
    3. Generating per-class mineral overlays and combined grids
    4. Merging labels across multiple scans
    
    The service maintains console output consistency by accepting an optional
    Console instance, allowing CLI commands to use their existing console while
    programmatic consumers can provide their own or use a default.
    
    Attributes:
        console: Rich Console instance for progress/output (defaults to new Console)
        
    Example:
        >>> service = SpatialService()
        >>> result = service.render_overlay("0921", "Amherst_Point", "detail_1")
        >>> print(result.summary)
        'Spatial overlay generation complete'
        >>> print(f"Generated {len(result.artifacts)} overlay files")
    """
    
    def __init__(self, console: Optional[Console] = None, *, context: Optional[RuntimeContext] = None):
        """Initialize spatial service.
        
        Args:
            console: Optional Rich Console instance. If None, creates a new Console.
            context: Optional RuntimeContext providing resolved configuration and paths. If None,
                a new context is bootstrapped.
        """
        self.console = console if console is not None else Console()
        self.context = context if context is not None else RuntimeContext.bootstrap()
    
    def render_overlay(
        self,
        sol: str,
        target: str,
        scan: str,
        layers: str = "minerals,organics,hydration",
        upscale: int = 1,
        save_debug: bool = True,
        use_reviewed: bool = False,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        working_dir_override: Optional[Path] = None,
        overlay_suffix: Optional[str] = None,
    ) -> ServiceResult:
        """Render accepted-peak locations on the ACI image for the selected layers.
        
        This method orchestrates the complete spatial overlay generation process:
        - Loads ACI image and spatial table
        - Archives existing overlays if using reviewed data
        - Renders overlays for organics, hydration, and minerals layers
        - Generates per-class mineral overlays and combined grids
        - Optionally writes debug CSV
        
        Args:
            sol: Sol number (e.g., "0921")
            target: Target name (e.g., "Amherst_Point")
            scan: Scan name (e.g., "detail_1")
            layers: Comma-separated layers to render (default: "minerals,organics,hydration")
            upscale: Upscale factor for overlays (1 = native, 3 = zoom)
            save_debug: Write spatial_debug.csv
            use_reviewed: Use scan-level accepted table and user_keep flags; archive existing overlays first
            data_dir: Base data directory. If None, uses config default.
            results_dir: Results directory. If None, uses config default.
            working_dir_override: If provided, use this Loupe working directory
                instead of discovering one from the filesystem. Useful for
                alternate imagery (e.g., colorized ACI) sharing the same spectra.
            overlay_suffix: If provided, append to the overlay output directory
                name (e.g., "colorized" → "spatial_overlays_colorized").

        Returns:
            ServiceResult with summary, artifacts (overlay PNGs, debug CSV if enabled), warnings, and metadata
            
        Raises:
            SpatialError: If overlay generation fails (missing imagery, invalid paths, etc.)
            
        Example:
            >>> service = SpatialService()
            >>> result = service.render_overlay(
            ...     sol="0921",
            ...     target="Amherst_Point",
            ...     scan="detail_1",
            ...     layers="minerals,organics",
            ...     upscale=3
            ... )
            >>> print(result.summary)
            'Spatial overlay generation complete'
        """
        run_context = self.context.with_overrides(
            data_dir=data_dir,
            results_dir=results_dir,
        )
        cfg = run_context.config

        try:
            from sherloc_pipeline.core.data_ingestion import DataIngestion
            from sherloc_pipeline.core.spatial import (
                load_spatial_table,
                build_spatial_crop,
                _save_rgb_image_pil,
                BASELINE_SOFTWARE_TAG,
            )
            from sherloc_pipeline.visualization.spatial import (
                overlay_points_on_aci,
                render_pointloc_full,
                render_pointloc_zoomed,
                render_pointloc_with_colorbar,
                build_combined_grid,
            )
        except ImportError as e:
            error = SpatialError(
                f"Failed to import required modules: {e}",
                exit_code=1,
            )
            raise enrich(error, sol=sol, target=target, scan=scan) from e

        if working_dir_override is not None:
            # Skip full scan context resolution — construct only what we need
            working_dir = Path(working_dir_override)
            if not working_dir.exists():
                error = SpatialError(
                    f"Working directory override does not exist: {working_dir}",
                    exit_code=1,
                )
                raise enrich(error, sol=sol, target=target, scan=scan)
            results_dir_path = Path(run_context.results_root)
            from sherloc_pipeline.core.data_ingestion import normalize_target_name
            norm_target = normalize_target_name(target)
            results_path = results_dir_path / norm_target / f"{sol}_{scan}"
            # Lightweight context stand-in
            context = type('_Ctx', (), {
                'base_data_dir': working_dir.parent.parent.parent,
                'results_dir': results_dir_path,
                'results_path': results_path,
            })()
            ingestion = DataIngestion(
                base_data_dir=context.base_data_dir,
                results_dir=context.results_dir,
                sol=sol,
                target=target,
                scan=scan,
            )
        else:
            try:
                context = resolve_scan_context(
                    sol=sol,
                    target=target,
                    scan=scan,
                    data_dir=data_dir,
                    results_dir=results_dir,
                    context=run_context,
                )
            except (FileNotFoundError, ValueError) as e:
                error = SpatialError(
                    f"Failed to resolve scan context: {e}",
                    exit_code=1,
                    context={"sol": sol, "target": target, "scan": scan},
                )
                raise enrich(error, sol=sol, target=target, scan=scan) from e

            ingestion = DataIngestion(
                base_data_dir=context.base_data_dir,
                results_dir=context.results_dir,
                sol=sol,
                target=target,
                scan=scan,
            )
            working_dir = ingestion.find_working_directory(sol, scan)
            if not working_dir:
                error = SpatialError(
                    f"No working directory found for sol {sol}, scan {scan}",
                    exit_code=1,
                )
                raise enrich(error, sol=sol, target=target, scan=scan)
        
        try:
            self.console.print(f"[blue]Working dir:[/blue] {working_dir}")
        except Exception:
            pass
        
        artifacts = []
        warnings = []
        
        # Use progress bars for overlay generation
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                
                task_load = progress.add_task("Loading ACI image and spatial table...", total=None)
                
                try:
                    # Load ACI image (optionally upscale)
                    image, meta = ingestion.load_aci_image(working_dir)
                    # Preserve native ACI for full-frame artifacts
                    image_native = image.copy() if hasattr(image, 'copy') else image
                    if upscale and upscale > 1:
                        image, up_meta = ingestion.upscale_image(image, factor=upscale)
                    
                    # Spatial table with pixel coords (respect upscaling)
                    spatial_df = load_spatial_table(working_dir, scale_factor=float(upscale or 1))
                    
                    progress.update(task_load, description="✅ ACI image and spatial table loaded")
                    
                    # Resolve base results and overlay dir
                    base = context.results_path
                    dir_name = "spatial_overlays"
                    if overlay_suffix:
                        dir_name = f"spatial_overlays_{overlay_suffix}"
                    overlay_dir = base / dir_name
                    
                    # If using reviewed peaks, archive existing overlays by renaming directory
                    if use_reviewed and overlay_dir.exists():
                        n = 0
                        while True:
                            suffix = f"spatial_overlays_archive_{n}" if n > 0 else "spatial_overlays_archive"
                            candidate = base / suffix
                            if not candidate.exists():
                                break
                            n += 1
                        try:
                            shutil.move(str(overlay_dir), str(candidate))
                            try:
                                self.console.print(f"[yellow]Archived existing overlays to {candidate}[/yellow]")
                            except Exception:
                                pass
                        except Exception as e:
                            warning_msg = f"failed to archive overlays: {e}"
                            warnings.append(warning_msg)
                            logger.warning(warning_msg)
                    overlay_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Load unified accepted table if present
                    reviewed_df = None
                    try:
                        acc_path = base / f"{sol}_{target}_{scan}_accepted_peaks.csv"
                        if acc_path.exists():
                            reviewed_df = pd.read_csv(acc_path)
                        # Normalize booleans if saved as strings
                        for col in ["keep", "user_keep"]:
                            if col in reviewed_df.columns:
                                reviewed_df[col] = reviewed_df[col].astype(str).str.lower().isin(["true", "1", "yes"])
                        # Attempt to backfill minerals label_id if missing
                        try:
                            need_map = ((reviewed_df['modality'] == 'minerals') & 
                                       (reviewed_df['label_id'].astype(str).str.strip() == '')).any()
                        except Exception:
                            need_map = False
                        if need_map:
                            try:
                                from sherloc_pipeline.core.mineral_id import load_mineral_rules, map_min_id_series
                                fitting_cfg = getattr(cfg, 'fitting', {})
                                try:
                                    library_path = fitting_cfg.get('library_path') if isinstance(fitting_cfg, dict) else getattr(fitting_cfg, 'library_path', None)
                                except Exception:
                                    library_path = None
                                try:
                                    inline_rules = fitting_cfg.get('mineral_rules') if isinstance(fitting_cfg, dict) else getattr(fitting_cfg, 'mineral_rules', None)
                                except Exception:
                                    inline_rules = None
                                _rules = load_mineral_rules(Path(library_path) if library_path else None, inline_rules=inline_rules)
                                mask = reviewed_df['modality'] == 'minerals'
                                reviewed_df.loc[mask, 'label_id'] = map_min_id_series(reviewed_df.loc[mask, 'mean'], _rules)
                            except Exception:
                                pass
                    except Exception:
                        reviewed_df = None
                    
                    # Helper: map accepted points to max SNR per point
                    def _load_points_from_csv(path: Path, point_col: str = 'point', snr_col: str = 'snr', where=None) -> pd.DataFrame:
                        if not path.exists():
                            return pd.DataFrame(columns=['point', 'snr']).astype({'point': int, 'snr': float})
                        df = pd.read_csv(path)
                        if where is not None:
                            df = df.query(where)
                        if point_col not in df.columns or snr_col not in df.columns:
                            return pd.DataFrame(columns=['point', 'snr']).astype({'point': int, 'snr': float})
                        # consolidate by point, taking max snr
                        agg = df.groupby(point_col)[snr_col].max().reset_index().rename(columns={point_col: 'point', snr_col: 'snr'})
                        return agg
                    
                    layers_set = set([s.strip().lower() for s in str(layers).split(',') if s.strip()])
                    
                    overlays = []
                    debug_rows = spatial_df.copy()
                    # Initialize debug flags
                    for col in ["organics_DG", "hydration_OH", "hydration_bend"]:
                        if col not in debug_rows.columns:
                            debug_rows[col] = False
                    # Also keep SNR columns for reference
                    for col in ["organics_snr", "hydration_oh_snr"]:
                        if col not in debug_rows.columns:
                            debug_rows[col] = np.nan
                    
                    # Build crop in native coordinates (pad=50 px), then upscale for zoomed panels
                    scale = float(upscale or 1.0)
                    spatial_native = spatial_df[["point", "x", "y"]].copy()
                    spatial_native["xPix"] = spatial_df["xPix"] / scale
                    spatial_native["yPix"] = spatial_df["yPix"] / scale
                    crop_native, (xmin, ymin, xmax, ymax) = build_spatial_crop(
                        image=image_native, spatial_df=spatial_native, upscale=1.0, pad_px=50.0
                    )
                    # Always create a 3× (or chosen upscale) crop to match notebook's cropped_original.png
                    crop_img = crop_native
                    if scale > 1.0:
                        try:
                            import skimage.transform as _sktr
                            if crop_native.ndim == 2:
                                crop_img = _sktr.resize(
                                    crop_native,
                                    (int(crop_native.shape[0] * scale), int(crop_native.shape[1] * scale)),
                                    mode='reflect',
                                    anti_aliasing=True
                                )
                            else:
                                crop_img = _sktr.resize(
                                    crop_native,
                                    (int(crop_native.shape[0] * scale), int(crop_native.shape[1] * scale), crop_native.shape[2]),
                                    mode='reflect',
                                    anti_aliasing=True
                                )
                        except Exception:
                            pass
                    # Save cropped_original.png as RGB (preserve original intensities; convert floats to 8-bit)
                    try:
                        out_crop = overlay_dir / "cropped_original.png"
                        arr = crop_img
                        if arr.dtype != np.float32 and arr.dtype != np.float64:
                            arr = arr.astype(np.float32) / 255.0
                        else:
                            arr = arr.astype(np.float32)
                        arr = np.clip(arr, 0.0, 1.0)
                        if arr.ndim == 2:
                            arr = np.stack([arr, arr, arr], axis=-1)
                        elif arr.ndim == 3 and arr.shape[2] > 3:
                            arr = arr[:, :, :3]
                        _save_rgb_image_pil(
                            arr,
                            out_crop,
                            metadata={"Software": BASELINE_SOFTWARE_TAG},
                        )
                        artifacts.append(out_crop)
                        try:
                            self.console.print(f"[blue]Cropped ACI:[/blue] {out_crop}")
                        except Exception:
                            pass
                    except Exception as e:
                        logger.debug(f"crop save skipped: {e}")
                    
                    # Process each layer (organics, hydration, minerals)
                    # Organics layer
                    if 'organics' in layers_set:
                        task_org = progress.add_task("Rendering organics overlay...", total=None)
                        org_csv = base / "organics_fit" / f"{sol}_{target}_{scan}_R1_organics_accepted_peaks.csv"
                        # Prefer reviewed unified table when requested; otherwise fall back to per-modality CSV
                        if use_reviewed and reviewed_df is not None:
                            _df = reviewed_df[
                                (reviewed_df['modality'] == 'organics') &
                                (reviewed_df.get('user_keep', True) == True) &
                                (reviewed_df['label_id'].astype(str).str.upper() == 'G')
                            ]
                            if not _df.empty:
                                org_pts = _df.groupby('point')['snr'].max().reset_index().rename(columns={'point': 'point', 'snr': 'snr'})
                                try:
                                    org_pts['point'] = org_pts['point'].astype(int)
                                    org_pts['snr'] = org_pts['snr'].astype(float)
                                except Exception:
                                    pass
                            else:
                                org_pts = pd.DataFrame(columns=['point', 'snr']).astype({'point': int, 'snr': float})
                        else:
                            # Use G (and legacy DG) rows per prior behavior
                            try:
                                if org_csv.exists():
                                    _df = pd.read_csv(org_csv)
                                    if 'band' in _df.columns:
                                        _df = _df[_df['band'].astype(str).str.upper().isin(['G', 'DG'])]
                                    org_pts = _df.groupby('point')['snr'].max().reset_index().rename(columns={'point': 'point', 'snr': 'snr'})
                                else:
                                    org_pts = pd.DataFrame(columns=['point', 'snr']).astype({'point': int, 'snr': float})
                            except Exception:
                                org_pts = _load_points_from_csv(org_csv)
                        
                        if not org_pts.empty:
                            m = spatial_df.merge(org_pts, on='point', how='inner')
                            pts = m[["xPix", "yPix"]].to_numpy(float)
                            snr_vals = m["snr"].to_numpy(float)
                            # Non-selected points: hollow white, alpha 0.3
                            sel_points = set(int(p) for p in m['point'].tolist())
                            non_sel = spatial_df[~spatial_df['point'].isin(sel_points)]
                            if not non_sel.empty:
                                pixel_scale = float(meta.get('pixel_scale', 10.1)) / float(upscale or 1)
                                diameter_um = 100.0
                                radius_px_ns = (diameter_um / pixel_scale) / 2.0
                                overlays.append({
                                    "points": non_sel[["xPix", "yPix"]].to_numpy(float),
                                    "facecolors": 'none',
                                    "edgecolor": 'white',
                                    "alpha": 0.3,
                                    "radius_px": radius_px_ns,
                                    "linewidth_px": 1.0,
                                    "antialiased": False,
                                    "label": None,
                                })
                            # Selected points: hollow, colored edge by SNR using viridis
                            import matplotlib
                            cmap = matplotlib.colormaps.get_cmap("viridis")
                            vmin = float(np.nanmin(snr_vals))
                            vmax = float(np.nanmax(snr_vals))
                            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                                vmin, vmax = 0.0, 1.0
                            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                            edge_colors = cmap(norm(snr_vals))
                            # Compute radius in pixels for ~100 µm diameter circles
                            pixel_scale = float(meta.get('pixel_scale', 10.1)) / float(upscale or 1)
                            diameter_um = 100.0
                            radius_px = (diameter_um / pixel_scale) / 2.0
                            overlays.append({
                                "points": pts,
                                "facecolors": 'none',
                                "edgecolor": edge_colors,
                                "radius_px": radius_px,
                                "ring_width_px": 2.0,
                                "linewidth_px": 1.0,
                                "antialiased": False,
                                "alpha": 1.0,
                                "label": "Organics (G)",
                                "detections": True,
                            })
                            # update debug flags
                            debug_rows.loc[debug_rows['point'].isin(m['point']), 'organics_DG'] = True
                            debug_rows.loc[debug_rows['point'].isin(m['point']), 'organics_snr'] = (
                                debug_rows.loc[debug_rows['point'].isin(m['point'])]
                                .merge(org_pts, on='point', how='left')['snr'].values
                            )
                            
                            # Triple artifacts for organics G/DG (notebook-style names)
                            scale = float(upscale or 1.0)
                            pts_native = pts / scale
                            ns_native = non_sel[["xPix", "yPix"]].to_numpy(float) / scale if not non_sel.empty else None
                            out_full = overlay_dir / f"{sol}_{target}_{scan}_organics_G_pointloc.png"
                            render_pointloc_full(
                                image_native=image_native,
                                pixel_scale_um_per_px=float(meta.get('pixel_scale', 10.1)),
                                selected_xy=pts_native,
                                selected_snr=snr_vals,
                                nonselected_xy=ns_native,
                                out_path=out_full
                            )
                            artifacts.append(out_full)
                            pts_crop = pts.copy()
                            pts_crop[:, 0] -= (xmin * scale)
                            pts_crop[:, 1] -= (ymin * scale)
                            ns_crop = None
                            if not non_sel.empty:
                                # copy=True: pandas returns a read-only view of
                                # contiguous frame data; we mutate in-place to
                                # translate into the crop frame.
                                ns_arr = non_sel[["xPix", "yPix"]].to_numpy(float, copy=True)
                                ns_arr[:, 0] -= (xmin * scale)
                                ns_arr[:, 1] -= (ymin * scale)
                                ns_crop = ns_arr
                            out_zoom = overlay_dir / f"{sol}_{target}_{scan}_organics_G_pointloc_zoomed.png"
                            render_pointloc_zoomed(
                                crop_image_native=crop_native,
                                selected_xy_crop_native=_np.column_stack([pts_native[:, 0] - xmin, pts_native[:, 1] - ymin]),
                                selected_snr=snr_vals,
                                nonselected_xy_crop_native=(
                                    _np.column_stack([ns_native[:, 0] - xmin, ns_native[:, 1] - ymin])
                                    if ns_native is not None else None
                                ),
                                out_path=out_zoom,
                                upscale=float(upscale or 1.0),
                            )
                            artifacts.append(out_zoom)
                            out_cb = overlay_dir / f"{sol}_{target}_{scan}_organics_G_pointloc_with_colorbar.png"
                            render_pointloc_with_colorbar(
                                image_native=image_native,
                                pixel_scale_um_per_px=float(meta.get('pixel_scale', 10.1)),
                                selected_xy=pts_native,
                                selected_snr=snr_vals,
                                nonselected_xy=ns_native,
                                title=f"{sol} {target} {scan} points with organics_G SNR > 3\nprocessed data, colorized by SNR",
                                out_path=out_cb
                            )
                            artifacts.append(out_cb)
                            overlays = []  # reset for next layer
                            progress.update(task_org, description="✅ Organics overlay complete")
                    
                    # Hydration layer (OH only)
                    if 'hydration' in layers_set:
                        task_hyd = progress.add_task("Rendering hydration overlay...", total=None)
                        hyd_csv = base / "hydration_fit" / f"{sol}_{target}_{scan}_R1_hydration_accepted_peaks.csv"
                        if use_reviewed and reviewed_df is not None:
                            _df = reviewed_df[
                                (reviewed_df['modality'] == 'hydration') &
                                (reviewed_df.get('user_keep', True) == True) &
                                (reviewed_df['label_id'].astype(str).str.upper() == 'OH')
                            ]
                            if not _df.empty:
                                hyd_pts = _df.groupby('point')['snr'].max().reset_index().rename(columns={'point': 'point', 'snr': 'snr'})
                                try:
                                    hyd_pts['point'] = hyd_pts['point'].astype(int)
                                    hyd_pts['snr'] = hyd_pts['snr'].astype(float)
                                except Exception:
                                    pass
                            else:
                                hyd_pts = pd.DataFrame(columns=['point', 'snr']).astype({'point': int, 'snr': float})
                        else:
                            hyd_pts = _load_points_from_csv(hyd_csv, where="band == 'OH'")
                        
                        if not hyd_pts.empty:
                            m = spatial_df.merge(hyd_pts, on='point', how='inner')
                            pts = m[["xPix", "yPix"]].to_numpy(float)
                            snr_vals = m["snr"].to_numpy(float)
                            # Non-selected points: hollow white, alpha 0.3
                            sel_points = set(int(p) for p in m['point'].tolist())
                            non_sel = spatial_df[~spatial_df['point'].isin(sel_points)]
                            if not non_sel.empty:
                                pixel_scale = float(meta.get('pixel_scale', 10.1)) / float(upscale or 1)
                                diameter_um = 100.0
                                radius_px_ns = (diameter_um / pixel_scale) / 2.0
                                overlays.append({
                                    "points": non_sel[["xPix", "yPix"]].to_numpy(float),
                                    "facecolors": 'none',
                                    "edgecolor": 'white',
                                    "alpha": 0.3,
                                    "radius_px": radius_px_ns,
                                    "linewidth_px": 1.0,
                                    "antialiased": False,
                                })
                            import matplotlib
                            cmap = matplotlib.colormaps.get_cmap("viridis")
                            vmin = float(np.nanmin(snr_vals))
                            vmax = float(np.nanmax(snr_vals))
                            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                                vmin, vmax = 0.0, 1.0
                            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                            edge_colors = cmap(norm(snr_vals))
                            pixel_scale = float(meta.get('pixel_scale', 10.1)) / float(upscale or 1)
                            diameter_um = 100.0
                            radius_px = (diameter_um / pixel_scale) / 2.0
                            overlays.append({
                                "points": pts,
                                "facecolors": 'none',
                                "edgecolor": edge_colors,
                                "radius_px": radius_px,
                                "ring_width_px": 2.0,
                                "linewidth_px": 1.0,
                                "antialiased": False,
                                "alpha": 1.0,
                                "label": "Hydration (OH)",
                                "detections": True,
                            })
                            debug_rows.loc[debug_rows['point'].isin(m['point']), 'hydration_OH'] = True
                            debug_rows.loc[debug_rows['point'].isin(m['point']), 'hydration_oh_snr'] = (
                                debug_rows.loc[debug_rows['point'].isin(m['point'])]
                                .merge(hyd_pts, on='point', how='left')['snr'].values
                            )
                            # Triple artifacts for hydration OH
                            scale = float(upscale or 1.0)
                            pts_native = pts / scale
                            ns_native = non_sel[["xPix", "yPix"]].to_numpy(float) / scale if not non_sel.empty else None
                            out_full = overlay_dir / f"{sol}_{target}_{scan}_hydration_OH_pointloc.png"
                            render_pointloc_full(
                                image_native=image_native,
                                pixel_scale_um_per_px=float(meta.get('pixel_scale', 10.1)),
                                selected_xy=pts_native,
                                selected_snr=snr_vals,
                                nonselected_xy=ns_native,
                                out_path=out_full
                            )
                            artifacts.append(out_full)
                            pts_crop = pts.copy()
                            pts_crop[:, 0] -= (xmin * scale)
                            pts_crop[:, 1] -= (ymin * scale)
                            ns_crop = None
                            if not non_sel.empty:
                                # copy=True: pandas returns a read-only view of
                                # contiguous frame data; we mutate in-place to
                                # translate into the crop frame.
                                ns_arr = non_sel[["xPix", "yPix"]].to_numpy(float, copy=True)
                                ns_arr[:, 0] -= (xmin * scale)
                                ns_arr[:, 1] -= (ymin * scale)
                                ns_crop = ns_arr
                            out_zoom = overlay_dir / f"{sol}_{target}_{scan}_hydration_OH_pointloc_zoomed.png"
                            render_pointloc_zoomed(
                                crop_image_native=crop_native,
                                selected_xy_crop_native=np.column_stack([pts_native[:, 0] - xmin, pts_native[:, 1] - ymin]),
                                selected_snr=snr_vals,
                                nonselected_xy_crop_native=(
                                    np.column_stack([ns_native[:, 0] - xmin, ns_native[:, 1] - ymin])
                                    if ns_native is not None else None
                                ),
                                out_path=out_zoom,
                                upscale=float(upscale or 1.0),
                            )
                            artifacts.append(out_zoom)
                            out_cb = overlay_dir / f"{sol}_{target}_{scan}_hydration_OH_pointloc_with_colorbar.png"
                            render_pointloc_with_colorbar(
                                image_native=image_native,
                                pixel_scale_um_per_px=float(meta.get('pixel_scale', 10.1)),
                                selected_xy=pts_native,
                                selected_snr=snr_vals,
                                nonselected_xy=ns_native,
                                title=f"{sol} {target} {scan} points with hydration_OH SNR > 3\nprocessed data, colorized by SNR",
                                out_path=out_cb
                            )
                            artifacts.append(out_cb)
                            progress.update(task_hyd, description="✅ Hydration overlay complete")
                    
                    # Minerals layer (raster circles) — per-class artifacts and combined grid using core APIs
                    if 'minerals' in layers_set:
                        task_min = progress.add_task("Rendering minerals overlays...", total=None)
                        min_csv = base / "minerals_fit" / f"{sol}_{target}_{scan}_R1_accepted_peaks.csv"
                        min_pts = _load_points_from_csv(min_csv)
                        # Notebook-aligned behavior: skip general minerals overlay; only per-class/aggregates below
                        # Build mapping dataframe `accepted_df` for class panels
                        if not min_csv.exists():
                            accepted_df = pd.DataFrame(columns=['point', 'center_cm1', 'snr', 'min_ID', 'r2'])
                        else:
                            if use_reviewed and reviewed_df is not None:
                                # Use reviewed unified table directly
                                _df = reviewed_df[
                                    (reviewed_df['modality'] == 'minerals') &
                                    (reviewed_df.get('user_keep', True) == True)
                                ].copy()
                                if _df.empty:
                                    accepted_df = pd.DataFrame(columns=['point', 'center_cm1', 'snr', 'min_ID', 'r2'])
                                else:
                                    _df = _df.rename(columns={'mean': 'center_cm1', 'label_id': 'min_ID'})
                                    if 'user_keep' not in _df.columns:
                                        _df['user_keep'] = _df.get('keep', True)
                                    # Normalize r2 column name from unified table
                                    try:
                                        if 'r2' not in _df.columns and 'r_squared' in _df.columns:
                                            _df['r2'] = pd.to_numeric(_df['r_squared'], errors='coerce')
                                    except Exception:
                                        pass
                                    accepted_df = _df[['point', 'center_cm1', 'snr', 'min_ID', 'r2', 'user_keep']].copy()
                                    try:
                                        accepted_df['point'] = accepted_df['point'].astype(int)
                                        accepted_df['snr'] = accepted_df['snr'].astype(float)
                                    except Exception:
                                        pass
                                    # Backfill missing/empty min_ID from rules if needed
                                    try:
                                        if accepted_df['min_ID'].astype(str).str.strip().eq('').any():
                                            from sherloc_pipeline.core.mineral_id import load_mineral_rules, map_min_id_series
                                            fitting_cfg = getattr(cfg, 'fitting', {})
                                            try:
                                                library_path = fitting_cfg.get('library_path') if isinstance(fitting_cfg, dict) else getattr(fitting_cfg, 'library_path', None)
                                            except Exception:
                                                library_path = None
                                            try:
                                                inline_rules = fitting_cfg.get('mineral_rules') if isinstance(fitting_cfg, dict) else getattr(fitting_cfg, 'mineral_rules', None)
                                            except Exception:
                                                inline_rules = None
                                            _rules = load_mineral_rules(Path(library_path) if library_path else None, inline_rules=inline_rules)
                                            mask = accepted_df['min_ID'].astype(str).str.strip().eq('')
                                            accepted_df.loc[mask, 'min_ID'] = map_min_id_series(accepted_df.loc[mask, 'center_cm1'], _rules)
                                    except Exception:
                                        pass
                            else:
                                try:
                                    from sherloc_pipeline.core.mineral_id import load_mineral_rules, map_min_id_series
                                    fitting_cfg = getattr(cfg, 'fitting', {})
                                    try:
                                        lib_path = fitting_cfg.get('library_path') if isinstance(fitting_cfg, dict) else getattr(fitting_cfg, 'library_path', None)
                                    except Exception:
                                        lib_path = None
                                    try:
                                        inline_rules = fitting_cfg.get('mineral_rules') if isinstance(fitting_cfg, dict) else getattr(fitting_cfg, 'mineral_rules', None)
                                    except Exception:
                                        inline_rules = None
                                    rules = load_mineral_rules(Path(lib_path) if lib_path else None, inline_rules=inline_rules)
                                    accepted_df = pd.read_csv(min_csv)
                                    accepted_df['min_ID'] = map_min_id_series(accepted_df['center_cm1'], rules)
                                except Exception:
                                    accepted_df = pd.read_csv(min_csv)
                                    if 'min_ID' not in accepted_df.columns:
                                        accepted_df['min_ID'] = ''
                        
                        if 'user_keep' not in accepted_df.columns:
                            accepted_df['user_keep'] = accepted_df.get('keep', True)

                        # DEBUG: Log mineral class distribution before filtering
                        if 'min_ID' in accepted_df.columns and not accepted_df.empty:
                            class_counts_before = accepted_df['min_ID'].value_counts().to_dict()
                            logger.debug(f"Mineral classes before R² filter: {class_counts_before}")

                        # Apply an R^2 gate to minerals overlays for consistency with fit PNG emission
                        # but preserve user_keep=TRUE entries regardless of R² value
                        try:
                            fitting_cfg = getattr(cfg, 'fitting', {})
                            try:
                                r2_min = float(fitting_cfg.get("r_squared_min")) if isinstance(fitting_cfg, dict) else float(getattr(fitting_cfg, "r_squared_min", 0.0))
                            except Exception:
                                r2_min = 0.0
                        
                            # Preserve user_keep=TRUE entries regardless of R²
                            user_keep_mask = accepted_df.get("user_keep", False) == True

                            if "r2" in accepted_df.columns:
                                accepted_df["r2"] = pd.to_numeric(accepted_df["r2"], errors="coerce")
                                r2_mask = accepted_df["r2"] >= r2_min
                                # Keep entries that either pass R² threshold OR have user_keep=TRUE
                                filtered_count = len(accepted_df)
                                accepted_df = accepted_df[r2_mask | user_keep_mask]
                                if filtered_count > len(accepted_df):
                                    logger.debug(f"R² filter removed {filtered_count - len(accepted_df)} peaks (r2_min={r2_min})")
                            elif "r_squared" in accepted_df.columns:
                                accepted_df["r_squared"] = pd.to_numeric(accepted_df["r_squared"], errors="coerce")
                                r2_mask = accepted_df["r_squared"] >= r2_min
                                # Keep entries that either pass R² threshold OR have user_keep=TRUE
                                filtered_count = len(accepted_df)
                                accepted_df = accepted_df[r2_mask | user_keep_mask]
                                if filtered_count > len(accepted_df):
                                    logger.debug(f"R² filter removed {filtered_count - len(accepted_df)} peaks (r2_min={r2_min})")
                        except Exception:
                            pass

                        # DEBUG: Log mineral class distribution after filtering
                        if 'min_ID' in accepted_df.columns and not accepted_df.empty:
                            class_counts_after = accepted_df['min_ID'].value_counts().to_dict()
                            logger.debug(f"Mineral classes after R² filter: {class_counts_after}")
                        
                        # Safety: ensure required columns exist to avoid KeyError during grouping
                        for _col in ['point', 'snr', 'min_ID']:
                            if _col not in accepted_df.columns:
                                accepted_df[_col] = [] if _col == 'min_ID' else 0
                        
                        # ---- Per-class artifacts and combined grid (outside try/except) ----
                        import matplotlib.pyplot as _plt
                        
                        # Build per-class SNR by point (max SNR per point within class)
                        class_snr = (
                            accepted_df.groupby(['min_ID', 'point'])['snr'].max().reset_index()
                            if ('snr' in accepted_df.columns and 'min_ID' in accepted_df.columns)
                            else pd.DataFrame(columns=['min_ID', 'point', 'snr'])
                        )

                        # DEBUG: Log which classes have data for rendering
                        if not class_snr.empty:
                            classes_with_data = class_snr['min_ID'].unique().tolist()
                            logger.debug(f"Classes with data for rendering: {classes_with_data}")
                            for cls in classes_with_data:
                                n_points = class_snr[class_snr['min_ID'] == cls]['point'].nunique()
                                logger.debug(f"  {cls}: {n_points} points")
                        else:
                            logger.warning("class_snr is empty - no mineral detections to render")

                        class_order = ['olivine', 'phosphate', 'pyroxene', 'sulf1_v1', 'sulf2_v1', 'sulf_v3', 'lo-carb', 'hi-carb', '1050']
                        
                        def _render_subset(label: str, pts_df: pd.DataFrame, base_name: str):
                            if pts_df.empty:
                                return
                            mm = spatial_df.merge(pts_df, on='point', how='inner')
                            if mm.empty:
                                return
                            pts_up = mm[["xPix", "yPix"]].to_numpy(float)
                            snr_vals = mm['snr'].astype(float).to_numpy()
                            sel_pts = set(int(p) for p in mm['point'].tolist())
                            ns_df = spatial_df[~spatial_df['point'].isin(sel_pts)]
                            # Native coordinates for full ACI
                            scale = float(upscale or 1.0)
                            if scale <= 0:
                                scale = 1.0
                            pts_native = pts_up / scale
                            ns_native = ns_df[["xPix", "yPix"]].to_numpy(float) / scale if not ns_df.empty else None
                            # Full ACI (no title/colorbar)
                            out_full = overlay_dir / f"{sol}_{target}_{scan}_{base_name}_pointloc.png"
                            render_pointloc_full(
                                image_native=image_native,
                                pixel_scale_um_per_px=float(meta.get('pixel_scale', 10.1)),
                                selected_xy=pts_native,
                                selected_snr=snr_vals,
                                nonselected_xy=ns_native,
                                out_path=out_full,
                            )
                            artifacts.append(out_full)
                            # Zoomed crop (render directly on crop)
                            out_zoom = overlay_dir / f"{sol}_{target}_{scan}_{base_name}_pointloc_zoomed.png"
                            render_pointloc_zoomed(
                                crop_image_native=crop_native,
                                selected_xy_crop_native=np.column_stack([pts_native[:, 0] - xmin, pts_native[:, 1] - ymin]),
                                selected_snr=snr_vals,
                                nonselected_xy_crop_native=(
                                    np.column_stack([ns_native[:, 0] - xmin, ns_native[:, 1] - ymin])
                                    if ns_native is not None else None
                                ),
                                out_path=out_zoom,
                                upscale=float(upscale or 1.0),
                            )
                            artifacts.append(out_zoom)
                            # With colorbar (full ACI) — notebook title format
                            out_cb = overlay_dir / f"{sol}_{target}_{scan}_{base_name}_pointloc_with_colorbar.png"
                            render_pointloc_with_colorbar(
                                image_native=image_native,
                                pixel_scale_um_per_px=float(meta.get('pixel_scale', 10.1)),
                                selected_xy=pts_native,
                                selected_snr=snr_vals,
                                nonselected_xy=ns_native,
                                title=f"{sol} {target} {scan} points with {base_name} SNR > 3\nprocessed data, colorized by SNR",
                                out_path=out_cb,
                            )
                            artifacts.append(out_cb)
                        
                        # Per-class panels: only generate artifacts when there are detections
                        for cls in class_order:
                            pts_cls = class_snr[class_snr['min_ID'] == cls][['point', 'snr']]
                            if not pts_cls.empty:
                                _render_subset(cls, pts_cls, cls)
                            else:
                                logger.debug(f"No detections for mineral class '{cls}' - skipping overlay")
                        
                        # Aggregates
                        def _aggregate_points(classes: List[str]) -> pd.DataFrame:
                            sub = class_snr[class_snr['min_ID'].isin(classes)]
                            if sub.empty:
                                return sub
                            return sub.groupby('point')['snr'].max().reset_index()
                        
                        agg = _aggregate_points(['lo-carb', 'hi-carb'])
                        if not agg.empty:
                            _render_subset('all-carb', agg, 'all-carb')
                        agg = _aggregate_points(['sulf1_v1', 'sulf2_v1'])
                        if not agg.empty:
                            _render_subset('sulf_all_v1', agg, 'sulf_all_v1')
                        
                        # Combined grid assembled from saved zoomed panels (exact notebook behavior)
                        snr_ranges = {}
                        if not class_snr.empty:
                            _min = class_snr.groupby('min_ID')["snr"].min()
                            _max = class_snr.groupby('min_ID')["snr"].max()
                            for _cls in _min.index:
                                snr_ranges[_cls] = (float(_min[_cls]), float(_max[_cls]))
                        # Build display labels from DEFAULT_RULES (which loads from config.yaml)
                        from sherloc_pipeline.core.mineral_id import DEFAULT_RULES
                        mineral_to_range = {r.label: f"{r.lo:g}-{r.hi:g} cm$^{-1}$" for r in DEFAULT_RULES}
                        fig, axs = _plt.subplots(3, 3, figsize=(15, 15))
                        for ax, cls in zip(axs.ravel(), class_order):
                            img_path = overlay_dir / "cropped_original.png"
                            cand = overlay_dir / f"{sol}_{target}_{scan}_{cls}_pointloc_zoomed.png"
                            if cand.exists():
                                img_path = cand
                            img = _plt.imread(img_path)
                            if img.ndim == 2:
                                ax.imshow(img, cmap='gray', origin='upper')
                            else:
                                ax.imshow(img, origin='upper')
                            ax.axis('off')
                            smin, smax = snr_ranges.get(cls, (None, None))
                            label = (
                                f"{mineral_to_range.get(cls, cls)}: '{cls}'"
                                if (smin is None or smax is None)
                                else f"{mineral_to_range.get(cls, cls)}: '{cls}' SNR {smin:.0f} to {smax:.0f}"
                            )
                            ax.set_title(label, color='black', fontsize=12, pad=5)

                        # Reserve space at top for suptitle to avoid overlap
                        _plt.tight_layout(rect=(0, 0, 1, 0.92))

                        # Position suptitle in the reserved space
                        fig.suptitle(
                            f"Peak detections in {sol} {target} {scan} colorized from low (purple) to high (yellow) SNR",
                            fontsize=14,
                            y=0.96
                        )

                        grid_out = overlay_dir / f"{sol}_{target}_{scan}_minerals_combined_grid.png"
                        from io import BytesIO
                        buf = BytesIO()
                        _plt.savefig(
                            buf,
                            format="png",
                            dpi=300,
                            bbox_inches='tight',
                            metadata={"Software": BASELINE_SOFTWARE_TAG},
                        )
                        _plt.close(fig)
                        buf.seek(0)
                        from PIL import Image as _PILImage

                        with _PILImage.open(buf) as buffered_img:
                            rgb = np.asarray(buffered_img.convert("RGB"), dtype=np.float32) / 255.0
                            dpi = buffered_img.info.get("dpi", (300, 300))
                        _save_rgb_image_pil(
                            rgb,
                            grid_out,
                            metadata={"Software": BASELINE_SOFTWARE_TAG},
                            dpi=dpi,
                        )
                        artifacts.append(grid_out)
                        progress.update(task_min, description="✅ Minerals overlays complete")
                    
                    # Debug CSV
                    if save_debug:
                        dbg_path = overlay_dir / "spatial_debug.csv"
                        cols = [
                            c for c in [
                                "point", "x", "y", "xPix", "yPix", "organics_DG", "hydration_OH",
                                "hydration_bend", "organics_snr", "hydration_oh_snr"
                            ]
                            if c in debug_rows.columns
                        ]
                        try:
                            debug_rows[cols].to_csv(dbg_path, index=False)
                            artifacts.append(dbg_path)
                            try:
                                self.console.print(f"[blue]Debug CSV:[/blue] {dbg_path}")
                            except Exception:
                                pass
                        except Exception as e:
                            warning_msg = f"failed to write spatial_debug.csv: {e}"
                            warnings.append(warning_msg)
                            logger.warning(warning_msg)
                    
                    summary = f"Spatial overlay generation complete for {sol}/{target}/{scan}"
                    metadata = {
                        "overlay_dir": str(overlay_dir),
                        "layers": layers,
                        "upscale": upscale,
                        "use_reviewed": use_reviewed,
                    }
                    
                    return ServiceResult(
                        summary=summary,
                        artifacts=artifacts,
                        warnings=warnings,
                        metadata=metadata,
                    )
                except Exception as inner_e:
                    # Re-raise as SpatialError to be caught by outer handler
                    raise SpatialError(
                        f"Failed during overlay generation: {inner_e}",
                        exit_code=1,
                        context={"sol": sol, "target": target, "scan": scan},
                    ) from inner_e
                
        except SpatialError:
            raise  # Re-raise SpatialError as-is
        except Exception as e:
            error = SpatialError(
                f"Failed to generate spatial overlays: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan},
            )
            raise enrich(error, sol=sol, target=target, scan=scan) from e
    
    def merge_label(
        self,
        sol: str,
        target: str,
        scan_a: str,
        scan_b: str,
        label: str = "hi-carb",
        use_reviewed: bool = True,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ) -> ServiceResult:
        """Compose a single pointloc overlay for a label across two scans on the same ACI image.
        
        This method uses the ACI image from the primary scan (scan_a) and plots
        points/detections from both scans for the specified mineral label.
        
        Args:
            sol: Sol number (e.g., "1626")
            target: Target name (e.g., "Klorne")
            scan_a: Primary scan (e.g., "detail")
            scan_b: Secondary scan to merge (e.g., "line")
            label: Minerals label_id to render (default: "hi-carb")
            use_reviewed: Use scan-level reviewed table and user_keep=True
            data_dir: Base data directory. If None, uses config default.
            results_dir: Results directory. If None, uses config default.
        
        Returns:
            ServiceResult with summary, artifacts (merged overlay PNG), warnings, and metadata
            
        Raises:
            SpatialError: If merge operation fails (missing scans, invalid paths, etc.)
            
        Example:
            >>> service = SpatialService()
            >>> result = service.merge_label(
            ...     sol="1626",
            ...     target="Klorne",
            ...     scan_a="detail",
            ...     scan_b="line",
            ...     label="hi-carb"
            ... )
            >>> print(result.summary)
            'Merged label overlay complete'
        """
        run_context = self.context.with_overrides(
            data_dir=data_dir,
            results_dir=results_dir,
        )
        
        try:
            from sherloc_pipeline.core.data_ingestion import DataIngestion
            from sherloc_pipeline.core.spatial import load_spatial_table
            from sherloc_pipeline.visualization.spatial import overlay_points_on_aci
        except ImportError as e:
            error = SpatialError(
                f"Failed to import required modules: {e}",
                exit_code=1,
            )
            raise enrich(error, sol=sol, target=target, scan=scan_a) from e
        
        try:
            # Resolve contexts for both scans
            context_a = resolve_scan_context(
                sol=sol,
                target=target,
                scan=scan_a,
                data_dir=run_context.data_root,
                results_dir=run_context.results_root,
                context=run_context,
            )
            context_b = resolve_scan_context(
                sol=sol,
                target=target,
                scan=scan_b,
                data_dir=run_context.data_root,
                results_dir=run_context.results_root,
                context=run_context,
            )
        except (FileNotFoundError, ValueError) as e:
            error = SpatialError(
                f"Failed to resolve scan context: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan_a": scan_a, "scan_b": scan_b},
            )
            raise enrich(error, sol=sol, target=target, scan=scan_a) from e
        
        artifacts = []
        warnings = []
        
        try:
            import matplotlib.pyplot as plt  # noqa: F401 (ensures backend availability)
            
            # Prepare ingestions and base paths
            ing_a = DataIngestion(
                base_data_dir=context_a.base_data_dir,
                results_dir=context_a.results_dir,
                sol=sol,
                target=target,
                scan=scan_a,
            )
            ing_b = DataIngestion(
                base_data_dir=context_b.base_data_dir,
                results_dir=context_b.results_dir,
                sol=sol,
                target=target,
                scan=scan_b,
            )
            base_a = context_a.results_path
            base_b = context_b.results_path
            
            # Load ACI image from primary scan
            working_dir_a = ing_a.find_working_directory(sol, scan_a)
            if not working_dir_a:
                error = SpatialError(
                    f"No working directory found for sol {sol}, scan {scan_a}",
                    exit_code=1,
                )
                raise enrich(error, sol=sol, target=target, scan=scan_a)
            
            image, meta = ing_a.load_aci_image(working_dir_a)
            
            # Per-scan helper
            def gather(scan_name: str, base: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                scan_acc = base / f"{sol}_{target}_{scan_name}_accepted_peaks.csv"
                acc = pd.read_csv(scan_acc) if scan_acc.exists() else pd.DataFrame(columns=['modality', 'point', 'label_id', 'snr'])
                if use_reviewed and 'user_keep' in acc.columns:
                    acc = acc[acc['user_keep'] == True]
                # Minerals only, specific label
                acc = acc[(acc['modality'] == 'minerals') & (acc['label_id'].astype(str) == str(label))]
                # SNR per point (max if multiple rows)
                snr_map = acc.groupby('point')['snr'].max() if not acc.empty and 'snr' in acc.columns else pd.Series(dtype=float)
                # Spatial table
                work = ing_a.find_working_directory(sol, scan_name) if scan_name == scan_a else ing_b.find_working_directory(sol, scan_name)
                if not work:
                    return np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
                spat = load_spatial_table(work)
                # Selected points
                sel_pts = sorted(set(int(p) for p in snr_map.index.tolist()))
                sel = spat[spat['point'].isin(sel_pts)].copy()
                sel_x = sel['xPix'].to_numpy(float)
                sel_y = sel['yPix'].to_numpy(float)
                sel_snr = sel['point'].map(snr_map).astype(float).fillna(0.0).to_numpy()
                # Nonselected points (all points in this scan minus selected)
                ns = spat[~spat['point'].isin(sel_pts)]
                ns_x = ns['xPix'].to_numpy(float)
                ns_y = ns['yPix'].to_numpy(float)
                return sel_x, sel_y, sel_snr, ns_x, ns_y
            
            sel_x_a, sel_y_a, sel_snr_a, ns_x_a, ns_y_a = gather(scan_a, base_a)
            sel_x_b, sel_y_b, sel_snr_b, ns_x_b, ns_y_b = gather(scan_b, base_b)
            
            # Compose overlays (use raster rings to match existing style)
            overlays = []
            if sel_x_a.size:
                overlays.append({
                    'points': np.stack([sel_x_a, sel_y_a], axis=1),
                    'colors': sel_snr_a,
                    'cmap': 'viridis',
                    'radius_px': 5,
                    'ring_width_px': 1.0,
                    'detections': True,
                    'raster': True,
                })
            if ns_x_a.size:
                overlays.append({
                    'points': np.stack([ns_x_a, ns_y_a], axis=1),
                    'edgecolor': 'white',
                    'alpha': 0.3,
                    'radius_px': 5,
                    'ring_width_px': 1.0,
                    'raster': True,
                })
            if sel_x_b.size:
                overlays.append({
                    'points': np.stack([sel_x_b, sel_y_b], axis=1),
                    'colors': sel_snr_b,
                    'cmap': 'viridis',
                    'radius_px': 5,
                    'ring_width_px': 1.0,
                    'detections': True,
                    'raster': True,
                })
            if ns_x_b.size:
                overlays.append({
                    'points': np.stack([ns_x_b, ns_y_b], axis=1),
                    'edgecolor': 'white',
                    'alpha': 0.3,
                    'radius_px': 5,
                    'ring_width_px': 1.0,
                    'raster': True,
                })
            
            overlay_dir = base_a / 'spatial_overlays'
            overlay_dir.mkdir(parents=True, exist_ok=True)
            out = overlay_dir / f"{sol}_{target}_{scan_a}+{scan_b}_{label}_pointloc.png"
            overlay_points_on_aci(image=image, overlays=overlays, output_path=out)
            artifacts.append(out)
            
            try:
                self.console.print(f"[green]Wrote combined overlay:[/green] {out}")
            except Exception:
                pass
            
            summary = f"Merged label overlay complete for {sol}/{target}/{scan_a}+{scan_b} (label: {label})"
            metadata = {
                "overlay_file": str(out),
                "label": label,
                "use_reviewed": use_reviewed,
            }
            
            return ServiceResult(
                summary=summary,
                artifacts=artifacts,
                warnings=warnings,
                metadata=metadata,
            )
            
        except SpatialError:
            raise  # Re-raise SpatialError as-is
        except Exception as e:
            error = SpatialError(
                f"Failed to merge label overlay: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan_a": scan_a, "scan_b": scan_b, "label": label},
            )
            raise enrich(error, sol=sol, target=target, scan=scan_a) from e


