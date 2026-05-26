"""Spatial overlay visualization functions for ACI images.

These functions were moved from core/spatial.py as part of the
core/visualization layer separation (Public Release v3).

Pure data/computation functions (coordinate transforms, spatial table loading,
image composition helpers) remain in core/spatial.py.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sherloc_pipeline.core.spatial import (
    _to_rgba,
    _save_rgb_image_pil,
    BASELINE_SOFTWARE_TAG,
)
from sherloc_pipeline.visualization.plotting import apply_plot_config


def _compose_with_skimage_rings(
    base_image: np.ndarray,
    selected_xy,
    selected_snr,
    nonselected_xy,
    radius_px: float = 5.0,
    ring_width_px: float = 1.0,
) -> np.ndarray:
    """Return RGB float image with rings using skimage.draw.circle_perimeter.

    Moved from core/spatial.py to eliminate matplotlib imports in core/.
    """
    import skimage.draw as _skdraw
    import matplotlib

    rgba = _to_rgba(base_image)
    h, w = rgba.shape[:2]
    overlay = np.zeros_like(rgba, dtype=np.float32)

    # Non-detections: white, alpha 0.3
    if nonselected_xy is not None and np.size(nonselected_xy) > 0:
        pts = np.asarray(nonselected_xy, dtype=float)
        for i in range(pts.shape[0]):
            cy = int(round(pts[i, 1])); cx = int(round(pts[i, 0]))
            rr, cc = _skdraw.circle_perimeter(cy, cx, int(round(radius_px)))
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            overlay[rr[valid], cc[valid], :3] = 1.0
            overlay[rr[valid], cc[valid], 3] = 0.3

    # Detections: colored by viridis, alpha 1.0
    if selected_xy is not None and np.size(selected_xy) > 0:
        pts = np.asarray(selected_xy, dtype=float)
        snr = np.asarray(selected_snr, dtype=float) if selected_snr is not None else np.ones(pts.shape[0], dtype=float)
        vmin = float(np.nanmin(snr)) if np.isfinite(snr).any() else 0.0
        vmax = float(np.nanmax(snr)) if np.isfinite(snr).any() else 1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        cmap = matplotlib.colormaps.get_cmap("viridis")
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cols = cmap(norm(snr))
        for i in range(pts.shape[0]):
            cy = int(round(pts[i, 1])); cx = int(round(pts[i, 0]))
            rr, cc = _skdraw.circle_perimeter(cy, cx, int(round(radius_px)))
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            color = np.array(cols[i][:3], dtype=np.float32)
            overlay[rr[valid], cc[valid], :3] = color
            overlay[rr[valid], cc[valid], 3] = 1.0

    # Composite
    a = overlay[..., 3:4]
    comp = overlay[..., :3] * a + rgba[..., :3] * (1.0 - a)
    return np.clip(comp, 0.0, 1.0)


def _compose_rings_on_image(
    image: np.ndarray,
    points_selected,
    snr_selected,
    points_nondet,
    pixel_scale_um_per_px: float,
    use_raster: bool = True,
    cmap_name: str = "viridis",
    ring_width_det_px: float = 2.0,
    ring_width_nondet_px: float = 1.0,
    alpha_nondet: float = 0.3,
) -> np.ndarray:
    """Return an RGB image with notebook-style rings composited over base image.

    Moved from core/spatial.py to eliminate matplotlib imports in core/.
    """
    import matplotlib

    # Prepare base RGB float 0..1
    if image.ndim == 2:
        base_rgb = np.stack([image, image, image], axis=-1)
    else:
        base_rgb = image
    if base_rgb.dtype != np.float32 and base_rgb.dtype != np.float64:
        base_rgb = base_rgb.astype(np.float32) / 255.0
    h, w = base_rgb.shape[0], base_rgb.shape[1]
    raster = np.zeros((h, w, 4), dtype=np.float32)

    # Compute radius in pixels for ~100 µm diameter
    diameter_um = 100.0
    radius_px = (diameter_um / float(pixel_scale_um_per_px)) / 2.0

    # Non-detections
    if points_nondet is not None and points_nondet.size:
        pts = np.asarray(points_nondet, dtype=float)
        color = np.array([1.0, 1.0, 1.0, alpha_nondet], dtype=np.float32)
        rrw = float(max(1.0, ring_width_nondet_px))
        for i in range(pts.shape[0]):
            cx = int(round(pts[i, 0])); cy = int(round(pts[i, 1]))
            rr = int(max(1, np.floor(radius_px)))
            x0 = max(0, cx - rr - 1); x1 = min(w, cx + rr + 2)
            y0 = max(0, cy - rr - 1); y1 = min(h, cy + rr + 2)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            ring = np.abs(d - radius_px) <= (rrw * 0.5)
            if not np.any(ring):
                continue
            raster[y0:y1, x0:x1, :3][ring] = color[:3]
            raster[y0:y1, x0:x1, 3][ring] = color[3]

    # Detections colorized by SNR
    if points_selected is not None and points_selected.size:
        pts = np.asarray(points_selected, dtype=float)
        snr = np.asarray(snr_selected, dtype=float) if snr_selected is not None else np.ones(pts.shape[0], dtype=float)
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        vmin = float(np.nanmin(snr)) if np.isfinite(snr).any() else 0.0
        vmax = float(np.nanmax(snr)) if np.isfinite(snr).any() else 1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        edge_colors = cmap(norm(snr))
        rrw = float(max(1.0, ring_width_det_px))
        for i in range(pts.shape[0]):
            cx = int(round(pts[i, 0])); cy = int(round(pts[i, 1]))
            rr = int(max(1, np.floor(radius_px)))
            x0 = max(0, cx - rr - 1); x1 = min(w, cx + rr + 2)
            y0 = max(0, cy - rr - 1); y1 = min(h, cy + rr + 2)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            ring = np.abs(d - radius_px) <= (rrw * 0.5)
            if not np.any(ring):
                continue
            col = np.array(edge_colors[i], dtype=np.float32)
            col[3] = 1.0
            raster[y0:y1, x0:x1, :3][ring] = col[:3]
            raster[y0:y1, x0:x1, 3][ring] = col[3]

    # Composite
    a = raster[..., 3:4]
    comp = raster[..., :3] * a + base_rgb * (1.0 - a)
    return np.clip(comp, 0.0, 1.0)


def overlay_points_on_aci(
    image: np.ndarray,
    overlays: List[Dict[str, Any]],
    output_path: Path,
    title: Optional[str] = None,
    colorbar: Optional[Dict[str, Any]] = None,
) -> Path:
    """Render overlays on an ACI image and save to `output_path`.

    Moved from core/spatial.py. Callers should import from
    sherloc_pipeline.visualization.spatial.

    overlays: list of dicts each with keys:
      - points: np.ndarray of shape (N, 2) with columns [xPix, yPix]
      - colors: array-like of length N (values or color strings)
      - markersize: float
      - edgecolor: str
      - label: Optional[str]
      - cmap: Optional[str] if colors are numeric
      - vmin, vmax: Optional[float] for colormap scaling
    colorbar: Optional dict with keys: label, cmap, vmin, vmax
    """
    from matplotlib import pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib import cm as mcm
    from matplotlib import patches as mpatches

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    # Base image as RGB float 0..1 for composite
    if image.ndim == 2:
        base_rgb = np.stack([image, image, image], axis=-1)
    else:
        base_rgb = image
    if base_rgb.dtype != np.float32 and base_rgb.dtype != np.float64:
        base_rgb = base_rgb.astype(np.float32) / 255.0
    plt.imshow(base_rgb, origin="upper")

    mappables = []
    fig = plt.gcf()
    current_dpi = float(fig.get_dpi() if hasattr(fig, "get_dpi") else 300.0)

    # Raster overlay buffer for 1-pixel rings (notebook-like)
    raster = np.zeros((base_rgb.shape[0], base_rgb.shape[1], 4), dtype=np.float32)

    for spec in overlays:
        pts = np.asarray(spec.get("points", []))
        if pts.size == 0:
            continue
        xs, ys = pts[:, 0], pts[:, 1]
        colors = spec.get("colors", None)
        mk = spec.get("marker", "o")
        ms = float(spec.get("markersize", 18.0))
        ec = spec.get("edgecolor", "white")
        lw = float(spec.get("linewidth", 0.7))
        # Allow pixel-exact linewidth via linewidth_px (convert px→pt using savefig DPI)
        lw_px = spec.get("linewidth_px", None)
        if lw_px is not None:
            try:
                lw = float(lw_px) * (72.0 / current_dpi)
            except Exception:
                pass
        alpha = float(spec.get("alpha", 0.95))
        cmap = spec.get("cmap", None)
        vmin = spec.get("vmin", None)
        vmax = spec.get("vmax", None)
        facecolors = spec.get("facecolors", None)
        norm = spec.get("norm", None)
        radius_px = spec.get("radius_px", None)
        if radius_px is not None and bool(spec.get("raster", True)):
            # Paint 1-pixel ring per point into raster buffer (closest to notebook behavior)
            # Determine edge color per point
            edge_colors = None
            if isinstance(ec, (list, tuple, np.ndarray)):
                arr = np.asarray(ec)
                if arr.ndim == 2:
                    edge_colors = arr
            if edge_colors is None and colors is not None and cmap is not None and (vmin is not None and vmax is not None):
                norm2 = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
                cm = mcm.get_cmap(cmap)
                edge_colors = cm(norm2(np.asarray(colors)))
                try:
                    edge_colors = np.asarray(edge_colors)
                    if edge_colors.ndim == 2 and edge_colors.shape[1] >= 4:
                        edge_colors[:, 3] = 1.0
                except Exception:
                    pass
                sm = mcm.ScalarMappable(norm=norm2, cmap=cmap)
                sm.set_array([])
                mappables.append(sm)
            # Fallback single color
            if edge_colors is None:
                rgba = mcolors.to_rgba(ec if isinstance(ec, str) else (ec if ec is not None else 'white'), alpha=1.0 if spec.get("detections", False) else alpha)
                edge_colors = np.tile(np.array(rgba)[None, :], (len(xs), 1))
            r = float(radius_px)
            ring_w = float(spec.get("ring_width_px", 1.0))
            for i in range(len(xs)):
                cx = int(round(xs[i])); cy = int(round(ys[i]))
                rr = int(max(1, np.floor(r)))
                x0 = max(0, cx - rr - 1); x1 = min(base_rgb.shape[1], cx + rr + 2)
                y0 = max(0, cy - rr - 1); y1 = min(base_rgb.shape[0], cy + rr + 2)
                if x0 >= x1 or y0 >= y1:
                    continue
                yy, xx = np.mgrid[y0:y1, x0:x1]
                d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                ring = np.abs(d - r) <= (ring_w * 0.5)
                if not np.any(ring):
                    continue
                col = edge_colors[i]
                # Force detections to full opacity if requested
                a = 1.0 if spec.get("detections", False) else (float(col[3]) if len(col) == 4 else alpha)
                rgb = np.array(col[:3], dtype=np.float32)
                # Write color and alpha (overwrite suffices for thin lines)
                raster[y0:y1, x0:x1, :3][ring] = rgb
                raster[y0:y1, x0:x1, 3][ring] = a
            continue
        if radius_px is not None:
            # Draw pixel-accurate circle outlines using patches
            ax = plt.gca()
            # If edgecolor is a single color, broadcast; if array, use per-point
            if isinstance(ec, (list, tuple, np.ndarray)) and np.asarray(ec).ndim == 2:
                edge_colors = np.asarray(ec)
            else:
                edge_colors = None
            for i in range(len(xs)):
                ec_i = edge_colors[i] if edge_colors is not None else ec
                circ = mpatches.Circle(
                    (xs[i], ys[i]),
                    radius=float(radius_px),
                    facecolor='none' if facecolors is None or facecolors == 'none' else facecolors,
                    edgecolor=ec_i,
                    linewidth=lw,
                    alpha=alpha,
                    antialiased=bool(spec.get("antialiased", False)),
                )
                ax.add_patch(circ)
            # Create a dummy mappable if cmap provided for colorbar reference
            if cmap is not None and (vmin is not None and vmax is not None):
                norm2 = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
                sm = mcm.ScalarMappable(norm=norm2, cmap=cmap)
                sm.set_array([])
                mappables.append(sm)
        else:
            # Draw scatter; support hollow markers (facecolors='none') and numeric color mapping for c
            kwargs = dict(s=ms, marker=mk, edgecolors=ec, linewidths=lw, alpha=alpha, label=spec.get("label", None))
            if facecolors is not None:
                kwargs["facecolors"] = facecolors
            if colors is not None:
                kwargs["c"] = colors
            if cmap is not None:
                kwargs["cmap"] = cmap
            if norm is not None:
                kwargs["norm"] = norm
            if vmin is not None:
                kwargs["vmin"] = vmin
            if vmax is not None:
                kwargs["vmax"] = vmax
            sc = plt.scatter(xs, ys, **kwargs)
            # Only mappable if a cmap provided with numeric colors
            if cmap is not None and (colors is not None or norm is not None):
                mappables.append(sc)

    # Composite raster overlay on top of base RGB
    comp = None
    if np.any(raster[..., 3] > 0):
        a = raster[..., 3:4]
        comp = raster[..., :3] * a + base_rgb * (1.0 - a)
        plt.cla()
        plt.imshow(comp, origin="upper")

    if title:
        plt.title(title)
    plt.axis("off")

    # Single colorbar if requested. If no mappable, create one from cmap+norm.
    if colorbar is not None:
        ax = plt.gca()
        # Place a colorbar axis with the same height as the image axes
        try:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.02)
        except Exception:
            cax = None
        if mappables:
            cb = plt.colorbar(mappables[-1], ax=ax if cax is None else None, cax=cax)
        else:
            cmap_name = colorbar.get("cmap", "plasma")
            vmin = colorbar.get("vmin", None)
            vmax = colorbar.get("vmax", None)
            if vmin is not None and vmax is not None:
                norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
                sm = mcm.ScalarMappable(norm=norm, cmap=cmap_name)
                sm.set_array([])
                cb = plt.colorbar(sm, ax=ax if cax is None else None, cax=cax)
            else:
                cb = None
        if colorbar.get("label") and cb is not None:
            cb.set_label(str(colorbar["label"]))

    # If no title and no colorbar are requested, save a borderless image directly for exact pixel retention
    if (title is None) and (colorbar is None) and (comp is not None):
        arr = np.clip(comp, 0.0, 1.0).astype(np.float32)
        _save_rgb_image_pil(arr, output_path, metadata={"Software": BASELINE_SOFTWARE_TAG})
        plt.close()
        return output_path

    # Use tight_layout (default behavior)
    plot_config, bbox_inches = apply_plot_config(fig)
    buf = BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=plot_config.savefig_dpi,
        bbox_inches=bbox_inches,
        pad_inches=0,
        metadata={"Software": BASELINE_SOFTWARE_TAG},
    )
    plt.close()
    buf.seek(0)
    from PIL import Image as _PILImage

    with _PILImage.open(buf) as buffered_img:
        rgb = np.asarray(buffered_img.convert("RGB"), dtype=np.float32) / 255.0
        dpi = buffered_img.info.get("dpi", (plot_config.savefig_dpi, plot_config.savefig_dpi))
    _save_rgb_image_pil(
        rgb,
        output_path,
        metadata={"Software": BASELINE_SOFTWARE_TAG},
        dpi=dpi,
    )
    return output_path


def render_pointloc_full(
    image_native: np.ndarray,
    pixel_scale_um_per_px: float,
    selected_xy: np.ndarray,
    selected_snr: np.ndarray,
    nonselected_xy: Optional[np.ndarray],
    out_path: Path,
) -> Path:
    """Render full-frame point location image with rings.

    Moved from core/spatial.py.
    """
    comp = _compose_with_skimage_rings(
        base_image=image_native,
        selected_xy=selected_xy,
        selected_snr=selected_snr,
        nonselected_xy=nonselected_xy,
        radius_px=5.0,
        ring_width_px=3.0,
    )
    return _save_rgb_image_pil(
        comp,
        out_path,
        metadata={"Software": BASELINE_SOFTWARE_TAG},
    )


def render_pointloc_zoomed(
    crop_image_native: np.ndarray,
    selected_xy_crop_native: np.ndarray,
    selected_snr: np.ndarray,
    nonselected_xy_crop_native: Optional[np.ndarray],
    out_path: Path,
    upscale: float = 3.0,
) -> Path:
    """Render zoomed point location image with rings and upscaling.

    Moved from core/spatial.py.
    """
    comp_native = _compose_with_skimage_rings(
        base_image=crop_image_native,
        selected_xy=selected_xy_crop_native,
        selected_snr=selected_snr,
        nonselected_xy=nonselected_xy_crop_native,
        radius_px=5.0,
        ring_width_px=3.0,
    )
    # Resize after composition to match notebook
    try:
        import skimage.transform as _sktr
        comp_up = _sktr.resize(comp_native, (int(comp_native.shape[0]*upscale), int(comp_native.shape[1]*upscale), comp_native.shape[2]), mode='reflect', anti_aliasing=True)
    except Exception:
        comp_up = comp_native
    return _save_rgb_image_pil(
        comp_up,
        out_path,
        metadata={"Software": BASELINE_SOFTWARE_TAG},
    )


def render_pointloc_with_colorbar(
    image_native: np.ndarray,
    pixel_scale_um_per_px: float,
    selected_xy: np.ndarray,
    selected_snr: np.ndarray,
    nonselected_xy: Optional[np.ndarray],
    title: str,
    out_path: Path,
) -> Path:
    """Render point location image with SNR colorbar.

    Moved from core/spatial.py.
    """
    import matplotlib.pyplot as _plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib

    vmin = float(np.nanmin(selected_snr)) if np.isfinite(selected_snr).any() else 0.0
    vmax = float(np.nanmax(selected_snr)) if np.isfinite(selected_snr).any() else 1.0
    comp = _compose_with_skimage_rings(
        base_image=image_native,
        selected_xy=selected_xy,
        selected_snr=selected_snr,
        nonselected_xy=nonselected_xy,
        radius_px=5.0,
        ring_width_px=1.0,
    )
    fig, ax = _plt.subplots(figsize=(6.0, 6.0))
    ax.imshow(comp, origin='upper')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.02)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    _plt.colorbar(sm, cax=cax, label="SNR")
    plot_config, bbox_inches = apply_plot_config(
        fig,
        margins_override={"left": 0.01, "right": 0.995, "top": 0.995, "bottom": 0.01},
        bbox_override="tight",
        use_default_margins=False,
    )
    tmp = BytesIO()
    # Use pad_inches=0.1 to add minimal whitespace around edges (prevents title/colorbar from touching border)
    _plt.savefig(
        tmp,
        format="png",
        dpi=plot_config.savefig_dpi,
        bbox_inches=bbox_inches,
        pad_inches=0.1,
        metadata={"Software": BASELINE_SOFTWARE_TAG},
    )
    _plt.close(fig)
    tmp.seek(0)
    from PIL import Image as _PILImage

    with _PILImage.open(tmp) as buffered_img:
        rgb = np.asarray(buffered_img.convert("RGB"), dtype=np.float32) / 255.0
        dpi = buffered_img.info.get("dpi", (plot_config.savefig_dpi, plot_config.savefig_dpi))
    out_path = _save_rgb_image_pil(
        rgb,
        out_path,
        metadata={"Software": BASELINE_SOFTWARE_TAG},
        dpi=dpi,
    )
    return out_path


def build_combined_grid(
    crop_image: np.ndarray,
    class_to_points: Dict[str, Dict[str, np.ndarray]],
    snr_ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
    out_path: Path,
    suptitle: Optional[str] = None,
    suptitle_y: float = 0.92,
) -> Path:
    """Build and save the 3×3 combined minerals grid using a shared crop image.

    Moved from core/spatial.py.

    class_to_points: { min_ID: { 'selected_xy': (N,2), 'selected_snr': (N,), 'nonselected_xy': (M,2) } }
    - Draw live rings for classes with detections; otherwise show crop only.
    - Panel titles at fontsize=12, left-aligned at (0,1), using Unicode cm⁻¹.
    - Suptitle fontsize=12; layout tight with rect to avoid overlap.
    """
    import matplotlib.pyplot as _plt

    # Build labels and order from config mineral_rules to keep single source of truth
    try:
        from sherloc_pipeline.config import get_config
        from sherloc_pipeline.core.mineral_id import load_mineral_rules
        cfg = get_config()
        lib_path = None
        try:
            lib_path = cfg.fitting.get('library_path') if isinstance(cfg.fitting, dict) else getattr(cfg.fitting, 'library_path', None)
        except Exception:
            lib_path = None
        inline_rules = None
        try:
            inline_rules = cfg.fitting.get('mineral_rules') if isinstance(cfg.fitting, dict) else getattr(cfg.fitting, 'mineral_rules', None)
        except Exception:
            inline_rules = None
        rules = load_mineral_rules(Path(lib_path) if lib_path else None, inline_rules=inline_rules)
        mineral_to_range = {r.label: f"{r.lo:g}-{r.hi:g} cm$^{-1}$" for r in rules}
        # Display order follows config order by default
        class_order = [r.label for r in rules]
    except Exception:
        # Fallback: use DEFAULT_RULES which loads from config.yaml
        # This ensures config.yaml remains the single source of truth even in fallback path
        from sherloc_pipeline.core.mineral_id import DEFAULT_RULES
        mineral_to_range = {r.label: f"{r.lo:g}-{r.hi:g} cm$^{-1}$" for r in DEFAULT_RULES}
        class_order = [r.label for r in DEFAULT_RULES]

    fig, axs = _plt.subplots(3, 3, figsize=(15, 15))
    # Estimate pixel scale for crop: assume crop is already upscaled relative to native as provided by caller.
    # Caller should pass points in crop coordinates already; pixel scale only affects ring diameter calculation.
    # We cannot know pixel scale here, so derive from a provided example if caller includes 'pixel_scale_upscaled'.
    pixel_scale_upscaled = None
    # Allow optional per-class pixel scale override
    if "__pixel_scale__" in class_to_points:
        try:
            pixel_scale_upscaled = float(class_to_points["__pixel_scale__"].get("value"))
        except Exception:
            pixel_scale_upscaled = None

    for ax, cls in zip(axs.ravel(), class_order):
        entry = class_to_points.get(cls, None)
        if entry is None or entry.get('selected_xy') is None or (entry.get('selected_xy').size == 0):
            # Empty panel: show only the crop
            if crop_image.ndim == 2:
                ax.imshow(crop_image, cmap='gray', origin='upper')
            else:
                ax.imshow(crop_image, origin='upper')
        else:
            sel_xy = entry.get('selected_xy')
            sel_snr = entry.get('selected_snr')
            ns_xy = entry.get('nonselected_xy', None)
            # Require pixel scale; if missing, fallback to 10.1 µm/px which matches defaults
            pxs = float(entry.get('pixel_scale_um_per_px', pixel_scale_upscaled if pixel_scale_upscaled is not None else 10.1))
            comp = _compose_rings_on_image(crop_image, sel_xy, sel_snr, ns_xy, pixel_scale_um_per_px=pxs)
            ax.imshow(comp, origin='upper')
        ax.axis('off')
        smin, smax = snr_ranges.get(cls, (None, None))
        if smin is None or smax is None:
            label = f"{mineral_to_range.get(cls, cls)}: '{cls}'"
        else:
            label = f"{mineral_to_range.get(cls, cls)}: '{cls}' SNR {smin:.0f} to {smax:.0f}"
        ax.set_title(label, color='black', fontsize=12, pad=5)

    if not suptitle:
        suptitle = "Peak detections colorized from low (purple) to high (yellow) SNR"

    # Reserve more space at top for suptitle to avoid overlap with subplot titles
    # rect=(left, bottom, right, top) - using 0.92 leaves 8% for suptitle + spacing
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    # Position suptitle in the reserved space with good vertical centering
    fig.suptitle(suptitle, fontsize=14, y=0.96)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Get plot config but don't call tight_layout again (we already did above)
    from sherloc_pipeline.visualization.plotting import configure_matplotlib
    plot_config = configure_matplotlib()
    bbox_inches = 'tight'
    buf = BytesIO()
    _plt.savefig(
        buf,
        format="png",
        dpi=plot_config.savefig_dpi,
        bbox_inches=bbox_inches,
        pad_inches=0,
        metadata={"Software": BASELINE_SOFTWARE_TAG},
    )
    _plt.close(fig)
    buf.seek(0)
    from PIL import Image as _PILImage

    with _PILImage.open(buf) as buffered_img:
        rgb = np.asarray(buffered_img.convert("RGB"), dtype=np.float32) / 255.0
        dpi = buffered_img.info.get("dpi", (plot_config.savefig_dpi, plot_config.savefig_dpi))
    _save_rgb_image_pil(
        rgb,
        out_path,
        metadata={"Software": BASELINE_SOFTWARE_TAG},
        dpi=dpi,
    )
    return out_path
