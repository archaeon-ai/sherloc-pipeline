"""
Spatial mapping module for SHERLOC pipeline.

This module handles spatial location of spectral data on ACI images using Loupe logic.
Implements APIs described in docs/spatial_next_steps.md.

Coordinate transforms and overlay conventions are adapted from Loupe V5.1.5a
(Apache License 2.0, © 2022 California Institute of Technology / JPL).
"""

from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
import pandas as pd
import numpy as np

from ..config import get_config
from sherloc_pipeline.core.utils import require_file


BASELINE_SOFTWARE_TAG = "Matplotlib version3.10.6, https://matplotlib.org/"
BASELINE_DPI = (299.9994, 299.9994)


def _to_rgba(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        h, w = image.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        if image.dtype != np.float32 and image.dtype != np.float64:
            img = image.astype(np.float32) / 255.0
        else:
            img = image.astype(np.float32)
        rgba[..., :3] = img[..., None]
        rgba[..., 3] = 1.0
        return rgba
    else:
        h, w, c = image.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        if image.dtype != np.float32 and image.dtype != np.float64:
            img = image.astype(np.float32) / 255.0
        else:
            img = image.astype(np.float32)
        if c >= 3:
            rgba[..., :3] = img[..., :3]
        else:
            rgba[..., :3] = np.repeat(img, 3, axis=2)
        rgba[..., 3] = 1.0
        return rgba

def _compose_with_skimage_rings(
    base_image: np.ndarray,
    selected_xy: Optional[np.ndarray],
    selected_snr: Optional[np.ndarray],
    nonselected_xy: Optional[np.ndarray],
    radius_px: float = 5.0,
    ring_width_px: float = 1.0,
) -> np.ndarray:
    """Return RGB float image with rings. Delegates to visualization.spatial.

    Implementation moved to visualization/spatial.py to keep core/ matplotlib-free.
    """
    from sherloc_pipeline.visualization.spatial import _compose_with_skimage_rings as _impl
    return _impl(base_image=base_image, selected_xy=selected_xy, selected_snr=selected_snr,
                 nonselected_xy=nonselected_xy, radius_px=radius_px, ring_width_px=ring_width_px)

def _save_rgb_image_pil(
    image_rgb_float: np.ndarray,
    out_path: Path,
    *,
    metadata: Optional[dict] = None,
    dpi: Optional[Tuple[float, float]] = None,
) -> Path:
    import zlib
    import struct

    arr = np.clip(image_rgb_float, 0.0, 1.0)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[2] == 1:
        color_type = 0  # grayscale
    elif arr.shape[2] == 3:
        color_type = 2  # truecolour
    elif arr.shape[2] == 4:
        color_type = 6  # truecolour with alpha
    else:
        # Fallback: convert to truecolour
        arr = np.repeat(arr[..., :1], 3, axis=2)
        color_type = 2

    data = (arr * 255.0 + 0.5).astype(np.uint8)
    height, width = data.shape[:2]
    channels = data.shape[2]
    packed = b"".join(b"\x00" + row.tobytes() for row in data.reshape(height, width * channels))
    compressed = zlib.compress(packed, level=6)

    def _chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
        return (
            struct.pack(">I", len(chunk_data))
            + chunk_type
            + chunk_data
            + struct.pack(">I", zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    chunks = [_chunk(b"IHDR", ihdr)]

    text_entries = metadata or {}
    for key, value in text_entries.items():
        text_data = f"{key}\x00{value}".encode("latin-1", errors="replace")
        chunks.append(_chunk(b"tEXt", text_data))

    if dpi is not None and dpi[0] and dpi[1]:
        ppm_x = int(round(dpi[0] / 0.0254))
        ppm_y = int(round(dpi[1] / 0.0254))
        phys = struct.pack(">IIB", ppm_x, ppm_y, 1)
        chunks.append(_chunk(b"pHYs", phys))

    chunk_size = 65536
    for offset in range(0, len(compressed), chunk_size):
        chunk_payload = compressed[offset : offset + chunk_size]
        chunks.append(_chunk(b"IDAT", chunk_payload))

    chunks.append(_chunk(b"IEND", b""))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        for chunk in chunks:
            handle.write(chunk)
    return out_path

def read_spatial_csv(path: Path) -> Dict[str, pd.DataFrame]:
    """Parse Loupe spatial.csv into DataFrames keyed by header lines (e.g., 'x,y', 'az,el').

    The file is structured as multiple blocks, each starting with a header like 'x,y' on its own
    line, followed by rows of numeric values until the next header or EOF.
    """
    txt = Path(path).read_text(errors="ignore").splitlines()
    blocks: Dict[str, pd.DataFrame] = {}
    i = 0
    while i < len(txt):
        line = txt[i].strip()
        if not line:
            i += 1
            continue
        # Header lines contain a comma and do not start with a number or sign
        if ("," in line) and not (line[0].isdigit() or line[0] in "+-."):
            header = ",".join([c.strip() for c in line.split(",")])
            rows: List[List[float]] = []
            i += 1
            while i < len(txt):
                s = txt[i].strip()
                if not s:
                    break
                c0 = s[0]
                if c0.isdigit() or c0 in "+-.":
                    parts = [p.strip() for p in s.split(",")]
                    try:
                        rows.append([float(p) for p in parts])
                    except ValueError:
                        break
                    i += 1
                else:
                    break
            if rows:
                cols = [c.strip() for c in header.split(",")]
                blocks[header] = pd.DataFrame(rows, columns=cols)
        else:
            i += 1
    return blocks


def read_loupe_csv(path: Path) -> Dict[str, str]:
    """Read Loupe loupe.csv file and return key-value pairs as strings."""
    df = pd.read_csv(path, header=None, index_col=0)
    series = df.squeeze() if hasattr(df, "squeeze") else df.iloc[:, 0]
    return series.fillna("N/A").to_dict()


def get_spatial_params_from_loupe_csv(loupe_csv: Path) -> Dict[str, float]:
    """Extract spatial parameters from a Loupe `loupe.csv`.

    Returns keys: 'laser_center' (tuple), 'pixel_scale' (float).
    Falls back to config defaults when missing.
    """
    cfg = get_config()
    params = read_loupe_csv(loupe_csv)
    laser_x = float(params.get("laser_x", cfg.spatial.get("laser_center", [809.0, 664.0])[0]))
    laser_y = float(params.get("laser_y", cfg.spatial.get("laser_center", [809.0, 664.0])[1]))
    # Loupe typically does not store pixel_scale explicitly; use config image.pixel_scale
    pixel_scale = float(params.get("pixel_scale", cfg.image.pixel_scale))
    return {
        "laser_center": (laser_x, laser_y),
        "pixel_scale": pixel_scale,
    }


def convert_coordinates_loupe(
    x: np.ndarray,
    y: np.ndarray,
    laser_center: Tuple[float, float],
    pixel_scale: float,
    scale_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert normalized (x, y) to pixel coordinates using Loupe-consistent transform.

    xPix = laser_center_x * scale_factor - x * scale_factor * 1000 / pixel_scale
    yPix = laser_center_y * scale_factor - y * scale_factor * 1000 / pixel_scale
    """
    xPix = laser_center[0] * scale_factor - x * scale_factor * 1000.0 / float(pixel_scale)
    yPix = laser_center[1] * scale_factor - y * scale_factor * 1000.0 / float(pixel_scale)
    return xPix, yPix


def load_spatial_table(working_dir: Path, scale_factor: float = 1.0) -> pd.DataFrame:
    """Load `spatial.csv` and `loupe.csv` to compute xPix/yPix for each point.

    Returns DataFrame with columns: ['point', 'x', 'y', 'xPix', 'yPix']
    """
    working_dir = Path(working_dir)
    spatial_path = working_dir / "spatial.csv"
    loupe_path = working_dir / "loupe.csv"
    require_file(spatial_path, "spatial.csv not found")
    require_file(loupe_path, "loupe.csv not found")

    blocks = read_spatial_csv(spatial_path)
    if "x,y" not in blocks:
        # Some files may use capital letters; try case-insensitive match
        key = next((k for k in blocks.keys() if k.lower().strip() == "x,y"), None)
        if key is None:
            raise ValueError("'x,y' block not found in spatial.csv")
        xy = blocks[key]
    else:
        xy = blocks["x,y"]

    params = get_spatial_params_from_loupe_csv(loupe_path)
    xPix, yPix = convert_coordinates_loupe(
        x=xy["x"].to_numpy(float),
        y=xy["y"].to_numpy(float),
        laser_center=params["laser_center"],
        pixel_scale=params["pixel_scale"],
        scale_factor=scale_factor,
    )
    df = pd.DataFrame({
        "point": np.arange(len(xy), dtype=int),
        "x": xy["x"].to_numpy(float),
        "y": xy["y"].to_numpy(float),
        "xPix": xPix,
        "yPix": yPix,
    })
    return df


def overlay_points_on_aci(
    image: np.ndarray,
    overlays: List[Dict[str, Any]],
    output_path: Path,
    title: Optional[str] = None,
    colorbar: Optional[Dict[str, Any]] = None,
) -> Path:
    """Render overlays on an ACI image and save to `output_path`.

    Moved to sherloc_pipeline.visualization.spatial. This shim delegates to that
    implementation for backward compatibility with existing callers.
    """
    from sherloc_pipeline.visualization.spatial import overlay_points_on_aci as _viz_overlay
    return _viz_overlay(image=image, overlays=overlays, output_path=output_path, title=title, colorbar=colorbar)


# ---- Core APIs for spatial overlay rendering ----

def build_spatial_crop(
    image: np.ndarray,
    spatial_df: pd.DataFrame,
    upscale: float = 1.0,
    pad_px: float = 50.0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Return (crop, (xmin, ymin, xmax, ymax)) around scan area (optional upscaling).

    - Computes bounding box from spatial_df['xPix','yPix'] with ±pad_px (in current image pixels).
    - If upscale != 1, resizes crop using skimage.transform.resize with anti-aliasing.
    - Preserves grayscale vs RGB.
    """
    xs = spatial_df["xPix"].to_numpy(float)
    ys = spatial_df["yPix"].to_numpy(float)
    h = int(image.shape[0]); w = int(image.shape[1])
    xmin = max(0, int(np.floor(np.nanmin(xs) - pad_px)))
    xmax = min(w, int(np.ceil(np.nanmax(xs) + pad_px)))
    ymin = max(0, int(np.floor(np.nanmin(ys) - pad_px)))
    ymax = min(h, int(np.ceil(np.nanmax(ys) + pad_px)))
    cropped = image[ymin:ymax, xmin:xmax, ...] if image.ndim == 3 else image[ymin:ymax, xmin:xmax]
    if upscale and float(upscale) != 1.0:
        try:
            import skimage.transform as _sktr
            if cropped.ndim == 2:
                cropped = _sktr.resize(cropped, (int(cropped.shape[0] * upscale), int(cropped.shape[1] * upscale)), mode='reflect', anti_aliasing=True)
            else:
                cropped = _sktr.resize(cropped, (int(cropped.shape[0] * upscale), int(cropped.shape[1] * upscale), int(cropped.shape[2])), mode='reflect', anti_aliasing=True)
        except Exception:
            # Fallback: nearest-neighbor via numpy repeat
            factor = int(round(float(upscale)))
            if factor > 1:
                if cropped.ndim == 2:
                    cropped = np.repeat(np.repeat(cropped, factor, axis=0), factor, axis=1)
                else:
                    cropped = np.repeat(np.repeat(cropped, factor, axis=0), factor, axis=1)
    return cropped, (xmin, ymin, xmax, ymax)


def _compose_rings_on_image(
    image: np.ndarray,
    points_selected: Optional[np.ndarray],
    snr_selected: Optional[np.ndarray],
    points_nondet: Optional[np.ndarray],
    pixel_scale_um_per_px: float,
    use_raster: bool = True,
    cmap_name: str = "viridis",
    ring_width_det_px: float = 2.0,
    ring_width_nondet_px: float = 1.0,
    alpha_nondet: float = 0.3,
) -> np.ndarray:
    """Return RGB image with rings. Delegates to visualization.spatial.

    Implementation moved to visualization/spatial.py to keep core/ matplotlib-free.
    """
    from sherloc_pipeline.visualization.spatial import _compose_rings_on_image as _impl
    return _impl(image=image, points_selected=points_selected, snr_selected=snr_selected,
                 points_nondet=points_nondet, pixel_scale_um_per_px=pixel_scale_um_per_px,
                 use_raster=use_raster, cmap_name=cmap_name,
                 ring_width_det_px=ring_width_det_px, ring_width_nondet_px=ring_width_nondet_px,
                 alpha_nondet=alpha_nondet)


def draw_rings(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    snr: Optional[np.ndarray] = None,
    *,
    pixel_scale_um_per_px: float,
    nonselected: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    style: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Compose notebook-style rings on an image and return RGB float array.

    Delegates to visualization.spatial._compose_rings_on_image.
    """
    style = style or {}
    pts_sel = None
    pts_ns = None
    if x is not None and y is not None:
        pts_sel = np.stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)], axis=1)
    if nonselected is not None and len(nonselected) == 2 and nonselected[0] is not None and nonselected[1] is not None:
        pts_ns = np.stack([np.asarray(nonselected[0], dtype=float), np.asarray(nonselected[1], dtype=float)], axis=1)
    return _compose_rings_on_image(
        image=image,
        points_selected=pts_sel,
        snr_selected=(np.asarray(snr, dtype=float) if snr is not None else None),
        points_nondet=pts_ns,
        pixel_scale_um_per_px=float(pixel_scale_um_per_px),
        use_raster=bool(style.get('raster', True)),
        cmap_name=str(style.get('cmap', 'viridis')),
        ring_width_det_px=float(style.get('ring_width_det_px', 2.0)),
        ring_width_nondet_px=float(style.get('ring_width_nondet_px', 1.0)),
        alpha_nondet=float(style.get('alpha_nondet', 0.3)),
    )


def render_pointloc_full(
    image_native: np.ndarray,
    pixel_scale_um_per_px: float,
    selected_xy: np.ndarray,
    selected_snr: np.ndarray,
    nonselected_xy: Optional[np.ndarray],
    out_path: Path,
) -> Path:
    """Render full-frame point location image with rings.

    Moved to sherloc_pipeline.visualization.spatial. This shim delegates for
    backward compatibility.
    """
    from sherloc_pipeline.visualization.spatial import render_pointloc_full as _viz_fn
    return _viz_fn(image_native=image_native, pixel_scale_um_per_px=pixel_scale_um_per_px,
                   selected_xy=selected_xy, selected_snr=selected_snr,
                   nonselected_xy=nonselected_xy, out_path=out_path)


def render_pointloc_zoomed(
    crop_image_native: np.ndarray,
    selected_xy_crop_native: np.ndarray,
    selected_snr: np.ndarray,
    nonselected_xy_crop_native: Optional[np.ndarray],
    out_path: Path,
    upscale: float = 3.0,
) -> Path:
    """Render zoomed point location image with rings.

    Moved to sherloc_pipeline.visualization.spatial. This shim delegates for
    backward compatibility.
    """
    from sherloc_pipeline.visualization.spatial import render_pointloc_zoomed as _viz_fn
    return _viz_fn(crop_image_native=crop_image_native,
                   selected_xy_crop_native=selected_xy_crop_native,
                   selected_snr=selected_snr,
                   nonselected_xy_crop_native=nonselected_xy_crop_native,
                   out_path=out_path, upscale=upscale)


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

    Moved to sherloc_pipeline.visualization.spatial. This shim delegates for
    backward compatibility.
    """
    from sherloc_pipeline.visualization.spatial import render_pointloc_with_colorbar as _viz_fn
    return _viz_fn(image_native=image_native, pixel_scale_um_per_px=pixel_scale_um_per_px,
                   selected_xy=selected_xy, selected_snr=selected_snr,
                   nonselected_xy=nonselected_xy, title=title, out_path=out_path)


def build_combined_grid(
    crop_image: np.ndarray,
    class_to_points: Dict[str, Dict[str, np.ndarray]],
    snr_ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
    out_path: Path,
    suptitle: Optional[str] = None,
    suptitle_y: float = 0.92,
) -> Path:
    """Build and save the 3×3 combined minerals grid.

    Moved to sherloc_pipeline.visualization.spatial. This shim delegates for
    backward compatibility.
    """
    from sherloc_pipeline.visualization.spatial import build_combined_grid as _viz_fn
    return _viz_fn(crop_image=crop_image, class_to_points=class_to_points,
                   snr_ranges=snr_ranges, out_path=out_path,
                   suptitle=suptitle, suptitle_y=suptitle_y)
