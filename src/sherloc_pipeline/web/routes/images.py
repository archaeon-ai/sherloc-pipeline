"""ACI context image endpoints.

v1.0-beta deploys SHERLOC ACI bytes from R2 (per m2020-phase platform
spec §3.9 — hierarchical-key model, Session 73 amendment). The route
handler does its existing DB-driven file_path resolution + colorized /
enhanced / VICAR pipeline; the storage layer changes from local FS to
R2 via per-tier strip-prefix + bucket selection.

PHASE_TIER drives the strip prefix + bucket:

- team:   <team-data-root>/<rest>  → phase-team/sherloc-aci/<rest>
- public: <public-pds-root>/<rest> → phase-public/sherloc-aci/<rest>

v4.1.9 refactor (Session 94, m2020-phase spec §3.9.8): the R2 client +
per-tier config machinery moved to ``web/r2_reader``; this module
imports the shared primitives. Behavior is unchanged.

VICAR ``.IMG`` files are converted in-process at request time (via the
existing ``read_aci_image()`` path through a ``NamedTemporaryFile``
shim, since the VICAR reader takes a Path, not a file-like). PNG bytes
flow through the existing flatten-alpha + enhancement pipeline.
"""

import io
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from sqlalchemy.orm import Session

from sherloc_pipeline.database.models import ContextImageORM, ScanORM
from sherloc_pipeline.web.data_access import DataAccessService
from sherloc_pipeline.web.r2_reader import (
    colorized_variant_exists,
    derive_r2_key,
    find_colorized_key,
    get_r2_client_and_config,
    r2_get_bytes,
)

# Backward-compat re-export: the public surface other modules grep for.
__all__ = ["router", "colorized_variant_exists", "select_served_aci"]


def select_served_aci(session: Session, scan_id: str) -> Optional[ContextImageORM]:
    """Return the ACI ``ContextImageORM`` that ``GET /api/images/{scan}/aci`` would serve.

    Centralises the candidate-selection rule used by ``get_aci_image``
    so other routes (e.g. ``GET /api/scans/{id}`` probing for the
    colorized sibling) can target the SAME row the image route serves
    rather than ``.first()``-ing into an angle-range variant. PR #31
    Codex Round 1 F1 — desync between probed row and served row
    silently lies about Colorized button availability when a scan has
    multiple ACI rows.

    Rule (mirrors ``routes/images.py`` lines 225-241):
    - All ``image_type == "ACI"`` rows for the scan are candidates.
    - Prefer the base image whose filename does NOT end with an
      angle-range suffix like ``_145-185``.
    - Fall back to the first candidate if no base image is found
      (defensive — production scans should always carry a base).
    - Return ``None`` when no ACI rows exist.
    """
    aci_candidates = (
        session.query(ContextImageORM)
        .filter(
            ContextImageORM.scan_id == scan_id,
            ContextImageORM.image_type == "ACI",
        )
        .all()
    )
    for candidate in aci_candidates:
        fname = Path(candidate.file_path or "").stem
        if not re.search(r"_\d+-\d+$", fname):
            return candidate
    if aci_candidates:
        return aci_candidates[0]
    return None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["images"])

# Cache header for successfully served images (24 hours).
_CACHE_CONTROL = "public, max-age=86400"

# Directory for enhanced ACI image cache.
_ENHANCED_CACHE_DIR = Path("/tmp/sherloc_aci_cache")

# Route-level scan_id path-traversal rejection (spec §3.9.1 + §2.9.1 Layer 1).
_SCAN_ID_BANNED = re.compile(r"\.\.|/|\\|%2e|%2f", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Image processing helpers (storage-layer-agnostic; bytes-in)
# ---------------------------------------------------------------------------

def _flatten_alpha(pil_img):
    """Convert images with alpha to opaque grayscale/RGB.

    Loupe exports ACI PNGs in LA mode where the actual image data lives
    in the *alpha* channel (luminance is a binary mask). Detect that
    pattern and extract the alpha as the grayscale image.
    """
    from PIL import Image as PILImage

    if pil_img.mode == "LA":
        L = pil_img.getchannel("L")
        A = pil_img.getchannel("A")
        l_arr = np.array(L)
        unique_l = np.unique(l_arr)
        if len(unique_l) <= 2 and set(unique_l).issubset({0, 255}):
            return A
        bg = PILImage.new("L", pil_img.size, 0)
        bg.paste(L, mask=A)
        return bg
    if pil_img.mode == "RGBA":
        bg = PILImage.new("RGB", pil_img.size, (0, 0, 0))
        bg.paste(pil_img, mask=pil_img.getchannel("A"))
        return bg
    if pil_img.mode == "PA":
        return _flatten_alpha(pil_img.convert("RGBA"))
    return pil_img


def _image_to_png_bytes(image_array) -> bytes:
    """Convert a numpy image array to PNG bytes using Pillow."""
    from PIL import Image as PILImage

    if image_array.ndim == 2:
        pil_img = PILImage.fromarray(image_array, mode="L")
    else:
        pil_img = PILImage.fromarray(image_array)
    pil_img = _flatten_alpha(pil_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _apply_enhancement(image_array: np.ndarray, upscale: int) -> np.ndarray:
    """Apply the ACI enhancement pipeline to a numpy image array.

    Steps:
    1. Adaptive histogram equalization (CLAHE, clip_limit=0.03).
    2. Lanczos upscaling if upscale > 1.
    3. Unsharp mask sharpening (radius=1.5, amount=0.5).
    """
    from skimage.exposure import equalize_adapthist
    from skimage.filters import unsharp_mask
    from skimage.transform import resize

    if image_array.dtype == np.uint8:
        img = image_array.astype(np.float64) / 255.0
    else:
        img = image_array.astype(np.float64)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

    img = equalize_adapthist(img, clip_limit=0.03)

    if upscale > 1:
        h, w = img.shape[:2]
        if img.ndim == 2:
            new_shape = (h * upscale, w * upscale)
        else:
            new_shape = (h * upscale, w * upscale, img.shape[2])
        img = resize(img, new_shape, order=3, anti_aliasing=True)

    img = unsharp_mask(img, radius=1.5, amount=0.5)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def _get_enhanced_png(scan_id: str, image_array: np.ndarray, upscale: int) -> bytes:
    """Return enhanced PNG bytes, serving from disk cache if available.

    Cache key: ``{scan_id}_enhanced_{upscale}x.png`` under
    ``_ENHANCED_CACHE_DIR``.
    """
    _ENHANCED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _ENHANCED_CACHE_DIR / f"{scan_id}_enhanced_{upscale}x.png"

    if cache_path.is_file():
        logger.debug("Serving enhanced ACI from cache: %s", cache_path)
        return cache_path.read_bytes()

    enhanced = _apply_enhancement(image_array, upscale)
    png_bytes = _image_to_png_bytes(enhanced)
    try:
        cache_path.write_bytes(png_bytes)
    except OSError:
        logger.warning("Could not write enhanced ACI cache to %s", cache_path)
    return png_bytes


def _decode_vicar_bytes(bytes_data: bytes) -> np.ndarray:
    """Write VICAR bytes to a NamedTemporaryFile and call read_aci_image.

    Used for ``.IMG`` keys served from R2 (PDS-archived public-tier data).
    The VICAR reader takes a Path; tempfile shim avoids a more invasive
    refactor for v1.0-beta.
    """
    from sherloc_pipeline.vision.img_reader import read_aci_image

    with tempfile.NamedTemporaryFile(suffix=".IMG", delete=False) as tmp:
        tmp.write(bytes_data)
        tmp_path = Path(tmp.name)
    try:
        image_array, _meta = read_aci_image(tmp_path)
        return image_array
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------

def _get_data_access(request: Request) -> DataAccessService:
    access_mode = getattr(request.app.state, "access_mode", "internal")
    return DataAccessService(access_mode=access_mode)


@router.get("/images/{scan_id}/aci")
def get_aci_image(
    request: Request,
    scan_id: str,
    colorized: bool = Query(False, description="Return colorized variant if available"),
    enhanced: bool = Query(False, description="Apply adaptive histogram equalization and sharpening"),
    upscale: int = Query(1, ge=1, le=5, description="Lanczos upscaling factor (1–5, only used when enhanced=true)"),
):
    """Serve the ACI context image for a scan.

    Resolves the per-tier R2 key from the ``context_images.file_path``
    column (spec §3.9.3 hierarchical-key model). Optional query params:

    - colorized: probe sibling ``sol_NNNN_colorized/<...>`` R2 key; serve
      that variant if present, else the base.
    - enhanced: apply CLAHE + optional Lanczos upscale + unsharp mask.
      Results cached at ``/tmp/sherloc_aci_cache/`` for 24h.
    - upscale: integer 1–5 (only used when ``enhanced=true``).

    Returns: ``image/png`` bytes with 24h ``Cache-Control``.
    """
    # Route-level path-traversal guard on scan_id (spec §3.9.1).
    if not scan_id or _SCAN_ID_BANNED.search(scan_id):
        raise HTTPException(status_code=400, detail="invalid_scan_id")

    session = request.state.db

    scan = session.query(ScanORM).filter(ScanORM.id == scan_id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    data_access = _get_data_access(request)
    data_access.validate_scan_access(scan)

    aci = select_served_aci(session, scan_id)
    if aci is None:
        raise HTTPException(
            status_code=404,
            detail=f"No ACI context image found for scan {scan_id}",
        )

    file_path_str = aci.file_path or ""

    # Derive R2 key from per-tier strip + base file_path. A `pds:` LIDVID
    # ref (legacy on-demand-fetch path) does not begin with the per-tier
    # strip prefix, so it surfaces as `misconfigured_path` 500 per spec
    # §3.9.4 — an unresolved `pds:` ref in production IS broken ingestion
    # (the rclone in ARCHITECTURE_LOCKED §14 D3.4 should have resolved
    # LIDVIDs to absolute paths before R2 population). v1.1 may revisit
    # the on-demand path; v1.0-beta rejects.
    _, cfg = get_r2_client_and_config()
    base_key = derive_r2_key(
        file_path_str, cfg["strip_prefix"], active_tier=cfg.get("tier")
    )

    # Optional colorized variant — sibling sol_NNNN_colorized/ key.
    key = base_key
    if colorized:
        colorized_key = find_colorized_key(base_key)
        if colorized_key is not None:
            key = colorized_key

    # R2 GET — surfaces 404/502 per spec §3.9.4.
    bytes_data = r2_get_bytes(key)

    suffix = Path(key).suffix.upper()
    if suffix == ".PNG":
        from PIL import Image as PILImage

        pil_img = PILImage.open(io.BytesIO(bytes_data))
        has_alpha = pil_img.mode in ("LA", "RGBA", "PA")
        if has_alpha:
            pil_img = _flatten_alpha(pil_img)
        if enhanced:
            image_array = np.array(pil_img)
            png_bytes = _get_enhanced_png(scan_id, image_array, upscale)
        elif has_alpha:
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
        else:
            png_bytes = bytes_data
    elif suffix == ".IMG":
        try:
            image_array = _decode_vicar_bytes(bytes_data)
        except Exception as exc:
            logger.exception("VICAR decode failed for key=%s", key)
            raise HTTPException(
                status_code=500, detail=f"Failed to read VICAR image: {exc}"
            ) from exc
        if enhanced:
            png_bytes = _get_enhanced_png(scan_id, image_array, upscale)
        else:
            png_bytes = _image_to_png_bytes(image_array)
    else:
        raise HTTPException(
            status_code=500, detail=f"Unsupported image format: {suffix}"
        )

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": _CACHE_CONTROL},
    )
