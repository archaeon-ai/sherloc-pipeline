"""GET /api/plots/{scan_id} endpoint -- server-rendered plots."""

import io
import zlib

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from sqlalchemy.orm import Session

from sherloc_pipeline.database.models import ScanORM

router = APIRouter(prefix="/api", tags=["plots"])

VALID_PLOT_TYPES = {"spectrogram", "averaged_spectrum", "fit_overlay"}
VALID_REGIONS = {"R1", "R2", "R3"}
VALID_FORMATS = {"png", "pdf", "svg"}


@router.get("/plots/{scan_id}")
def get_plot(
    request: Request,
    scan_id: str,
    plot_type: str = Query("spectrogram"),
    region: str = Query("R1"),
    domain: str = Query("raman"),
    format: str = Query("png"),
    dpi: int = Query(150, ge=50, le=300),
) -> Response:
    """Generate and return a server-rendered plot."""
    if plot_type not in VALID_PLOT_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid plot_type: {plot_type}")
    if region not in VALID_REGIONS:
        raise HTTPException(status_code=400, detail=f"Invalid region: {region}")
    if format not in VALID_FORMATS:
        raise HTTPException(status_code=400, detail=f"Invalid format: {format}")

    session: Session = request.state.db
    scan = session.query(ScanORM).filter(ScanORM.id == scan_id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from sherloc_pipeline.core.calibration import (
            calculate_loupe_wavelength_wavenumber,
            get_region_wavelength_mask,
        )
        from sherloc_pipeline.database.models import ScanPointORM, SpectrumORM

        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=2148)
        mask = get_region_wavelength_mask(wavelength, region)
        wn = wavenumber[mask]

        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "averaged_spectrum":
            spectra = (
                session.query(SpectrumORM)
                .join(ScanPointORM)
                .filter(
                    ScanPointORM.scan_id == scan_id,
                    SpectrumORM.region == region,
                    SpectrumORM.spectrum_type == "dark_subtracted",
                )
                .all()
            )
            if not spectra:
                raise HTTPException(status_code=404, detail="No spectra available")

            all_int = []
            for sp in spectra:
                raw = np.frombuffer(zlib.decompress(sp.intensities), dtype=np.float32)
                if len(raw) >= len(mask):
                    all_int.append(raw[mask])
            if all_int:
                avg = np.mean(np.stack(all_int), axis=0)
                ax.plot(wn, avg, linewidth=0.8, color="#1e293b")
                x_label = "Wavelength (nm)" if region in ("R2", "R3") else r"Raman Shift (cm$^{-1}$)"
                ax.set_xlabel(x_label)
                ax.set_ylabel("Intensity (counts)")
                ax.set_title(f"{scan.target} {scan.scan_name} — Averaged Spectrum ({region})")
        else:
            # Default: spectrogram placeholder
            ax.text(
                0.5, 0.5, f"Plot type '{plot_type}' placeholder",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title(f"{scan.target} {scan.scan_name} ({region})")

        buf = io.BytesIO()
        fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        media_types = {
            "png": "image/png",
            "pdf": "application/pdf",
            "svg": "image/svg+xml",
        }
        return Response(content=buf.read(), media_type=media_types[format])

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Plot generation failed: {exc}")
