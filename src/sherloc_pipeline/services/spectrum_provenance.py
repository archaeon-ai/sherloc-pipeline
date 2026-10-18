"""Provenance for the exact source selection and served spectrum."""

import hashlib
import json
import zlib

import numpy as np

from sherloc_pipeline import __version__


def build_spectrum_provenance(
    scan, rows, axes, region, spectrum_type, mode, requested_indices,
    contributor_indices, intensity, *, normalization_applied=False,
    max_photodiode=None, despike_applied=False, despike_method=None,
    despike_params=None, masked_positions=None, aggregation=None,
    baseline_params=None, wavelength_filter=None, despike_lineage=None,
):
    """Keep missing product evidence null; hash native arrays and lineage.

    The source digest identifies a reproducible response input and recipe,
    not an original product file. No original-file hash is inferred from DB
    arrays, and no calibration version is inferred from a PDS origin label.
    """
    contributors = set(contributor_indices)
    rows = [(sp, idx) for sp, idx in rows if idx in contributors]
    metadata = scan.pds4_metadata if scan.data_source == "pds4" else scan.loupe_metadata
    metadata = metadata or {}
    source = {
        "scan_id": scan.id,
        "sol": scan.sol_number,
        "target": scan.target,
        "scan_name": scan.scan_name,
        "data_origin": scan.data_source,
        "product_id": metadata.get("lidvid") if scan.data_source == "pds4" else None,
        "product_version": metadata.get("version"),
        "source_sha256": metadata.get("source_sha256"),
        "scan_excitation_nm": scan.laser_wavelength_nm,
        "scan_excitation_basis": "scan_record_or_nominal_default",
        "processing_applied": scan.processing_applied,
        "pipeline_version": scan.processing_pipeline_version,
        "config_hash": scan.processing_config_hash,
    }
    selection = {
        "mode": mode,
        "indices": requested_indices if requested_indices is not None else "all",
        "n_points": len(contributors),
        "contributor_indices": sorted(contributors),
        "omitted_indices": sorted(set(requested_indices if requested_indices is not None else range(scan.n_points)) - contributors),
    }
    native = []
    for sp, idx in rows:
        values = np.frombuffer(zlib.decompress(sp.intensities), dtype=np.float32)
        native.append({
            "spectrum_id": sp.id, "point_index": idx, "region": sp.region,
            "spectrum_type": sp.spectrum_type, "processing_level": sp.processing_level,
            "intensity_sha256": hashlib.sha256(values.astype("<f4").tobytes()).hexdigest(),
        })
    processing = []
    if despike_lineage:
        processing.append({"step": "despike", "method": "ml", "scope": "per_contributor",
                           "contributors": despike_lineage})
    if region == "R123":
        processing.append({"step": "r123_stitch", "method": "overlap_sum", "regions": ["R1", "R2", "R3"]})
    if spectrum_type == "dark_subtracted" and (scan.data_source or "loupe") == "loupe":
        pd_by_index = {sp.scan_point.point_index: sp.scan_point.photodiode_mean for sp, _ in rows}
        available = {idx: pd for idx, pd in pd_by_index.items()
                     if pd is not None and np.isfinite(pd) and pd > 0}
        processing.append({
            "step": "laser_normalization", "applied": normalization_applied,
            "max_photodiode": max_photodiode,
            "factors": {str(idx): max_photodiode / pd for idx, pd in available.items()}
                       if normalization_applied else {},
            "missing_indices": sorted(contributors - available.keys()),
            "unavailable_reason": None if normalization_applied else "missing_finite_positive_photodiode",
        })
    if aggregation is not None:
        processing.append({"step": "aggregate", **aggregation})
    if despike_applied and despike_method == "modz":
        processing.append({"step": "despike", "method": "modz", "scope": "served_array", "params": despike_params})
    if baseline_params is not None:
        processing.append({"step": "baseline", "method": "aspls", "params": baseline_params})
    provenance = {
        "data_origin": scan.data_source,
        "source": source,
        "selection": selection,
        "source_spectra": native,
        "source_processing": {"spectrum_type": spectrum_type, "processing_applied": scan.processing_applied,
                              "software_version": metadata.get("software_version")},
        "axis_source": axes.source,
        "axis_sha256": axes.sha256,
        "calibration_version": axes.calibration_version,
        "excitation_nm": axes.excitation_nm,
        "excitation_basis": axes.excitation_basis,
        "wavelength_unit": "nm",
        "wavenumber_unit": "cm-1",
        "intensity_unit": metadata.get("intensity_unit") or (
            "counts" if scan.data_source == "loupe" and spectrum_type == "dark_subtracted" else None
        ),
        "region": region,
        "wavelength_filter": wavelength_filter,
        "averaging_method": aggregation["method"] if aggregation else None,
        "processing_chain": processing,
        "application": {"version": __version__, "commit": None},
    }
    identity = {
        "provenance": provenance,
        "channel_indices": np.flatnonzero(axes.mask).tolist(),
        "masked_positions": masked_positions or [],
        "served_intensity_sha256": hashlib.sha256(np.asarray(intensity, dtype="<f8").tobytes()).hexdigest(),
    }
    provenance["source_digest"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return provenance
