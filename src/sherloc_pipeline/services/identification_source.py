"""Canonical raw point preparation from detached, bounded database fields."""

from types import SimpleNamespace
import zlib

import numpy as np

from sherloc_pipeline.services.identification_runtime import WorkRejected
from sherloc_pipeline.services.spectrum_axes import resolve_spectrum_axes
from sherloc_pipeline.services.spectrum_provenance import build_spectrum_provenance


def prepare_stored_point(detached, max_points):
    scan = SimpleNamespace(**detached["scan"])
    point = SimpleNamespace(**detached["point"])
    spectrum = SimpleNamespace(**detached["spectrum"], scan_point=point)
    for data in (spectrum.intensities, spectrum.wavelengths):
        if data is None:
            continue
        decoder = zlib.decompressobj()
        values = decoder.decompress(data, max_points * 4 + 1)
        if len(values) > max_points * 4 or not decoder.eof or decoder.unused_data:
            raise WorkRejected("source point limit exceeded")
    axes = resolve_spectrum_axes(scan, [spectrum], "R1")
    intensity = np.frombuffer(zlib.decompress(spectrum.intensities), dtype=np.float32)[axes.mask]
    normalized, max_pd = False, detached["max_photodiode"]
    if (scan.data_source or "loupe") == "loupe" and spectrum.spectrum_type == "dark_subtracted":
        pd = point.photodiode_mean
        if max_pd is not None and pd is not None and np.isfinite(pd) and pd > 0:
            intensity = intensity * (max_pd / pd)
            normalized = True
    else:
        max_pd = None
    provenance = build_spectrum_provenance(
        scan, [(spectrum, point.point_index)], axes, "R1", spectrum.spectrum_type, "point",
        [point.point_index], [point.point_index], intensity, normalization_applied=normalized,
        max_photodiode=max_pd, wavelength_filter=detached["wavelength_filter"])
    return axes.wavenumber[axes.mask], intensity, provenance
