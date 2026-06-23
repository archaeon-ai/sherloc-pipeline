// ACI pixel-coordinate resolution for scan points.
//
// A `ScanPoint` carries up to two coordinate pairs:
//   * `x_aci_pixel` / `y_aci_pixel` — always in ACI image-pixel frame when
//     present (populated server-side by `_compute_aci_pixels` from Loupe
//     `spatial.csv`, or by PDS ingestion for `coordinate_frame='aci_pixel'`).
//   * `x_pixel` / `y_pixel` — frame-dependent. Per
//     `models/spectra.py:101-104`, `coordinate_frame='scanner_workspace'`
//     means raw Loupe workspace units (~±0.5 range), and
//     `coordinate_frame='aci_pixel'` means already-resolved ACI pixels.
//
// The Workbench "Scan" zoom button bug (issue #16) traced to a naive
// `pt.x_aci_pixel ?? pt.x_pixel` fallback that interpreted
// scanner-workspace values (~±0.5) as ACI pixels, producing a tight
// zoom near the upper-left corner of the image where the ACI is
// typically dark. This helper enforces the frame discipline so callers
// only ever see image-pixel coords.

import type { ScanPoint } from './types';

export interface AciPixel {
  x: number;
  y: number;
}

/**
 * Returns the point's coordinates in the ACI image-pixel frame, or
 * `null` if they cannot be safely interpreted as such.
 *
 * Resolution order:
 *   1. `x_aci_pixel` / `y_aci_pixel` if both are present — these are
 *      authoritative ACI pixels regardless of `coordinate_frame`.
 *   2. Skip when `coordinate_frame === 'scanner_workspace'`: those
 *      values are in the ±0.5-ish range per `models/spectra.py:101-104`
 *      and would corrupt the bbox math (the original issue #16 defect).
 *   3. Otherwise — `coordinate_frame === 'aci_pixel'` (PDS RMO ingest)
 *      OR `coordinate_frame === null` (legacy / implicit, per the
 *      schema docstring) — accept `x_pixel` / `y_pixel`. Preserving
 *      the null-frame path keeps backward compatibility with legacy
 *      records ingested before `coordinate_frame` was added.
 */
export function getAciPixel(pt: ScanPoint): AciPixel | null {
  if (pt.x_aci_pixel !== null && pt.y_aci_pixel !== null) {
    return { x: pt.x_aci_pixel, y: pt.y_aci_pixel };
  }
  if (pt.x_pixel === null || pt.y_pixel === null) {
    return null;
  }
  if (pt.coordinate_frame === 'scanner_workspace') {
    return null;
  }
  // 'aci_pixel' or null (legacy implicit) — trust the value.
  return { x: pt.x_pixel, y: pt.y_pixel };
}

/**
 * Axis-aligned bounding box (image-pixel frame) covering every point
 * for which {@link getAciPixel} returns non-null. Returns `null` when
 * zero points have resolvable ACI coords.
 */
export interface AciBBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export function computeAciBBox(points: ScanPoint[]): AciBBox | null {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let any = false;

  for (const pt of points) {
    const coord = getAciPixel(pt);
    if (coord === null) continue;
    any = true;
    if (coord.x < minX) minX = coord.x;
    if (coord.x > maxX) maxX = coord.x;
    if (coord.y < minY) minY = coord.y;
    if (coord.y > maxY) maxY = coord.y;
  }

  if (!any || !isFinite(minX)) return null;
  return { minX, minY, maxX, maxY };
}
