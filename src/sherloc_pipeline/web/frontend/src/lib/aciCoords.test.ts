// Unit coverage for the ACI pixel-coordinate helper (issue #16).
// The Workbench "Scan" zoom button corrupted-view symptom traced to a
// naive `x_aci_pixel ?? x_pixel` fallback that interpreted Loupe
// scanner_workspace values (~±0.5) as ACI image pixels. These tests
// lock in the frame discipline so future changes can't regress.

import { describe, it, expect } from 'vitest';
import { getAciPixel, computeAciBBox } from './aciCoords';
import type { ScanPoint } from './types';

function makePoint(overrides: Partial<ScanPoint>): ScanPoint {
  return {
    id: 'p',
    point_index: 0,
    x_pixel: null,
    y_pixel: null,
    x_aci_pixel: null,
    y_aci_pixel: null,
    azimuth_dn: null,
    elevation_dn: null,
    azimuth_error: null,
    elevation_error: null,
    photodiode_mean: null,
    photodiode_std: null,
    coordinate_frame: null,
    ...overrides,
  };
}

describe('getAciPixel — frame discipline', () => {
  it('uses x_aci_pixel/y_aci_pixel when present (server-resolved)', () => {
    const pt = makePoint({
      x_aci_pixel: 820,
      y_aci_pixel: 640,
      x_pixel: 0.1,
      y_pixel: -0.2,
      coordinate_frame: 'scanner_workspace',
    });
    expect(getAciPixel(pt)).toEqual({ x: 820, y: 640 });
  });

  it('uses x_pixel/y_pixel when coordinate_frame is aci_pixel (PDS RMO case)', () => {
    const pt = makePoint({
      x_aci_pixel: null,
      y_aci_pixel: null,
      x_pixel: 810,
      y_pixel: 665,
      coordinate_frame: 'aci_pixel',
    });
    expect(getAciPixel(pt)).toEqual({ x: 810, y: 665 });
  });

  it('returns null for scanner_workspace points without server-resolved coords (issue #16 root cause)', () => {
    // Loupe scan whose working_dir lacks spatial.csv — backend leaves
    // x_aci_pixel null, x_pixel carries raw workspace units. Previously
    // the viewer treated 0.2 as an ACI pixel; we now skip such points.
    const pt = makePoint({
      x_aci_pixel: null,
      y_aci_pixel: null,
      x_pixel: 0.2,
      y_pixel: -0.1,
      coordinate_frame: 'scanner_workspace',
    });
    expect(getAciPixel(pt)).toBeNull();
  });

  it('accepts legacy null-frame points (R2 F2 fix — preserve pre-frame-field behavior)', () => {
    // Records ingested before `coordinate_frame` was added have
    // `frame=null` per `models/spectra.py:580-583`. The prior viewer
    // used x_pixel/y_pixel for these; R2 reinstates that path to
    // avoid regressing legacy records.
    const pt = makePoint({
      x_pixel: 820,
      y_pixel: 640,
      coordinate_frame: null,
    });
    expect(getAciPixel(pt)).toEqual({ x: 820, y: 640 });
  });

  it('rejects scanner_workspace-framed points even at survey-scale magnitudes', () => {
    // Survey scans can have larger scanner_workspace values than detail
    // scans (docstring: "±0.5 range for detail, larger for survey").
    // Explicit frame label is authoritative regardless of magnitude.
    const pt = makePoint({
      x_pixel: 50,
      y_pixel: 50,
      coordinate_frame: 'scanner_workspace',
    });
    expect(getAciPixel(pt)).toBeNull();
  });

  it('returns null when neither coord pair is fully populated', () => {
    expect(getAciPixel(makePoint({}))).toBeNull();
    expect(
      getAciPixel(makePoint({ x_aci_pixel: 100, y_aci_pixel: null })),
    ).toBeNull();
  });
});

describe('computeAciBBox', () => {
  it('returns the envelope of all resolvable points', () => {
    const points: ScanPoint[] = [
      makePoint({ x_aci_pixel: 800, y_aci_pixel: 600 }),
      makePoint({ x_aci_pixel: 850, y_aci_pixel: 620 }),
      makePoint({ x_aci_pixel: 830, y_aci_pixel: 680 }),
    ];
    expect(computeAciBBox(points)).toEqual({
      minX: 800,
      minY: 600,
      maxX: 850,
      maxY: 680,
    });
  });

  it('mixes resolvable and unresolvable points without skewing the bbox', () => {
    const points: ScanPoint[] = [
      // Valid PDS-frame point.
      makePoint({
        x_pixel: 810,
        y_pixel: 660,
        coordinate_frame: 'aci_pixel',
      }),
      // Loupe scanner_workspace point — would corrupt the bbox if
      // accidentally included. computeAciBBox must skip it.
      makePoint({
        x_pixel: 0.1,
        y_pixel: 0.2,
        coordinate_frame: 'scanner_workspace',
      }),
      // Resolved Loupe point.
      makePoint({ x_aci_pixel: 830, y_aci_pixel: 650 }),
    ];
    expect(computeAciBBox(points)).toEqual({
      minX: 810,
      minY: 650,
      maxX: 830,
      maxY: 660,
    });
  });

  it('returns null when no point has resolvable coords', () => {
    const points: ScanPoint[] = [
      makePoint({
        x_pixel: 0.1,
        y_pixel: 0.2,
        coordinate_frame: 'scanner_workspace',
      }),
      makePoint({}),
    ];
    expect(computeAciBBox(points)).toBeNull();
  });

  it('handles a single point (degenerate bbox)', () => {
    const points: ScanPoint[] = [
      makePoint({ x_aci_pixel: 820, y_aci_pixel: 640 }),
    ];
    expect(computeAciBBox(points)).toEqual({
      minX: 820,
      minY: 640,
      maxX: 820,
      maxY: 640,
    });
  });
});
