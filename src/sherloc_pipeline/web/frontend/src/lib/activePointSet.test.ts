// Coverage for the Map Mode image/point-set pairing (issue #8).
// The colorized point set must only be drawn when the colorized image is
// actually active — never merely because `useColorized` is true while the
// colorized image is still loading or failed to load.

import { describe, it, expect } from 'vitest';
import { selectActivePointSet } from './activePointSet';
import type { PointSet } from './types/map';

const gray: PointSet = {
  scan_id: 's',
  source: 'sherloc',
  coordinate_source: 'scanner_workspace_transformed',
  points: [{ index: 0, x: 820, y: 640 }],
  voronoi: null,
};
const color: PointSet = {
  scan_id: 's',
  source: 'sherloc',
  coordinate_source: 'scanner_workspace_transformed',
  points: [{ index: 0, x: 793, y: 632 }],
  voronoi: null,
};
const img = {} as unknown; // stand-in for a decoded HTMLImageElement

describe('selectActivePointSet — image/point-set pairing', () => {
  it('uses the colorized set only when colorized AND its image are active', () => {
    expect(selectActivePointSet(true, img, color, gray)).toBe(color);
  });

  it('keeps grayscale while the colorized image is still loading (null image)', () => {
    // The registration-bug edge path: useColorized true and the colorized
    // point set already assigned, but the colorized image has not arrived.
    expect(selectActivePointSet(true, null, color, gray)).toBe(gray);
  });

  it('keeps grayscale when the colorized image fetch failed (null image)', () => {
    expect(selectActivePointSet(true, undefined, color, gray)).toBe(gray);
  });

  it('keeps grayscale when no colorized point set exists', () => {
    expect(selectActivePointSet(true, img, null, gray)).toBe(gray);
  });

  it('uses grayscale when colorized is toggled off', () => {
    expect(selectActivePointSet(false, img, color, gray)).toBe(gray);
  });

  it('passes through null grayscale (nothing loaded yet)', () => {
    expect(selectActivePointSet(false, null, null, null)).toBeNull();
  });
});
