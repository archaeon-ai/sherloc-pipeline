// Point-set / active-image pairing for the Map Mode overlay (issue #8).
//
// A scan can carry two point sets: the grayscale `pointSet` and, when a
// colorized ACI variant exists, a crop-shifted `pointSetColorized`. The
// overlay must always draw the point set that matches the image currently on
// screen, or the points land in the wrong frame (the ~28px registration bug
// this fix exists to prevent).

import type { PointSet } from './types/map';

/**
 * Select the point set paired with the currently displayed ACI image.
 *
 * The colorized point set is used ONLY when the colorized image is actually
 * active (`useColorized && colorizedImage`). `useColorized` alone is
 * insufficient: `MapMode` assigns `pointSetColorized` before the colorized
 * image fetch resolves (so a slow/failed fetch doesn't strand the coords), so
 * keying off `useColorized` alone would draw crop-shifted coordinates over the
 * grayscale image during the load window or on fetch failure. This mirrors the
 * exact condition `MapCanvas` uses to pick `activeImage`.
 */
export function selectActivePointSet(
  useColorized: boolean,
  colorizedImage: unknown,
  pointSetColorized: PointSet | null,
  pointSet: PointSet | null,
): PointSet | null {
  return useColorized && colorizedImage && pointSetColorized
    ? pointSetColorized
    : pointSet;
}
