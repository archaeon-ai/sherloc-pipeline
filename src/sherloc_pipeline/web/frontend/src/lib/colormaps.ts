// ============================================================
// Colormap utilities for Map Mode overlays.
// No external dependencies — colormaps are embedded as lookup tables.
// ============================================================

/** Returns [r, g, b] in 0-255 range for a scalar t in [0, 1]. */
export type ColormapFn = (t: number) => [number, number, number];

/**
 * Get a colormap function by name.
 * Supported: 'viridis', 'red', 'green', 'blue'.
 */
export function getColormap(name: string): ColormapFn {
  switch (name) {
    case 'viridis':
      return viridis;
    case 'red':
      return monoRed;
    case 'green':
      return monoGreen;
    case 'blue':
      return monoBlue;
    default:
      return viridis;
  }
}

/** Normalize value to [0,1] given a range. Returns 0 if range is degenerate. */
export function normalizeValue(
  value: number,
  min: number,
  max: number,
): number {
  if (max <= min) return 0;
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

/** Convert colormap [r,g,b] output to CSS rgba string. */
export function colormapToRGBA(
  rgb: [number, number, number],
  alpha: number = 1,
): string {
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
}

// --- Mono-channel colormaps ---

function monoRed(t: number): [number, number, number] {
  const v = Math.round(clamp01(t) * 255);
  return [v, 0, 0];
}

function monoGreen(t: number): [number, number, number] {
  const v = Math.round(clamp01(t) * 255);
  return [0, v, 0];
}

function monoBlue(t: number): [number, number, number] {
  const v = Math.round(clamp01(t) * 255);
  return [0, 0, v];
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

// --- Viridis colormap ---
// 64-entry sampled LUT from matplotlib viridis (Stéfan van der Walt & Nathaniel Smith).
// Each entry is [R, G, B] in 0-255 range.
// Linearly interpolated between entries for smooth gradients.

// 64 entries sampled uniformly from matplotlib viridis.
// Generated via: cm.viridis(i/63) for i in 0..63, rounded to nearest int RGB 0-255.
const VIRIDIS_LUT: [number, number, number][] = [
  [68, 1, 84],
  [70, 7, 90],
  [71, 13, 96],
  [71, 19, 101],
  [72, 24, 106],
  [72, 29, 111],
  [72, 35, 116],
  [72, 40, 120],
  [71, 45, 123],
  [70, 50, 126],
  [69, 55, 129],
  [68, 59, 132],
  [66, 64, 134],
  [64, 69, 136],
  [62, 73, 137],
  [61, 78, 138],
  [58, 83, 139],
  [56, 88, 140],
  [54, 92, 141],
  [52, 96, 141],
  [50, 100, 142],
  [49, 104, 142],
  [47, 108, 142],
  [45, 112, 142],
  [44, 115, 142],
  [42, 119, 142],
  [41, 123, 142],
  [39, 127, 142],
  [38, 130, 142],
  [36, 134, 142],
  [35, 138, 141],
  [33, 142, 141],
  [32, 146, 140],
  [31, 150, 139],
  [31, 154, 138],
  [31, 158, 137],
  [31, 161, 135],
  [33, 165, 133],
  [35, 169, 131],
  [38, 173, 129],
  [42, 176, 127],
  [47, 180, 124],
  [53, 183, 121],
  [59, 187, 117],
  [66, 190, 113],
  [74, 193, 109],
  [82, 197, 105],
  [90, 200, 100],
  [101, 203, 94],
  [110, 206, 88],
  [119, 209, 83],
  [129, 211, 77],
  [139, 214, 70],
  [149, 216, 64],
  [160, 218, 57],
  [170, 220, 50],
  [181, 222, 43],
  [192, 223, 37],
  [202, 225, 31],
  [213, 226, 26],
  [223, 227, 24],
  [234, 229, 26],
  [244, 230, 30],
  [253, 231, 37],
];

/**
 * Viridis colormap: perceptually uniform, colorblind-safe.
 * Input t in [0, 1], returns [R, G, B] in 0-255 range.
 */
function viridis(t: number): [number, number, number] {
  const clamped = clamp01(t);
  const n = VIRIDIS_LUT.length - 1;
  const idx = clamped * n;
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, n);
  const frac = idx - lo;

  const a = VIRIDIS_LUT[lo];
  const b = VIRIDIS_LUT[hi];

  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  ];
}
