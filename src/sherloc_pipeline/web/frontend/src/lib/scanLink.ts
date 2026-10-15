import type { ScanPoint } from './types';

export type LinkSelection =
  | { mode: 'average'; indices: number[] }
  | { mode: 'point' | 'subset'; indices: number[] };

function nonnegativeInteger(value: string | undefined, label: string): number {
  if (value === undefined || !/^\d+$/.test(value) || !Number.isSafeInteger(Number(value))) {
    throw new Error(`${label} must be a nonnegative whole number.`);
  }
  return Number(value);
}

export function parseLinkSelection(params: Record<string, string>): LinkSelection {
  if (params.point !== undefined && params.points !== undefined) {
    throw new Error('Use either point or points in the link, not both.');
  }
  if (params.point !== undefined) {
    return { mode: 'point', indices: [nonnegativeInteger(params.point, 'Point index')] };
  }
  if (params.points !== undefined) {
    const indices = params.points.split(',').map((value) => nonnegativeInteger(value, 'Point index'));
    return { mode: 'subset', indices: [...new Set(indices)] };
  }
  return { mode: 'average', indices: [] };
}

export function parseScanLink(params: Record<string, string>) {
  if (!params.scan_name?.trim()) throw new Error('The link must include a scan_name.');
  return {
    scanName: params.scan_name,
    sol: nonnegativeInteger(params.sol, 'Sol'),
    selection: parseLinkSelection(params),
  };
}

export function validateLinkSelection(
  selection: LinkSelection,
  points: Pick<ScanPoint, 'point_index'>[],
): void {
  const available = new Set(points.map((point) => point.point_index));
  const missing = selection.indices.filter((index) => !available.has(index));
  if (missing.length) {
    throw new Error(`Point index out of range for this scan: ${missing.join(', ')}. Choose an available point from the scan browser.`);
  }
}

export function workbenchLink(scanId: string, selection: LinkSelection): string {
  const query = selection.mode === 'average'
    ? ''
    : `?${selection.mode === 'point' ? 'point' : 'points'}=${selection.indices.join(',')}`;
  return `#/scan/${encodeURIComponent(scanId)}/workbench${query}`;
}
