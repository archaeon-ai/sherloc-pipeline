import { describe, expect, it } from 'vitest';
import { get } from 'svelte/store';
import { currentHash, parsedRoute } from './stores';
import { parseLinkSelection, parseScanLink, validateLinkSelection, workbenchLink } from './scanLink';

describe('scan deep links', () => {
  it('recognizes the resolver route and decodes the scan name once', () => {
    currentHash.set('#/open?scan_name=Testrock%20%26%20%2B%25&sol=42424&point=0');
    const route = get(parsedRoute);
    expect(route.page).toBe('open');
    expect(parseScanLink(route.queryParams!)).toEqual({
      scanName: 'Testrock & +%', sol: 42424, selection: { mode: 'point', indices: [0] },
    });
  });

  it('keeps the existing Workbench route and reads a subset', () => {
    currentHash.set('#/scan/test-id/workbench?points=0,2,2');
    const route = get(parsedRoute);
    expect(route.page).toBe('workbench');
    expect(route.scanId).toBe('test-id');
    expect(parseLinkSelection(route.queryParams!)).toEqual({ mode: 'subset', indices: [0, 2] });
  });

  it.each(['', '-1', '1.5', '2tail', 'NaN', 'Infinity', '1e2', ' 2', '9007199254740992'])(
    'rejects invalid sol and point values: %s', (value) => {
      expect(() => parseScanLink({ scan_name: 'Testrock', sol: value })).toThrow('Sol');
      expect(() => parseLinkSelection({ point: value })).toThrow('Point index');
      expect(() => parseLinkSelection({ points: `0,${value}` })).toThrow('Point index');
    },
  );

  it('requires a name and sol and rejects competing selections', () => {
    expect(() => parseScanLink({ sol: '42424' })).toThrow('scan_name');
    expect(() => parseScanLink({ scan_name: '  ', sol: '42424' })).toThrow('scan_name');
    expect(() => parseScanLink({ scan_name: 'Testrock' })).toThrow('Sol');
    expect(() => parseLinkSelection({ point: '0', points: '0,2' })).toThrow('not both');
  });

  it('validates against actual indices, including sparse scans', () => {
    const points = [{ point_index: 0 }, { point_index: 7 }];
    expect(() => validateLinkSelection(parseLinkSelection({ points: '7,0' }), points)).not.toThrow();
    expect(() => validateLinkSelection(parseLinkSelection({ points: '0,1,8' }), points))
      .toThrow('Point index out of range for this scan: 1, 8');
    expect(() => validateLinkSelection(parseLinkSelection({ point: '0' }), []))
      .toThrow('out of range');
  });

  it.each([
    [{}, '#/scan/test-id/workbench'],
    [{ point: '0' }, '#/scan/test-id/workbench?point=0'],
    [{ points: '0,2,2' }, '#/scan/test-id/workbench?points=0,2'],
    [{ points: '2' }, '#/scan/test-id/workbench?points=2'],
  ])('builds the Workbench link for %j', (params, expected) => {
    expect(workbenchLink('test-id', parseLinkSelection(params))).toBe(expected);
  });
});
