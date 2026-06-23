// CSV spike-mask export across despike methods (issue #8).
//
// Pre-#8 the CSV `spike_mask` column was emitted ONLY at the modz `despiked`
// stage (from artifacts). In ML mode the fetched array is already cleaned and
// the stage is `raw`, so the column never appeared. Issue #8 routes the column
// off an explicit `spikeMask`/`spikeMethod` prop (the Workbench's effective
// mask) so it exports in ML mode too, with a `# spike_mask_method:` provenance
// note recording which despike produced it.
//
// The CSV is generated into a Blob and downloaded; we capture the Blob's text
// by stubbing the Blob constructor and the URL/anchor download plumbing.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent } from '@testing-library/svelte';
import { tick } from 'svelte';
import ExportMenu from './ExportMenu.svelte';

let capturedCsv = '';

beforeEach(() => {
  capturedCsv = '';
  // Capture the CSV text the component hands to `new Blob([...])`.
  vi.stubGlobal(
    'Blob',
    class {
      constructor(parts: string[]) {
        capturedCsv = parts.join('');
      }
    },
  );
  // Stub the download plumbing so jsdom doesn't choke on navigation.
  vi.stubGlobal('URL', {
    createObjectURL: vi.fn(() => 'blob:mock'),
    revokeObjectURL: vi.fn(),
  });
  // The component triggers `a.click()` on a blob href; jsdom logs a noisy
  // "Not implemented: navigation" for that. No-op the anchor click — we assert
  // on the captured Blob text, not the download itself.
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
});

const wavenumber = [800, 900, 1000];
const intensity = [5, 6, 7];

async function exportCsv(props: Record<string, unknown>): Promise<string> {
  const { getByText } = render(ExportMenu, {
    props: { scanId: 's1', scanName: 'detail_1', target: 'Amherst', solNumber: 921, wavenumber, intensity, ...props },
  });
  await tick();
  await fireEvent.click(getByText(/Export/i));
  await tick();
  await fireEvent.click(getByText(/CSV at current stage/i));
  await tick();
  return capturedCsv;
}

describe('ExportMenu CSV spike-mask column (issue #8)', () => {
  it('emits spike_mask + ML provenance note in ML mode at the raw stage', async () => {
    const csv = await exportCsv({
      stage: 'raw',
      spikeMask: [true, false, true],
      spikeMethod: 'ml',
    });
    expect(csv).toContain('# spike_mask_method: ml');
    const header = csv.split('\n').find((l) => l.startsWith('wavenumber'));
    expect(header).toContain('spike_mask');
    // Data rows: mask 1/0/1 in the spike_mask column.
    const rows = csv.split('\n').filter((l) => /^\d/.test(l));
    expect(rows[0].split(',')).toEqual(['800', '5', '1']);
    expect(rows[1].split(',')).toEqual(['900', '6', '0']);
    expect(rows[2].split(',')).toEqual(['1000', '7', '1']);
  });

  it('emits spike_mask + modz note from the explicit props at the despiked stage', async () => {
    const csv = await exportCsv({
      stage: 'despiked',
      spikeMask: [false, true, false],
      spikeMethod: 'modz',
    });
    expect(csv).toContain('# spike_mask_method: modz');
    expect(csv.split('\n').find((l) => l.startsWith('wavenumber'))).toContain('spike_mask');
    const rows = csv.split('\n').filter((l) => /^\d/.test(l));
    expect(rows.map((r) => r.split(',')[2])).toEqual(['0', '1', '0']);
  });

  it('ignores a stale despiked-stage artifact when the props are null (PR #10 F1)', async () => {
    // An undo-restored 'despiked' snapshot still carries the modz spikeMask
    // artifact even when the active method is 'none' — the Workbench then
    // passes null props, and the column/header must NOT be inferred from
    // stage/artifacts.
    const csv = await exportCsv({
      stage: 'despiked',
      spikeMask: null,
      spikeMethod: null,
      artifacts: { spikeMask: [false, true, false] },
    });
    expect(csv).not.toContain('spike_mask');
    expect(csv).not.toContain('spike_mask_method');
  });

  it('omits the column when the mask is present but the method is null (strict pairing)', async () => {
    const csv = await exportCsv({
      stage: 'raw',
      spikeMask: [true, false, true],
      spikeMethod: null,
    });
    expect(csv).not.toContain('spike_mask');
  });

  it('omits the column when no mask is present', async () => {
    const csv = await exportCsv({ stage: 'raw' });
    expect(csv).not.toContain('spike_mask');
    expect(csv).not.toContain('spike_mask_method');
  });

  it('omits the column in ML mode when the mask is null (no positions)', async () => {
    const csv = await exportCsv({ stage: 'raw', spikeMask: null, spikeMethod: 'ml' });
    expect(csv).not.toContain('spike_mask');
  });
});

describe('ExportMenu CSV badpix column (issue #9)', () => {
  // wavenumber/intensity from the outer scope are [800,900,1000]/[5,6,7].
  it('emits badpix column + source header when the badpix prop is set', async () => {
    const csv = await exportCsv({
      stage: 'raw',
      badpix: [
        { position: 0, tier: 1, source: 'g5_eps' },
        { position: 2, tier: 2, source: 'jb25' },
      ],
    });
    // Provenance header records the annotation source set (sorted, de-duped).
    expect(csv).toContain('# badpix_source: g5_eps,jb25');
    const header = csv.split('\n').find((l) => l.startsWith('wavenumber'));
    expect(header).toContain('badpix');
    // 0/1 per served row: positions 0 and 2 flagged, position 1 clear.
    const rows = csv.split('\n').filter((l) => /^\d/.test(l));
    expect(rows.map((r) => r.split(',').pop())).toEqual(['1', '0', '1']);
  });

  it('badpix column is distinct from spike_mask (both can be present)', async () => {
    const csv = await exportCsv({
      stage: 'raw',
      spikeMask: [true, false, false],
      spikeMethod: 'ml',
      badpix: [{ position: 2, tier: 1, source: 'both' }],
    });
    const header = csv.split('\n').find((l) => l.startsWith('wavenumber'))!.split(',');
    expect(header).toContain('spike_mask');
    expect(header).toContain('badpix');
    // spike_mask flags row 0; badpix flags row 2 — independent columns.
    const rows = csv.split('\n').filter((l) => /^\d/.test(l)).map((r) => r.split(','));
    const sIdx = header.indexOf('spike_mask');
    const bIdx = header.indexOf('badpix');
    expect(rows.map((r) => r[sIdx])).toEqual(['1', '0', '0']);
    expect(rows.map((r) => r[bIdx])).toEqual(['0', '0', '1']);
  });

  it('omits the column + header when the badpix prop is null (toggle OFF gate)', async () => {
    // The Workbench passes null when the overlay toggle is OFF.
    const csv = await exportCsv({ stage: 'raw', badpix: null });
    expect(csv).not.toContain('badpix_source');
    const header = csv.split('\n').find((l) => l.startsWith('wavenumber'));
    expect(header).not.toContain('badpix');
  });

  it('omits the column when the badpix list is empty', async () => {
    const csv = await exportCsv({ stage: 'raw', badpix: [] });
    expect(csv).not.toContain('badpix_source');
    expect(csv.split('\n').find((l) => l.startsWith('wavenumber'))).not.toContain('badpix');
  });
});
