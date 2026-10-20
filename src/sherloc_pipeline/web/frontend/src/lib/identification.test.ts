import { expect, it } from 'vitest';
import { identificationPlot } from './identification';
import type { IdentificationEvidence } from './identification';

function synthetic(): IdentificationEvidence {
  return { result_digest: 'a'.repeat(64), source_digest: 'b'.repeat(64), status: 'candidate', reasons: [],
    method: { method_id: 'raman-pearson', release_id: 'c'.repeat(64), support_min: .8, status: 'provisional' },
    reference_release_id: 'd'.repeat(64), reference_subset_id: 'e'.repeat(64), ranking: [],
    evidence: { grid: [810, 815, 820], mask: [true, false, true], prepared_trace: [.2, .3, .4],
      inspection: { mask: [true, false, true], fit: [.1, .2, .3], residual: [.1, .1, .1] } } };
}

it('keeps Pearson inspection separate from mixture fit', () => {
  const result = synthetic();
  const plot = identificationPlot(result)!;
  expect(plot.label).toBe('Pearson inspection fit');
  expect(plot.traces[2].name).toBe('Pearson inspection fit residual');
  expect(plot.traces[0].y).toEqual([.2, null, .4]);
  result.method.method_id = 'raman-mixture';
  expect(identificationPlot(result)).toBeNull();
});

it('declines absent or mismatched fit arrays', () => {
  expect(identificationPlot(null)).toBeNull();
  const result = synthetic();
  result.evidence!.inspection!.residual = [1.];
  expect(identificationPlot(result)).toBeNull();
});
