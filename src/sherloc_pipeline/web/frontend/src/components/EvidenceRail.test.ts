import { beforeEach, expect, it } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import { clearEvidence } from '../lib/evidenceStore';
import type { IdentificationEvidence } from '../lib/identification';
import EvidenceRail from './EvidenceRail.svelte';

beforeEach(() => { clearEvidence(); });

function result(): IdentificationEvidence {
  return {
    result_digest: 'a'.repeat(64), source_digest: 'b'.repeat(64), status: 'candidate', reasons: [],
    method: { release_id: 'c'.repeat(64), support_min: .8, status: 'provisional' },
    reference_release_id: 'd'.repeat(64), reference_subset_id: 'e'.repeat(64),
    ranking: [
      { template_id: '1'.repeat(64), species_name: 'Syntheticite', support: .91,
        trace_revision_ids: [], citations: [] },
      { template_id: '2'.repeat(64), species_name: 'Alternatite', support: .85,
        trace_revision_ids: [], citations: [] },
    ],
  };
}

it.each([false, true])('labels the top mixture retrieval according to retention in the fit (%s)', retained => {
  const evidence = result();
  evidence.method.support_min = .6;
  evidence.ranking.forEach(candidate => { candidate.support = null; candidate.retrieval_score = .95; });
  const fitted = evidence.ranking[retained ? 0 : 1];
  evidence.mixture = {
    support: .9, diagnostics: { converged: true, finite_solution: true },
    components: [{ template_id: fitted.template_id, species_name: fitted.species_name, coefficient: .8, weight: 1 }],
  };
  render(EvidenceRail, { result: evidence, contextKey: 'synthetic-context' });
  expect(screen.getByText(new RegExp(`^${retained ? 'Leading candidate' : 'Top retrieval \\(pruned by fit\\)'}: Syntheticite`))).toBeTruthy();
  if (!retained) expect(screen.queryByText(/^Leading candidate:/)).toBeNull();
  expect(screen.getByText(/^Alternative: Alternatite/)).toBeTruthy();
});

it('preserves the Pearson leading-candidate wording and support score', () => {
  render(EvidenceRail, { result: result(), contextKey: 'synthetic-context' });
  expect(screen.getByText(/^Leading candidate: Syntheticite · support 0.910$/)).toBeTruthy();
  expect(screen.queryByText(/pruned by fit/)).toBeNull();
});
