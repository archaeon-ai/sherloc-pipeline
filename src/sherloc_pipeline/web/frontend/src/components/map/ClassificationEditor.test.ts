// The Classification Rules editor keeps its edits in page state only (the
// /api/user profile routes are retired), so it says so next to Apply/Reset.

import { describe, it, expect, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';
import classWindows from '../../lib/classWindows.json';
import ClassificationEditor from './ClassificationEditor.svelte';

describe('ClassificationEditor', () => {
  it('applies the default windows as center and half-width', async () => {
    const { component } = render(ClassificationEditor, { props: { hasFittedData: true } });
    const apply = vi.fn();
    component.$on('profileApply', (event) => apply(event.detail.profile));
    await fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(apply).toHaveBeenCalledWith({
      name: 'custom',
      rules: classWindows.map(({ lo, hi, ...rule }) => ({
        ...rule, center: (lo + hi) / 2, range: (hi - lo) / 2, disabled: false,
      })),
    });
  });

  it('says edits apply to this page only', () => {
    render(ClassificationEditor, { props: { hasFittedData: true } });
    const note = screen.getByText('Edits apply to this page only; they are lost on reload.');
    expect(note).toBeVisible();
    expect(screen.getByRole('button', { name: 'Apply' })).toBeInTheDocument();
  });
});
