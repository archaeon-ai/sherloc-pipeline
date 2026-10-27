// The Classification Rules editor keeps its edits in page state only (the
// /api/user profile routes are retired), so it says so next to Apply/Reset.

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import ClassificationEditor from './ClassificationEditor.svelte';

describe('ClassificationEditor', () => {
  it('says edits apply to this page only', () => {
    render(ClassificationEditor, { props: { hasFittedData: true } });
    const note = screen.getByText('Edits apply to this page only; they are lost on reload.');
    expect(note).toBeVisible();
    expect(screen.getByRole('button', { name: 'Apply' })).toBeInTheDocument();
  });
});
