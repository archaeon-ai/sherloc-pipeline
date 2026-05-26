// Regression coverage for issue #34 — Workbench Baseline checkbox and
// Background selector must follow applied spectrum state, not last user
// intent. Codex /code-review PR #35 R1 F1 (Major / Tests) required
// these tests; PR #35 R1 F2 (Major / Correctness) added the stale-async
// generation guard that the fourth test below exercises.
//
// Scope: integration over ProcessingChain + its BaselineStep and
// BackgroundStep children. The bug is parent/child reactive-flow, so a
// pure unit test on either child alone wouldn't catch the regression.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent } from '@testing-library/svelte';
import { tick } from 'svelte';
import ProcessingChain from './ProcessingChain.svelte';
import * as api from '../lib/api';
import type { BackgroundResponse } from '../lib/types';

const WAVENUMBER_A = [800, 900, 1000, 1100, 1200];
const INTENSITY_A = [10, 12, 15, 13, 11];
const WAVENUMBER_B = [810, 910, 1010, 1110, 1210];
const INTENSITY_B = [9, 11, 14, 12, 10];

// Find the BaselineStep header checkbox. DespikeStep and BaselineStep
// both render a `.step-toggle input[type=checkbox]` in their header;
// they appear in DOM order [Despike(0), Baseline(1)]. BackgroundStep
// uses radios so it does not contribute a `.step-toggle` checkbox.
function getBaselineCheckbox(container: HTMLElement): HTMLInputElement {
  const toggles = container.querySelectorAll<HTMLInputElement>(
    '.step-toggle input[type="checkbox"]',
  );
  expect(toggles.length).toBe(2);
  return toggles[1];
}

// Locate a step card by its title text ("1. Despike", "2. Background", etc.).
function getStepCard(container: HTMLElement, title: string): HTMLElement {
  const titles = Array.from(container.querySelectorAll<HTMLElement>('.step-title'));
  const match = titles.find((t) => t.textContent?.trim() === title);
  expect(match, `no step card titled "${title}"`).toBeDefined();
  return match!.closest('.step-card') as HTMLElement;
}

// BackgroundStep's body (including the radios) only renders when
// `!collapsed`. Click the header to expand before reaching for radios.
async function expandBackgroundPanel(container: HTMLElement): Promise<void> {
  const card = getStepCard(container, '2. Background');
  const header = card.querySelector<HTMLButtonElement>('.step-header')!;
  await fireEvent.click(header);
  await tick();
}

function getBgRadio(container: HTMLElement, value: 'none' | 'as' | 'fs'): HTMLInputElement {
  const card = getStepCard(container, '2. Background');
  const radio = card.querySelector<HTMLInputElement>(
    `input[type="radio"][value="${value}"]`,
  );
  expect(radio, `no bg radio for value=${value}`).not.toBeNull();
  return radio as HTMLInputElement;
}

// The header's blue indicator dot is visible regardless of collapsed
// state; useful for asserting "bg is armed / disarmed" without expanding.
function bgIndicatorActive(container: HTMLElement): boolean {
  const card = getStepCard(container, '2. Background');
  const dot = card.querySelector('.step-indicator');
  return dot?.classList.contains('active') ?? false;
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe('ProcessingChain — UI state tracks applied spectrum state (issue #34)', () => {
  it('renders with Baseline unchecked and Background indicator inactive on initial mount', async () => {
    const { container } = render(ProcessingChain, {
      props: { wavenumber: WAVENUMBER_A, intensity: INTENSITY_A },
    });
    expect(getBaselineCheckbox(container).checked).toBe(false);
    expect(bgIndicatorActive(container)).toBe(false);
    // Expand the panel to confirm the "None" radio is the selected option.
    await expandBackgroundPanel(container);
    expect(getBgRadio(container, 'none').checked).toBe(true);
    expect(getBgRadio(container, 'as').checked).toBe(false);
  });

  it('emits stateUpdate(baseline_corrected) after the Baseline checkbox is checked', async () => {
    const postBaselineSpy = vi
      .spyOn(api, 'postBaseline')
      .mockResolvedValue({
        schema_version: '1',
        raw: INTENSITY_A,
        wavenumber: WAVENUMBER_A,
        corrected: [1, 2, 3, 4, 5],
        baseline: [9, 10, 12, 9, 6],
        params_used: { method: 'aspls', lam: 1e6, max_iter: 10 },
      });
    const stateUpdate = vi.fn();
    const { container, component } = render(ProcessingChain, {
      props: { wavenumber: WAVENUMBER_A, intensity: INTENSITY_A },
    });
    component.$on('stateUpdate', (e) => stateUpdate(e.detail));

    await fireEvent.click(getBaselineCheckbox(container));
    // Let postBaseline resolve + the apply handler run.
    await new Promise((r) => setTimeout(r, 0));
    await tick();

    expect(postBaselineSpy).toHaveBeenCalledTimes(1);
    expect(stateUpdate).toHaveBeenCalledTimes(1);
    expect(stateUpdate.mock.calls[0][0].stage).toBe('baseline_corrected');
  });

  it('resets Baseline checkbox and Background radio when raw input props change', async () => {
    vi.spyOn(api, 'postBaseline').mockResolvedValue({
      schema_version: '1',
      raw: INTENSITY_A,
      wavenumber: WAVENUMBER_A,
      corrected: [1, 2, 3, 4, 5],
      baseline: [9, 10, 12, 9, 6],
      params_used: { method: 'aspls', lam: 1e6, max_iter: 10 },
    });
    vi.spyOn(api, 'postBackground').mockResolvedValue({
      schema_version: '1',
      subtracted: [4, 5, 6, 5, 4],
      background_scaled: [6, 7, 9, 8, 7],
      scale_used: 1.0,
      bg_type: 'as',
    });

    const { container, component } = render(ProcessingChain, {
      props: { wavenumber: WAVENUMBER_A, intensity: INTENSITY_A },
    });

    // Arm both controls (simulates user checking baseline + selecting Arm-Stowed).
    await fireEvent.click(getBaselineCheckbox(container));
    await expandBackgroundPanel(container);
    await fireEvent.click(getBgRadio(container, 'as'));
    await new Promise((r) => setTimeout(r, 0));
    await tick();
    expect(getBaselineCheckbox(container).checked).toBe(true);
    expect(bgIndicatorActive(container)).toBe(true);
    expect(getBgRadio(container, 'as').checked).toBe(true);
    expect(getBgRadio(container, 'none').checked).toBe(false);

    // Raw input changes (point switch / modality-triggered region reload).
    component.$set({ wavenumber: WAVENUMBER_B, intensity: INTENSITY_B });
    await tick();

    expect(getBaselineCheckbox(container).checked).toBe(false);
    expect(bgIndicatorActive(container)).toBe(false);
    expect(getBgRadio(container, 'none').checked).toBe(true);
    expect(getBgRadio(container, 'as').checked).toBe(false);
  });

  it('drops a stale Background apply response that resolves after a raw-input change (F2 generation guard)', async () => {
    // Hand-controlled promise so we can land the response AFTER $set.
    let resolveBackground: (v: BackgroundResponse) => void = () => {};
    const pending = new Promise<BackgroundResponse>((resolve) => {
      resolveBackground = resolve;
    });
    vi.spyOn(api, 'postBackground').mockReturnValue(pending);

    const stateUpdate = vi.fn();
    const { container, component } = render(ProcessingChain, {
      props: { wavenumber: WAVENUMBER_A, intensity: INTENSITY_A },
    });
    component.$on('stateUpdate', (e) => stateUpdate(e.detail));

    // Start an in-flight background request.
    await expandBackgroundPanel(container);
    await fireEvent.click(getBgRadio(container, 'as'));
    await tick();

    // Raw input changes before the response lands — bumps the input
    // generation counter, so the in-flight response should be dropped.
    component.$set({ wavenumber: WAVENUMBER_B, intensity: INTENSITY_B });
    await tick();

    // Land the stale response.
    resolveBackground({
      schema_version: '1',
      subtracted: [4, 5, 6, 5, 4],
      background_scaled: [6, 7, 9, 8, 7],
      scale_used: 1.0,
      bg_type: 'as',
    });
    await new Promise((r) => setTimeout(r, 0));
    await tick();

    // Parent should have received only the toggle-driven raw emission
    // from the bg radio click (or none at all) — and crucially no
    // `bg_subtracted` stage from the stale response.
    const subtractedEmits = stateUpdate.mock.calls.filter(
      (call) => call[0].stage === 'bg_subtracted',
    );
    expect(subtractedEmits.length).toBe(0);
  });

  it('cleanly re-applies Baseline after a reset (no leftover-state regression)', async () => {
    const postBaselineSpy = vi
      .spyOn(api, 'postBaseline')
      .mockResolvedValue({
        schema_version: '1',
        raw: INTENSITY_A,
        wavenumber: WAVENUMBER_A,
        corrected: [1, 2, 3, 4, 5],
        baseline: [9, 10, 12, 9, 6],
        params_used: { method: 'aspls', lam: 1e6, max_iter: 10 },
      });
    const { container, component } = render(ProcessingChain, {
      props: { wavenumber: WAVENUMBER_A, intensity: INTENSITY_A },
    });

    // First apply.
    await fireEvent.click(getBaselineCheckbox(container));
    await new Promise((r) => setTimeout(r, 0));
    await tick();
    expect(postBaselineSpy).toHaveBeenCalledTimes(1);

    // Reset via input change.
    component.$set({ wavenumber: WAVENUMBER_B, intensity: INTENSITY_B });
    await tick();
    expect(getBaselineCheckbox(container).checked).toBe(false);

    // Re-arm: second apply should fire cleanly.
    await fireEvent.click(getBaselineCheckbox(container));
    await new Promise((r) => setTimeout(r, 0));
    await tick();
    expect(postBaselineSpy).toHaveBeenCalledTimes(2);
    expect(getBaselineCheckbox(container).checked).toBe(true);
  });
});
