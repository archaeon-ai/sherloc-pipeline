// Regression coverage for issue #34 — Workbench Baseline checkbox and
// Background selector must follow applied spectrum state, not last user
// intent. These tests cover that behavior; a stale-async generation
// guard (exercised by the fourth test below) prevents an out-of-order
// async result from stomping applied state.
//
// Scope: integration over ProcessingChain + its BaselineStep and
// BackgroundStep children. The bug is parent/child reactive-flow, so a
// pure unit test on either child alone wouldn't catch the regression.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent } from '@testing-library/svelte';
import { tick } from 'svelte';
import ProcessingChain from './ProcessingChain.svelte';
import * as api from '../lib/api';
import type { BackgroundResponse, DespikeResponse } from '../lib/types';

const WAVENUMBER_A = [800, 900, 1000, 1100, 1200];
const INTENSITY_A = [10, 12, 15, 13, 11];
const WAVENUMBER_B = [810, 910, 1010, 1110, 1210];
const INTENSITY_B = [9, 11, 14, 12, 10];

// Find the BaselineStep header checkbox. As of issue #6 the DespikeStep no
// longer renders a `.step-toggle` checkbox (it is a None|ML|modz method
// selector built from radios), and BackgroundStep uses radios for bgType, so
// BaselineStep is the only step contributing a `.step-toggle` checkbox.
// Locate it by its step card to stay robust against future step changes.
function getBaselineCheckbox(container: HTMLElement): HTMLInputElement {
  const card = getStepCard(container, '3. Baseline');
  const cb = card.querySelector<HTMLInputElement>(
    '.step-toggle input[type="checkbox"]',
  );
  expect(cb, 'no baseline checkbox in the "3. Baseline" step card').not.toBeNull();
  return cb as HTMLInputElement;
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
    expect(postBaselineSpy.mock.calls[0][0].params?.wavenumber_range).toBeUndefined();
    expect(stateUpdate).toHaveBeenCalledTimes(1);
    expect(stateUpdate.mock.calls[0][0].stage).toBe('baseline_corrected');
  });

  it('fits the current zoom only when selected and resets to full when it is cleared', async () => {
    const zoomRange: [number, number] = [850, 1150];
    const postBaselineSpy = vi.spyOn(api, 'postBaseline').mockResolvedValue({
      schema_version: '1',
      raw: INTENSITY_A,
      wavenumber: WAVENUMBER_A,
      corrected: [10, 2, 3, 4, 11],
      baseline: [0, 10, 12, 9, 0],
      params_used: {
        method: 'aspls',
        lam: 1e6,
        max_iter: 10,
        wavenumber_range: zoomRange,
      },
    });
    const stateUpdate = vi.fn();
    const { container, component } = render(ProcessingChain, {
      props: {
        wavenumber: WAVENUMBER_A,
        intensity: INTENSITY_A,
        visibleRange: zoomRange,
      },
    });
    component.$on('stateUpdate', (e) => stateUpdate(e.detail));

    const card = getStepCard(container, '3. Baseline');
    await fireEvent.click(card.querySelector<HTMLButtonElement>('.step-header')!);
    await tick();
    const visibleRadio = card.querySelector<HTMLInputElement>(
      'input[name="baseline-range"][value="visible"]',
    );
    expect(visibleRadio).not.toBeNull();
    await fireEvent.change(visibleRadio!);
    await fireEvent.click(getBaselineCheckbox(container));
    await new Promise((r) => setTimeout(r, 0));
    await tick();

    expect(postBaselineSpy).toHaveBeenCalledTimes(1);
    expect(postBaselineSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        params: expect.objectContaining({ wavenumber_range: zoomRange }),
      }),
    );
    const snapshot = stateUpdate.mock.calls[0][0];
    expect(snapshot.artifacts.baselineRange).toEqual(zoomRange);
    expect(snapshot.params.wavenumber_range).toEqual(zoomRange);

    component.$set({ visibleRange: null });
    await tick();

    const fullRadio = card.querySelector<HTMLInputElement>(
      'input[name="baseline-range"][value="full"]',
    );
    expect(fullRadio?.checked).toBe(true);
    expect(visibleRadio?.checked).toBe(false);
    expect(visibleRadio?.disabled).toBe(true);

    postBaselineSpy.mockClear();
    const lamInput = card.querySelector<HTMLInputElement>('#bl-lam');
    await fireEvent.input(lamInput!, { target: { value: '6.1' } });
    await new Promise((r) => setTimeout(r, 350));
    await tick();

    expect(postBaselineSpy).toHaveBeenCalledTimes(1);
    expect(postBaselineSpy.mock.calls[0][0].params?.wavenumber_range).toBeUndefined();
    expect(card.querySelector('.step-error')).toBeNull();
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

describe('ProcessingChain — despike method selector kick', () => {
  // none→modz causes NO refetch (both methods fetch raw), so nothing upstream
  // replaces the chain's input props — the chain itself must kick the client
  // compute on that transition or the selector shows modz active while the
  // plot stays raw.
  it('selecting modz from none runs the client compute and emits a despiked state', async () => {
    const DESPIKED = [9, 11, 14, 12, 10];
    const postDespikeSpy = vi.spyOn(api, 'postDespike').mockResolvedValue({
      schema_version: '1',
      despiked: DESPIKED,
      spike_mask: [false, false, true, false, false],
      n_spikes: 1,
      params_used: { window_size: 7, zscore_threshold: 6, max_iterations: 1, sulfate_guard: true },
    });
    const stateUpdate = vi.fn();
    const { container, component } = render(ProcessingChain, {
      props: {
        wavenumber: WAVENUMBER_A,
        intensity: INTENSITY_A,
        despikeMethod: 'none',
        region: 'R1',
      },
    });
    component.$on('stateUpdate', (e) => stateUpdate(e.detail));
    // Simulate the workbench round-trip: it owns despikeMethod and updates it
    // synchronously in its despikeMethodChange handler (no refetch for
    // none→modz, so no new wavenumber/intensity props arrive).
    component.$on('despikeMethodChange', (e) =>
      component.$set({ despikeMethod: e.detail.method }),
    );

    // Expand the despike panel and select modz.
    const card = getStepCard(container, '1. Despike');
    await fireEvent.click(card.querySelector<HTMLButtonElement>('.step-header')!);
    await tick();
    const modzRadio = card.querySelector<HTMLInputElement>(
      'input[name="despike-method"][value="modz"]',
    );
    expect(modzRadio, 'no modz radio in the despike step').not.toBeNull();
    await fireEvent.change(modzRadio!);
    await new Promise((r) => setTimeout(r, 0));
    await tick();

    // The compute ran without any prop change from above...
    expect(postDespikeSpy).toHaveBeenCalledTimes(1);
    // ...and the chain emitted the despiked stage carrying the modz output
    // (the methodChange first emits a 'raw' reset, then the apply lands).
    const lastEmit = stateUpdate.mock.calls[stateUpdate.mock.calls.length - 1][0];
    expect(lastEmit.stage).toBe('despiked');
    expect(lastEmit.raman.intensity).toEqual(DESPIKED);
  });

  it('selecting ml does NOT kick the client compute (refetch path owns ML)', async () => {
    const postDespikeSpy = vi.spyOn(api, 'postDespike');
    const { container, component } = render(ProcessingChain, {
      props: {
        wavenumber: WAVENUMBER_A,
        intensity: INTENSITY_A,
        despikeMethod: 'none',
        region: 'R1',
        mlMaskCount: 5,
      },
    });
    component.$on('despikeMethodChange', (e) =>
      component.$set({ despikeMethod: e.detail.method }),
    );
    const card = getStepCard(container, '1. Despike');
    await fireEvent.click(card.querySelector<HTMLButtonElement>('.step-header')!);
    await tick();
    const mlRadio = card.querySelector<HTMLInputElement>(
      'input[name="despike-method"][value="ml"]',
    );
    await fireEvent.change(mlRadio!);
    await new Promise((r) => setTimeout(r, 0));
    await tick();
    expect(postDespikeSpy).not.toHaveBeenCalled();
  });

  // A modz POST in flight across an input switch must be dropped.
  // Without the generation guard, the older response (computed from
  // the PREVIOUS point's array) resolving last would be accepted — method is
  // still 'modz' — and render the previous spectrum's despiked intensity for
  // the current selection.
  it('drops a stale modz response that resolves after an input switch (generation guard)', async () => {
    const DESPIKED_STALE = [1, 2, 3, 4, 5];
    const DESPIKED_NEW = [6, 7, 8, 9, 10];
    let resolveStale: (v: DespikeResponse) => void = () => {};
    const pendingStale = new Promise<DespikeResponse>((r) => {
      resolveStale = r;
    });
    let resolveNew: (v: DespikeResponse) => void = () => {};
    const pendingNew = new Promise<DespikeResponse>((r) => {
      resolveNew = r;
    });
    const postDespikeSpy = vi
      .spyOn(api, 'postDespike')
      .mockReturnValueOnce(pendingStale)
      .mockReturnValueOnce(pendingNew);

    const stateUpdate = vi.fn();
    const { container, component } = render(ProcessingChain, {
      props: {
        wavenumber: WAVENUMBER_A,
        intensity: INTENSITY_A,
        despikeMethod: 'none',
        region: 'R1',
      },
    });
    component.$on('stateUpdate', (e) => stateUpdate(e.detail));
    component.$on('despikeMethodChange', (e) =>
      component.$set({ despikeMethod: e.detail.method }),
    );

    // Activate modz → compute 1 starts against input A (suspends on
    // pendingStale).
    const card = getStepCard(container, '1. Despike');
    await fireEvent.click(card.querySelector<HTMLButtonElement>('.step-header')!);
    await tick();
    await fireEvent.change(
      card.querySelector<HTMLInputElement>('input[name="despike-method"][value="modz"]')!,
    );
    await new Promise((r) => setTimeout(r, 0));
    await tick();
    expect(postDespikeSpy).toHaveBeenCalledTimes(1);

    // Input switch while compute 1 is in flight (point switch refetch):
    // resets the chain, bumps the generation, kicks compute 2 on input B.
    component.$set({ wavenumber: WAVENUMBER_B, intensity: INTENSITY_B });
    await new Promise((r) => setTimeout(r, 0));
    await tick();
    expect(postDespikeSpy).toHaveBeenCalledTimes(2);
    // Compute 2 ran against the NEW input array.
    expect(postDespikeSpy.mock.calls[1][0].intensity).toEqual(INTENSITY_B);

    // Newer compute resolves first → accepted.
    resolveNew({
      schema_version: '1',
      despiked: DESPIKED_NEW,
      spike_mask: [false, true, false, false, false],
      n_spikes: 1,
      params_used: { window_size: 7, zscore_threshold: 6, max_iterations: 1, sulfate_guard: true },
    });
    await new Promise((r) => setTimeout(r, 0));
    await tick();
    let despikedEmits = stateUpdate.mock.calls.filter((c) => c[0].stage === 'despiked');
    expect(despikedEmits.length).toBe(1);
    expect(despikedEmits[0][0].raman.intensity).toEqual(DESPIKED_NEW);

    // Stale compute (input A's array) resolves LAST — must be dropped: no new
    // emission, and the chain state still reflects the new input's compute.
    resolveStale({
      schema_version: '1',
      despiked: DESPIKED_STALE,
      spike_mask: [true, false, false, false, false],
      n_spikes: 1,
      params_used: { window_size: 7, zscore_threshold: 6, max_iterations: 1, sulfate_guard: true },
    });
    await new Promise((r) => setTimeout(r, 0));
    await tick();
    despikedEmits = stateUpdate.mock.calls.filter((c) => c[0].stage === 'despiked');
    expect(despikedEmits.length).toBe(1);
    const lastEmit = stateUpdate.mock.calls[stateUpdate.mock.calls.length - 1][0];
    expect(lastEmit.raman.intensity).toEqual(DESPIKED_NEW);
  });
});
