// Despike method-selector coverage for issue #6 — the Workbench chain's
// "1. Despike" step is a three-way method selector (None | ML | modz)
// replacing the legacy on/off checkbox. Locks:
//   - the three radio options render;
//   - ML is gated on mlMaskCount (disabled + hint when 0);
//   - modz is gated on region (disabled + "R1 only" hint when not R1);
//   - selecting a method dispatches `methodChange` (a named handler, never a
//     reactive write — FRONTEND_HAZARDS H1);
//   - the modz tunable params panel shows ONLY for modz; the ML provenance
//     line shows ONLY for ml;
//   - the no-double-despike invariant's client half: selecting ML never runs
//     the client-side modz compute (postDespike is not called).
//
// The step body only renders when expanded; tests click the header first.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent } from '@testing-library/svelte';
import { tick } from 'svelte';
import * as api from '../lib/api';
import DespikeStep from './DespikeStep.svelte';

const WAVENUMBER = [800, 900, 1000, 1100, 1200];
const INTENSITY = [10, 12, 15, 13, 11];

function expandBody(container: HTMLElement): Promise<void> {
  const header = container.querySelector<HTMLButtonElement>('.step-header')!;
  return fireEvent.click(header).then(() => tick());
}

function getRadio(container: HTMLElement, value: 'none' | 'ml' | 'modz'): HTMLInputElement {
  const r = container.querySelector<HTMLInputElement>(
    `input[type="radio"][value="${value}"]`,
  );
  expect(r, `no despike-method radio for value=${value}`).not.toBeNull();
  return r as HTMLInputElement;
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe('DespikeStep method selector (issue #6)', () => {
  it('renders three method options: None, ML, modz', async () => {
    const { container } = render(DespikeStep, {
      props: { wavenumber: WAVENUMBER, intensity: INTENSITY, method: 'none', mlMaskCount: 5, region: 'R1' },
    });
    await expandBody(container);
    expect(getRadio(container, 'none')).toBeTruthy();
    expect(getRadio(container, 'ml')).toBeTruthy();
    expect(getRadio(container, 'modz')).toBeTruthy();
  });

  it('disables ML and shows a hint when no masks are stored (mlMaskCount=0)', async () => {
    const { container } = render(DespikeStep, {
      props: { wavenumber: WAVENUMBER, intensity: INTENSITY, method: 'none', mlMaskCount: 0, region: 'R1' },
    });
    await expandBody(container);
    expect(getRadio(container, 'ml').disabled).toBe(true);
    expect(container.textContent).toMatch(/No ML masks stored for this scan/i);
  });

  it('enables ML when masks are stored (mlMaskCount>0)', async () => {
    const { container } = render(DespikeStep, {
      props: { wavenumber: WAVENUMBER, intensity: INTENSITY, method: 'none', mlMaskCount: 12, region: 'R1' },
    });
    await expandBody(container);
    expect(getRadio(container, 'ml').disabled).toBe(false);
    expect(container.textContent).not.toMatch(/No ML masks stored/i);
  });

  it('disables modz and shows "R1 only" when region is not R1', async () => {
    const { container } = render(DespikeStep, {
      props: { wavenumber: WAVENUMBER, intensity: INTENSITY, method: 'none', mlMaskCount: 5, region: 'R2' },
    });
    await expandBody(container);
    expect(getRadio(container, 'modz').disabled).toBe(true);
    expect(container.textContent).toMatch(/R1 only/i);
  });

  it('keeps modz enabled on R1', async () => {
    const { container } = render(DespikeStep, {
      props: { wavenumber: WAVENUMBER, intensity: INTENSITY, method: 'none', mlMaskCount: 0, region: 'R1' },
    });
    await expandBody(container);
    expect(getRadio(container, 'modz').disabled).toBe(false);
  });

  it('dispatches methodChange when the user selects ML', async () => {
    const { container, component } = render(DespikeStep, {
      props: { wavenumber: WAVENUMBER, intensity: INTENSITY, method: 'none', mlMaskCount: 5, region: 'R1' },
    });
    const methodChange = vi.fn();
    component.$on('methodChange', (e) => methodChange(e.detail));
    await expandBody(container);
    await fireEvent.change(getRadio(container, 'ml'));
    expect(methodChange).toHaveBeenCalledTimes(1);
    // `prev` rides along so the chain can distinguish none→modz (kick the
    // client compute locally) from ml→modz (refetch path kicks it).
    expect(methodChange.mock.calls[0][0]).toEqual({ method: 'ml', prev: 'none' });
  });

  it('shows the modz tunable params only when modz is the selected method', async () => {
    const { container, component } = render(DespikeStep, {
      props: { wavenumber: WAVENUMBER, intensity: INTENSITY, method: 'none', mlMaskCount: 5, region: 'R1' },
    });
    await expandBody(container);
    // None selected → no Window Size slider.
    expect(container.querySelector('#ds-window')).toBeNull();
    // Switch to modz.
    component.$set({ method: 'modz' });
    await tick();
    expect(container.querySelector('#ds-window')).not.toBeNull();
    expect(container.textContent).toMatch(/Sulfate Guard/i);
  });

  it('shows ML provenance (method label + masked-channel count) when ml is applied', async () => {
    const { container } = render(DespikeStep, {
      props: {
        wavenumber: WAVENUMBER,
        intensity: INTENSITY,
        method: 'ml',
        mlMaskCount: 5,
        region: 'R1',
        mlApplied: true,
        mlMethodLabel: 'ml_v1.3_tau_matched',
        mlNMaskedChannels: 7,
        mlMissingRegions: [],
      },
    });
    await expandBody(container);
    const prov = container.querySelector('[data-testid="ml-provenance"]');
    expect(prov).not.toBeNull();
    expect(prov!.textContent).toMatch(/ml_v1\.3_tau_matched/);
    expect(prov!.textContent).toMatch(/7 channels masked/i);
  });

  // Issue #12: on an aggregate (average/subset) ML view the union mask spans
  // the contributing points, so per-position provenance is replaced by an
  // aggregate-count line that steers the user to a single point.
  it('shows the aggregate-count provenance line when mlAggregateView is set (issue #12)', async () => {
    const { container } = render(DespikeStep, {
      props: {
        wavenumber: WAVENUMBER,
        intensity: INTENSITY,
        method: 'ml',
        mlMaskCount: 100,
        region: 'R1',
        mlApplied: true,
        mlMethodLabel: 'ml_v1.3_tau_matched',
        mlNMaskedChannels: 247,
        mlMissingRegions: [],
        mlAggregateView: true,
      },
    });
    await expandBody(container);
    const note = container.querySelector('[data-testid="ml-aggregate-note"]');
    expect(note, 'expected the aggregate provenance note').not.toBeNull();
    expect(note!.textContent).toMatch(
      /Masks applied on 247 channels across the contributing points — select a single point to see positions\./,
    );
    // The per-position "channels masked" copy is replaced, not appended.
    expect(container.textContent).not.toMatch(/CR despiked/);
  });

  it('shows per-position provenance (not the aggregate line) when mlAggregateView is false', async () => {
    const { container } = render(DespikeStep, {
      props: {
        wavenumber: WAVENUMBER,
        intensity: INTENSITY,
        method: 'ml',
        mlMaskCount: 5,
        region: 'R1',
        mlApplied: true,
        mlMethodLabel: 'ml_v1.3_tau_matched',
        mlNMaskedChannels: 7,
        mlMissingRegions: [],
        mlAggregateView: false,
      },
    });
    await expandBody(container);
    expect(container.querySelector('[data-testid="ml-aggregate-note"]')).toBeNull();
    expect(container.textContent).toMatch(/7 channels masked/i);
  });

  it('surfaces the missing-regions warning for an all-or-none composite miss', async () => {
    const { container } = render(DespikeStep, {
      props: {
        wavenumber: WAVENUMBER,
        intensity: INTENSITY,
        method: 'ml',
        mlMaskCount: 5,
        region: 'R123',
        mlApplied: false,
        mlMethodLabel: null,
        mlNMaskedChannels: 0,
        mlMissingRegions: ['R2'],
      },
    });
    await expandBody(container);
    const warn = container.querySelector('[data-testid="ml-missing-regions"]');
    expect(warn).not.toBeNull();
    expect(warn!.textContent).toMatch(/missing masks for R2/i);
    expect(warn!.textContent).toMatch(/all-or-none/i);
  });

  it('NEVER runs the client-side modz compute when ML is selected (no-double-despike)', async () => {
    const postDespikeSpy = vi.spyOn(api, 'postDespike');
    const { container, component } = render(DespikeStep, {
      props: {
        wavenumber: WAVENUMBER,
        intensity: INTENSITY,
        method: 'ml',
        mlMaskCount: 5,
        region: 'R1',
        mlApplied: true,
        mlNMaskedChannels: 3,
      },
    });
    await expandBody(container);
    // runModz() is the only entry to the client compute and it is hard-gated
    // on method==='modz'. Calling it while ML is the active method is a no-op,
    // so postDespike (the modz network call) is never reached.
    (component as unknown as { runModz: () => void }).runModz();
    await new Promise((r) => setTimeout(r, 0));
    await tick();
    expect(postDespikeSpy).not.toHaveBeenCalled();
  });

  it('runModz() runs the client compute only under modz and emits apply', async () => {
    const postDespikeSpy = vi.spyOn(api, 'postDespike').mockResolvedValue({
      schema_version: '1',
      despiked: [9, 11, 14, 12, 10],
      spike_mask: [false, false, true, false, false],
      n_spikes: 1,
      params_used: { window_size: 7, zscore_threshold: 6, max_iterations: 1, sulfate_guard: true },
    });
    const { component } = render(DespikeStep, {
      props: { wavenumber: WAVENUMBER, intensity: INTENSITY, method: 'modz', mlMaskCount: 0, region: 'R1' },
    });
    const apply = vi.fn();
    component.$on('apply', (e) => apply(e.detail));
    // Drive the public entry the workbench calls after a raw fetch lands.
    (component as unknown as { runModz: () => void }).runModz();
    await new Promise((r) => setTimeout(r, 0));
    await tick();
    expect(postDespikeSpy).toHaveBeenCalledTimes(1);
    expect(apply).toHaveBeenCalledTimes(1);
    expect(apply.mock.calls[0][0].nSpikes).toBe(1);
  });
});
