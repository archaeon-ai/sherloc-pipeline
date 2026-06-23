<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { postFit } from '../lib/api';
  import { ApiError } from '../lib/api';
  import type { Peak, BaselineParams, FitParams } from '../lib/types';
  import InfoTooltip from './InfoTooltip.svelte';

  const fitRef = `<div class="ref-title">Multi-Gaussian peak fitting</div>
Fits a sum-of-Gaussians model using Trust Region Reflective optimization
(scipy.optimize.least_squares). FWHM is soft-constrained toward the
SHERLOC slit width (34.1 cm⁻¹).
<div class="ref-section"><b>F-test</b></div>
Sequential nested-model F-test. Adds peaks one at a time; each addition
is retained iff <i>F</i>&nbsp;=&nbsp;((RSS<sub>n</sub>&minus;RSS<sub>n+1</sub>)/Δk)/(RSS<sub>n+1</sub>/dof)
gives p&nbsp;&lt;&nbsp;0.01 against the F-distribution (Δk=3 per Gaussian).
First non-significant addition halts the search.
<div class="ref-section"><b>AICc</b></div>
Information criterion. Fits every candidate peak count and selects the
one minimizing AICc&nbsp;=&nbsp;<i>n</i>&nbsp;ln(RSS/<i>n</i>) + 2<i>k</i> + 2<i>k</i>(<i>k</i>+1)/(<i>n</i>&minus;<i>k</i>&minus;1).
<div class="ref-cite">Optimization: Trust Region Reflective (Branch, Coleman &amp; Li, 1999).
<br>Model selection: Sequential F-test for nested models (standard).
<br>AICc: Burnham &amp; Anderson (2002). <i>Model Selection and Multimodel Inference</i>.
<br>Instrument: Bhartia, R. et al. (2021). Perseverance's SHERLOC Investigation. <i>Space Sci. Rev.</i>, 217, 58.
<a href="https://doi.org/10.1007/s11214-021-00812-z" target="_blank">doi:10.1007/s11214-021-00812-z</a></div>`;

  export let wavenumber: number[] = [];
  export let wavelength: number[] | null = null;
  export let intensity: number[] = [];
  export let baselineParams: BaselineParams = { method: 'aspls', lam: 1e6, max_iter: 10 };
  export let collapsed: boolean = true;
  // Scan target_type forwarded to /api/process/fit so the backend quality
  // classifier downgrades calibration / engineering scans to "review"
  // (no Mars-target ground truth). Null = no scan context (e.g. legacy
  // callers); the backend treats this as mars_target.
  export let targetType: string | null = null;

  const dispatch = createEventDispatcher<{
    apply: {
      peaks: Peak[];
      fitCurve: number[];
      residual: number[];
      corrected: number[];
      baseline: number[];
      rSquared: number;
      modelSelectionMethod: string;
      fitWavenumber: number[];
      params: FitParams;
    };
  }>();

  // Domain presets
  const domainDefaults: Record<string, { range: [number, number]; fwhm: [number, number]; snr: number; peaks: number; unit: string }> = {
    minerals:      { range: [700, 1200],   fwhm: [22, 90],   snr: 3.0,  peaks: 5, unit: 'cm⁻¹' },
    organics:      { range: [1250, 1850],  fwhm: [40, 200],  snr: 2.0,  peaks: 2, unit: 'cm⁻¹' },
    hydration:     { range: [2800, 3900],  fwhm: [50, 300],  snr: 2.0,  peaks: 3, unit: 'cm⁻¹' },
    fluorescence:  { range: [276, 357],    fwhm: [7, 35],    snr: 10.0, peaks: 4, unit: 'nm' },
  };

  // Parameters
  let domain: 'minerals' | 'organics' | 'hydration' | 'fluorescence' = 'minerals';
  let waveMin = 700;
  let waveMax = 1200;
  let maxPeaks = 5;
  let minSnr = 3.0;
  let fwhmMin = 22;
  let fwhmMax = 90;
  let modelSelection: 'f-test' | 'aicc' = 'f-test';

  let fitting = false;
  let error = '';
  let lastNPeaks: number | null = null;
  let lastRSquared: number | null = null;

  $: isFluor = domain === 'fluorescence';
  $: unitLabel = domainDefaults[domain]?.unit ?? 'cm⁻¹';

  // Dispatch region switch when fluorescence is selected
  export let onRegionSwitch: ((region: string) => void) | null = null;

  function applyDomainDefaults() {
    const d = domainDefaults[domain];
    if (d) {
      waveMin = d.range[0];
      waveMax = d.range[1];
      fwhmMin = d.fwhm[0];
      fwhmMax = d.fwhm[1];
      minSnr = d.snr;
      maxPeaks = d.peaks;
    }
    // Auto-switch region for fluorescence
    if (domain === 'fluorescence' && onRegionSwitch) {
      onRegionSwitch('R123');
    } else if (domain !== 'fluorescence' && onRegionSwitch) {
      onRegionSwitch('R1');
    }
  }

  $: if (domain) applyDomainDefaults();

  function toggleCollapsed() {
    collapsed = !collapsed;
  }

  function buildFitCurve(corrected: number[], residual: number[]): number[] {
    return corrected.map((c, i) => c - residual[i]);
  }

  async function runFit() {
    // For fluorescence, use wavelength as x-axis; for Raman domains, use wavenumber
    const xData = isFluor && wavelength ? wavelength : wavenumber;
    if (xData.length === 0) return;
    fitting = true;
    error = '';
    try {
      const result = await postFit({
        wavenumber: xData,
        intensity,
        target_type: targetType ?? undefined,
        params: {
          // Workbench is sequential: BaselineStep is the canonical baseline stage.
          // The fit step receives intensity that has already been baseline-corrected
          // (or intentionally not, if BaselineStep is disabled). Passing baseline=undefined
          // tells the backend to skip baseline correction and use the input as-is.
          baseline: undefined,
          fitting: {
            domain,
            wavenumber_range: [waveMin, waveMax],
            max_peaks: maxPeaks,
            min_snr: minSnr,
            fwhm_bounds: [fwhmMin, fwhmMax],
            model_selection: isFluor ? 'aicc' : modelSelection,
          },
        },
      });
      lastNPeaks = result.n_peaks;
      lastRSquared = result.r_squared;
      const fitCurve = buildFitCurve(result.corrected, result.residual);
      dispatch('apply', {
        peaks: result.peaks,
        fitCurve,
        residual: result.residual,
        corrected: result.corrected,
        baseline: result.baseline,
        rSquared: result.r_squared,
        modelSelectionMethod: result.model_selection_method,
        fitWavenumber: result.wavenumber,
        params: {
          domain,
          wavenumber_range: [waveMin, waveMax],
          max_peaks: maxPeaks,
          min_snr: minSnr,
          fwhm_bounds: [fwhmMin, fwhmMax],
          model_selection: modelSelection,
        },
      });
    } catch (e) {
      if (e instanceof ApiError) {
        error = e.message;
      } else {
        error = 'Fitting failed';
      }
    } finally {
      fitting = false;
    }
  }
</script>

<div class="step-card">
  <button class="step-header" on:click={toggleCollapsed}>
    <div class="step-header-left">
      <span class="step-indicator" class:active={lastNPeaks !== null}></span>
      <span class="step-title">4. Peak Fit</span>
      {#if fitting}
        <span class="spinner"></span>
      {/if}
    </div>
    <div class="step-header-right">
      {#if lastNPeaks !== null}
        <span class="step-badge mono">{lastNPeaks} peak{lastNPeaks !== 1 ? 's' : ''}</span>
        {#if lastRSquared !== null}
          <span class="step-badge mono">R²={lastRSquared.toFixed(3)}</span>
        {/if}
      {/if}
      <span class="collapse-icon">{collapsed ? '+' : '-'}</span>
    </div>
  </button>

  {#if !collapsed}
    <div class="step-body">
      {#if error}
        <div class="step-error">{error}</div>
      {/if}

      <div class="param-group">
        <label for="rf-domain">Domain</label>
        <select id="rf-domain" bind:value={domain}>
          <option value="minerals">Minerals</option>
          <option value="organics">Organics</option>
          <option value="hydration">Hydration</option>
          <option value="fluorescence">Fluorescence</option>
        </select>
      </div>

      <div class="param-group">
        <label>Fit Range ({unitLabel})</label>
        <div class="range-inputs">
          <input type="number" bind:value={waveMin} style="width: 80px" />
          <span>&ndash;</span>
          <input type="number" bind:value={waveMax} style="width: 80px" />
        </div>
      </div>

      <div class="param-group">
        <label for="rf-peaks">Max Peaks: {maxPeaks}</label>
        <input id="rf-peaks" type="range" min="1" max="10" bind:value={maxPeaks} />
      </div>

      <div class="param-group">
        <label for="rf-snr">Min SNR: {minSnr.toFixed(1)}</label>
        <input id="rf-snr" type="range" min="1" max="20" step="0.5" bind:value={minSnr} />
      </div>

      <div class="param-group">
        <label>FWHM Bounds ({unitLabel})</label>
        <div class="range-inputs">
          <input type="number" bind:value={fwhmMin} style="width: 70px" />
          <span>&ndash;</span>
          <input type="number" bind:value={fwhmMax} style="width: 70px" />
        </div>
      </div>

      {#if !isFluor}
      <div class="param-group">
        <label for="rf-model" class="label-with-info">
          Model Selection
          <InfoTooltip text={fitRef} />
        </label>
        <select id="rf-model" bind:value={modelSelection}>
          <option value="f-test">F-test</option>
          <option value="aicc">AICc</option>
        </select>
      </div>
      {/if}

      {#if isFluor}
        <div class="step-info" style="font-size: 0.78rem; color: var(--color-text-secondary)">
          Agnostic multi-Gaussian fitting with AICc model selection. Auto-switches to full spectrum view.
        </div>
      {/if}

      <button class="btn-primary fit-btn" on:click={runFit} disabled={fitting || wavenumber.length === 0}>
        {#if fitting}
          <span class="spinner"></span> Fitting...
        {:else}
          Fit Spectrum
        {/if}
      </button>
    </div>
  {/if}
</div>

<style>
  .label-with-info {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .step-card {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-surface);
    overflow: hidden;
  }

  .step-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    padding: 10px 12px;
    background: var(--color-background);
    border: none;
    border-radius: 0;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 600;
  }

  .step-header:hover {
    background: var(--color-primary-light);
  }

  .step-header-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .step-header-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .step-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--color-border);
    border: 1px solid var(--color-border-strong);
  }

  .step-indicator.active {
    background: var(--color-success);
    border-color: var(--color-success);
  }

  .step-title {
    color: var(--color-text);
  }

  .step-badge {
    font-size: 0.75rem;
    padding: 1px 6px;
    background: var(--color-info-light);
    color: var(--color-info);
    border-radius: 9999px;
  }

  .collapse-icon {
    font-size: 1rem;
    color: var(--color-text-tertiary);
    width: 20px;
    text-align: center;
    font-family: var(--font-mono);
  }

  .step-body {
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    border-top: 1px solid var(--color-border);
  }

  .step-error {
    background: var(--color-error-light);
    color: var(--color-error);
    padding: 6px 10px;
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
  }

  .param-group select,
  .param-group input[type="range"] {
    width: 100%;
  }

  .range-inputs {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .fit-btn {
    width: 100%;
    margin-top: 4px;
  }
</style>
