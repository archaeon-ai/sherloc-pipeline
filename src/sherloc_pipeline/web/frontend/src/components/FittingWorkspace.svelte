<script lang="ts">
  import { onMount } from 'svelte';
  import { navigate, baselineParams, selectedSpectrum, lastFitResult } from '../lib/stores';
  import { getScan, getAverageSpectrum, postFit } from '../lib/api';
  import { ApiError } from '../lib/api';
  import type { ScanDetail, SpectrumResponse, FitResponse, Peak } from '../lib/types';
  import SpectrumPlot from './SpectrumPlot.svelte';
  import PeakTable from './PeakTable.svelte';

  export let scanId: string;

  let scan: ScanDetail | null = null;
  let spectrum: SpectrumResponse | null = null;
  let fitResult: FitResponse | null = null;
  let loading = true;
  let fitting = false;
  let error = '';
  let fitError = '';

  // Fitting parameters
  let domain: 'minerals' | 'organics' | 'hydration' = 'minerals';
  let waveMin = 700;
  let waveMax = 1200;
  let maxPeaks = 5;
  let minSnr = 3.0;
  let fwhmMin = 22;
  let fwhmMax = 90;
  let modelSelection: 'f-test' | 'aicc' = 'f-test';

  let showResidual = false;
  let showFit = false;
  let showPeakLabels = true;

  // Domain presets
  const domainDefaults: Record<string, { range: [number, number]; fwhm: [number, number]; snr: number; peaks: number }> = {
    minerals: { range: [700, 1200], fwhm: [22, 90], snr: 3.0, peaks: 5 },
    organics: { range: [1250, 1850], fwhm: [40, 200], snr: 2.0, peaks: 2 },
    hydration: { range: [2800, 3900], fwhm: [50, 300], snr: 2.0, peaks: 3 },
  };

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
  }

  $: if (domain) applyDomainDefaults();

  onMount(async () => {
    loading = true;
    try {
      const res = await getScan(scanId);
      scan = res.scan;
      spectrum = await getAverageSpectrum(scanId, { region: 'R1' });
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load scan';
    } finally {
      loading = false;
    }
  });

  async function runFit() {
    if (!spectrum) return;
    fitting = true;
    fitError = '';
    try {
      fitResult = await postFit({
        wavenumber: spectrum.wavenumber,
        intensity: spectrum.intensity,
        // Forward scan context so the backend quality classifier applies
        // the calibration-scan downgrade rule (v4.1.12). Without this the
        // legacy Fitting Workspace would render green "pass" pills for
        // calibration scans, which is the exact failure the Workbench fix
        // closes.
        target_type: scan?.target_type ?? undefined,
        params: {
          baseline: $baselineParams,
          fitting: {
            domain,
            wavenumber_range: [waveMin, waveMax],
            max_peaks: maxPeaks,
            min_snr: minSnr,
            fwhm_bounds: [fwhmMin, fwhmMax],
            model_selection: modelSelection,
          },
        },
      });
      lastFitResult.set(fitResult);
      showFit = true;
    } catch (e) {
      if (e instanceof ApiError) {
        fitError = e.message;
      } else {
        fitError = 'Fitting failed';
      }
    } finally {
      fitting = false;
    }
  }

  // Build fit curve from peaks for overlay
  $: fitCurve = fitResult ? buildFitCurve(fitResult) : null;

  function buildFitCurve(fr: FitResponse): number[] {
    // The fit curve = corrected - residual
    return fr.corrected.map((c, i) => c - fr.residual[i]);
  }

  $: displayPeaks = fitResult?.peaks ?? [];

  function exportSpectrumCsv() {
    if (!fitResult) return;
    const header = 'wavenumber,raw,baseline,corrected,fit,residual';
    const fitCurveArr = buildFitCurve(fitResult);
    const rows = fitResult.wavenumber.map((wn, i) =>
      [
        wn.toFixed(4),
        fitResult!.corrected[i] !== undefined ? (fitResult!.corrected[i] + fitResult!.baseline[i]).toFixed(4) : '',
        fitResult!.baseline[i].toFixed(4),
        fitResult!.corrected[i].toFixed(4),
        fitCurveArr[i].toFixed(4),
        fitResult!.residual[i].toFixed(4),
      ].join(',')
    );
    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${scan?.target ?? 'scan'}_${domain}_fit.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<div class="page-container">
  <div class="breadcrumb">
    <button class="btn-link" on:click={() => navigate('#/')}>Scans</button>
    <span class="breadcrumb-sep">/</span>
    <button class="btn-link" on:click={() => navigate(`#/scan/${scanId}`)}>
      {scan?.target ?? '...'} Sol {scan?.sol_number ?? ''}
    </button>
    <span class="breadcrumb-sep">/</span>
    <span>Fitting</span>
  </div>

  {#if error}
    <div class="error-message">{error}</div>
  {/if}

  {#if loading}
    <div class="empty-state"><span class="spinner"></span> Loading...</div>
  {:else}
    <div class="workspace-layout">
      <!-- Spectrum + results (left) -->
      <div class="workspace-main">
        <div class="card">
          <div class="card-header flex items-center justify-between">
            <span>
              {scan?.target ?? 'Unknown'} Sol {scan?.sol_number ?? ''} {scan?.scan_name ?? ''} &mdash;
              {domain} fitting
            </span>
            <div class="flex gap-2">
              <label class="inline-toggle">
                <input type="checkbox" bind:checked={showFit} disabled={!fitResult} />
                Show fitted peaks
              </label>
              <label class="inline-toggle">
                <input type="checkbox" bind:checked={showPeakLabels} disabled={!fitResult} />
                Show peak labels
              </label>
              <label class="inline-toggle">
                <input type="checkbox" bind:checked={showResidual} disabled={!fitResult} />
                Residual
              </label>
            </div>
          </div>
          <div class="card-body">
            {#if spectrum}
              <SpectrumPlot
                wavenumber={showFit && fitResult ? fitResult.wavenumber : spectrum.wavenumber}
                intensity={showFit && fitResult ? fitResult.corrected : spectrum.intensity}
                baseline={fitResult?.baseline ?? null}
                corrected={fitResult?.corrected ?? null}
                residual={fitResult?.residual ?? null}
                {fitCurve}
                peaks={displayPeaks}
                {showResidual}
                showBaseline={false}
                {showFit}
                {showPeakLabels}
                height={480}
              />
            {/if}
          </div>
        </div>

        {#if fitResult}
          <div class="card" style="margin-top: 16px">
            <div class="card-header flex items-center justify-between">
              <span>Peak Results</span>
              <span class="mono" style="font-size: 0.8rem; color: var(--color-text-secondary)">
                R&sup2; = {fitResult.r_squared.toFixed(4)} | {fitResult.model_selection_method}
              </span>
            </div>
            <div class="card-body">
              <PeakTable peaks={fitResult.peaks} />
              <div style="margin-top: 8px; display: flex; gap: 6px; justify-content: flex-end">
                <button class="btn-secondary btn-sm" on:click={exportSpectrumCsv}>
                  Export Spectrum CSV
                </button>
              </div>
            </div>
          </div>
        {/if}

        {#if fitError}
          <div class="error-message" style="margin-top: 16px">{fitError}</div>
        {/if}
      </div>

      <!-- Parameters (right) -->
      <div class="workspace-sidebar">
        <div class="card">
          <div class="card-header">Fitting Parameters</div>
          <div class="card-body param-controls">
            <div class="param-group">
              <label for="p-domain">Domain</label>
              <select id="p-domain" bind:value={domain}>
                <option value="minerals">Minerals</option>
                <option value="organics">Organics</option>
                <option value="hydration">Hydration</option>
              </select>
            </div>

            <div class="param-group">
              <label>Wavenumber Range (cm<sup>-1</sup>)</label>
              <div class="range-inputs">
                <input type="number" bind:value={waveMin} style="width: 80px" />
                <span>&ndash;</span>
                <input type="number" bind:value={waveMax} style="width: 80px" />
              </div>
            </div>

            <div class="param-group">
              <label for="p-peaks">Max Peaks: {maxPeaks}</label>
              <input id="p-peaks" type="range" min="1" max="10" bind:value={maxPeaks} />
            </div>

            <div class="param-group">
              <label for="p-snr">Min SNR: {minSnr.toFixed(1)}</label>
              <input id="p-snr" type="range" min="1" max="20" step="0.5" bind:value={minSnr} />
            </div>

            <div class="param-group">
              <label>FWHM Bounds (cm<sup>-1</sup>)</label>
              <div class="range-inputs">
                <input type="number" bind:value={fwhmMin} style="width: 70px" />
                <span>&ndash;</span>
                <input type="number" bind:value={fwhmMax} style="width: 70px" />
              </div>
            </div>

            <div class="param-group">
              <label for="p-model">Model Selection</label>
              <select id="p-model" bind:value={modelSelection}>
                <option value="f-test">F-test</option>
                <option value="aicc">AICc</option>
              </select>
            </div>

            <hr style="border: none; border-top: 1px solid var(--color-border); margin: 8px 0" />

            <div class="param-group">
              <label>Baseline Params</label>
              <div class="meta-info mono" style="font-size: 0.8rem; color: var(--color-text-secondary)">
                {$baselineParams.method} | &lambda;={$baselineParams.lam?.toExponential(0)} | iter={$baselineParams.max_iter}
              </div>
              <button
                class="btn-secondary btn-sm"
                style="margin-top: 4px"
                on:click={() => navigate(`#/scan/${scanId}/baseline`)}
              >
                Tune Baseline
              </button>
            </div>

            <button class="btn-primary fit-btn" on:click={runFit} disabled={fitting || !spectrum}>
              {#if fitting}
                <span class="spinner"></span> Fitting...
              {:else}
                Fit Spectrum
              {/if}
            </button>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .breadcrumb {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 16px;
    font-size: 0.9rem;
    color: var(--color-text-secondary);
  }

  .breadcrumb-sep {
    color: var(--color-text-tertiary);
  }

  .btn-link {
    background: none;
    border: none;
    color: var(--color-primary);
    padding: 0;
    cursor: pointer;
    font-size: inherit;
  }

  .btn-link:hover {
    text-decoration: underline;
  }

  .workspace-layout {
    display: grid;
    grid-template-columns: 1fr 280px;
    gap: 16px;
    align-items: start;
  }

  @media (max-width: 1024px) {
    .workspace-layout {
      grid-template-columns: 1fr;
    }
  }

  .workspace-main {
    min-width: 0;
  }

  .param-controls {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .param-group label {
    margin-bottom: 2px;
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

  .inline-toggle {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.8rem;
    color: var(--color-text-secondary);
    cursor: pointer;
    margin-bottom: 0;
  }
</style>
