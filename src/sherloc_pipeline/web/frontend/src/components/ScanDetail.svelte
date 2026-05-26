<script lang="ts">
  import { onMount } from 'svelte';
  import { navigate, selectedScan, selectedSpectrum } from '../lib/stores';
  import { getScan, getScanPoints, getAverageSpectrum, getPointSpectrum } from '../lib/api';
  import { ApiError } from '../lib/api';
  import type { ScanDetail as ScanDetailType, SpectrumResponse, ScanPoint } from '../lib/types';
  import SpectrumPlot from './SpectrumPlot.svelte';
  import PointSelector from './PointSelector.svelte';

  export let scanId: string;

  let scan: ScanDetailType | null = null;
  let spectrum: SpectrumResponse | null = null;
  let points: ScanPoint[] = [];
  let loading = true;
  let error = '';
  let baselineCorrected = false;
  let region = 'R1';

  // Point spectrum mode
  let viewMode: 'average' | 'point' = 'average';
  let selectedPointIdx: number | null = null;

  onMount(() => {
    loadScan();
  });

  async function loadScan() {
    loading = true;
    error = '';
    try {
      const res = await getScan(scanId);
      scan = res.scan;
      selectedScan.set(scan);
      // Load points for point selector
      const ptRes = await getScanPoints(scanId);
      points = ptRes.points;
      await loadSpectrum();
    } catch (e) {
      if (e instanceof ApiError) {
        error = e.message;
      } else {
        error = 'Failed to load scan';
      }
    } finally {
      loading = false;
    }
  }

  async function loadSpectrum() {
    try {
      if (viewMode === 'point' && selectedPointIdx !== null) {
        spectrum = await getPointSpectrum(scanId, selectedPointIdx, {
          region,
        });
      } else {
        spectrum = await getAverageSpectrum(scanId, {
          region,
          baseline_corrected: baselineCorrected,
        });
      }
      selectedSpectrum.set(spectrum);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        spectrum = null;
      } else {
        throw e;
      }
    }
  }

  async function toggleBaseline() {
    baselineCorrected = !baselineCorrected;
    await loadSpectrum();
  }

  async function changeRegion(newRegion: string) {
    region = newRegion;
    await loadSpectrum();
  }

  async function switchToAverage() {
    viewMode = 'average';
    selectedPointIdx = null;
    baselineCorrected = false;
    await loadSpectrum();
  }

  async function selectPoint(idx: number) {
    viewMode = 'point';
    selectedPointIdx = idx;
    baselineCorrected = false;
    await loadSpectrum();
  }

  function getXLabel(reg: string): string {
    if (reg === 'R2' || reg === 'R3') return 'Wavelength (nm)';
    return 'Raman Shift (cm\u207B\u00B9)';
  }

  function getSpectrumTitle(): string {
    const base = `${scan?.target ?? 'Unknown'} Sol ${scan?.sol_number ?? ''} ${scan?.scan_name ?? ''} (${region})`;
    if (viewMode === 'point' && selectedPointIdx !== null) {
      return `${base} — Point ${selectedPointIdx}`;
    }
    return base;
  }

  function exportSpectrumCsv() {
    if (!spectrum) return;
    const header = 'wavenumber,intensity';
    const rows = spectrum.wavenumber.map((wn, i) =>
      `${wn.toFixed(4)},${spectrum!.intensity[i].toFixed(4)}`
    );
    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const mode = viewMode === 'point' ? `pt${selectedPointIdx}` : 'avg';
    a.download = `${scan?.target ?? 'scan'}_${scan?.scan_name ?? ''}_${region}_${mode}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<div class="page-container">
  <div class="breadcrumb">
    <button class="btn-link" on:click={() => navigate('#/')}>Scans</button>
    <span class="breadcrumb-sep">/</span>
    <span>{scan?.target ?? '...'} &mdash; Sol {scan?.sol_number ?? '...'}</span>
  </div>

  {#if error}
    <div class="error-message">{error}</div>
  {/if}

  {#if loading}
    <div class="empty-state"><span class="spinner"></span> Loading scan...</div>
  {:else if scan}
    <div class="detail-layout">
      <!-- Metadata panel -->
      <div class="card metadata-panel">
        <div class="card-header">Scan Metadata</div>
        <div class="card-body">
          <dl class="meta-list">
            <dt>Sol</dt><dd class="mono">{scan.sol_number}</dd>
            <dt>Target</dt><dd>{scan.target ?? '--'}</dd>
            <dt>Scan Name</dt><dd class="mono">{scan.scan_name}</dd>
            <dt>Scan ID</dt><dd class="mono" style="font-size: 0.8rem">{scan.scan_id}</dd>
            <dt>Points</dt><dd class="mono">{scan.n_points}</dd>
            <dt>Channels</dt><dd class="mono">{scan.n_channels}</dd>
            <dt>Shots/Point</dt><dd class="mono">{scan.shots_per_point}</dd>
            <dt>Class</dt><dd><span class="badge badge-neutral">{scan.scan_class}</span></dd>
            <dt>Type</dt><dd>{scan.scan_type ?? '--'}</dd>
            <dt>Target Type</dt><dd>{scan.target_type}</dd>
            <dt>Data Source</dt><dd>{scan.data_source}</dd>
            <dt>Status</dt>
            <dd>
              <span
                class="badge"
                class:badge-success={scan.processing_status === 'completed'}
                class:badge-error={scan.processing_status === 'failed'}
                class:badge-neutral={!scan.processing_status}
              >
                {scan.processing_status ?? 'unprocessed'}
              </span>
            </dd>
            {#if scan.processed_at}
              <dt>Processed</dt><dd class="mono" style="font-size: 0.8rem">{new Date(scan.processed_at).toLocaleString()}</dd>
            {/if}
            {#if scan.parent_scan_id}
              <dt>Parent Scan</dt><dd class="mono" style="font-size: 0.75rem">{scan.parent_scan_id}</dd>
            {/if}
          </dl>

          <div class="action-buttons">
            <button class="btn-primary" on:click={() => navigate(`#/scan/${scanId}/workbench`)}>
              Processing Workbench
            </button>
            <button class="btn-secondary" on:click={() => navigate(`#/scan/${scanId}/fit`)}>
              Fitting Workspace
            </button>
            <button class="btn-secondary" on:click={() => navigate(`#/scan/${scanId}/baseline`)}>
              Baseline Workspace
            </button>
            <button class="btn-secondary" on:click={() => navigate(`#/scan/${scanId}/map`)}>
              Map Mode
            </button>
          </div>
        </div>

        <!-- Point selector -->
        <div class="card-header" style="margin-top: 1px">Point Spectra</div>
        <div class="card-body">
          <div class="view-toggle">
            <button
              class="btn-sm"
              class:btn-primary={viewMode === 'average'}
              class:btn-secondary={viewMode !== 'average'}
              on:click={switchToAverage}
            >
              Average
            </button>
            <span class="mono" style="font-size: 0.8rem; color: var(--color-text-secondary)">
              {viewMode === 'point' && selectedPointIdx !== null ? `Point ${selectedPointIdx}` : 'All points'}
            </span>
          </div>
          <div class="point-grid-compact">
            {#each points as point}
              <button
                class="point-chip"
                class:selected={viewMode === 'point' && selectedPointIdx === point.point_index}
                on:click={() => selectPoint(point.point_index)}
                title="Point {point.point_index}"
              >
                {point.point_index}
              </button>
            {/each}
          </div>
        </div>
      </div>

      <!-- Spectrum panel -->
      <div class="spectrum-panel">
        <div class="card">
          <div class="card-header flex items-center justify-between">
            <span>{viewMode === 'point' ? `Point ${selectedPointIdx} Spectrum` : 'Averaged Spectrum'}</span>
            <div class="flex gap-2 items-center">
              <div class="region-tabs">
                {#each ['R1', 'R2', 'R3'] as r}
                  <button
                    class="region-tab"
                    class:active={region === r}
                    on:click={() => changeRegion(r)}
                  >
                    {r}
                  </button>
                {/each}
              </div>
              {#if viewMode === 'average'}
                <button
                  class="btn-sm"
                  class:btn-primary={baselineCorrected}
                  class:btn-secondary={!baselineCorrected}
                  on:click={toggleBaseline}
                >
                  {baselineCorrected ? 'Baseline Corrected' : 'Raw'}
                </button>
              {/if}
            </div>
          </div>
          <div class="card-body">
            {#if spectrum}
              <SpectrumPlot
                wavenumber={spectrum.wavenumber}
                intensity={spectrum.intensity}
                xLabel={getXLabel(region)}
                title={getSpectrumTitle()}
                height={420}
              />
              <div class="spectrum-footer">
                <div class="provenance mono">
                  {spectrum.n_channels} ch
                  {#if spectrum.n_points_averaged}
                    | {spectrum.n_points_averaged} pts averaged
                    | trim {((spectrum.effective_trim_pct_per_tail ?? 0) * 100).toFixed(1)}%/tail
                  {/if}
                  | {spectrum.provenance.calibration_version}
                </div>
                <div class="export-buttons">
                  <button class="btn-secondary btn-sm" on:click={exportSpectrumCsv}>
                    Export CSV
                  </button>
                </div>
              </div>
            {:else}
              <div class="empty-state" style="padding: 60px">
                No spectrum data available for this scan and region
              </div>
            {/if}
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

  .detail-layout {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 16px;
    align-items: start;
  }

  @media (max-width: 1024px) {
    .detail-layout {
      grid-template-columns: 1fr;
    }
  }

  .meta-list {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 6px 12px;
    font-size: 0.85rem;
  }

  .meta-list dt {
    color: var(--color-text-secondary);
    font-weight: 500;
  }

  .meta-list dd {
    margin: 0;
  }

  .action-buttons {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 16px;
  }

  .view-toggle {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .point-grid-compact {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
    max-height: 160px;
    overflow-y: auto;
  }

  .point-chip {
    width: 30px;
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    color: var(--color-text-secondary);
    cursor: pointer;
    padding: 0;
  }

  .point-chip:hover {
    border-color: var(--color-primary);
  }

  .point-chip.selected {
    background: var(--color-primary);
    color: white;
    border-color: var(--color-primary);
  }

  .region-tabs {
    display: flex;
    gap: 2px;
    background: var(--color-background);
    border-radius: var(--radius-md);
    padding: 2px;
  }

  .region-tab {
    padding: 4px 10px;
    font-size: 0.8rem;
    font-weight: 500;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--color-text-secondary);
  }

  .region-tab.active {
    background: var(--color-surface);
    color: var(--color-text);
    box-shadow: var(--shadow-sm);
  }

  .spectrum-panel {
    min-width: 0;
  }

  .spectrum-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 8px;
  }

  .provenance {
    font-size: 0.75rem;
    color: var(--color-text-tertiary);
  }

  .export-buttons {
    display: flex;
    gap: 6px;
  }
</style>
