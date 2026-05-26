<script lang="ts">
  import type { ProcessingArtifacts, Peak } from '../lib/types';

  export let scanId: string = '';
  export let scanName: string = '';
  export let target: string = '';
  export let solNumber: number = 0;
  export let stage: string = 'raw';
  export let wavenumber: number[] = [];
  export let intensity: number[] = [];
  export let artifacts: ProcessingArtifacts | null = null;
  export let processingChain: Record<string, unknown>[] = [];
  export let scanNPoints: number = 0;
  export let selectionMode: 'average' | 'subset' | 'point' = 'average';
  export let selectedIndices: number[] = [];
  export let selectedPointIdx: number | null = null;

  let open = false;
  let showProvenanceModal = false;

  function toggleMenu() {
    open = !open;
  }

  function closeMenu() {
    open = false;
  }

  // --- Provenance JSON ---

  function buildSelection(): Record<string, unknown> {
    if (selectionMode === 'point' && selectedPointIdx !== null) {
      return {
        mode: 'point',
        n_points: 1,
        indices: [selectedPointIdx],
      };
    }
    if (selectionMode === 'subset') {
      const sorted = [...selectedIndices].sort((a, b) => a - b);
      return {
        mode: 'subset',
        n_points: sorted.length,
        indices: sorted,
      };
    }
    // average: all points in the scan
    return {
      mode: 'average',
      n_points: scanNPoints,
      indices: 'all',
    };
  }

  function buildProvenanceJson(): Record<string, unknown> {
    return {
      source: {
        scan_id: scanId,
        sol: solNumber,
        target: target,
        scan_name: scanName,
        data_origin: 'loupe',
      },
      selection: buildSelection(),
      processing_chain: processingChain,
      calibration: {
        version: 'loupe_v5.1.5a',
        laser_nm: 248.6,
      },
      application: {
        version: '3.1.0',
        timestamp: new Date().toISOString(),
      },
    };
  }

  // --- CSV generation ---

  function buildCsvContent(): string {
    const provenance = buildProvenanceJson();
    const lines: string[] = [];

    // Header comments with provenance
    lines.push(`# SHERLOC Web Workbench Export`);
    lines.push(`# scan_id: ${provenance.source ? (provenance.source as Record<string, unknown>).scan_id : ''}`);
    lines.push(`# sol: ${solNumber}`);
    lines.push(`# target: ${target}`);
    lines.push(`# scan_name: ${scanName}`);
    lines.push(`# stage: ${stage}`);
    lines.push(`# calibration_version: loupe_v5.1.5a`);
    lines.push(`# laser_nm: 248.6`);
    lines.push(`# exported_at: ${new Date().toISOString()}`);
    lines.push(`# n_channels: ${wavenumber.length}`);

    // Selection provenance
    const sel = provenance.selection as Record<string, unknown>;
    lines.push(`# selection_mode: ${sel.mode}`);
    lines.push(`# selection_n_points: ${sel.n_points}`);
    const idx = sel.indices;
    const idxStr = Array.isArray(idx) ? (idx as number[]).join(',') : String(idx);
    lines.push(`# selection_indices: ${idxStr}`);

    // Processing chain provenance (one JSON line per step)
    const chain = provenance.processing_chain as Record<string, unknown>[];
    if (chain && chain.length > 0) {
      lines.push(`# processing_chain:`);
      for (const step of chain) {
        lines.push(`#   ${JSON.stringify(step)}`);
      }
    } else {
      lines.push(`# processing_chain: (none)`);
    }

    // Build columns based on stage
    const columns: string[] = ['wavenumber'];
    const data: number[][] = [wavenumber];

    columns.push('intensity');
    data.push(intensity);

    if (stage === 'despiked' && artifacts?.spikeMask) {
      columns.push('spike_mask');
      data.push(artifacts.spikeMask.map(v => v ? 1 : 0));
    }

    if (stage === 'bg_subtracted') {
      if (artifacts?.background) {
        columns.push('background');
        data.push(artifacts.background);
      }
      if (artifacts?.backgroundScaled) {
        columns.push('background_scaled');
        data.push(artifacts.backgroundScaled);
      }
    }

    if (stage === 'baseline_corrected') {
      if (artifacts?.baseline) {
        columns.push('baseline');
        data.push(artifacts.baseline);
      }
    }

    if (stage === 'raman_fitted') {
      if (artifacts?.fitCurve) {
        columns.push('fit_curve');
        data.push(artifacts.fitCurve);
      }
      if (artifacts?.residual) {
        columns.push('residual');
        data.push(artifacts.residual);
      }
    }

    lines.push(columns.join(','));

    // Data rows
    const n = wavenumber.length;
    for (let i = 0; i < n; i++) {
      const row: string[] = [];
      for (const col of data) {
        row.push(i < col.length ? col[i].toString() : '');
      }
      lines.push(row.join(','));
    }

    return lines.join('\n');
  }

  function buildFilename(ext: string): string {
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const safeTarget = (target || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_');
    return `${safeTarget}_sol${solNumber}_${scanName || scanId}_${stage}_${ts}.${ext}`;
  }

  function downloadCsv() {
    const csv = buildCsvContent();
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = buildFilename('csv');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    closeMenu();
  }

  function showProvenance() {
    showProvenanceModal = true;
    closeMenu();
  }

  function closeProvenance() {
    showProvenanceModal = false;
  }

  function copyProvenance() {
    const json = JSON.stringify(buildProvenanceJson(), null, 2);
    navigator.clipboard.writeText(json).catch(() => {
      // Fallback: select text in the pre element
      const el = document.querySelector('.provenance-json');
      if (el) {
        const range = document.createRange();
        range.selectNodeContents(el);
        const sel = window.getSelection();
        if (sel) {
          sel.removeAllRanges();
          sel.addRange(range);
        }
      }
    });
  }

  // Close menu on outside click
  function handleWindowClick(e: MouseEvent) {
    const target = e.target as HTMLElement;
    if (!target.closest('.export-menu-container')) {
      open = false;
    }
  }
</script>

<svelte:window on:click={handleWindowClick} />

<div class="export-menu-container">
  <button
    class="btn-secondary btn-sm export-trigger"
    on:click|stopPropagation={toggleMenu}
    disabled={wavenumber.length === 0}
  >
    Export &#9660;
  </button>

  {#if open}
    <!-- svelte-ignore a11y-click-events-have-key-events -->
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="export-dropdown" on:click|stopPropagation>
      <button class="dropdown-item" on:click={downloadCsv}>
        <span class="dropdown-icon">&#128196;</span>
        CSV at current stage
        <span class="dropdown-hint">{stage}</span>
      </button>
      <button class="dropdown-item" on:click={showProvenance}>
        <span class="dropdown-icon">&#128203;</span>
        Full provenance JSON
      </button>
    </div>
  {/if}
</div>

{#if showProvenanceModal}
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div class="modal-overlay" on:click={closeProvenance}>
    <div class="modal-content" on:click|stopPropagation>
      <div class="modal-header">
        <h3>Processing Provenance</h3>
        <button class="btn-secondary btn-sm" on:click={closeProvenance}>Close</button>
      </div>
      <div class="modal-body">
        <pre class="provenance-json">{JSON.stringify(buildProvenanceJson(), null, 2)}</pre>
      </div>
      <div class="modal-footer">
        <button class="btn-primary btn-sm" on:click={copyProvenance}>
          Copy to Clipboard
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .export-menu-container {
    position: relative;
    display: inline-block;
  }

  .export-trigger {
    font-family: var(--font-sans);
    white-space: nowrap;
  }

  .export-dropdown {
    position: absolute;
    top: calc(100% + 4px);
    right: 0;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-md);
    min-width: 220px;
    z-index: 100;
    overflow: hidden;
  }

  .dropdown-item {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 10px 14px;
    font-size: 0.85rem;
    font-family: var(--font-sans);
    color: var(--color-text);
    background: none;
    border: none;
    border-radius: 0;
    cursor: pointer;
    text-align: left;
  }

  .dropdown-item:hover {
    background: var(--color-background);
  }

  .dropdown-item + .dropdown-item {
    border-top: 1px solid var(--color-border);
  }

  .dropdown-icon {
    font-size: 1rem;
    flex-shrink: 0;
  }

  .dropdown-hint {
    margin-left: auto;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--color-text-tertiary);
  }

  /* Provenance modal */
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 200;
  }

  .modal-content {
    background: var(--color-surface);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    width: 90%;
    max-width: 600px;
    max-height: 80vh;
    display: flex;
    flex-direction: column;
  }

  .modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px;
    border-bottom: 1px solid var(--color-border);
  }

  .modal-header h3 {
    font-size: 1rem;
    font-weight: 600;
    margin: 0;
  }

  .modal-body {
    padding: 16px 18px;
    overflow-y: auto;
    flex: 1;
  }

  .provenance-json {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    line-height: 1.5;
    background: var(--color-background);
    padding: 12px;
    border-radius: var(--radius-md);
    border: 1px solid var(--color-border);
    white-space: pre-wrap;
    word-break: break-all;
    margin: 0;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    padding: 12px 18px;
    border-top: 1px solid var(--color-border);
  }
</style>
