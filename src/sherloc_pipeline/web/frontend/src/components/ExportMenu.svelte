<script lang="ts">
  import type { EvidenceDisplay } from '../lib/evidenceTypes';
  import { evidenceCsv, evidenceExport } from '../lib/evidenceStore';
  import type { SharedEvidence } from '../lib/evidenceStore';
  import type { ExportTrace, ProcessingArtifacts } from '../lib/types';

  import type { IdentificationEvidence } from '../lib/identification';
  export let identification: IdentificationEvidence | null = null;
  export let evidenceDesk = false;
  export let evidenceMode = false;
  export let evidenceDisplays: EvidenceDisplay[] = [];
  export let evidenceShared: SharedEvidence | null = null;
  export let exportFigure: ((format: 'png' | 'svg') => void) | null = null;
  export let trace: ExportTrace | null = null;
  export let scanId: string = '';
  export let scanName: string = '';
  export let target: string = '';
  export let solNumber: number = 0;
  export let stage: string = 'raw';
  export let wavenumber: number[] = [];
  export let intensity: number[] = [];
  $: channelCount = evidenceDesk ? intensity.length : wavenumber.length;
  export let artifacts: ProcessingArtifacts | null = null;
  export let processingChain: Record<string, unknown>[] = [];
  export let scanNPoints: number = 0;
  export let selectionMode: 'average' | 'subset' | 'point' = 'average';
  export let selectedIndices: number[] = [];
  export let selectedPointIdx: number | null = null;
  // Spike-mask export (issue #8). The mask column is emitted iff BOTH props
  // are set — the Workbench's strict three-way method gating is the single
  // authority (ml mask only in ML mode, modz mask only in modz mode, null in
  // none mode). ML produces the mask at the 'raw' stage (the fetch is already
  // cleaned), modz at the 'despiked' stage; `spikeMethod` records which
  // despike produced it in the provenance header. Null = no column, no
  // header — deliberately NOT inferred from `artifacts`/`stage`, because a
  // despiked snapshot restored via undo still carries the modz spikeMask
  // artifact even when the active method is 'none' (PR #10 review F1).
  export let spikeMask: boolean[] | null = null;
  export let spikeMethod: 'ml' | 'modz' | null = null;
  // Known-noisy channel annotation export (issue #9). The badpix column is
  // emitted iff this prop is non-null — the Workbench passes the list only when
  // the overlay toggle is ON (mirroring the spike-mask explicit-props gate:
  // ProcessingWorkbench is the single authority). Each entry's `position`
  // indexes the served array, so the column is 0/1 per served row. A
  // `# badpix_source:` header records the annotation source set. Fully separate
  // from despiking — these channels are annotated, never masked.
  export let badpix: { position: number; tier: number; source: string }[] | null = null;

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

  // One timestamp for both formats while this export snapshot is displayed.
  $: exportTimestamp = snapshotTimestamp(trace, wavenumber, intensity, stage, processingChain);
  function snapshotTimestamp(..._snapshot: unknown[]): string {
    return new Date().toISOString();
  }

  function buildProvenanceJson() {
    const provenance = trace?.provenance;
    return {
      ...provenance,
      ...(identification ? { identification } : {}),
      source: trace?.source ?? {
        scan_id: scanId, sol: solNumber, target, scan_name: scanName, data_origin: null,
      },
      selection: trace?.selection ?? buildSelection(),
      processing_chain: provenance?.processing_chain ?? [],
      user_processing_chain: processingChain,
      calibration: {
        version: provenance?.calibration_version ?? null,
        axis_source: provenance?.axis_source ?? null,
        axis_sha256: provenance?.axis_sha256 ?? null,
        laser_nm: provenance?.excitation_nm ?? null,
        excitation_basis: provenance?.excitation_basis ?? null,
      },
      application: {
        ...provenance?.application,
        version: provenance?.application?.version ?? null,
        timestamp: exportTimestamp,
      },
      stage,
      axes: { wavenumber_unit: provenance?.wavenumber_unit ?? (evidenceDesk && trace ? null : 'cm-1'),
        wavelength_unit: provenance?.wavelength_unit ?? (evidenceDesk && trace ? null : 'nm'), intensity_unit: provenance?.intensity_unit ?? null },
      data: { wavenumber, wavelength: trace?.wavelength ?? null, intensity,
        channel_indices: trace?.channel_indices ?? [], source_intensity: trace?.intensity ?? null },
      annotations: {
        ...(spikeMask && spikeMethod ? { spike_method: spikeMethod, spike_mask: spikeMask } : {}),
        ...(badpix?.length ? { badpix } : {}),
      },
    };
  }

  // --- CSV generation ---

  function buildCsvContent(): string {
    if (evidenceMode) return evidenceCsv(evidenceDisplays, identification, evidenceShared);
    const provenance = buildProvenanceJson();
    const lines: string[] = [];

    // CSV and JSON serialize the same committed trace and processing state.
    lines.push(`# SHERLOC Web Workbench Export`);
    lines.push(`# provenance: ${JSON.stringify(provenance)}`);
    lines.push(`# scan_id: ${provenance.source.scan_id}`);
    lines.push(`# sol: ${provenance.source.sol}`);
    lines.push(`# target: ${provenance.source.target}`);
    lines.push(`# scan_name: ${provenance.source.scan_name}`);
    lines.push(`# stage: ${stage}`);
    lines.push(`# data_origin: ${provenance.source.data_origin ?? 'unknown'}`);
    lines.push(`# axis_source: ${provenance.calibration.axis_source ?? 'unknown'}`);
    lines.push(`# calibration_version: ${provenance.calibration.version ?? 'unknown'}`);
    lines.push(`# laser_nm: ${provenance.calibration.laser_nm ?? 'unknown'}`);
    lines.push(`# application_version: ${provenance.application.version ?? 'unknown'}`);
    lines.push(`# wavenumber_unit: ${provenance.axes.wavenumber_unit ?? 'unknown'}`);
    lines.push(`# wavelength_unit: ${provenance.axes.wavelength_unit ?? 'unknown'}`);
    lines.push(`# intensity_unit: ${provenance.axes.intensity_unit ?? 'unknown'}`);
    lines.push(`# exported_at: ${provenance.application.timestamp}`);
    lines.push(`# n_channels: ${channelCount}`);

    // Selection provenance
    const sel = provenance.selection as Record<string, unknown>;
    lines.push(`# selection_mode: ${sel.mode}`);
    lines.push(`# selection_n_points: ${sel.n_points}`);
    const idx = sel.indices;
    const idxStr = Array.isArray(idx) ? (idx as number[]).join(',') : String(idx);
    lines.push(`# selection_indices: ${idxStr}`);

    // Processing chain provenance (one JSON line per step)
    const chain = [...provenance.processing_chain, ...provenance.user_processing_chain];
    if (chain && chain.length > 0) {
      lines.push(`# processing_chain:`);
      for (const step of chain) {
        lines.push(`#   ${JSON.stringify(step)}`);
      }
    } else {
      lines.push(`# processing_chain: (none)`);
    }

    // Spike mask (issue #8): exported iff the explicit props carry it — never
    // inferred from stage/artifacts (PR #10 review F1: an undo-restored
    // 'despiked' snapshot still holds the modz artifact while the active
    // method may be 'none'; the caller's method gating decides, not us).
    const hasSpikeMask = spikeMask !== null && spikeMethod !== null;
    if (hasSpikeMask) {
      // Provenance note so the consumer knows what spike_mask=1 means: ML marks
      // channels REMOVED by the stored mask (the served trace is already clean);
      // modz marks spikes detected in the displayed trace.
      lines.push(`# spike_mask_method: ${spikeMethod}`);
    }

    // Known-noisy channel annotation (issue #9): emitted iff the explicit
    // `badpix` prop carries the list (the Workbench gates on the overlay
    // toggle, not us). The badpix=1 column marks ANNOTATED channels, NOT
    // removed ones — distinct from spike_mask, which marks despiked channels.
    const hasBadpix = badpix !== null && badpix.length > 0;
    if (hasBadpix) {
      const sources = [...new Set(badpix!.map((b) => b.source))].sort();
      lines.push(`# badpix_source: ${sources.join(',')}`);
    }

    // Build columns based on stage
    const columns: string[] = ['wavenumber'];
    const data: number[][] = [wavenumber];

    if (trace?.wavelength) {
      columns.push('wavelength');
      data.push(trace.wavelength);
    }
    if (trace?.channel_indices.length) {
      columns.push('channel');
      data.push(trace.channel_indices);
    }
    columns.push('intensity');
    data.push(intensity);

    if (hasSpikeMask) {
      columns.push('spike_mask');
      data.push(spikeMask!.map(v => v ? 1 : 0));
    }

    if (hasBadpix) {
      // 0/1 per served row: 1 where a served position is annotated known-noisy.
      const badpixCol = new Array<number>(channelCount).fill(0);
      for (const b of badpix!) {
        if (b.position >= 0 && b.position < badpixCol.length) badpixCol[b.position] = 1;
      }
      columns.push('badpix');
      data.push(badpixCol);
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
    const n = channelCount;
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
    const source = buildProvenanceJson().source;
    const safeTarget = String(source.target || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_');
    const safeScan = String(source.scan_name || source.scan_id).replace(/[^a-zA-Z0-9_-]/g, '_');
    return `${safeTarget}_sol${source.sol}_${safeScan}_${stage}_${ts}.${ext}`;
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
    const json = JSON.stringify(evidenceMode ? evidenceExport(evidenceDisplays, identification, evidenceShared) : buildProvenanceJson(), null, 2);
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
    disabled={evidenceMode ? evidenceDisplays.length === 0 : channelCount === 0}
  >
    Export &#9660;
  </button>

  {#if open}
    <!-- svelte-ignore a11y-click-events-have-key-events -->
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="export-dropdown" on:click|stopPropagation>
      <button class="dropdown-item" on:click={downloadCsv}>
        <span class="dropdown-icon">&#128196;</span>
        {evidenceMode ? 'Comparison CSV' : 'CSV at current stage'}
        <span class="dropdown-hint">{stage}</span>
      </button>
      {#if exportFigure}
        <button class="dropdown-item" on:click={() => { exportFigure?.('svg'); closeMenu(); }}>SVG + provenance</button>
        <button class="dropdown-item" on:click={() => { exportFigure?.('png'); closeMenu(); }}>PNG + provenance</button>
      {/if}
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
        <pre class="provenance-json">{JSON.stringify(evidenceMode ? evidenceExport(evidenceDisplays, identification, evidenceShared) : buildProvenanceJson(), null, 2)}</pre>
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
