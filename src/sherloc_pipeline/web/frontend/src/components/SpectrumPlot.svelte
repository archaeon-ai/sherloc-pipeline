<script lang="ts">
  import { onMount, afterUpdate, onDestroy } from 'svelte';
  import type { Peak } from '../lib/types';
  import { buildPeakElements } from '../lib/spectrumLabels';

  // We use dynamic import for plotly to handle SSR gracefully
  let Plotly: typeof import('plotly.js-basic-dist-min') | null = null;

  export let wavenumber: number[] = [];
  export let intensity: number[] = [];
  export let baseline: number[] | null = null;
  export let corrected: number[] | null = null;
  export let residual: number[] | null = null;
  export let fitCurve: number[] | null = null;
  export let peaks: Peak[] = [];
  export let xLabel: string = 'Raman Shift (cm\u207B\u00B9)';
  export let yLabel: string = 'Intensity (counts)';
  export let title: string = '';
  export let showResidual: boolean = false;
  export let showBaseline: boolean = false;
  export let showFit: boolean = false;
  export let showPeakLabels: boolean = true;
  export let compact: boolean = false;
  export let height: number = 400;

  let plotDiv: HTMLDivElement;
  let plotInitialized = false;

  onMount(async () => {
    Plotly = await import('plotly.js-basic-dist-min');
    plotInitialized = true;
    renderPlot();
  });

  onDestroy(() => {
    if (plotDiv && Plotly) {
      try {
        Plotly.purge(plotDiv);
      } catch {
        // ignore
      }
    }
  });

  afterUpdate(() => {
    if (plotInitialized) renderPlot();
  });

  function renderPlot() {
    if (!Plotly || !plotDiv || wavenumber.length === 0) return;

    const traces: Plotly.Data[] = [];

    // Main spectrum trace
    const yData = corrected && showFit ? corrected : intensity;
    traces.push({
      x: wavenumber,
      y: yData,
      type: 'scatter',
      mode: 'lines',
      name: corrected && showFit ? 'Baseline-corrected' : 'Spectrum',
      line: { color: '#1e293b', width: 1.2 },
      hovertemplate: `${xLabel}: %{x:.1f}<br>${yLabel}: %{y:.1f}<extra></extra>`,
    });

    // Baseline overlay
    if (baseline && showBaseline && !showFit) {
      traces.push({
        x: wavenumber,
        y: baseline,
        type: 'scatter',
        mode: 'lines',
        name: 'Baseline',
        line: { color: '#dc2626', width: 1.5, dash: 'dash' },
      });
    }

    // Fit curve overlay
    if (fitCurve && showFit) {
      traces.push({
        x: wavenumber,
        y: fitCurve,
        type: 'scatter',
        mode: 'lines',
        name: 'Fit',
        line: { color: '#dc2626', width: 1.8 },
      });
    }

    // Peak vertical-line markers (shapes, gated by showFit) +
    // peak text annotations (gated by showPeakLabels) — toggles are independent.
    const peakElements = buildPeakElements(peaks, { showFit, showPeakLabels });
    const shapes = peakElements.shapes as Partial<Plotly.Shape>[];
    const annotations = peakElements.annotations as unknown as Partial<Plotly.Annotations>[];

    const mainLayout: Partial<Plotly.Layout> = {
      title: title ? { text: title, font: { size: 14 } } : undefined,
      xaxis: {
        title: compact ? undefined : { text: xLabel, font: { size: 12 } },
        tickfont: { family: 'JetBrains Mono, monospace', size: compact ? 9 : 10 },
        gridcolor: '#f1f5f9',
        showline: true,
        linecolor: '#cbd5e1',
        ticks: 'outside',
        tickcolor: '#cbd5e1',
      },
      yaxis: {
        title: compact ? undefined : { text: yLabel, font: { size: 12 } },
        tickfont: { family: 'JetBrains Mono, monospace', size: 10 },
        gridcolor: '#f1f5f9',
        showline: true,
        linecolor: '#cbd5e1',
        ticks: 'outside',
        tickcolor: '#cbd5e1',
      },
      margin: compact
        ? { l: 45, r: 10, t: 5, b: 35 }
        : { l: 60, r: 20, t: title ? 40 : 20, b: 50 },
      height: showResidual && residual ? height * 0.7 : height,
      shapes,
      annotations,
      hovermode: 'x unified',
      plot_bgcolor: '#ffffff',
      paper_bgcolor: '#ffffff',
      font: { family: 'Inter, system-ui, sans-serif' },
      showlegend: traces.length > 1,
      legend: { x: 1, y: 1, xanchor: 'right', bgcolor: 'rgba(255,255,255,0.8)' },
    };

    const config: Partial<Plotly.Config> = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      toImageButtonOptions: {
        format: 'png',
        filename: 'sherloc_spectrum',
        height: 600,
        width: 1000,
        scale: 2,
      },
      displaylogo: false,
    };

    if (showResidual && residual) {
      // Create subplots: main + residual
      const residualTrace: Plotly.Data = {
        x: wavenumber,
        y: residual,
        type: 'scatter',
        mode: 'lines',
        name: 'Residual',
        line: { color: '#64748b', width: 1 },
        xaxis: 'x2',
        yaxis: 'y2',
      };
      traces.push(residualTrace);

      const subplotLayout = {
        ...mainLayout,
        height,
        xaxis: { ...mainLayout.xaxis, domain: [0, 1], anchor: 'y' },
        yaxis: { ...mainLayout.yaxis, domain: [0.35, 1], anchor: 'x' },
        xaxis2: {
          title: { text: xLabel, font: { size: 12 } },
          tickfont: { family: 'JetBrains Mono, monospace', size: 10 },
          gridcolor: '#f1f5f9',
          showline: true,
          linecolor: '#cbd5e1',
          ticks: 'outside',
          tickcolor: '#cbd5e1',
          domain: [0, 1],
          anchor: 'y2',
        },
        yaxis2: {
          title: { text: 'Residual', font: { size: 12 } },
          tickfont: { family: 'JetBrains Mono, monospace', size: 10 },
          gridcolor: '#f1f5f9',
          showline: true,
          linecolor: '#cbd5e1',
          ticks: 'outside',
          tickcolor: '#cbd5e1',
          domain: [0, 0.25],
          anchor: 'x2',
        },
      };

      Plotly.react(plotDiv, traces, subplotLayout as Plotly.Layout, config);
    } else {
      Plotly.react(plotDiv, traces, mainLayout as Plotly.Layout, config);
    }
  }

  function downloadAs(format: 'png' | 'svg' | 'pdf') {
    if (!Plotly || !plotDiv) return;
    if (format === 'pdf') {
      // Plotly.js doesn't support PDF directly — export SVG and let user print/convert
      // Fall back to SVG for PDF request
      format = 'svg';
    }
    Plotly.downloadImage(plotDiv, {
      format,
      filename: `sherloc_spectrum`,
      height: 600,
      width: 1000,
      scale: 2,
    } as Plotly.DownloadImgopts);
  }
</script>

<div class="plot-container">
  <div bind:this={plotDiv} class="plot-div"></div>
  {#if wavenumber.length === 0}
    <div class="plot-empty">No spectrum data</div>
  {:else}
    <div class="export-bar">
      <span class="export-label">Export plot:</span>
      <button class="btn-export" on:click={() => downloadAs('png')} title="Download as PNG">PNG</button>
      <button class="btn-export" on:click={() => downloadAs('svg')} title="Download as SVG">SVG</button>
    </div>
  {/if}
</div>

<style>
  .plot-container {
    position: relative;
    width: 100%;
  }

  .plot-div {
    width: 100%;
  }

  .plot-empty {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: var(--color-text-tertiary);
    font-size: 0.9rem;
  }

  .export-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    justify-content: flex-end;
    margin-top: 4px;
  }

  .export-label {
    font-size: 0.75rem;
    color: var(--color-text-tertiary);
  }

  .btn-export {
    padding: 2px 8px;
    font-size: 0.75rem;
    font-family: var(--font-mono);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    color: var(--color-text-secondary);
    cursor: pointer;
  }

  .btn-export:hover {
    border-color: var(--color-primary);
    color: var(--color-primary);
  }
</style>
