<script lang="ts">
  import { onMount, afterUpdate, onDestroy } from 'svelte';
  import type { Peak } from '../lib/types';
  import { buildPeakElements } from '../lib/spectrumLabels';

  // Dynamic import for plotly (same pattern as SpectrumPlot.svelte)
  let Plotly: typeof import('plotly.js-basic-dist-min') | null = null;

  export let wavenumber: number[] = [];
  export let intensity: number[] = [];     // current processed spectrum
  export let rawIntensity: number[] = [];  // original raw for overlay
  export let stage: string = 'raw';        // current processing stage
  // Overlays (all optional)
  export let spikeMask: boolean[] | null = null;    // red triangles at spike locations
  export let background: number[] | null = null;     // dashed gray background line
  export let backgroundScaled: number[] | null = null;
  export let baseline: number[] | null = null;       // dashed red baseline curve
  export let fitCurve: number[] | null = null;       // green fit curve
  export let peaks: Peak[] = [];                     // peak annotations
  export let residual: number[] | null = null;
  // Display toggles
  export let showRaw: boolean = false;
  export let showResidual: boolean = false;
  export let title: string = '';
  export let height: number = 420;
  export let fitRange: [number, number] | null = null;
  export let wavelength: number[] | null = null;
  export let region: string = 'R1';

  // Use wavelength for x-axis on non-R1 regions
  $: useWavelength = region !== 'R1' && wavelength !== null && wavelength.length > 0;
  $: xData = useWavelength ? wavelength! : wavenumber;
  $: xLabel = useWavelength ? 'Wavelength (nm)' : 'Raman Shift (cm\u207B\u00B9)';

  let plotDiv: HTMLDivElement;
  let plotInitialized = false;

  function _validRange(minStr: string, maxStr: string): boolean {
    return minStr !== '' && maxStr !== '' && !isNaN(parseFloat(minStr)) && !isNaN(parseFloat(maxStr));
  }

  // Overlay visibility toggles (user can override for export)
  let showOverlayRaw = true;
  let showOverlayBaseline = true;
  let showOverlayFit = true;
  let showOverlayPeaks = true;
  let showOverlayPeakLabels = true;
  let showOverlayResidual = true;
  let showOverlaySpikes = true;
  let showOverlayBg = true;
  let showGridlines = true;
  let showPointLabel = false;
  let showProcessingLabel = false;

  // Axis limits (empty string = auto)
  let xMin = '';
  let xMax = '';
  let yMin = '';
  let yMax = '';

  // Point selection info (for display)
  export let selectionMode: 'average' | 'subset' | 'point' = 'average';
  export let selectedIndices: number[] = [];
  export let selectedPointIdx: number | null = null;
  export let nPoints: number = 0;
  export let averagingMethod: string = 'trim_mean';

  // Stage display names
  const stageLabels: Record<string, string> = {
    raw: 'raw',
    despiked: 'despiked',
    bg_subtracted: 'bg subtracted',
    baseline_corrected: 'baseline corrected',
    raman_fitted: 'fitted',
  };

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

    // Preserve current axis ranges from interactive zoom/pan,
    // but only if the data length hasn't changed (same region).
    const curLayout = (plotDiv as unknown as { layout?: Record<string, unknown> }).layout;
    const prevNChannels = curLayout?._nChannels as number | undefined;
    const sameRegion = prevNChannels === xData.length;
    const prevXRange = sameRegion && curLayout?.xaxis
      ? (curLayout.xaxis as Record<string, unknown>).range as [number, number] | undefined
      : undefined;
    const prevYRange = sameRegion && curLayout?.yaxis
      ? (curLayout.yaxis as Record<string, unknown>).range as [number, number] | undefined
      : undefined;

    const traces: Plotly.Data[] = [];

    // Main trace: current processed spectrum
    traces.push({
      x: xData,
      y: intensity,
      type: 'scatter',
      mode: 'lines',
      name: stage === 'raw' ? 'raw' : 'processed',
      line: { color: '#1e293b', width: 1.4 },
      hovertemplate: useWavelength
        ? 'Wavelength: %{x:.1f} nm<br>Intensity: %{y:.1f}<extra></extra>'
        : 'Raman Shift: %{x:.1f} cm\u207B\u00B9<br>Intensity: %{y:.1f}<extra></extra>',
    });

    // Raw overlay (faded)
    if (showRaw && showOverlayRaw && rawIntensity.length > 0 && stage !== 'raw') {
      traces.push({
        x: xData,
        y: rawIntensity,
        type: 'scatter',
        mode: 'lines',
        name: 'raw',
        line: { color: '#94a3b8', width: 0.8 },
        opacity: 0.5,
      });
    }

    // Background overlay (dashed gray)
    if (showOverlayBg && background && (stage === 'bg_subtracted' || stage === 'despiked')) {
      traces.push({
        x: xData,
        y: background,
        type: 'scatter',
        mode: 'lines',
        name: 'background',
        line: { color: '#6b7280', width: 1.5, dash: 'dash' },
      });
    }

    // Background scaled overlay
    if (showOverlayBg && backgroundScaled && stage === 'bg_subtracted') {
      traces.push({
        x: xData,
        y: backgroundScaled,
        type: 'scatter',
        mode: 'lines',
        name: 'bg (scaled)',
        line: { color: '#9ca3af', width: 1.2, dash: 'dot' },
      });
    }

    // Baseline overlay (dashed red)
    if (showOverlayBaseline && baseline && (stage === 'baseline_corrected' || stage === 'raman_fitted')) {
      traces.push({
        x: xData,
        y: baseline,
        type: 'scatter',
        mode: 'lines',
        name: 'baseline',
        line: { color: '#dc2626', width: 1.5, dash: 'dash' },
      });
    }

    // Fit curve (green) — only within the fitting range
    if (showOverlayFit && fitCurve && stage === 'raman_fitted') {
      let fitX = xData;
      let fitY = fitCurve;
      if (fitRange) {
        const indices: number[] = [];
        for (let i = 0; i < wavenumber.length; i++) {
          if (wavenumber[i] >= fitRange[0] && wavenumber[i] <= fitRange[1]) {
            indices.push(i);
          }
        }
        fitX = indices.map((i) => xData[i]);
        fitY = indices.map((i) => fitCurve[i]);
      }
      traces.push({
        x: fitX,
        y: fitY,
        type: 'scatter',
        mode: 'lines',
        name: 'fit',
        line: { color: '#16a34a', width: 1.8 },
      });
    }

    // Spike markers: red triangles at positions where spikeMask is true
    if (showOverlaySpikes && spikeMask && spikeMask.length === wavenumber.length) {
      const spikeX: number[] = [];
      const spikeY: number[] = [];
      for (let i = 0; i < spikeMask.length; i++) {
        if (spikeMask[i]) {
          spikeX.push(xData[i]);
          spikeY.push(intensity[i]);
        }
      }
      if (spikeX.length > 0) {
        traces.push({
          x: spikeX,
          y: spikeY,
          type: 'scatter',
          mode: 'markers',
          name: `Spikes (${spikeX.length})`,
          marker: {
            symbol: 'triangle-up',
            size: 8,
            color: '#dc2626',
            line: { color: '#991b1b', width: 1 },
          },
          hovertemplate: 'Spike at %{x:.1f} cm\u207B\u00B9<br>Intensity: %{y:.1f}<extra></extra>',
        });
      }
    }

    // Peak vertical-line markers (shapes, gated by showOverlayPeaks) +
    // peak text annotations (gated by showOverlayPeakLabels, new in issue #18).
    // The two toggles are independent — operators can show labels without
    // the vertical-line clutter, or vice versa.
    const peakElements = buildPeakElements(peaks, {
      showFit: showOverlayPeaks,
      showPeakLabels: showOverlayPeakLabels,
      shapeY1: 0.95,
      labelFormat: 'paren',
    });
    const shapes = peakElements.shapes as Partial<Plotly.Shape>[];
    const annotations = peakElements.annotations as unknown as Partial<Plotly.Annotations>[];

    // Individual peak colors (cycle through for multiple peaks)
    const peakColors = ['#7c3aed', '#0891b2', '#c026d3', '#ea580c', '#4f46e5'];

    for (let pi = 0; pi < (showOverlayPeaks ? peaks : []).length; pi++) {
      const peak = peaks[pi];
      if (peak.center_cm1 === null) continue;

      // Individual Gaussian curve for this peak
      if (peak.fwhm_cm1 !== null && peak.amplitude > 0) {
        const sigma = peak.fwhm_cm1 / (2 * Math.sqrt(2 * Math.log(2)));
        const c = peak.center_cm1;
        const a = peak.amplitude;
        const color = peakColors[pi % peakColors.length];
        // Generate curve over ±4*FWHM around center
        const lo = Math.max(c - 4 * peak.fwhm_cm1, fitRange ? fitRange[0] : 640);
        const hi = Math.min(c + 4 * peak.fwhm_cm1, fitRange ? fitRange[1] : 4200);
        const nPts = 200;
        const step = (hi - lo) / nPts;
        const pX: number[] = [];
        const pY: number[] = [];
        for (let j = 0; j <= nPts; j++) {
          const xj = lo + j * step;
          pX.push(xj);
          // Gaussian: a * exp(-0.5 * ((x - c) / sigma)^2)
          const z = (xj - c) / sigma;
          pY.push(a * Math.exp(-0.5 * z * z));
        }
        traces.push({
          x: pX,
          y: pY,
          type: 'scatter',
          mode: 'lines',
          name: peak.mineral_assignment || `Peak ${peak.center_cm1.toFixed(0)}`,
          line: { color, width: 1.2, dash: 'dash' },
          showlegend: false,
        });
      }
    }

    // Point label annotation on the figure
    if (showPointLabel) {
      let pointText = '';
      if (selectionMode === 'point' && selectedPointIdx !== null) {
        pointText = `Point ${selectedPointIdx}`;
      } else if (selectionMode === 'subset' && selectedIndices.length > 0) {
        pointText = `Points ${[...selectedIndices].sort((a, b) => a - b).join(', ')}`;
      } else {
        pointText = `Average (${nPoints} pts)`;
      }
      annotations.push({
        x: 0,
        y: 0,
        xref: 'paper',
        yref: 'paper',
        xanchor: 'left',
        yanchor: 'top',
        text: pointText,
        showarrow: false,
        font: { size: 10, color: '#64748b', family: 'JetBrains Mono, monospace' },
        yshift: -28,
      });
    }

    // Processing label annotation on the figure
    if (showProcessingLabel) {
      const stageText = stageLabels[stage] || stage;
      const avgText = selectionMode !== 'point' ? ` · ${averagingMethod.replace('_', ' ')}` : '';
      annotations.push({
        x: 1,
        y: 0,
        xref: 'paper',
        yref: 'paper',
        xanchor: 'right',
        yanchor: 'top',
        text: `${stageText}${avgText}`,
        showarrow: false,
        font: { size: 10, color: '#64748b', family: 'JetBrains Mono, monospace' },
        yshift: -28,
      });
    }

    const mainLayout: Partial<Plotly.Layout> & { _nChannels?: number } = {
      _nChannels: xData.length,
      title: title ? { text: title, font: { size: 14 } } : undefined,
      xaxis: {
        title: { text: xLabel, font: { size: 12 } },
        tickfont: { family: 'JetBrains Mono, monospace', size: 10 },
        gridcolor: '#f1f5f9',
        showgrid: showGridlines,
        showline: true,
        linecolor: '#cbd5e1',
        ticks: 'outside',
        tickcolor: '#cbd5e1',
        ...(_validRange(xMin, xMax)
          ? { range: [parseFloat(xMin), parseFloat(xMax)], autorange: false }
          : prevXRange
            ? { range: [...prevXRange], autorange: false }
            : { autorange: true }),
      },
      yaxis: {
        title: { text: 'Intensity (counts)', font: { size: 12 } },
        tickfont: { family: 'JetBrains Mono, monospace', size: 10 },
        gridcolor: '#f1f5f9',
        showgrid: showGridlines,
        showline: true,
        linecolor: '#cbd5e1',
        ticks: 'outside',
        tickcolor: '#cbd5e1',
        ...(_validRange(yMin, yMax)
          ? { range: [parseFloat(yMin), parseFloat(yMax)], autorange: false }
          : { autorange: true }),
      },
      margin: { l: 65, r: 20, t: title ? 40 : 20, b: 50 },
      height: showResidual && residual ? height * 0.7 : height,
      shapes,
      annotations,
      hovermode: 'x unified',
      plot_bgcolor: '#ffffff',
      paper_bgcolor: '#ffffff',
      font: { family: 'Inter, system-ui, sans-serif' },
      showlegend: traces.length > 1,
      legend: {
        x: 1,
        y: 1,
        xanchor: 'right',
        bgcolor: 'rgba(255,255,255,0.85)',
        bordercolor: '#e2e8f0',
        borderwidth: 1,
        font: { size: 11 },
      },
    };

    const config: Partial<Plotly.Config> = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      toImageButtonOptions: {
        format: 'png',
        filename: `sherloc_raman_${stage}`,
        height: 600,
        width: 1000,
        scale: 2,
      },
      displaylogo: false,
    };

    if (showResidual && showOverlayResidual && residual) {
      // Subplot: main spectrum + residual below
      const residualTrace: Plotly.Data = {
        x: xData,
        y: residual,
        type: 'scatter',
        mode: 'lines',
        name: 'residual',
        line: { color: '#64748b', width: 1 },
        xaxis: 'x2',
        yaxis: 'y2',
      };
      traces.push(residualTrace);

      // Zero line for residual
      const residualZero: Plotly.Data = {
        x: [xData[0], xData[xData.length - 1]],
        y: [0, 0],
        type: 'scatter',
        mode: 'lines',
        name: 'Zero',
        line: { color: '#d1d5db', width: 0.8, dash: 'dash' },
        xaxis: 'x2',
        yaxis: 'y2',
        showlegend: false,
      };
      traces.push(residualZero);

      const subplotLayout = {
        ...mainLayout,
        height,
        xaxis: {
          ...mainLayout.xaxis,
          domain: [0, 1],
          anchor: 'y' as Plotly.YAxisName,
        },
        yaxis: {
          ...mainLayout.yaxis,
          domain: [0.35, 1],
          anchor: 'x' as Plotly.XAxisName,
        },
        xaxis2: {
          title: { text: xLabel, font: { size: 12 } },
          tickfont: { family: 'JetBrains Mono, monospace', size: 10 },
          gridcolor: '#f1f5f9',
          showline: true,
          linecolor: '#cbd5e1',
          ticks: 'outside',
          tickcolor: '#cbd5e1',
          domain: [0, 1],
          anchor: 'y2' as Plotly.YAxisName,
          range: [640, 4200],
          autorange: false,
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
          anchor: 'x2' as Plotly.XAxisName,
        },
      };

      Plotly.react(plotDiv, traces, subplotLayout as Plotly.Layout, config);
    } else {
      Plotly.react(plotDiv, traces, mainLayout as Plotly.Layout, config);
    }
  }

  function downloadAs(format: 'png' | 'svg') {
    if (!Plotly || !plotDiv) return;
    Plotly.downloadImage(plotDiv, {
      format,
      filename: `sherloc_raman_${stage}`,
      height: 600,
      width: 1000,
      scale: 2,
    } as Plotly.DownloadImgopts);
  }
</script>

<div class="raman-view">
  <div bind:this={plotDiv} class="plot-div"></div>
  {#if wavenumber.length === 0}
    <div class="plot-empty">No spectrum data loaded</div>
  {:else}
    <div class="axis-controls">
      <span class="axis-label">x:</span>
      <input class="axis-input" type="number" bind:value={xMin} on:change={renderPlot} placeholder="640" />
      <span class="axis-sep">-</span>
      <input class="axis-input" type="number" bind:value={xMax} on:change={renderPlot} placeholder="4200" />
      <span class="axis-label" style="margin-left: 8px">y:</span>
      <input class="axis-input" type="number" bind:value={yMin} on:change={renderPlot} placeholder="auto" />
      <span class="axis-sep">-</span>
      <input class="axis-input" type="number" bind:value={yMax} on:change={renderPlot} placeholder="auto" />
    </div>
    <div class="overlay-toggles">
      <label class="toggle-label"><input type="checkbox" bind:checked={showGridlines} on:change={renderPlot} /> Grid</label>
      <label class="toggle-label"><input type="checkbox" bind:checked={showPointLabel} on:change={renderPlot} /> Point label</label>
      <label class="toggle-label"><input type="checkbox" bind:checked={showProcessingLabel} on:change={renderPlot} /> Processing</label>
      {#if showRaw && rawIntensity.length > 0 && stage !== 'raw'}
        <label class="toggle-label"><input type="checkbox" bind:checked={showOverlayRaw} on:change={renderPlot} /> Raw</label>
      {/if}
      {#if background && (stage === 'bg_subtracted' || stage === 'despiked')}
        <label class="toggle-label"><input type="checkbox" bind:checked={showOverlayBg} on:change={renderPlot} /> BG</label>
      {/if}
      {#if baseline && (stage === 'baseline_corrected' || stage === 'raman_fitted')}
        <label class="toggle-label"><input type="checkbox" bind:checked={showOverlayBaseline} on:change={renderPlot} /> Baseline</label>
      {/if}
      {#if fitCurve && stage === 'raman_fitted'}
        <label class="toggle-label"><input type="checkbox" bind:checked={showOverlayFit} on:change={renderPlot} /> Fit</label>
      {/if}
      {#if peaks.length > 0}
        <label class="toggle-label"><input type="checkbox" bind:checked={showOverlayPeaks} on:change={renderPlot} /> Fitted peaks</label>
        <label class="toggle-label"><input type="checkbox" bind:checked={showOverlayPeakLabels} on:change={renderPlot} /> Peak labels</label>
      {/if}
      {#if showResidual && residual}
        <label class="toggle-label"><input type="checkbox" bind:checked={showOverlayResidual} on:change={renderPlot} /> Residual</label>
      {/if}
      {#if spikeMask}
        <label class="toggle-label"><input type="checkbox" bind:checked={showOverlaySpikes} on:change={renderPlot} /> Spikes</label>
      {/if}
    </div>
    <div class="export-bar">
      <span class="stage-badge">{stageLabels[stage] || stage}{selectionMode !== 'point' ? ` · ${averagingMethod.replace('_', ' ')}` : ''}</span>
      <span class="point-info mono">
        {#if selectionMode === 'point' && selectedPointIdx !== null}
          Point {selectedPointIdx}
        {:else if selectionMode === 'subset' && selectedIndices.length > 0}
          Points {[...selectedIndices].sort((a, b) => a - b).join(', ')}
        {:else}
          {nPoints} Points
        {/if}
      </span>
      <span class="export-label">Export plot:</span>
      <button class="btn-export" on:click={() => downloadAs('png')} title="Download as PNG">PNG</button>
      <button class="btn-export" on:click={() => downloadAs('svg')} title="Download as SVG">SVG</button>
    </div>
  {/if}
</div>

<style>
  .raman-view {
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

  .stage-badge {
    font-size: 0.75rem;
    font-family: var(--font-mono);
    padding: 2px 8px;
    border-radius: 9999px;
    background: var(--color-primary-light);
    color: var(--color-primary);
  }

  .point-info {
    font-size: 0.72rem;
    color: var(--color-text-secondary);
    margin-right: auto;
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

  .overlay-toggles {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: flex-end;
    margin-top: 2px;
  }

  .toggle-label {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-size: 0.72rem;
    color: var(--color-text-secondary);
    cursor: pointer;
    margin-bottom: 0;
    white-space: nowrap;
    user-select: none;
  }

  .toggle-label input[type='checkbox'] {
    cursor: pointer;
    margin: 0;
  }

  .axis-controls {
    display: flex;
    align-items: center;
    gap: 3px;
    justify-content: flex-end;
    margin-top: 2px;
  }

  .axis-label {
    font-size: 0.72rem;
    font-family: var(--font-mono);
    color: var(--color-text-secondary);
    font-weight: 500;
  }

  .axis-input {
    width: 58px;
    font-size: 0.72rem;
    font-family: var(--font-mono);
    padding: 1px 4px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    text-align: center;
  }

  .axis-input::placeholder {
    color: var(--color-text-tertiary);
  }

  .axis-sep {
    font-size: 0.72rem;
    color: var(--color-text-tertiary);
  }
</style>
