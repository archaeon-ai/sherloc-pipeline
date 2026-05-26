<script lang="ts">
  import { mapDisplayMode, mapLayers } from '../../lib/stores/mapStore';
  import type { LayerRef, RGBChannelAssignment, ScalarLayer } from '../../lib/types/map';

  // Built-in presets
  const PRESETS: { name: string; red: LayerRef; green: LayerRef; blue: LayerRef }[] = [
    {
      name: 'Sulfate Mineralogy',
      red: { domain: 'minerals', class_id: 'sulfate_v1', value_type: 'snr' },
      green: { domain: 'minerals', class_id: 'hi_carb', value_type: 'snr' },
      blue: { domain: 'fluorescence', class_id: 'group1a', value_type: 'snr' },
    },
    {
      name: 'Organics + Hydration',
      red: { domain: 'organics', class_id: 'D_band', value_type: 'snr' },
      green: { domain: 'organics', class_id: 'G_band', value_type: 'snr' },
      blue: { domain: 'hydration', class_id: 'oh_stretch', value_type: 'snr' },
    },
    {
      name: 'Fluorescence Groups',
      red: { domain: 'fluorescence', class_id: 'group1a', value_type: 'snr' },
      green: { domain: 'fluorescence', class_id: 'group2', value_type: 'snr' },
      blue: { domain: 'fluorescence', class_id: 'group3', value_type: 'snr' },
    },
  ];

  // Channel state
  let redValue = '';
  let greenValue = '';
  let blueValue = '';

  // Range sliders per channel (0-1 min/max)
  let redMin = 0;
  let redMax = 1;
  let greenMin = 0;
  let greenMax = 1;
  let blueMin = 0;
  let blueMax = 1;

  // Build available layer options
  $: layerOptions = buildOptions($mapLayers);

  interface LayerOption {
    value: string;
    label: string;
    ref: LayerRef;
  }

  function buildOptions(layers: ScalarLayer[]): LayerOption[] {
    const seen = new Set<string>();
    const options: LayerOption[] = [];
    for (const layer of layers) {
      const key = `${layer.domain}:${layer.class_id ?? ''}:${layer.value_type}`;
      if (seen.has(key)) continue;
      seen.add(key);
      options.push({
        value: key,
        label: `${layer.domain} / ${layer.class_id ?? 'total'} (${layer.value_type})`,
        ref: {
          domain: layer.domain,
          class_id: layer.class_id,
          value_type: layer.value_type,
        },
      });
    }
    return options;
  }

  function refToValue(ref: LayerRef | null): string {
    if (!ref) return '';
    return `${ref.domain}:${ref.class_id ?? ''}:${ref.value_type}`;
  }

  function valueToRef(value: string): LayerRef | null {
    const opt = layerOptions.find((o) => o.value === value);
    return opt?.ref ?? null;
  }

  function updateDisplayMode() {
    const channels: RGBChannelAssignment = {
      red: valueToRef(redValue),
      green: valueToRef(greenValue),
      blue: valueToRef(blueValue),
    };
    mapDisplayMode.set({ type: 'rgb_mix', channels });
  }

  function applyPreset(preset: typeof PRESETS[number]) {
    redValue = refToValue(preset.red);
    greenValue = refToValue(preset.green);
    blueValue = refToValue(preset.blue);
    // Reset ranges
    redMin = 0; redMax = 1;
    greenMin = 0; greenMax = 1;
    blueMin = 0; blueMax = 1;
    updateDisplayMode();
  }

  // Reactively update display mode when channel selections change
  $: if (redValue || greenValue || blueValue) {
    updateDisplayMode();
  }

  // When range sliders change, update the corresponding layer's colormap range.
  // Slider values are 0-1 normalized; they scale the layer's full data range.
  $: updateLayerRanges(redMin, redMax, greenMin, greenMax, blueMin, blueMax);

  function updateLayerRanges(..._deps: number[]) {
    void _deps; // trigger reactivity
    const channelRefs = [
      { value: redValue, min: redMin, max: redMax },
      { value: greenValue, min: greenMin, max: greenMax },
      { value: blueValue, min: blueMin, max: blueMax },
    ];

    mapLayers.update((currentLayers) => {
      let changed = false;
      const updated = currentLayers.map((layer) => {
        const key = `${layer.domain}:${layer.class_id ?? ''}:${layer.value_type}`;
        for (const ch of channelRefs) {
          if (ch.value === key) {
            // Compute the full data range from measured values
            const measured = layer.values
              .filter((v) => v.status === 'measured' && v.value != null)
              .map((v) => v.value as number);
            if (measured.length === 0) break;
            const dataMin = Math.min(...measured);
            const dataMax = Math.max(...measured);
            const span = dataMax - dataMin;
            const newMin = dataMin + ch.min * span;
            const newMax = dataMin + ch.max * span;
            if (
              Math.abs(layer.colormap.range[0] - newMin) > 0.01 ||
              Math.abs(layer.colormap.range[1] - newMax) > 0.01
            ) {
              changed = true;
              return {
                ...layer,
                colormap: { ...layer.colormap, range: [newMin, newMax] as [number, number] },
              };
            }
            break;
          }
        }
        return layer;
      });
      return changed ? updated : currentLayers;
    });
  }
</script>

<div class="rgb-panel">
  <div class="section-header">RGB Channel Mix</div>

  <!-- Presets -->
  <div class="presets">
    <span class="preset-label">Presets:</span>
    {#each PRESETS as preset}
      <button class="preset-btn" on:click={() => applyPreset(preset)}>
        {preset.name}
      </button>
    {/each}
  </div>

  <!-- Channel selectors -->
  <div class="channel-row">
    <span class="channel-indicator red"></span>
    <select class="channel-select" bind:value={redValue} on:change={updateDisplayMode}>
      <option value="">-- None --</option>
      {#each layerOptions as opt}
        <option value={opt.value}>{opt.label}</option>
      {/each}
    </select>
  </div>

  <div class="channel-row">
    <span class="channel-indicator green"></span>
    <select class="channel-select" bind:value={greenValue} on:change={updateDisplayMode}>
      <option value="">-- None --</option>
      {#each layerOptions as opt}
        <option value={opt.value}>{opt.label}</option>
      {/each}
    </select>
  </div>

  <div class="channel-row">
    <span class="channel-indicator blue"></span>
    <select class="channel-select" bind:value={blueValue} on:change={updateDisplayMode}>
      <option value="">-- None --</option>
      {#each layerOptions as opt}
        <option value={opt.value}>{opt.label}</option>
      {/each}
    </select>
  </div>

  <!-- Per-channel stretch -->
  <div class="range-section">
    <div class="range-header">
      <span class="section-subheader">Channel Stretch</span>
      <span class="range-hint">Clip low / saturate high</span>
    </div>
    <div class="range-channel">
      <div class="range-row">
        <span class="range-label red-text">R</span>
        <input type="range" min="0" max="1" step="0.01" bind:value={redMin} class="range-slider" />
        <input type="range" min="0" max="1" step="0.01" bind:value={redMax} class="range-slider" />
      </div>
      <span class="range-values">{(redMin * 100).toFixed(0)}% – {(redMax * 100).toFixed(0)}%</span>
    </div>
    <div class="range-channel">
      <div class="range-row">
        <span class="range-label green-text">G</span>
        <input type="range" min="0" max="1" step="0.01" bind:value={greenMin} class="range-slider" />
        <input type="range" min="0" max="1" step="0.01" bind:value={greenMax} class="range-slider" />
      </div>
      <span class="range-values">{(greenMin * 100).toFixed(0)}% – {(greenMax * 100).toFixed(0)}%</span>
    </div>
    <div class="range-channel">
      <div class="range-row">
        <span class="range-label blue-text">B</span>
        <input type="range" min="0" max="1" step="0.01" bind:value={blueMin} class="range-slider" />
        <input type="range" min="0" max="1" step="0.01" bind:value={blueMax} class="range-slider" />
      </div>
      <span class="range-values">{(blueMin * 100).toFixed(0)}% – {(blueMax * 100).toFixed(0)}%</span>
    </div>
  </div>
</div>

<style>
  .rgb-panel {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .section-header {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .presets {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px;
  }

  .preset-label {
    font-size: 0.75rem;
    color: var(--color-text-tertiary);
  }

  .preset-btn {
    font-size: 0.72rem;
    padding: 2px 8px;
    background: rgba(255, 255, 255, 0.06);
    color: var(--color-text-secondary);
    border: 1px solid var(--color-border);
    border-radius: 9999px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .preset-btn:hover {
    background: var(--color-primary-light, rgba(59, 130, 246, 0.1));
    color: var(--color-primary);
    border-color: var(--color-primary);
  }

  .channel-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .channel-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .channel-indicator.red { background: #ef4444; }
  .channel-indicator.green { background: #22c55e; }
  .channel-indicator.blue { background: #3b82f6; }

  .channel-select {
    flex: 1;
    padding: 4px 6px;
    font-size: 0.78rem;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    color: var(--color-text);
  }

  .channel-select:focus {
    outline: none;
    border-color: var(--color-primary);
  }

  .range-section {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding-top: 4px;
    border-top: 1px solid var(--color-border);
  }

  .range-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 2px;
  }

  .section-subheader {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--color-text-secondary);
  }

  .range-hint {
    font-size: 0.68rem;
    color: var(--color-text-tertiary);
    font-style: italic;
  }

  .range-row {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .range-channel {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .range-values {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--color-text-tertiary);
    text-align: right;
    padding-right: 2px;
  }

  .range-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 700;
    width: 14px;
    text-align: center;
  }

  .red-text { color: #ef4444; }
  .green-text { color: #22c55e; }
  .blue-text { color: #3b82f6; }

  .range-slider {
    flex: 1;
    height: 4px;
    appearance: none;
    -webkit-appearance: none;
    background: var(--color-border);
    border-radius: 2px;
    outline: none;
  }

  .range-slider::-webkit-slider-thumb {
    appearance: none;
    -webkit-appearance: none;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--color-primary);
    cursor: pointer;
  }
</style>
