<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { mapDisplayMode, mapLayers } from '../../lib/stores/mapStore';
  import type { DisplayMode } from '../../lib/types/map';

  const dispatch = createEventDispatcher<{
    rgbMix: void;
  }>();

  // Build the hierarchical options from available layers
  interface DomainGroup {
    domain: string;
    label: string;
    classes: { class_id: string; label: string; maxSnr: number }[];
  }

  $: domainGroups = buildDomainGroups($mapLayers);

  function buildDomainGroups(
    layers: typeof $mapLayers,
  ): DomainGroup[] {
    const domainMap = new Map<string, DomainGroup>();
    const domainLabels: Record<string, string> = {
      minerals: 'Minerals',
      organics: 'Organics',
      hydration: 'Hydration',
      fluorescence: 'Fluorescence',
    };

    for (const layer of layers) {
      if (!domainMap.has(layer.domain)) {
        domainMap.set(layer.domain, {
          domain: layer.domain,
          label: domainLabels[layer.domain] ?? layer.domain,
          classes: [],
        });
      }
      if (layer.class_id) {
        const group = domainMap.get(layer.domain)!;
        if (!group.classes.find((c) => c.class_id === layer.class_id)) {
          // Compute max SNR from measured values in this layer
          const measured = layer.values.filter(
            (v) => v.status === 'measured' && v.value != null,
          );
          const maxSnr =
            measured.length > 0
              ? Math.max(...measured.map((v) => v.value!))
              : -Infinity;
          group.classes.push({
            class_id: layer.class_id,
            label: layer.label,
            maxSnr,
          });
        }
      }
    }

    // Classes are already in wavenumber/wavelength order from KNOWN_CLASSES

    return Array.from(domainMap.values());
  }

  // Encode display mode as a select value string
  $: currentValue = encodeMode($mapDisplayMode);

  function encodeMode(mode: DisplayMode): string {
    switch (mode.type) {
      case 'all_domains':
        return 'all';
      case 'domain':
        return `domain:${mode.domain}`;
      case 'class':
        return `class:${mode.domain}:${mode.class_id}`;
      case 'rgb_mix':
        return 'rgb_mix';
      default:
        return 'all';
    }
  }

  function handleChange(e: Event) {
    const value = (e.target as HTMLSelectElement).value;

    if (value === 'all') {
      mapDisplayMode.set({ type: 'all_domains' });
      return;
    }

    if (value === 'rgb_mix') {
      dispatch('rgbMix');
      // Set display mode to rgb_mix with null channels (panel will populate)
      mapDisplayMode.set({
        type: 'rgb_mix',
        channels: { red: null, green: null, blue: null },
      });
      return;
    }

    if (value.startsWith('domain:')) {
      const domain = value.slice('domain:'.length);
      mapDisplayMode.set({ type: 'domain', domain });
      return;
    }

    if (value.startsWith('class:')) {
      const parts = value.slice('class:'.length).split(':');
      mapDisplayMode.set({
        type: 'class',
        domain: parts[0],
        class_id: parts[1],
      });
      return;
    }
  }
</script>

<div class="display-mode-selector">
  <div class="section-label">Display Mode</div>
  <select class="mode-select" value={currentValue} on:change={handleChange}>
    <option value="all">Select a class</option>

    {#each domainGroups as group}
      <optgroup label={group.label}>
        {#each group.classes as cls}
          <option value="class:{group.domain}:{cls.class_id}">
            {cls.label}
          </option>
        {/each}
        <option value="domain:{group.domain}">[{group.label}] All</option>
      </optgroup>
    {/each}

    <option value="rgb_mix">RGB Mix</option>
  </select>
</div>

<style>
  .display-mode-selector {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .section-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .mode-select {
    width: 100%;
    padding: 6px 8px;
    font-size: 0.82rem;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    color: var(--color-text);
    cursor: pointer;
  }

  .mode-select:focus {
    outline: none;
    border-color: var(--color-primary);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15);
  }
</style>
