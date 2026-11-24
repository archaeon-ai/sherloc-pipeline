// Every RGB preset must name class ids the map actually has as layers;
// a preset id that matches no layer leaves its channel empty.

import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import { get } from 'svelte/store';
import RGBMixPanel, { PRESETS } from './RGBMixPanel.svelte';
import { DOMAIN_CLASSES } from '../../lib/mapClasses';
import { mapDisplayMode, mapLayers } from '../../lib/stores/mapStore';
import type { ScalarLayer } from '../../lib/types/map';

function layersFromDomainClasses(): ScalarLayer[] {
  const layers: ScalarLayer[] = [];
  for (const [domain, classes] of Object.entries(DOMAIN_CLASSES)) {
    for (const cls of classes) {
      layers.push({
        id: `${domain}:${cls}:snr`,
        point_set_id: 'scan',
        label: cls,
        domain,
        class_id: cls,
        value_type: 'snr',
        values: [],
        colormap: { type: 'sequential', name: 'viridis', range: [0, 20] },
        opacity: 1,
        visible: true,
      } as ScalarLayer);
    }
  }
  return layers;
}

describe('RGBMixPanel presets', () => {
  beforeEach(() => {
    mapLayers.set(layersFromDomainClasses());
  });

  it('names only class ids that are map layers', () => {
    for (const preset of PRESETS) {
      for (const ref of [preset.red, preset.green, preset.blue]) {
        expect(DOMAIN_CLASSES[ref.domain] ?? [], `${preset.name}: ${ref.domain}/${ref.class_id}`)
          .toContain(ref.class_id);
      }
    }
  });

  it.each(PRESETS.map((p) => [p.name, p]))('%s fills all three channels', async (_name, preset) => {
    render(RGBMixPanel);
    await fireEvent.click(screen.getByRole('button', { name: preset.name }));
    const mode = get(mapDisplayMode);
    expect(mode.type).toBe('rgb_mix');
    if (mode.type !== 'rgb_mix') return;
    expect(mode.channels.red?.class_id).toBe(preset.red.class_id);
    expect(mode.channels.green?.class_id).toBe(preset.green.class_id);
    expect(mode.channels.blue?.class_id).toBe(preset.blue.class_id);
  });
});
