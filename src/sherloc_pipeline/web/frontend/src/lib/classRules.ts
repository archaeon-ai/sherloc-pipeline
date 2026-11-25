import classWindows from './classWindows.json';
import type { ClassificationRule } from './types/map';

export function matchRule(domain: string, center: number, rules: ClassificationRule[]) {
  let best: ClassificationRule | undefined;
  let bestLo = -Infinity;
  let bestHi = -Infinity;
  for (const rule of rules) {
    if (rule.disabled || rule.domain !== domain) continue;
    const lo = rule.center - rule.range;
    const hi = rule.center + rule.range;
    if (!(lo <= center && center <= hi)) continue;
    // core/fluor_id.py: assign_fluor_group
    if (domain !== 'minerals') return rule;
    // core/mineral_id.py: assign_min_id
    if (!best || lo > bestLo || (lo === bestLo && hi > bestHi)) {
      best = rule;
      bestLo = lo;
      bestHi = hi;
    }
  }
  return best;
}

export function classUnit(domain: string) {
  return domain === 'fluorescence' ? 'nm' : 'cm⁻¹';
}

export const defaultCenters = new Map(
  classWindows.map(({ class_id, domain, lo, hi }) => [
    class_id, { center: (lo + hi) / 2, unit: classUnit(domain) },
  ] as const),
);
defaultCenters.set('g1_doublet', { center: 315, unit: 'nm' });

export function classLabel(
  cls: string, center = defaultCenters.get(cls)?.center, unit = defaultCenters.get(cls)?.unit,
) {
  return cls !== 'unidentified' && center != null && unit ? `${cls} ~${center} ${unit}` : cls;
}
