import { expect, it } from 'vitest';
import { classLabel, matchRule } from './classRules';
import edges from './classWindowEdges.json';
import windows from './classWindows.json';

const rules = windows.map(({ lo, hi, ...rule }) => ({
  ...rule, center: (lo + hi) / 2, range: (hi - lo) / 2, disabled: false,
}));

it.each(edges)('assigns $domain at $value to $expected', ({ domain, value, expected }) => {
  expect(matchRule(domain, value, rules)?.class_id).toBe(expected);
});

it.each(rules)('matches the midpoint of $class_id', (rule) => {
  expect(matchRule(rule.domain, rule.center, rules)).toBe(rule);
});

it('skips a disabled window at a shared endpoint', () => {
  const edited = rules.map((r) => ({ ...r, disabled: r.class_id === 'group1b' }));
  expect(matchRule('fluorescence', 329, edited)?.class_id).toBe('group2');
  expect(matchRule('fluorescence', 325, edited)).toBeUndefined();
});

it('prefers the higher upper bound when two mineral windows start together', () => {
  const narrow = { ...rules[0], center: 850, range: 10 };
  const wide = { ...narrow, center: 860, range: 20 };
  expect(matchRule('minerals', 850, [narrow, wide])).toBe(wide);
});

it('labels OH_stretch with its default or edited center', () => {
  const rule = rules.find((r) => r.class_id === 'OH_stretch')!;
  expect(classLabel(rule.class_id)).toBe('OH_stretch ~3500 cm⁻¹');
  const edited = { ...rule, center: 3300 };
  expect(classLabel(edited.class_id, edited.center)).toBe('OH_stretch ~3300 cm⁻¹');
});
