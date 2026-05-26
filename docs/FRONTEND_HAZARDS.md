# Frontend hazards — patterns to recognize and avoid

This doc captures non-obvious failure modes specific to the SHERLOC web frontend stack (Svelte 4 + Vite + TypeScript). Each entry pairs the hazard pattern with a recommended idiom for new code.

Read this before authoring or substantially modifying any `*.svelte` file in `src/sherloc_pipeline/web/frontend/src/`.

---

## H1 — Svelte 4 reactive-block invalidation tracking ("`$:` writes become tracked deps")

### The hazard

Svelte 4 compiles a reactive statement that writes a variable into something equivalent to:

```js
$: { x = newVal; }
// becomes (roughly):
$: $$invalidate('x', x = newVal);
```

The runtime `$$invalidate` includes an "only update if changed" guard. In Svelte 4 this materializes — depending on compiler version and the specific source shape — as a comparison against the current value of `x`, which can register `x` as a tracked dependency of the same block that wrote it.

**Result:** any *other* code that subsequently writes `x` (a user toggle, a `bind:value` input, a different reactive block) re-fires the original block, which immediately stomps the new value back to `newVal`. The user observes the bug as "my input gets reset every time I touch it."

### When it actually bites

The hazard manifests reliably when **all three** conditions hold:

1. The `$:` block writes one or more variables
2. At least one of those variables is also written elsewhere (a `bind:value` input, an `on:click` handler, another `$:` block, a parent prop)
3. The block writes *multiple* variables, OR the single written variable is heavily user-mutated

Single-write blocks with rarely-mutated targets are typically safe in practice because the invalidation comparison short-circuits when the value hasn't changed.

### Live case study — ProcessingChain.svelte (v4.1.13)

**Symptom:** The Workbench Baseline checkbox and Background selector would visibly toggle on user click, then snap back to the previous state moments later. Worse: applying a Baseline correction would briefly succeed, then the checkbox would deselect itself while the visualization continued to show the baselined data — UI and applied state out of sync.

**Root cause:** The reset block was multi-write and included user-toggled state:

```svelte
$: {
  rawWavenumber = wavenumber;
  rawIntensity = intensity;
  afterDespikeIntensity = null;
  // ...
  baselineEnabled = false;   // ← user-toggled
  bgEnabled = false;         // ← user-toggled
  bgType = 'none';           // ← user-toggled
}
```

User clicks bgEnabled checkbox → `bgEnabled = true` → block re-fires (bgEnabled is now a tracked dep via invalidation) → block writes `bgEnabled = false` → STOMP.

**Fix:** Funnel writes through a function call. The function body is opaque to Svelte's dependency analyzer, so the writes inside it don't register as deps of the calling reactive block. Use plain function parameters to pin the *real* deps explicitly.

```svelte
function applyRawInputReset(w: number[], i: number[]): void {
  rawWavenumber = w;
  rawIntensity = i;
  // ... all the resets
  baselineEnabled = false;
  bgEnabled = false;
  bgType = 'none';
  inputGeneration += 1;
}

// Only wavenumber + intensity are tracked dependencies; the writes hide inside the helper.
$: applyRawInputReset(wavenumber, intensity);
```

See `src/sherloc_pipeline/web/frontend/src/components/ProcessingChain.svelte:62-90` for the canonical implementation including the `inputGeneration` async-staleness guard.

### Recommended idiom

When a `$:` block needs to write **any** variable that is also written elsewhere in the file:

```svelte
// Always funnel through a named helper.
function doTheUpdate(deps...): void {
  // writes here are opaque to Svelte's dep tracker
}
$: doTheUpdate(realDep1, realDep2);
```

**Do not:**

```svelte
$: {
  writeUserToggledVar = false;  // ← will re-fire on every user toggle
}

$: if (someCondition) {
  writeUserToggledVar = derivedValue;  // ← also vulnerable
}
```

### Audited safe sites (as of 2026-05-22)

These `$: if`-pattern sites were audited and judged safe by inspection — either the write target is not user-mutated elsewhere, or the condition self-stabilizes after one write:

| Site | Why safe |
|---|---|
| `App.svelte:23` (route guard) | Writes via store, not local |
| `AciViewer.svelte:44` (`!colorizedAvailable && colorized` → `colorized = false`) | Condition self-stabilizes; one-shot write |
| `DespikeStep.svelte:37` | Empty body |
| `RGBMixPanel.svelte:100` | Calls function — writes are opaque |
| `map/MapCanvas.svelte:131-152` | Each block calls a method; no top-level user-mutated writes |
| `map/MapControls.svelte:15` | Single-write to internal state |
| `PointSelector.svelte:47` | Refactored 2026-05-22 to use `resetRangeEnd(p)` helper as a defensive measure — see file |

### Long-term resolution

The entire class of `$:`-invalidation hazards is eliminated by Svelte 5's runes (`$state`, `$derived`, `$effect`), which use explicit dependency declaration instead of compiler-magic auto-tracking. **A migration to Svelte 5 runes is tracked separately as a tech-debt issue** (search GitHub issues for label `svelte5-migration`). Migration is scoped post-v1.0.1.

Until that migration lands, the helper-function idiom above is the load-bearing workaround.

---

## H2 — Reserved for the next pattern

Add new hazards here as they're discovered, with the same structure: hazard → when it bites → live case study → recommended idiom.
