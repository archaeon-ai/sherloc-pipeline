# SHERLOC Pipeline — Invariants

System-wide constraints that must hold across changes. Both human contributors
and AI agents working in this repo are expected to read this file before
modifying core paths, and to update it in the same commit when an invariant
is intentionally extended or refined.

> If a change appears to violate one of these invariants, treat that as a
> design question, not an implementation detail. Either revise the change to
> conform, or update the invariant here with the rationale.

---

## 1. CLI surface stability

**Constraint.** The behavior of every published `sherloc <command>` is
backward-compatible across patch and minor versions. Argument names,
defaults, output locations, and exit-code semantics do not change without
either (a) deprecation in `CHANGELOG.md` for at least one minor version, or
(b) a major-version bump.

**Current commands** (see `src/sherloc_pipeline/cli/app.py`):
`full-pipeline`, `apply-review`, `plot`, `ingest`, `process-new`, `db-stats`,
`pds-ingest`, `pixl-ingest`, `fit-fluor`, `persist-peaks`, `backfill`,
`extract-training`.

**How to verify.** `pytest -m "not slow"` includes CLI integration tests.

---

## 2. Config-default stability

**Constraint.** Modifications to `src/sherloc_pipeline/config.yaml` must not
alter behavior for existing users. New sections may be added; existing values
must remain unless the change is documented in `CHANGELOG.md`.

**How to verify.** `git diff config.yaml` against the previous release tag.
Run the golden-baseline regression test (`pytest -m "slow"`) when changing
fitting or preprocessing parameters.

---

## 3. Service return-pattern consistency

**Constraint.** Service-layer functions return `ServiceResult(summary,
artifacts, warnings, metadata)` rather than raw datatypes.

**Where defined.** `src/sherloc_pipeline/services/base.py`.

**Why.** Uniform plumbing of side-effects (artifacts, warnings) lets the CLI
and web layers render results consistently and surface partial failures.

---

## 4. Error hierarchy

**Constraint.** All recoverable errors raised from the service or web layer
extend `SherlocServiceError`. Specific subclasses
(`SpectralFittingError`, `IngestionError`, etc.) carry domain context via
`enrich(sol=…, target=…, scan=…)`.

**Where defined.** `src/sherloc_pipeline/services/errors.py`.

**Why.** One catch-clause in the CLI / web layer can render any service
error to the user with full provenance.

---

## 5. No silent failures

**Constraint.** No bare `except:` clauses. Catches name the exception types
they expect and either log + add to `ServiceResult.warnings`, or re-raise
through the `SherlocServiceError` hierarchy.

**Why.** Silent failures surface as cryptic downstream symptoms (empty
results, missing fits, NaN propagation). A single
`except (FooError, BarError):` makes the failure mode legible to the next
reader.

**How to verify.** `git grep -nE '^[[:space:]]*except:' src/` — should return
no matches.

---

## 6. Test isolation

**Constraint.** Tests run from the bundled `tests/fixtures/` data and never
require operator-local paths or external network access. Production code
reads paths from `RuntimeContext` (`src/sherloc_pipeline/services/runtime.py`)
or env-driven configuration; it never hardcodes operator-local paths.

**How to verify.**

```bash
pytest -m "not slow"                                                # passes offline
scripts/check-forbidden-strings.sh --tree                           # no operator paths
```

CI runs the same checks on every push.

---

## 7. Output-location separation

**Constraint.** `sherloc plot` writes its outputs to `results/<target>/plots/`,
not into the per-scan output folders. Re-running `full-pipeline` archives the
per-scan folder (`<scan>_archive/`); plot outputs live outside that path so
they are not archived alongside.

**Why.** Plots are commonly regenerated against fresh data; per-scan folders
are immutable artifacts.

---

## 8. Forbidden-string discipline

**Constraint.** Tracked content must not contain operator-local
infrastructure strings (absolute paths, operator hostnames, internal
codenames). Substitutions are documented in `CONTRIBUTING.md` "Public-repo
discipline".

**Where enforced.**

- Pre-commit hook (`.pre-commit-config.yaml` → `scripts/check-forbidden-strings.sh --staged`)
- CI workflow (`.github/workflows/ci.yml` → same script with `--tree`)

**How to verify.** `scripts/check-forbidden-strings.sh --tree` exits 0.

---

## 9. Spectral calibration: no `np.linspace` for wavenumbers

**Constraint.** Wavenumber axes are derived from the Loupe polynomial
calibration (`core/normalization.calculate_loupe_wavelength_wavenumber`).
**Never** construct a wavenumber axis with `np.linspace`. R1 (Raman) is
523 channels (52–574) after wavelength filtering; R123 stitching uses Loupe
overlap summation.

**Where defined.** `src/sherloc_pipeline/core/normalization.py`,
`docs/schema/SPECTRAL_REGIONS.md` (the canonical reference).

**Why.** Linear approximation introduces ≤15 cm⁻¹ peak position errors. This
is the calibration error that motivated the Sprint 4 rebuild.

---

## 10. Calibration vs normalization terminology

**Constraint.** Two distinct operations are not conflated:

- **Wavelength/wavenumber calibration** — channel-to-wavenumber mapping via
  Loupe polynomial.
- **Laser normalization** — photodiode-based intensity correction across
  scan points.

Both live in `core/normalization.py` for historical reasons; their
docstrings and error messages must distinguish them. New code should not
reintroduce shared abstractions that erase the distinction.

---

## When to update this document

Update INVARIANTS.md in the same commit when:

- A new system-wide constraint is being established.
- An existing invariant is intentionally being extended or refined.
- A constraint here turns out to be wrong, and the code is the authority.

Do **not** update for:

- Bug fixes that conform to an existing invariant.
- Implementation detail changes.
- One-off task notes.

---

## Pre-commit checklist for contributors

- [ ] `pytest -m "not slow"` passes locally.
- [ ] `scripts/check-forbidden-strings.sh --tree` passes.
- [ ] `ruff check .` passes.
- [ ] No new bare `except:` clauses.
- [ ] No new hardcoded operator-local paths.
- [ ] Service functions return `ServiceResult`; new errors extend `SherlocServiceError`.
- [ ] If a config default changed, `CHANGELOG.md` updated.
- [ ] If an invariant here was extended, this file updated in the same commit.
