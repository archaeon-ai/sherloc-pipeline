# Hydration cosmic-ray veto — evidence and proposed thresholds

**Issue:** #38 — hydration fitting fits cosmic rays as OH-stretch peaks, with the
FWHM pinned at the 50 cm⁻¹ floor.
**Status:** feature flag implemented and **DEFAULT OFF**. The thresholds below
are *proposals*. Flipping `fitting.hydration_cr_veto.enabled` to `true` is a
science decision and requires explicit operator ratification recorded on the
issue.

## What was built

Two independent signals, both behind `fitting.hydration_cr_veto` in
`config.yaml`, implemented once in `core/hydration_veto.py` and applied
consistently across all three hydration paths:

| Path | Entry point | Input the fit sees | Input the veto sees |
|------|-------------|--------------------|---------------------|
| Pipeline / CLI `fit-hydration` | `services/fitting.py::_fit_point_hydration` | `R1_normalized` (pre-baseline, **not** despiked) | the same array |
| Map-mode quick fit | `services/map_fitting.py::_fit_raman_domain` | despiked + baselined | the pre-despike intensity, passed through explicitly |
| Web point fit | `web/routes/processing.py::process_fit` | caller-supplied, optionally baselined | the caller's raw intensity |

**Mechanism (a) — despike-mask veto.** The fit still runs on the non-despiked
spectrum, preserving the published-method fidelity that motivated the no-despike
choice. The existing classical despiker runs on the *same* array purely to
harvest the spike mask that `despike_r1_spectrum` already returns and that every
call site currently discards. A candidate is implicated when a masked,
positive-going bin near its centre accounts for a material share of its local
height, or when the overall raw→despiked drop at the centre exceeds a ratio.

**Mechanism (b) — bound pinning.** A fit that converged within ε of the FWHM
floor is unreliable by construction: the optimiser wanted a narrower peak than
the model permits. This is reported as a flag and **never** rejects on its own.

`action: "reject"` drops an implicated candidate; `action: "flag"` keeps it and
only annotates it. Bound pinning is flag-only under both settings.

**Despiker parameters are shared, not re-declared.** The mask a verdict rests on
is only comparable across paths if every path despikes the same way, so all of
them — the three above, the web modz preview, and the sweep utility below —
build their `DespikeParams` through `core/preprocessing.despike_params_from_config()`.
It resolves *every* field of `preprocessing.despike` (including `run_length_max`
and the laser/sulfate guard windows), so an operator override changes the veto
identically everywhere or nowhere. Pass `--config` to the sweep if the database
was fitted under a non-default config.

**Turning the flag on for an already-fitted scan.** Re-fitting deletes the
per-point CSV, overlay PNG and accepted-peaks summary for any point that no
longer has an accepted peak (and, on the organics path, the mutually exclusive
D+G / G-only export the run did not write), and `persist_raman_peaks()` reads a
completed run with no per-point CSVs as a zero result that clears the domain's
existing database rows. Without both, the previous run's artifacts would be
rediscovered and the vetoed peak restored. A run marker written as `running` at
fit start and `completed` at the end is what distinguishes that zero result from
an interrupted rerun; a fit directory with no marker predates the change and
persists exactly as it did before.

## Proposed thresholds

| Parameter | Proposed | Rationale |
|-----------|----------|-----------|
| `center_window_cm1` | 15.0 | ≈ two detector channels either side of the fitted centre at R1 spacing (~8.6 cm⁻¹/channel), so a centre that lands one bin off the spike still resolves. |
| `amplitude_drop_ratio_max` | 0.5 | A candidate more than half of whose local height the despiker removes is spike-dominated, not a band with a spike on it. |
| `mask_min_drop_ratio` | 0.10 | Floor below which a masked bin is ignored. See the false-positive finding below — without it the mask signal alone vetoes authentic broad bands. |
| `fwhm_floor_epsilon_cm1` | 0.5 | Matches the ±0.5 cm⁻¹ band used to count the floor pile-up in the issue. |

## Measured behaviour (synthetic bench)

Measured on a detector-realistic R1 axis (~8.6 cm⁻¹/channel), reproduced by the
test suite (`tests/unit/core/test_hydration_veto.py`,
`tests/unit/services/test_hydration_veto_paths.py`,
`tests/unit/web/test_processing.py`):

- **Defect reproduced flag-off.** A two-channel spike inside 2800–3900 cm⁻¹ on
  flat noise, with no real band present, is fit and accepted: R² ≈ 0.43 (gate is
  0.25), sharpness gate passed, FWHM = 50.0 cm⁻¹ — pinned exactly at the floor.
  This is the reported signature, and the existing quality gates do not catch it.
- **Flag-on, all three paths.** The same spike is vetoed by both mechanism (a)
  signals (mask hit and drop ratio ≈ 1.0) and additionally carries the
  bound-pinning flag.
- **Authentic control.** A broad OH band (FWHM ≈ 280 cm⁻¹, centre ≈ 3400 cm⁻¹)
  survives flag-on on every path, with drop ratio ≈ 0.013 and no bound-pinning
  flag.

## False-positive risk found during development

The mask signal *on its own* is too aggressive. On the authentic broad-band
control, the rolling-median despiker masked a single noise channel near the band
apex, and a bare "centre falls on a masked bin" rule therefore vetoed a genuine
hydration feature. Two changes fixed it, and both are load-bearing:

1. Only **positive-going** masked bins implicate a cosmic ray. Spikes are
   additive; a masked bin the despiker *raised* is a downward outlier and says
   nothing about the candidate.
2. A masked bin must account for at least `mask_min_drop_ratio` of the
   candidate's local height. At the proposed 0.10 the observed nick (≈ 0.013 of
   the band height) is correctly ignored while the synthetic cosmic ray (≈ 1.0)
   is still caught.

This is the sharpest reason the flag ships default-off: the mask mechanism's
false-positive rate is a property of the despiker's behaviour on real broad
bands, and only the corpus sweep below can measure it.

## Sweep results — to be produced by the operator

The corpus this needs (the floor-pinned hydration rows and the reproduction line
scan) lives in the team-tier database, which is not reachable from this repo's
test environment and holds pre-archive measurement values. The sweep is
therefore delivered as a read-only script rather than as pre-computed numbers:

```
python3 scripts/hydration_cr_veto_report.py --db <phase.db>
python3 scripts/hydration_cr_veto_report.py --db <phase.db> --scan <reproduction-scan-uuid>
```

It reports, for the floor-pinned rows and optionally for all hydration rows:
mask-veto catches, amplitude-ratio catches, their overlap, bound-pinning flags,
total vetoed vs. survived, and the drop-ratio distribution. It writes nothing
back and prunes nothing; pruning existing spurious rows is the separate,
operator-gated companion activity.

Two numbers decide ratification:

1. **Catch rate** — what share of the floor-pinned pile-up each mechanism
   accounts for, and how much the two overlap. A mechanism that adds no
   non-overlapping catches does not need to be on.
2. **False-positive risk** — how many *interior-width* hydration rows (well
   above the floor, i.e. authentic-looking broad bands) either mechanism would
   veto. Run with `--all-rows` to measure this. A non-trivial count means the
   thresholds are too tight, not that the veto is wrong.

Record both, together with the ratified threshold values, on issue #38 before
changing the default.
