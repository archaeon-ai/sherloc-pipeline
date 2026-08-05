# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Map Mode fitting no longer looks frozen on slow or queued scans (#6).**
  Map fitting runs on a single-threaded executor, so a job started while
  another scan is still fitting waited its turn while the UI sat on an empty
  progress panel; the WebSocket's only liveness signal was a contentless
  `{"type": "heartbeat"}` the client ignored. The heartbeat frame now carries
  the server-side job snapshot (`status`, `fitted`/`total`, `queue_position`,
  `elapsed_s`, `since_last_message_s`, `stalled`) and is also sent once
  immediately on connect, and a `job_started` frame is emitted when the job
  actually leaves the executor queue. Map Mode surfaces both as progress-panel
  status and log lines ("Queued behind N active fit jobs", "No new results for
  Ns"). The client-message read is now a single long-lived receive instead of a
  re-armed 10 ms `wait_for`, which could drop the frame it had just picked up —
  including a user's `cancel`. `GET /api/map/jobs/{job_id}` reports fitted
  points from an authoritative counter rather than counting messages in the
  reconnect ring buffer, which undercounted once the buffer wrapped
  (>2000 messages). Terminal jobs are reaped when a new fit starts.

### Changed
- **Map Mode fitting is faster on large scans (#6).** Despiking and the asPLS
  baseline depend only on a point's R1 spectrum, but were recomputed inside
  every requested Raman domain — three times per point for the default
  minerals+organics+hydration selection. They now run once per point and are
  shared across domains (fit results are unchanged; pinned by test).
  Fluorescence fitting batch-loads R1/R2/R3 for the whole scan instead of
  issuing three queries per point. Runs also log their scale up front
  (`Fitting N points x M domain(s)`, flagged when N > 200, where the sequential
  design stops being comfortable).

### Added
- **Zoomed mineral-region companion plot for average fits (#30).** Every
  average-spectrum fit overlay `fit-averages` (and so `process-new`) emits now
  has a companion `<stem>_fit_<lo>-<hi>.png` beside the full-range 500–4000
  cm⁻¹ plot, for both average kinds. The window is the configured
  `fitting.r1_fit_range` (default `[700, 1200]`, reproducing the historical
  `_700-1200` naming) and the filename tokens derive from it, so a non-default
  range self-describes. Render-only: same peaks, same model array, no refit —
  only R² is rescoped to the displayed window. No new flags.
- **Optional atomic delivery handoff.** A successful `full-pipeline` run now
  records an explicit `not_configured` skip by default. When
  `SHERLOC_HANDOFF_DIR` is set, it inventories the completed scan's raw ACI
  PNGs, any mirrored colorized workspace PNGs, and required `spatial.csv` /
  `loupe.csv` companions, then atomically publishes a versioned
  `<run_id>.ready` manifest with portable locators and byte evidence. Invalid
  sources or an unwritable destination fail the run; no existing serving,
  ingestion, or database behavior is changed when the variable is absent.

### Fixed
- **Scan-context resolution now tolerates raw Loupe dir/manifest typos
  (#16).** `process-new`/`full-pipeline` preprocessing could fail to
  resolve a scan's working directory when the raw Loupe directory name
  (and its `loupe.csv` `human_readable_workspace` field) is misspelled
  relative to the DB-corrected `scan_name` — e.g. sol 1521's composite
  reductions live under `meteroite_sum_active_median_dark` on disk while
  the DB scan is `meteorite_sum_active_median_dark`. Manifest-based
  resolution now falls back to a typo-tolerant (edit-distance) match when
  no exact match is found, resolving only when a single candidate is
  unambiguously closest — ambiguous or too-distant cases still fail with
  the existing clear error rather than guessing.

## [5.5.0] - 2026-07-10

Closes #7: the transitional `context_images.file_path` column is
dropped. Ingestion, the web serve path, and the processing-side disk
resolver now use `r2_rel_key` exclusively — `file_path` carried no
information `r2_rel_key` lacked (both serving databases were fully
backfilled: team 1107/1107, public 1529/1529 including the 5 `pds:`
sentinel rows). Deploying this release requires running the new Alembic
migration (`17db1a1940d6`, `DROP COLUMN` via SQLite batch mode) against
both the team and public databases; no separate data re-sync is needed
since the locator was already the live read path before the drop.

### Removed
- **`context_images.file_path` column** (migration `17db1a1940d6`,
  down-revision `931df60632cb`). `downgrade()` re-adds it as nullable
  Text — the absolute paths cannot be reconstructed from `r2_rel_key`
  alone, so a downgrade cannot restore the original data.
- **`ImageInfo.file_path`** (`services/image_query.py`) — not part of
  the frozen `--json` CLI contract (verified via
  `tests/contract/test_cli_schema_contract.py` and a repo-wide grep of
  `tests/contract/`), so dropped outright rather than kept as a
  compatibility shim.

Processing-side half of #7 (prior entry, still accurate): the
disk-read sites derive their path from the stored relative locator
(`context_images.r2_rel_key`) via `<root> + locator`, exactly as the
serve path derives R2 keys.

### Added
- **`core.r2_keys.resolve_disk_path(rel_locator, *, data_root,
  pds_cache_dir)`** — the disk-edge inverse of `derive_rel_locator`:
  `loupe/…` locators resolve under `data_root`, `sol_NNNN/data_aci/…`
  under `pds_cache_dir`. Returns `None` (caller raises its own
  missing-file-equivalent error) for a missing / `pds:`-schemed /
  unrecognized locator, a failed traversal guard (mirrors
  `derive_r2_key`), or an unmounted (`None`) root. The `sol_NNNN` anchor
  accepts the `_colorized` variant.

### Changed
- **Processing-side disk reads use the locator, not `file_path`:**
  - `services/image_query.py` — `ImageInfo` gains `r2_rel_key` (populated
    from the ORM; `to_dict()` output unchanged for API compat); the
    service takes `data_root` / `pds_cache_dir` (config-derived defaults)
    and resolves every disk read (`load_image`, export, VICAR-label
    fallback) through `resolve_disk_path`. A `None` resolution raises the
    site's existing error type naming the locator — no silent fallback.
  - `services/segmentation.py` — the batch/single image queries select
    `r2_rel_key` (no longer `file_path`); reads resolve through
    `resolve_disk_path`. Service takes `data_root` / `pds_cache_dir`.
  - `core/coordinates.py` — the legacy local-FS fallback derives the
    Loupe workspace dir from `r2_rel_key` (colorized variant via
    `colorize_sol_segment` on the locator) + `data_root`/`pds_cache_dir`
    (threaded params, config-derived when unset). The R2 path is
    untouched.

## [5.4.0] - 2026-07-10

Context images are now identified by a stored **relative locator**
(`context_images.r2_rel_key`) — the string after `sherloc-aci/` in the
object's R2 key — instead of translating a machine-specific absolute
`file_path` at serve time. Part of #7 (the serve-path transition; the
processing-side `data_root + locator` conversion and the `file_path`
column drop remain tracked there).

### Added
- **`context_images.r2_rel_key` column + backfill** (migration
  `931df60632cb`): idempotent one-time translation of existing
  `file_path` rows using exactly the strip logic the serve path ran
  through v5.3.x (canonical per-tier roots + the known legacy team
  alias + any env-var overrides). `pds:<lidvid>` sentinel rows
  round-trip; rows matching no known layout stay NULL (they were not
  servable before either) and are logged.
- **`core.r2_keys.derive_rel_locator`** — structural locator derivation
  (anchored on the `sol_NNNN` segment) used by all ingestion writers
  going forward; no deployment paths in source.

### Changed
- **R2 key derivation is a single concatenation** (`sherloc-aci/` +
  locator). The per-tier strip-prefix table, `PHASE_*_STRIP_PREFIX`
  env fallbacks, legacy-ingestion aliases, and tier inference are
  deleted from `core/r2_keys.py`; `derive_workspace_key` is now a pure
  locator transform (no `PHASE_TIER` lookup). Tier isolation remains
  credential-side (bucket-scoped tokens) + per-tier databases.
- **Ingestion dual-writes** `file_path` (absolute, kept transitionally
  for processing-side disk reads and old-code rollback) and
  `r2_rel_key`. The PDS download step records the locator relative to
  the cache root when it resolves a `pds:` reference.
- **Env vars retired** (§5.5): `PHASE_TEAM_STRIP_PREFIX`,
  `PHASE_PUBLIC_STRIP_PREFIX`, `PHASE_TEAM_LEGACY_STRIP_ALIASES`,
  `PHASE_PUBLIC_LEGACY_STRIP_ALIASES` — no effect on a migrated
  database (the backfill migration alone consults them for parity).

### Notes
- **Schema + migration change**: deploys must run `alembic upgrade
  head` (the boot sequence does this) and then **re-sync both tier
  databases** to the VPS per the established DB-sync pattern. Web ACI
  serving and Map Mode read `r2_rel_key` exclusively. A database whose
  schema was never migrated (column absent) fails at ORM query time —
  the boot-sequence `alembic upgrade head` is the guard for that case;
  a *migrated* row whose locator is NULL (matched no known layout)
  fails with the controlled `misconfigured_path` 500.
- `file_path` semantics are unchanged for processing-side disk reads;
  dropping the column (and converting those readers to
  `data_root + locator`) is deferred until the locator is proven on
  both tiers.

## [5.3.0] - 2026-06-27

Widens the name-authoritative `scan_type` resolver's calibration vocabulary so
the full mission corpus classifies without spuriously quarantining
calibration-target and engineering acquisitions. A follow-up to 5.2.0, applied
before the corpus was ever reclassified: validating the 5.2.0 resolver against
the full corpus (not just the golden fixture) showed it would have left ~half of
all scans with a `NULL` `scan_type` — the calibration-target, instrument-
diagnostic, and `detailed_*` naming families never appear in the golden fixture.

### Changed
- **`classify_scan_type` recognizes the calibration / engineering scan-name
  families** as `calibration` (regardless of spectrum count), in addition to
  the SRLC sequence code and AlGaN name: the AlGaN internal-cal lamp (now a
  substring match, so `SRLC*_AlGaN_*` / `passive_AlGaN_*` are caught, not only
  `AlGaN_*` prefixes); laser-off darks (`*laser_disabled*` / `*no_laser*`);
  cal-target materials (Teflon, Vectran, Orthofabric, Polycarbonate, Diffusil,
  nGimat); the external-cal meteorite; the maze focus/intensity target; laser
  power-state housekeeping (`power_on` / `power_off`); passive observations; and
  bare PPP intensity/SNR tests (`^\d+ppp`, anchored so science `detail_*ppp` /
  `survey_*ppp` acquisitions keep their geometry).
- **`detailed_*` now resolves to `detail` and `lines` to `line`** — variant
  spellings that the token-boundaried prefix rule did not match (and which the
  5.2.0 resolver would have quarantined).

### Notes
- `scan_type` (acquisition geometry) and `target_type` (mars/cal/engineering
  purpose) remain **orthogonal**: a real `survey`/`detail`/`line`/`HDR` geometry
  performed at a calibration or engineering target keeps its geometry. The
  widened vocabulary was verified value-blind against the full corpus to never
  assign `scan_type=calibration` to a `target_type=mars_target` scan; a
  cross-consistency test locks this so the two vocabularies cannot drift apart.
- Behavior-only change to the resolver (no CLI surface, schema, or migration
  change). Re-run `reclassify-scan-types` (dry-run first) to apply the corrected
  classification to an existing corpus.

## [5.2.0] - 2026-06-27

Name-authoritative scan classification at the source: the three classification
axes (`scan_type`, `scan_class`, `product_role`) are now derived from the scan
name rather than a spectrum-count heuristic, and a new analytical product-role
axis represents multishot acquisitions. All changes are additive and
backward-compatible; values for the existing corpus are corrected in place with
the new `reclassify-*` commands.

### Added
- **`ScanType` gains `line` and `HDR`.** These observation kinds were
  previously unrepresentable, so every `line`/`HDR` scan was mislabeled
  `detail`/`survey` by the count rule.
- **`classify_scan_type(scan_name, sequence_code, n_spectra)`** — a
  name-authoritative resolver (`models.spectra`). Calibration is keyed on the
  sequence code first (unshadowable); the kind is then matched from the name by
  a case-normalized, token-boundaried, ordered map (`survey` > `detail` >
  `line` > `HDR`; `cross`/`asterisk` inherit `line`). The spectrum-count rule
  is a fallback for explicitly-uninformative names only (empty / synthetic
  `pds_*` names); an informative-but-unrecognized name is **quarantined**
  (no guessed type written), guarding against silently mis-typing a future
  acquisition kind.
- **`product_role` axis** (`raw` / `canonical` / `alternate`; `NULL` for every
  non-multishot scan) on `ScanORM` and the `Scan` model, with a DB CHECK
  (Alembic revision `9b2e7c4a1f08`) enforcing enum membership and the role ⇒
  class/parent/sources couplings. For a multishot acquisition the
  `*_sum_active_median_dark` reduction is the counted `canonical` product; the
  raw is retained but not counted.
- **`classify_product_role` + multishot helpers** that distinguish the
  `*_median_all` / `*_sum_active_median_dark` reductions from the bare `*_all`
  spatial union (the `_all` naming collision).
- **`reclassify-scan-types`, `reclassify-scan-classes`, `reclassify-product-roles`
  CLI commands.** Each re-derives one axis in place with a dry-run default
  (`--apply` to write), a value-blind transition-count diff, a single
  transaction, a `--snapshot`/`--i-have-a-backup` gate before applying, a
  schema/migration preflight, a measurement-table no-mutation assertion
  (`spectra` / `fitted_peaks` content hash), and idempotence.

### Changed
- **`scan_class` now classifies bare trailing-underscore unions** (e.g.
  `detail_`, `line_`) as `composite` (previously stored `primary`).
- **Loupe ingestion sets `scan_type` name-authoritatively** at write time
  (`to_scan()`), so new ingests are classified correctly going forward.
- The PDS observation classifier (`PDSObservationGrouper.classify`) now routes
  through the shared resolver; PDS observations carry synthetic, kind-blind
  names, so they continue to fall back to sequence-code + spectrum count, with
  the historical "missing count ⇒ unclassified" contract preserved.

## [5.1.0] - 2026-06-25

### Fixed
- **Scan-point overlay stays registered when toggling the colorized ACI.** In
  Map Mode and the Workbench ACI viewer, switching to the colorized image
  previously left every overlay point offset by a near-constant ~28 px (the
  colorized ACI is a pure crop of the grayscale image, but the overlay kept
  using grayscale-frame coordinates). The coordinate resolver now reads the
  colorized workspace's own `spatial.csv` / `loupe.csv` so each variant's
  points are drawn in its own image frame.

### Added
- `GET /api/map/layers/{scan_id}` returns a `point_set_colorized` field (same
  shape as `point_set`, with its own Voronoi geometry) when a colorized ACI
  variant exists; otherwise `null`.
- `GET /api/scans/{scan_id}/points` returns `x_aci_pixel_colorized` /
  `y_aci_pixel_colorized` on each point when a colorized ACI variant exists.
- `resolve_display_coordinates(..., colorized=True)` resolves coordinates
  against the `sol_NNNN_colorized/` Loupe workspace. All of the above are
  additive and backward-compatible.

### Changed
- The `map_display_coordinates` cache is now keyed by
  `(scan_point_id, colorized)` so grayscale and colorized coordinates are
  cached independently (Alembic revision `0c0107a1bed5`). The table is a
  recomputable cache; the migration recreates it and existing entries
  repopulate on first access.

## [5.0.1] - 2026-06-23

### Fixed
- **`pds-ingest` no longer exits non-zero on a fully successful ingest.** A sol
  containing a zpz-only observation (a calibration intermediate with no RRS/RCS
  spectral product — e.g. Sol 921) previously recorded that expected skip as an
  error, so the command exited `1` despite ingesting all real data. Such
  observations are now classified as a non-fatal skip: reported as a warning,
  counted under a new `observations_no_spectral` field (CLI summary, `--json`,
  and `--report-json`), and excluded from the failure exit code. Genuine
  failures still exit non-zero.

## [5.0.0] - 2026-06-21

### Removed (BREAKING — CLI contract change)
- **`detect-confidence` command withdrawn.** The experimental false-alarm-probability
  (FAP) CLI command (shipped in v4.5.0) is removed. Per the CLI stability invariant
  (`docs/INVARIANTS.md` §1), removing a published command is breaking and triggers
  this **major-version bump**. The command's intended applications fell outside the
  validated scope of the underlying method, so it carried no supported user surface.

### Changed
- Documentation updated to reflect the removed command (README,
  `docs/architecture.md`, `docs/METHODS.md`, `docs/INVARIANTS.md`). The despike/ML
  cosmic-ray path and all other commands are unaffected — this is a
  CLI-surface-only change.

## [4.5.0] - 2026-06-18

### Changed
- **ML cosmic-ray despike model promoted to v1.3 ("v13c").** The v1.3 retrain uses
  the same recipe as v1.1 with cleaned training labels: it lifts blind-spot
  generalization recall while preserving protection (no regression) and removes
  cosmic rays v1.1 missed. The frozen in-code model identity
  (`sherloc_pipeline.ml_despike.manifest.ModelManifest`) changes:
  - name / artifact: `v1_stageB_v11` → `v1_stageB_v13c`
  - provenance label: `ml_v1.1_tau_matched` → `ml_v1.3_tau_matched`
  - ONNX sha256: `3bff18ec…f0f9` → `9668a0b2ca257ce333d57e3f76598dda8cb5c1839e2fde6bd955086d959be0ba`
  - source checkpoint sha256: `002970cd…ce28` → `a77cd435d65631a8728c9d39c01c31dd30805ac37062b8c48937be6fb3594881`
  - tau R1: `0.29101562500044187` → `0.29882812500038747`; fluorescence
    (R2/R3): `0.273437499999351` → `0.2656250000008831`
  - download URL: `model-cr-despike-1.3/v1_stageB_v13c.onnx`

  Region windows, `n_channels` (2148), `opset` (18), and the runtime
  (onnxruntime 1.26.0, CPUExecutionProvider, fp32) are **unchanged**. The
  integration parity gate (`scripts/verify_ml_despike_parity.py`) passes at zero
  symmetric difference on the golden batch at the v1.3 operating point.
- **`cosmic_ray_masks` regenerated at v1.3** across all science scans; the
  superseded `ml_v1.1_tau_matched` rows were removed. The display despike and ML
  spike markers now serve v1.3 masks. This is **masks-only**: fitted peaks, mineral
  IDs, fluorescence groups, and manual review decisions are unchanged — a full ML
  re-fit on v1.3 is a separate, gated effort. The public tier is unchanged (ML
  disabled until PDS EDR availability).
- Golden `ml` regression anchor (`tests/golden/sol_921_detail_1_ml/`) regenerated
  at v1.3. Fitted peaks, review decisions, and calibration arrays are
  **byte-identical** to the v1.1 anchor; the v1.3 difference is additional
  cosmic-ray removal that does not disturb any fitted band.

### Fixed
- `sherloc_pipeline.__version__` corrected from a stale `2.0.0` to track the
  package version (`pyproject.toml`).

## [4.4.2] - 2026-06-17

### Fixed
- Badpix annotation layer rebuilt on a **dark-plane veto**. The prior tier-1 list
  was selected by the ε=(active−dark) outlier rate, which flags active-only REAL
  Raman/fluorescence bands identically to flickering pixels and misses true RTS
  defects (active≈dark, so they cancel in ε). It had wrongly flagged real spectral
  bands as bad pixels while missing the genuine defects flanking them. Tier-1 is now
  derived from the dark plane (a real defect fires with the laser OFF; a spectral
  band cannot): dark |z|>4 rate ≥ max(0.5%, 6× region median) over 177,067 frames.
  Removed 17 dark-quiet false positives (incl. real-band channels); added 420
  dark-firing defects across R1/R2/R3; JB25-attested channels preserved; new
  `dark_veto` source. **121 → 524 channels.** This is an annotation-only asset — it
  never masks, replaces, or alters any spectral value (CR despiking is unaffected).
  Both tiers serve the corrected list after this release.

## [4.4.1] - 2026-06-12

### Fixed
- ML despike spike markers now render only on single-point views, where "this
  displayed trace was repaired at these positions" is literally true. Multi-point
  views (average/subset) previously rendered the union of per-point masked channels
  as on-trace markers (e.g. 247 markers on a 100-point trim-mean), overstating the
  effect on the displayed trace; they now show a one-line provenance note with the
  union count. The CSV `spike_mask` column follows the same rule. Marker eligibility
  is bound to the committed spectrum's selection context (not live UI state), so
  selection transitions and failed refetches cannot render a stale aggregate mask as
  a point mask. modz markers (computed on the displayed trace) and the badpix overlay
  are unchanged; the `masked_positions` API field is unchanged. (#12)

## [4.4.0] - 2026-06-12

### Added
- `masked_positions` on the average/point/subset spectra responses — indices into
  the **served** arrays where stored-mask (ml) despike replaced channels (empty on
  `none`/`modz`; R123 stitch is position-preserving so positions equal absolute
  channels). The Workbench now renders the same spike-removal markers in ML mode as
  in modz mode, behind the existing overlay toggle, with method-aware labels; the CSV
  export emits the `spike_mask` column in ML mode with a `# spike_mask_method:`
  provenance header. (#8)
- **Badpix annotation layer**: curated known-noisy-channel asset (121 channels — 108
  CR-confusable elevated-noise + 13 stable hot — from the 2026 detector
  characterization cross-checked against the published mission bad-pixel table,
  Jakubek et al. 2025, Appl. Spectrosc. 79(6), 904–918), served at
  `GET /api/spectra/badpix?region=` as served-array positions; rendered as a separate
  default-off overlay (hollow blue diamonds, tier+source on hover), fully independent
  of CR despiking; optional toggle-gated `badpix` CSV export column. (#9)

### Fixed
- References panel: corrected the Jakubek et al. citation year (2024 → 2025;
  DOI-verified).

## [4.3.0] - 2026-06-12

### Added
- `despike_method` selector (`none | ml | modz`) on the spectra read endpoints —
  query param on `GET /api/spectra/{scan_id}/average` and
  `GET /api/spectra/{scan_id}/point/{idx}`, body field on
  `POST /api/spectra/{scan_id}/subset`. The legacy `despike: bool` keeps working
  (`true` → `ml`); an explicit `despike_method` takes precedence. `modz` runs the
  legacy rolling-median modified-z despike live on the **served** R1 array
  (display-level, matching the legacy Workbench client step; the CLI pipeline's
  per-spectrum pre-fit modz is unchanged). Non-R1 regions under `modz` are served
  non-despiked with `despike_missing_regions` disclosing the gap. New additive
  response field `despike_params_used` (populated on the modz path only).
- `ml_mask_count` on the scan detail response: count of stored cosmic-ray mask rows
  for the scan. 0-safe on mask-less scans and pre-migration databases (never a 500).
  Clients gate ML despike availability on `> 0`.
- Workbench **despike method selector (None | ML | modz)** in the processing chain.
  ML applies the stored masks server-side at fetch, with provenance displayed
  (precise method string, masked-channel count, composite all-or-none missing-regions
  warning); modz keeps its tunable client-side step; the method enum selects exactly
  one application site, so two despike methods can never apply to one rendered
  spectrum. The ML option is disabled with a hint when the scan has no stored masks;
  modz is disabled outside R1.

### Fixed
- The ML despike shipped in 4.2.x was only reachable through the Scan Detail view,
  which nothing in the app routes to. The despike controls are now first-class in
  the Workbench, where scans actually open. (#6)

## [4.2.1] - 2026-06-11

### Added
- `backfill-masks` CLI command: populates the `cosmic_ray_masks` table for
  already-processed scans by running the ML despike detector over each scan's
  normalized spectra and persisting the per-(point, region) channel masks —
  **without re-fitting**. Masks-only: `fitted_peaks`, mineral IDs, fluorescence
  groups, and manual review decisions are left untouched (preprocessing performs no
  DB writes; the only write is to `cosmic_ray_masks`). Idempotent; supports `--sol`,
  `--science`, `--no-engineering`, `--data-dir`, `--results-dir`, and `--dry-run`.
  Requires the `ml-despike` extra (onnxruntime). The masks produced are identical to
  those `full-pipeline` would persist (same despike path and provenance/method
  string).

  This is the supported path to make the web despike toggle functional for the
  existing mission dataset after upgrading to 4.2.x — deploying the 4.2.x image alone
  leaves the toggle present but empty until masks are backfilled. The full re-fit (ML
  despike feeding the fitting stage) remains deferred pending a golden-baseline
  fit-sensitivity assessment.

## [4.2.0] - 2026-06-11

The cosmic-ray despike step gains a method selector and the ML CR detector (v1.1)
becomes the **default method**: `despike_method ∈ {ml, modz, none}`, resolved
CLI > config > default `ml`. Detection runs on the raw ACTIVE/DARK planes from the
Loupe workspace at the frozen operating point; replacement uses the existing
interpolation at the existing despike stage, so `modz` output remains byte-identical
to pre-change behavior (verified against the frozen `tests/golden/sol_921_detail_1/`
anchor).

### Added
- `--despike-method {ml,modz,none}` on `full-pipeline`, `process-new`, and `plot`
  (additive; default defers to config). Invalid values are rejected at parse time
  with the choice list.
- New `sherloc_pipeline.ml_despike` package: frozen model manifest (sha256-pinned
  ONNX artifact, per-region thresholds), sha256-verified fetch-and-cache, 8-channel
  featurization port, ONNX CPU detector. Requires the new optional extra:
  `pip install 'sherloc-pipeline[ml-despike]'` (onnxruntime; the base install stays
  lean and `modz`/`none` never import any ML runtime).
- `cosmic_ray_masks` satellite table (Alembic migration included): per-(point,
  region) channel-index masks with full provenance (method `ml_v1.1_tau_matched`,
  model sha256, per-region tau), attached to the DARK_SUBTRACTED spectrum row with
  CASCADE lifecycle. Run provenance (including the installed onnxruntime version) is
  recorded in pipeline metadata and a `cr_masks.json` run artifact.
- `plot` applies despiking per the selected method: `ml` applies **stored** masks
  from the database (`SHERLOC_DB_PATH`; `plot` never runs inference), `modz` computes
  the legacy method live, `none` renders raw. Spectra without stored masks render
  non-despiked with an aggregated once-per-invocation note; every render prints an
  effective-despike-state summary line.
- New ml default-path golden anchor `tests/golden/sol_921_detail_1_ml/` and a
  `--despike-method` flag on `scripts/generate_golden_baseline.py`.
- **Web stored-mask despike toggle**: a `despike` query parameter on
  `GET /api/spectra/{scan}/average`, `GET /api/spectra/{scan}/point/{idx}`, and
  `POST /api/spectra/{scan}/subset` (plus `despike` on `SubsetRequest`). When set, the
  server applies the **persisted** cosmic-ray masks using the same interpolation
  helper the pipeline uses — no model inference ever runs on the serving host.
  Responses gain additive, defaulted fields (`despike_applied`, `despike_method`,
  `n_masked_channels`, `masked_channels` on the point endpoint,
  `despike_missing_regions` and `n_uncovered_contributor_channels` on composite
  views). R123 is despiked constituent-first and all-or-none; composite views disclose
  the derived uncovered-contributor count (207 of 2148 for the R123 summation view).
  Because the new fields are optional with defaults, `API_SCHEMA_VERSION` stays
  `1.0.0`. The Scan Detail spectrum view gains a despiked-by-default toggle with a
  no-stored-mask indicator. `POST /api/process/despike` is unchanged (modz-only in v1).
- `docs/METHODS.md` gains a power-user section on the ML detector (input contract,
  featurization, frozen operating point, artifact identity, runtime policy, and
  coverage limits).

### Changed
- **Default despike method is `ml`** (config `preprocessing.despike.method: "ml"`).
  Pipeline outputs produced with the default change accordingly; `--despike-method
  modz` reproduces legacy outputs byte-identically. The legacy golden suite is
  verified under explicit `modz` (`pytest -m slow`).
- Internal service API: `PreprocessingService.run_scan(despike_r1: bool)` was replaced
  by `run_scan(despike_method: Optional[str])`; `none` reproduces the former
  `despike_r1=False` behavior exactly. (Internal kwargs only — the CLI surface is
  unchanged and additive.)

### Removed
- **Compatibility note:** the unwired `fluorescence_fitting.despike_enabled` config
  key was removed. It was read by no code path and had zero behavioral effect; the
  unified `despike_method` surface replaces it. Configs still carrying the key are
  unaffected (unknown keys are ignored), and fluorescence-frame despiking is now
  governed by `despike_method` (`ml` screens the covered R2/R3 contributions; `modz`
  preserves the legacy never-despiked fluorescence behavior).

## [4.1.17] - 2026-05-24

Backend fix for a UI regression: any scan whose `context_images.file_path` carried a
pre-v4.1.9 NAS-mount prefix (recorded as the new `PHASE_TEAM_LEGACY_STRIP_ALIASES`
default) returned **HTTP 500 `misconfigured_path`** on every ACI-resolving endpoint
(`/api/images/{id}/aci`, `/api/scans/{id}`, `/api/map/layers/{id}`), making the
affected scans unbrowsable in Workbench + Map Mode. These rows had been failing since
the v4.1.9 R2 migration shipped.

`core/r2_keys.py` now accepts per-tier *legacy aliases* in addition to the canonical
strip prefix. Legacy aliases share the same R2 byte layout under `sherloc-aci/<rel>/`
(rclone preserves the post-prefix tree), so the alias resolves to an identical key
without code- or data-layer changes elsewhere. The default carries the single
known-in-flight team-tier alias; deployments may add more via the new
`PHASE_TEAM_LEGACY_STRIP_ALIASES` / `PHASE_PUBLIC_LEGACY_STRIP_ALIASES` env vars
(colon-separated).

Tier-isolation invariants preserved — the public tier rejects team-tier legacy
aliases (and vice versa); the path-traversal guard still applies post-strip.

### Added
- `TIER_TO_LEGACY_STRIP_ALIASES` table + `_resolve_strip_prefix` helper in
  `src/sherloc_pipeline/core/r2_keys.py`.
- `PHASE_TEAM_LEGACY_STRIP_ALIASES` + `PHASE_PUBLIC_LEGACY_STRIP_ALIASES` env vars
  (colon-separated path list).
- 4 new tests in `tests/unit/web/test_r2_reader.py` covering legacy-alias acceptance
  + tier-isolation + post-strip traversal-guard preservation.

### Changed
- `derive_r2_key()` and `derive_workspace_key()` now accept either the canonical strip
  prefix OR any per-tier legacy alias. Behavior for canonical-prefix paths is
  unchanged.
- `pyproject.toml` version bumped 4.1.16 → 4.1.17.

## [4.1.9] - 2026-05-18

Backend fix: `/api/map/layers/<scan_id>` returned **HTTP 400** for every
scanner-workspace scan in the containerized deployment, because `core/coordinates.py`
did direct local-FS reads against the per-tier strip-prefix root +
`loupe/<sol>/<scan>/<workspace>/{spatial,loupe}.csv`, which works on the legacy
runtime but fails in the docker container that has no local SHERLOC data mount
(production is pure-R2).

v4.1.9 extends the R2-resolver pattern to cover Loupe-workspace companion files via a
new shared module `web/r2_reader`, adds the `spatial.csv` + `loupe.csv` fetch path to
`coordinates.py`, and refactors `web/routes/images.py` to import the shared primitives
(zero behavior change for ACI fetches).

### Added
- `src/sherloc_pipeline/web/r2_reader.py`: shared R2-reader module. Public API:
  `get_r2_client_and_config`, `is_r2_mode`, `derive_r2_key`, `r2_get_bytes`,
  `r2_head_exists`, `find_colorized_key`, `colorized_variant_exists`,
  **`get_working_file(file_path, filename)`** (NEW — companion-file fetch),
  `set_r2_client_for_tests`, `reset_r2_client_for_tests`. Tier→strip-prefix +
  tier→bucket constants preserved unchanged from v4.1.7. Tier isolation remains
  dual-enforced (code-side strip-prefix table + credential-side R2 bucket scoping).
- `core/coordinates.py:resolve_display_coordinates` and `_resolve_scanner_workspace`:
  new `workspace_reader` keyword arg (`Callable[[str, str], bytes] | None`). When
  provided, the resolver fetches `spatial.csv` + `loupe.csv` via the callable
  (production: `r2_reader.get_working_file`; tests: moto-backed mock), materializes the
  bytes through a `tempfile.TemporaryDirectory()`, then calls the unchanged FS-bound
  `load_spatial_table`. R2-404 from the reader maps to `CoordinatesUnavailableError`
  with an explicit "Loupe workspace files not found in R2 for scan `<id>`" message;
  500/502/504 propagate unchanged.
- `web/routes/map.py:get_map_layers` + `start_map_fit`: branch on
  `r2_reader.is_r2_mode()` to inject `get_working_file` in production (PHASE_TIER +
  AWS_* env set) and `None` for legacy local FS reads in dev.
- `tests/unit/web/test_r2_reader.py` (NEW): moto-backed per-tier resolve +
  cross-tier 403→502 + missing 404 + misconfigured_path + traversal + timeout 504;
  `is_r2_mode` env-var branching.
- `tests/unit/core/test_coordinates.py` (NEW): R2-path happy + 404 + 5xx-propagation
  + malformed-CSV + legacy-FS-path happy + missing.

### Changed
- `src/sherloc_pipeline/web/routes/images.py`: removed the inlined R2 client +
  key-derivation + GET/HEAD machinery and imports the same primitives from
  `web.r2_reader`. Behavior is unchanged; all existing
  `tests/unit/web/test_images.py` cases pass against the refactored module without
  modification.
- `tests/unit/web/test_images.py`: reaches into `web.r2_reader` instead of
  `web.routes.images` for the shared test-injection helpers. The route-level
  `_SCAN_ID_BANNED` regex stays in `images.py`.

### Migrated R2 path
Production containers reach R2 from two code sites; the same per-tier strip-prefix +
bucket scoping applies to both:

| Caller | Endpoint | R2 key shape (team example) |
|---|---|---|
| `web/routes/images.py:get_aci_image` | `GET /api/images/<scan_id>/aci` | `phase-team/sherloc-aci/<rest>/img/<aci>.{PNG,IMG}` |
| `core/coordinates.py` (via `routes/map.py:get_map_layers` + `start_map_fit`) | `GET /api/map/layers/<scan_id>`, `POST /api/map/fit` | `phase-team/sherloc-aci/<rest>/{spatial.csv,loupe.csv}` |

`<rest>` derives from `Path(file_path).parent.parent.relative_to(strip_prefix)` where
`strip_prefix` is the per-tier `TIER_TO_STRIP_PREFIX[tier]` value (distinct team /
public roots; see `web/r2_reader.py`). No new R2 prefix shape introduced; only a new
filename within the existing per-scan workspace directory.

## [4.1.8] - 2026-05-18

Frontend-only fix for two Auth0-related bugs:

- **Bug A — cross-tool SSO silent flow:** the SPA now calls
  `getTokenSilently({ cacheMode: 'on' })` on mount, so users navigating from the apex
  dashboard or the viewer inherit the Auth0 session via hidden iframe without clicking
  "Log in". Failure paths (`login_required`, `consent_required`,
  `interaction_required`) are silenced; unexpected errors get a sanitized
  console.warn (no token/URL leak).
- **Bug B — ACI image + map-layer fetches lacked Bearer auth:** the frontend
  previously used `new Image() + img.src = url` (browser-native, no `Authorization`
  header) and raw `fetch('/api/map/layers/...')` (bypassed the auth-attaching
  `fetchJson` wrapper). Both produced 401 `authn failed reason=no_credential` under
  Auth0 mode (worked under legacy CF Zero Trust cookie auth). v4.1.8 introduces
  authenticated helpers `fetchAciImage()` (fetch → blob → decoded HTMLImageElement)
  and `getMapLayers()` (typed wrapper over `fetchJson`), and a typed
  `AuthRequiredError` for "Log in required" UI states.

The backend resolver (`src/sherloc_pipeline/web/routes/images.py`) is **unchanged**
from v4.1.7 — the bug is purely in frontend HTTP discipline. Backend
`tests/unit/web/test_images.py` still pass without modification.

### Changed
- `src/sherloc_pipeline/web/frontend/src/lib/auth.ts`: add `bootstrapAuthReady`
  promise (auth-readiness gate for protected helpers). Add silent-SSO call in
  `buildAuth0Session()` after `createAuth0Client` returns and BEFORE the
  redirect-callback handler path. Narrow error catch (only expected Auth0 errors
  silenced).
- `src/sherloc_pipeline/web/frontend/src/lib/api.ts`: add `AuthRequiredError` class;
  add private `ensureAuthenticated()` gate; add `fetchAciImage(scanId, opts)`
  (replaces direct `<img src=>` usage) and `getMapLayers(scanId)` (typed,
  auth-attaching wrapper). Mark `getAciImageUrl()` `@deprecated` (kept for any
  unmigrated caller).
- `src/sherloc_pipeline/web/frontend/src/components/AciViewer.svelte`: `loadImage()`
  switches to `fetchAciImage`; preserves the stale-load guard; handles
  `AuthRequiredError` → renders a "Log in to view ACI image" placeholder.
- `src/sherloc_pipeline/web/frontend/src/lib/renderers/BaseImageRenderer.ts`:
  `loadImage(url)` replaced with synchronous `setImage(img)` (the caller now owns the
  fetch).
- `src/sherloc_pipeline/web/frontend/src/components/map/MapMode.svelte`: swap
  `aciUrl: string` → `aciImage: HTMLImageElement | null`; both
  `/api/map/layers/...` raw fetches → `getMapLayers()`; add a stale-load guard and a
  map auth-required UI state.
- `src/sherloc_pipeline/web/frontend/src/components/map/MapCanvas.svelte`: accepts
  `aciImage` / `colorizedAciImage` / `aciLoading` props instead of URLs;
  `loadImage()` replaced with `applyActiveImage()` (sync).

### Note on test coverage
The repo did not yet carry a JS test runner (vitest) at v4.1.8; coverage for the new
auth helpers was added in a later release. Backend `pytest
tests/unit/web/test_images.py` passes (resolver unchanged from v4.1.7), and
`svelte-check` is clean.

## [4.1.7] - 2026-05-15

Implements the R2-aware SHERLOC ACI resolver (hierarchical-key model). Produces the
`ghcr.io/archaeon-ai/sherloc-pipeline:v4.1.7` image.

### Added
- `src/sherloc_pipeline/web/routes/images.py`: full rewrite of the ACI endpoint.
  Replaces local-FS reads with R2 GET via a boto3 S3 client cached at module level.
  A per-tier strip-prefix table (team data root → `phase-team`; public PDS data root
  → `phase-public`; defaults documented in the source and overridable via
  `PHASE_TEAM_STRIP_PREFIX` / `PHASE_PUBLIC_STRIP_PREFIX` env vars) derives the R2 key
  from the DB-stored `context_images.file_path`. Preserves the existing route surface
  (`colorized`, `enhanced`, `upscale` query params; VICAR `.IMG` → PNG in-process
  conversion via tempfile shim through the existing `read_aci_image` helper). New
  module-level helpers: `_get_r2_client_and_config`, `_derive_r2_key`,
  `_r2_get_bytes`, `_r2_head_exists`, `_find_colorized_key`,
  `colorized_variant_exists` (public predicate for `routes/map.py`).
- `boto3>=1.34.0` in the `[web]` optional-dependency extra.
- `moto[s3]>=5.0.0` in the `[dev]` extra for the new R2 test fixtures.
- `tests/unit/web/test_images.py`: full rewrite around a moto-backed S3 mock. New
  tests: team-tier resolve (200), public-tier VICAR convert, missing-object 404,
  cross-tier credential 403→502, R2 timeout → 504 upstream_timeout, R2 non-timeout
  BotoCoreError → 500 upstream_error, misconfigured-path 500 (DB tier ≠ file_path
  prefix), `pds:` LIDVID returns 500 misconfigured_path (an unresolved on-demand ref
  is broken ingestion), colorized variant via R2 (SHA-asserted against base),
  colorized fall-back to base, resolver-config edge cases (PHASE_TIER unset, invalid
  tier, key derivation, path-traversal).
- `src/sherloc_pipeline/web/config_check.py`: new `_check_r2()` — enforces PHASE_TIER
  + AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_ENDPOINT_URL when
  `SHERLOC_AUTH_MODE=auth0`. Fails container startup loudly rather than at first ACI
  request. Defense-in-depth for the same condition the resolver maps to HTTP 500
  "tier_unset" at request time.
- `tests/unit/web/test_config_check.py`: new tests covering the R2 env-var requirement
  under auth0 mode + the dev-mode bypass.
- `Dockerfile`: build-time smoke check extended with `import boto3`.

### Changed
- `src/sherloc_pipeline/web/routes/map.py`: colorized-variant availability now probes
  R2 (via `colorized_variant_exists`) instead of the local FS. Same API-stable
  predicate; the backing store differs.
- `pyproject.toml` version: 4.1.6 → 4.1.7.
- `requirements-lock.txt` regenerated (boto3 + botocore + s3transfer + jmespath added;
  urllib3 bumped per the boto3 transitive requirement).

## [4.1.6] - 2026-05-15

GHCR publish workflow. Produces the `ghcr.io/archaeon-ai/sherloc-pipeline:v4.1.6`
image.

### Added
- `.github/workflows/publish.yml`: tag-triggered GHCR publish for `v*` tags. Builds
  the Dockerfile `runtime` stage on `linux/amd64`, logs into GHCR with
  `GITHUB_TOKEN`, and pushes `ghcr.io/archaeon-ai/sherloc-pipeline:<tag>`.
  Supply-chain extras (cosign sign, SPDX SBOM, SLSA provenance) intentionally
  deferred to a follow-up — see the comment header in the workflow file.

### Fixed
- `Dockerfile` stage 2: align the pre-built `phase-platform-auth` wheel URL with the
  `pyproject.toml` pin so both reference `archaeon-ai/phase-platform-auth@v0.1.0`.

## [4.1.5] - 2026-05-08

### Fixed
- `core/data_ingestion.restructure_fluorescence_data` and `create_r123_spectrum`
  skipped past interleaved `R{1,2,3}_Channel*` header rows in raw Loupe
  `darkSubSpectra*.csv`, restoring `sherloc plot --domain fluor` and `--domain both`
  against R2/R3 from raw workspaces. The database ingest path (which has its own
  section-header guard) was unaffected.

## [4.1.4] - 2026-05-07

### Fixed
- `docker-entrypoint.sh`: launch uvicorn with
  `--factory sherloc_pipeline.web.app:create_app` rather than importing `app`
  directly. The `runtime` stage's smoke check already imports `create_app`, but the
  entrypoint was still resolving `sherloc_pipeline.web.app:app`, which raised
  `AttributeError` at container start.

## [4.1.3] - 2026-05-07

### Fixed
- `Dockerfile` stage 3: drop `--no-index` from the final `pip install`. The PEP 517
  build for the direct-URL `phase-platform-auth` extra needs to fetch its build
  backend (hatchling) from PyPI; `--no-index` blocked that. `/wheels/` is still
  authoritative via `--find-links` for locked deps.

## [4.1.2] - 2026-05-07

### Fixed
- `Dockerfile` stage 3: install `git` in the runtime image too. pip resolves the
  `phase-platform-auth @ git+https://...` direct-URL specifier by cloning the URL even
  when a matching wheel exists in `/wheels/`; `--no-index` does not disable direct-URL
  fetches. Stage 2's pre-built wheel is kept as defense in depth.

## [4.1.1] - 2026-05-07

### Fixed
- `Dockerfile` stage 2: pre-build a wheel for the `[web]` extra's git-URL
  `phase-platform-auth` dep into `/wheels/` so stage 3 can resolve it offline
  alongside the other locked-dep wheels.

## [4.1.0] - 2026-05-06

Auth0 JWT validator switchover. The validator now lives in
[`archaeon-ai/phase-platform-auth`](https://github.com/archaeon-ai/phase-platform-auth)
v0.1.0; SHERLOC consumes it as a runtime dependency rather than shipping a parallel
implementation. CFAccessValidator, DevValidator, and the FastAPI dependency continue
to live in this repo.

### Added
- New runtime dependency `phase-platform-auth >= 0.1.0, < 1` (web extra). Pinned to
  the v0.1.0 git tag until PyPI publish.

### Changed
- `sherloc_pipeline.web.auth.Auth0Validator` is now a re-export of
  `phase_platform_auth.Auth0Validator`. `TokenClaims`, `AuthError`,
  `JWKSUnavailableError`, and `build_www_authenticate` likewise route to the package.
  The SHERLOC import surface is unchanged.
- DevValidator synthetic claims now carry `phase:team-member`; the legacy
  `sherloc:internal` was retired.
- `SHERLOC_AUTH0_IDENTITY_CLAIM_URI` is now mandatory in `auth0` mode; startup fails
  fast if unset. Better than a silent role-name mismatch at runtime.
- The default `WWW-Authenticate` realm is now `m2020-phase`. Override via
  `SHERLOC_AUTH_REALM`.

### Removed
- The legacy `{role_claim_uri}/roles` backward-compat path. An instance running
  without `SHERLOC_AUTH0_IDENTITY_CLAIM_URI` will refuse to start.
- `SHERLOC_AUTH0_ROLE_CLAIM_URI` env var (no longer read by the validator factory or
  the `/api/config` builder).
- Conformance test suite for the Auth0 validator (`tests/unit/test_auth0_validator.py`)
  — replaced with a smoke test that confirms the import surface routes to the package.
  The conformance suite itself lives in the `phase-platform-auth` package.

## [4.0.0] - 2026-04-28

First public release. v4.0.0 supersedes the prior v3.0.0 stable release; the CLI
surface (`full-pipeline`, `plot`, `apply-review`) and Python API
(`sherloc_pipeline.api.spectral`) of v3.0 are preserved.

### Security
- CF Access JWT signature validation now enforced on all authenticated routes.
  Validation covers signature (against the live JWKS), issuer, audience, and expiry.
  The `Cf-Access-Authenticated-User-Email` convenience header is no longer trusted.
- The CORS allowlist is now env-driven via `SHERLOC_CORS_ALLOWED_ORIGINS`; the
  default is empty (no cross-origin requests).
- JWKS unavailability returns HTTP 503 (not 401), with a 24-hour grace window during
  which a stale cache is reused.
- Dev escape hatch: `SHERLOC_AUTH_MODE=dev` bypasses validation and logs a prominent
  startup warning.

### Added
- Web UI (FastAPI + Svelte) with averaged- and per-point spectral exploration,
  classification profiles, and user preferences.
- Map Mode: WebGL-rendered scan-point map with on-demand fitting, push-WebSocket job
  updates, and an inline spectrum viewer.
- PDS4 ingestion (`sherloc pds-ingest`) for Mars 2020 SHERLOC archive data published
  through the PDS Geosciences Node.
- PIXL Pixlise ingestion (`sherloc pixl-ingest`) for cross-instrument context.
- Fluorescence fitting engine (`fit-fluor`, `core/fluor_fitting.py`) with an agnostic
  AICc default and an optional hypothesis-driven strategy. Group assignment for Ce³⁺
  doublet (anhydrite), Ce³⁺ phosphate, and silicate-defect bands.
- Unified peak persistence: `fitted_peaks.fit_modality` discriminates `minerals`,
  `organics`, `hydration`, and `fluorescence`. New `backfill`, `persist-peaks`, and
  `extract-training` CLI commands.
- Cross-modal annotation: fluorescence groups co-scored against Raman mineral
  assignments at the same scan point.
- R1 (523-channel) and R123 (2148-channel) spectrogram visualization pipelines.
- Grain segmentation and morphometry (optional SAM-based, behind the `[ml]` extra).
- Loupe-polynomial wavelength/wavenumber calibration; R123 stitching via Loupe
  overlap summation.
- Parallel per-point fitting for all four domains (configurable via
  `fitting.parallel_workers` and `fluorescence_fitting.parallel_workers`).

### Changed
- The database default location is now repository-relative (`./phase.db`,
  `./phase_pds.db`) rather than a hardcoded local path.
- Web UI configuration is now fully env-driven (`SHERLOC_DB`, `SHERLOC_ACCESS_MODE`,
  `SHERLOC_CORS_ALLOWED_ORIGINS`, `SHERLOC_CF_TEAM_DOMAIN`, `SHERLOC_CF_AUDIENCE`,
  `SHERLOC_AUTH_MODE`).
- Tests are installable on a fresh clone (no absolute paths).

### Removed
- Personal-infrastructure references (paths, hostnames) scrubbed from tracked content.
- Internal coordination tooling and scratch files untracked.
- Experimental research scripts are no longer shipped (kept locally by maintainers;
  not a public surface).

## [3.0.0] - 2024-12-02

### Added

#### New `sherloc plot` Command
A flexible spectral plotting command with three modes:

- **Averaged mode**: Average all points in a scan with optional processing
  ```bash
  sherloc plot --sol 0921 --target Amherst_Point --scan detail_1 \
    --background fs --baseline --fit --export both
  ```

- **Subset mode**: Average specific points (ad-hoc label averaging)
  ```bash
  sherloc plot --sol 0921 --target Amherst_Point --scan detail_1 \
    --points 21,41,49,71,86 --avg trim-mean --baseline --fit
  ```

- **Point mode**: Process single point from Loupe data
  ```bash
  sherloc plot --sol 0921 --target Amherst_Point --scan detail_1 \
    --point 91 --background fs --baseline --fit
  ```

#### Processing Options
- `--background as|fs`: Background subtraction (arm stowed or fused silica)
- `--bgscale auto|<float>`: Automatic PPP-based scaling or explicit value
- `--baseline`: asPLS baseline correction
- `--fit`: Gaussian peak fitting with AICc model selection
- `--single-peak <center>`: Fit exactly one Gaussian near specified position
- `--n-peaks <n>`: Limit automatic peak detection to N peaks maximum
- `--min-snr <float>`: Override minimum SNR threshold (default: 3.0)
- `--fwhm-min <float>`: Override minimum FWHM filter (default: 30 cm⁻¹)
- `--fwhm-max <float>`: Override maximum FWHM constraint (default: 90 cm⁻¹)

#### Python API
New `sherloc_pipeline.api.spectral` module for Jupyter notebook workflows:

- `process_scan_average()`: Process averaged spectrum from Loupe data
- `process_subset_average()`: Process subset of points
- `process_point()`: Process single point from Loupe data
- `load_point_spectrum()`: Load from existing pipeline outputs
- `load_reference_spectrum()`: Load reference mineral spectra
- `plot_spectrum()`: Generate single spectrum plots
- `plot_overlay()`: Generate multi-spectrum comparison plots

#### Example Notebook
- `notebooks/spectral_analysis_example.ipynb`: Complete API usage examples

#### Background Configuration
- Background file paths now configurable via `config.yaml`
- Column name mappings for different background formats
- Interpolation sanity checks with warnings for edge cases

### Changed

#### Dependencies
- Requires Python 3.9+ (was 3.8+) due to `list[int]` type hints
- Relaxed version constraints: `matplotlib>=3.5.0`, `Pillow>=9.0.0`
- Development dependencies (`pytest`, `ruff`, `jupyter`, `ipykernel`) moved to optional `[dev]` extra

#### Baseline Correction
- **CRITICAL FIX**: `baseline_aspls()` now uses all `BaselineParams` fields
  - Previously ignored: `asymmetric_coef`, `iters`, `tol`
  - Now correctly passed to `pybaselines.Baseline.aspls()`
  - Ensures consistent baseline behavior across full-pipeline and sherloc plot

#### Documentation
- PRD renamed to `docs/PRD.md`
- README updated with Python API section and examples
- New `docs/API.md` for API reference

### Fixed
- Background interpolation now warns when spectrum exceeds background range
- PPP scaling warns for missing or zero values
- Baseline parameters properly propagated to asPLS algorithm

### Removed
- Development prototype scripts in `scripts/` directory
- Vestigial plotting code comments about "deterministic" output

---

## [1.0.0] - 2024-11-26

Initial release with `full-pipeline` and `apply-review` commands.

### Added
- `sherloc full-pipeline`: Complete processing from Loupe data to spatial overlays
- `sherloc apply-review`: Manual review workflow with overlay regeneration
- Automated spectral preprocessing (despike, baseline, background subtraction)
- Gaussian peak fitting with quality flags
- Mineral classification by wavenumber ranges
- Spatial overlay rendering on ACI context images
