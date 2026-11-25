# Architecture

This document describes the runtime structure of the SHERLOC pipeline:
how the packages fit together, where data flows on a typical request,
and what an operator needs to know to deploy the web UI safely. It is
intended to be a living document — when the structure here drifts from
the code, the code is authoritative; please update this file in the
same change.

The codebase is organized as 12 first-party packages under
`src/sherloc_pipeline/` (11 code packages plus a `data/` reference-data
package). They form three rough layers:

```
                ┌──────────────────────────────────────────────┐
                │  cli  ◄──────►  api  ◄──────►  web (FastAPI) │   entry points
                └────┬───────────────┬──────────────────┬──────┘
                     │               │                  │
                ┌────▼───────────────▼──────────────────▼──────┐
                │  services (orchestration)                    │
                │  pipeline / ingestion / spectral / map_fit / │   business logic
                │  spectrogram / identification / preprocessing│
                │  / pds_ingestion / pixl_ingestion / ...      │
                └────┬─────────────────────────────────────────┘
                     │
                ┌────▼─────────────────────────────────────────┐
                │  core (algorithms)            visualization   │
                │  fitting / fluor_fitting /    fitting_plots / │   primitives
                │  baseline / preprocessing /   spatial /       │
                │  calibration / r1_extraction  cooccurrence    │
                │  / r123_stitching / mineral_id                │
                │                                               │
                │  vision (ACI image reading)                   │
                ├──────────────────────────────────────────────┤
                │  database (SQLAlchemy ORM, Alembic)           │
                │  models  (Pydantic DTOs + type registry)      │
                └──────────────────────────────────────────────┘
```

## Packages

| Package | Purpose | Key modules |
|---------|---------|-------------|
| `cli/` | Typer entry point exposing every batch command. | `app.py` — 23 commands: `apply-review`, `backfill`, `backfill-fluor-inputs`, `backfill-masks`, `backfill-public-release`, `db-stats`, `extract-training`, `fit-fluor`, `full-pipeline`, `ingest`, `init`, `pds-download`, `pds-ingest`, `persist-peaks`, `pixl-ingest`, `plot`, `process-new`, `reclassify-product-roles`, `reclassify-scan-classes`, `reclassify-scan-types`, `reclassify-targets`, `refit`, `serve`; and two command groups, `references` and `identify`. |
| `api/` | Notebook-friendly functional API. Stateless wrappers around services and core, intended for Jupyter use. | `spectral.py` — `process_scan_average`, `process_subset_average`, `process_point`, `load_point_spectrum`, `load_reference_spectrum`, `plot_spectrum`, `plot_overlay`. |
| `web/` | FastAPI app + Svelte frontend. Read-mostly exploration UI. | `app.py` (factory + middlewares), `auth.py` (CF Access JWT validation), `routes/` (12 modules), `frontend/` (Svelte). |
| `services/` | Orchestrators that compose `core` primitives and persist results. | `pipeline.py` (the 7-step run), `handoff.py` (optional atomic delivery manifest), `ingestion.py` / `pds_ingestion.py` / `pixl_ingestion.py`, `spectral.py`, `map_fitting.py`, `spectrogram.py`, `preprocessing.py`. |
| `core/` | Pure algorithmic primitives. Should not depend on `services/`. | `preprocessing.py`, `baseline.py`, `fitting.py` (Raman cm⁻¹), `fluor_fitting.py` / `fluor_id.py` (fluorescence in nm), `calibration.py`, `r1_extraction.py`, `r123_stitching.py`, `mineral_id.py`, `pds_client.py`, `pds_parsers.py`. |
| `models/` | Pydantic DTOs + a small type registry. Shared between CLI/API/web/services. | `spectra.py`, `fitting.py`, `instrument.py`, `pds.py`, `pixl.py`, `spectrogram.py`. |
| `database/` | SQLAlchemy ORM, connection management, Alembic glue. | `models.py` (17 ORMs), `connection.py` (`get_engine`, `init_pds_database`), `pixl_models.py`. |
| `visualization/` | Matplotlib figure generation. | `fitting_plots.py`, `spatial.py`, `preprocessing_plots.py`, `cooccurrence.py`. |
| `ml_despike/` | ML cosmic-ray detector: frozen manifest (sha256-pinned ONNX), sha256-verified fetch/cache, certified 8-channel featurization, ONNX CPU inference. Behind the `[ml-despike]` extra; never imported by `modz`/`none` paths. | `manifest.py`, `featurize.py`, `artifact.py`, `detector.py`. |
| `vision/` | ACI image reading (VICAR/PDS3 `.IMG`). | `img_reader.py`. |
| `data/` | Packaged spectral references. The Workbench overlay marks anomalous pixels (Jakubek et al. 2025, Supplemental Tables S3 and S4) without changing spectral values. | `background/` (arm-stowed and fused-silica calibration spectra), `anomalous_pixels.csv`. |

The full Pydantic + ORM schema and the Loupe wavelength polynomial
are in [`docs/schema/UNIFIED_SCHEMA.md`](schema/UNIFIED_SCHEMA.md) and
[`docs/schema/SPECTRAL_REGIONS.md`](schema/SPECTRAL_REGIONS.md).

## Data model

Two SQLite databases share an identical schema (21 tables: the 17 in `database/models.py` and the four reference
catalog tables in `database/reference_models.py`; 21 Alembic migrations
on the head chain):

- `phase.db` — Loupe-source data (proprietary tier; not redistributed).
- `phase_pds.db` — PDS4 archive data only (publicly released).

The web app's `SHERLOC_ACCESS_MODE=public` profile rejects any DB
whose path or URL still references `phase.db`, so the public deployment
is structurally prevented from serving Loupe data.

Key tables: `sols`, `scans`, `scan_points`, `spectra`,
`instrument_states`, `ccd_configurations`, `scanner_calibrations`,
`context_images`, `regions_of_interest`, `fitted_peaks`,
`spectrograms`, `map_display_coordinates`, `users`,
`user_preferences`, `classification_profiles`, `map_fit_cache`,
`cosmic_ray_masks`, `reference_releases`, `reference_files`,
`reference_traces`, `reference_templates`.

`cosmic_ray_masks` — per-(spectrum, region, method) channel-index arrays
with full provenance (method label, model sha256, per-region tau).
Satellite to `spectra` with CASCADE lifecycle; `UNIQUE(spectrum_id, method)`
prevents duplicate runs. The web `?despike=true` toggle reads these stored
masks; no model inference ever runs on the serving host.

`users`, `user_preferences` and `classification_profiles` have no writer:
the `/api/user/*` routes that created them are retired (they now
return 404). The tables and their migrations stay so existing databases,
the nightly pull and backups keep their rows. `map_fit_cache` is read by
`routes/map.py` but never written.

`fitted_peaks.fit_modality` discriminates four peak domains:
`minerals`, `organics`, `hydration`, `fluorescence`. Raman peaks carry
`center_cm1` / `fwhm_cm1`; fluorescence peaks carry `center_nm` /
`fwhm_nm` (with an `is_saturated` flag). DB triggers enforce that
each row is consistent with its `fit_modality`.

## Pipeline flow

`services/pipeline.py:PipelineService.run_full_pipeline()` is the
canonical batch path. Seven steps:

1. **Preprocessing** — despike, baseline, background subtraction.
2. **Raman fitting** (cm⁻¹ space) for `minerals`, `organics`,
   `hydration`. F-test or AICc model selection.
3. **Raman peak persistence** for the three Raman domains.
4. **Fluorescence fitting** (nm space) — AICc selects the number of Gaussian
   peaks. After fitting, the pipeline assigns a group label to each peak using
   the group numbers of [Sharma et al. (2023)](https://doi.org/10.1038/s41586-023-06143-z).
   This pipeline splits group 1 into 1a (shorter wavelength) and 1b (longer
   wavelength) and labels each peak on its own; Haney et al. (2026,
   https://doi.org/10.1029/2025JE009385) discuss this wavelength range as
   Ce³⁺-bearing anhydrite. Only the training export pairs peaks into doublets.
   Group assignments use the fluorescence peak wavelengths. New fluorescence
   peaks have NULL assignment confidence in the database.
5. **Per-scan averages** — fits of the scan's R1 mean and trimmed-mean
   spectra. The trimmed mean cuts 2 % from each tail, raised for small
   scans so that at least one point is cut from each tail (a 25-point
   scan cuts 4 %).
6. **Spatial overlay rendering** on ACI context images.
7. **Summary** — prints the scan's fit counts.

After step 7 succeeds, the pipeline records an explicit delivery-handoff skip
unless `SHERLOC_HANDOFF_DIR` is configured. A configured run inventories all
raw ACI PNGs in that scan workspace and any mirrored colorized PNGs plus their
`spatial.csv` and `loupe.csv` companions. It hashes the source bytes, records
portable locators and dimensions, and publishes `<run_id>.tmp` to
`<run_id>.ready` with one atomic rename. Per-scan manifests select the exact
product IDs they contain, rather than claiming every product from the sol.
Publication failure is fatal; the hook does not write a database or alter the
web-serving path.

Steps 2 and 3 each run their per-point fits in a `ProcessPoolExecutor`.
Worker count is resolved by `core.utils.resolve_parallel_workers()`:
`0` = half the cores, `1` = sequential, any other value = explicit.

## Web request flow

```
client (browser, Svelte SPA)
        │
        ▼
Cloudflare Access  ── JWT (Cf-Access-Jwt-Assertion header)
        │
        ▼
FastAPI app (uvicorn, 127.0.0.1:8000)
  ├── CORS middleware (env-driven allowlist — empty by default)
  ├── public_guards_middleware (rate limit + body-size cap, public mode only)
  ├── db_session_middleware (per-request SQLAlchemy session)
  └── routes:
      ├── /api/health             (health.py)
      ├── /api/scans              (scans.py)
      ├── /api/spectra            (spectra.py)
      ├── /api/process            (processing.py — baseline / fit / despike / bg-sub)
      ├── /api/plots              (plots.py)
      ├── /api/images             (images.py)
      ├── /api/pds                (pds.py)
      ├── /api/jobs               (jobs.py — generic job queue)
      ├── /api/map                (map.py — Map Mode + WebSocket)
      ├── /api/identify           (identification.py)
      ├── /api/references         (references.py)
      └── /api/config             (config.py)
```

Authentication: `web/auth.py:validate_cf_jwt` verifies the JWT
signature against Cloudflare's live JWKS, plus issuer
(`https://<SHERLOC_CF_TEAM_DOMAIN>`), audience (`SHERLOC_CF_AUDIENCE`),
and expiry. JWKS is cached with a 1-hour TTL and a 24-hour grace
window on fetch failure; outside the grace window — or with no cache —
authenticated requests return HTTP 503, never 401. Setting
`SHERLOC_AUTH_MODE=dev` short-circuits to a hardcoded `dev@local`
identity (with a startup warning logged).

The `Cf-Access-Authenticated-User-Email` convenience header is
**not** trusted; identity is read from the validated JWT claims.

Long-running fits in Map Mode are dispatched onto a 1-worker
`ThreadPoolExecutor`; progress streams to clients via
`ws_map.MapJobRegistry` over a WebSocket.

### Browser routes and links from PHASE

The SPA uses hash routes. PHASE can open a scan without knowing its pipeline
UUID by linking to `#/open?scan_name=<encoded-name>&sol=<sol>&point=<index>`.
Use `points=<index>,<index>,...` for a subset of a **single scan**. These examples
use an invented scan, sol and hosts (`sherloc.example.org` for the team
tier, `sherloc-public.example.org` for the public tier):

```text
https://sherloc.example.org/#/open?scan_name=Testrock_detail&sol=42424&point=0
https://sherloc-public.example.org/#/open?scan_name=Testrock_detail&sol=42424&points=0,2,7
```

Choose the host for the viewer's tier: team → `sherloc.example.org`, public
→ `sherloc-public.example.org`. Authentication uses that host's existing
flow. The resolver searches only scans visible in that tier; it never retries
on another host. Build query strings with `URLSearchParams` (or encode each
value with `encodeURIComponent`) so spaces, `&`, `+`, and `%` in names survive:

```js
const query = new URLSearchParams({ scan_name: 'Testrock_detail', sol: '42424', points: '0,2,7' });
const url = `https://sherloc.example.org/#/open?${query}`;
```

The resolver calls `GET /api/scans?scan_name=<encoded-name>&sol=<sol>` using
the existing list API. `scan_name` is an exact, case-sensitive match (no
substring, wildcard, or space/underscore normalization). The response retains
the list API's `scans`, `total`, `offset`, and `limit` fields; the resolver reads
all pages before deciding whether the match is unique. An unavailable scan
returns an empty list and displays **No scan found ... in this tier**. Multiple
matches are listed with target, scan name, sol, point count, class, type, and IDs;
the user must choose one. The resolver does not guess from target or point count.

A unique match or explicit choice replaces the resolver history entry with
`#/scan/<uuid>/workbench?point=N` or `#/scan/<uuid>/workbench?points=A,B,C`.
Indices are zero-based, nonnegative whole numbers. `point` selects one spectrum;
`points` starts the Workbench in subset averaging mode, even for a one-item
list. Repeated indices are removed, preserving their first occurrence. Sol is
also a nonnegative whole number. Missing name/sol, malformed numbers, empty
lists, and simultaneous `point` and `points` parameters show an error. Missing
point indices show an out-of-range error, including holes in sparse scans;
the resolver and direct Workbench links both reject them. Omitting both point
parameters opens the scan average. Changing either selection parameter for an
already open Workbench reloads that selection. Subsets that span more than one
scan are not supported.

## Deployment notes

- `scripts/serve.sh` is the canonical launch script. It binds uvicorn
  to `127.0.0.1:8000` and reads configuration from environment
  variables.
- Production deployments must front uvicorn with an authenticating
  proxy (Cloudflare Access or equivalent). See
  [`SECURITY.md`](../SECURITY.md) for the full posture and the
  required environment variables (`SHERLOC_CF_TEAM_DOMAIN`,
  `SHERLOC_CF_AUDIENCE`, etc.).
- The Svelte frontend is built once via `npm run build` in
  `src/sherloc_pipeline/web/frontend/`; the resulting `dist/` is
  served by FastAPI as static assets. Both the team-internal and
  PDS-only deployments share the same `dist/`.
- Database schema migrations are managed by Alembic (batch mode for
  SQLite). See [`docs/schema/UNIFIED_SCHEMA.md`](schema/UNIFIED_SCHEMA.md)
  for the current head and a migration walk-through.

## Calibration boundary

> **Never use `np.linspace()` for wavenumber axes.** Channel-to-wavenumber
> mapping uses the Loupe V5.1.5a polynomial. The repository distinguishes
> two operations that are sometimes conflated:
>
> - *Wavelength / wavenumber calibration* — channel index → physical units
>   via the Loupe polynomial coefficients (Raman vs fluorescence sets).
> - *Laser normalization* — photodiode-based intensity correction across
>   points within a scan.
>
> Calibration lives in `core/calibration.py`
> (`calculate_loupe_wavelength_wavenumber`); laser normalization lives
> in `core/laser_normalization.py` (`process_laser_normalization`).
> They are not interchangeable. The reference is
> [`docs/schema/SPECTRAL_REGIONS.md`](schema/SPECTRAL_REGIONS.md).
