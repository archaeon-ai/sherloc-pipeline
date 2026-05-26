# Architecture

This document describes the runtime structure of the SHERLOC pipeline:
how the packages fit together, where data flows on a typical request,
and what an operator needs to know to deploy the web UI safely. It is
intended to be a living document — when the structure here drifts from
the code, the code is authoritative; please update this file in the
same change.

The codebase is organized as nine first-party packages under
`src/sherloc_pipeline/`. They form three rough layers:

```
                ┌──────────────────────────────────────────────┐
                │  cli  ◄──────►  api  ◄──────►  web (FastAPI) │   entry points
                └────┬───────────────┬──────────────────┬──────┘
                     │               │                  │
                ┌────▼───────────────▼──────────────────▼──────┐
                │  services (orchestration)                    │
                │  pipeline / ingestion / spectral / map_fit / │   business logic
                │  spectrogram / classification / preprocessing│
                │  / pds_ingestion / pixl_ingestion / ...      │
                └────┬─────────────────────────────────────────┘
                     │
                ┌────▼─────────────────────────────────────────┐
                │  core (algorithms)            visualization   │
                │  fitting / fluor_fitting /    fitting_plots / │   primitives
                │  baseline / preprocessing /   spatial /       │
                │  calibration / r1_extraction  spectrograms    │
                │  / r123_stitching / mineral_id                │
                │                                               │
                │  vision (segmentation, optional)              │
                │  ml (clustering, similarity, features)        │
                ├──────────────────────────────────────────────┤
                │  database (SQLAlchemy ORM, Alembic)           │
                │  models  (Pydantic DTOs + type registry)      │
                └──────────────────────────────────────────────┘
```

## Packages

| Package | Purpose | Key modules |
|---------|---------|-------------|
| `cli/` | Typer entry point exposing every batch command. | `app.py` — 14 commands: `full-pipeline`, `plot`, `apply-review`, `ingest`, `process-new`, `db-stats`, `pds-ingest`, `pixl-ingest`, `fit-fluor`, `persist-peaks`, `backfill`, `extract-training`, `reclassify-targets`, `serve`. |
| `api/` | Notebook-friendly functional API. Stateless wrappers around services and core, intended for Jupyter use. | `spectral.py` — `process_scan_average`, `process_subset_average`, `process_point`, `load_point_spectrum`, `load_reference_spectrum`, `plot_spectrum`, `plot_overlay`. |
| `web/` | FastAPI app + Svelte frontend. Read-mostly exploration UI. | `app.py` (factory + middlewares), `auth.py` (CF Access JWT validation), `routes/` (12 modules), `frontend/` (Svelte). |
| `services/` | Orchestrators that compose `core` primitives and persist results. | `pipeline.py` (the 7-step run), `ingestion.py` / `pds_ingestion.py` / `pixl_ingestion.py`, `spectral.py`, `map_fitting.py`, `spectrogram.py`, `classification.py`, `preprocessing.py`. |
| `core/` | Pure algorithmic primitives. Should not depend on `services/`. | `preprocessing.py`, `baseline.py`, `fitting.py` (Raman cm⁻¹), `fluor_fitting.py` / `fluor_id.py` / `fluor_detection.py` (fluorescence in nm), `calibration.py`, `r1_extraction.py`, `r123_stitching.py`, `mineral_id.py`, `pds_client.py`, `pds_parsers.py`. |
| `models/` | Pydantic DTOs + a small type registry. Shared between CLI/API/web/services. | `spectra.py`, `fitting.py`, `instrument.py`, `pds.py`, `pixl.py`, `spectrogram.py`. |
| `database/` | SQLAlchemy ORM, connection management, Alembic glue. | `models.py` (16 ORMs), `connection.py` (`get_engine`, `init_pds_database`), `pixl_models.py`. |
| `visualization/` | Matplotlib/Plotly figure generation. | `fitting_plots.py`, `spatial.py`, `spectrograms.py`, `preprocessing_plots.py`, `cooccurrence.py`. |
| `vision/` | ACI image segmentation and morphometry. SAM-based; behind the `[ml]` extra. | `segmentation.py`, `morphometry.py`, `img_reader.py`. |
| `ml/` | Cross-spectrum analytics: clustering, distance, feature extraction, similarity. | `clustering.py`, `distance.py`, `features.py`, `similarity.py`. |

The full Pydantic + ORM schema and the Loupe wavelength polynomial
are in [`docs/schema/UNIFIED_SCHEMA.md`](schema/UNIFIED_SCHEMA.md) and
[`docs/schema/SPECTRAL_REGIONS.md`](schema/SPECTRAL_REGIONS.md).

## Data model

Two SQLite databases share an identical schema (16 tables, 11 Alembic
migrations on the head chain):

- `phase.db` — Loupe-source data (proprietary tier; not redistributed).
- `phase_pds.db` — PDS4 archive data only (publicly released).

The web app's `SHERLOC_ACCESS_MODE=public` profile rejects any DB
whose path or URL still references `phase.db`, so the public deployment
is structurally prevented from serving Loupe data.

Key tables: `sols`, `scans`, `scan_points`, `spectra`,
`instrument_states`, `ccd_configurations`, `scanner_calibrations`,
`context_images`, `regions_of_interest`, `fitted_peaks`,
`spectrograms`, `map_display_coordinates`, `users`,
`user_preferences`, `classification_profiles`, `map_fit_cache`.

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
3. **Fluorescence fitting** (nm space) — agnostic AICc default, or
   an optional hypothesis-driven strategy with falling-back-to-agnostic
   when constrained models score poorly. Doublet detection only fires
   in the Group-1 wavelength window (Ce³⁺ in anhydrite at Berry Hollow).
4. **Raman peak persistence** for the three Raman domains.
5. **Per-scan averages** (line / detail trim percentiles differ by scan
   type — line scans 4 %, detail scans 2 %).
6. **Spatial overlay rendering** on ACI context images.
7. **Summary** — companion JSON / Markdown report.

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
      ├── /api/user               (user.py — preferences, classification profiles)
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
> They are not interchangeable. The canonical reference is
> [`docs/schema/SPECTRAL_REGIONS.md`](schema/SPECTRAL_REGIONS.md).
