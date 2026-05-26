# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.1.17] - 2026-05-24

Backend fix for an operator-surfaced UI regression on the v1.0-beta VPS
deployment: any scan whose `context_images.file_path` carried a
pre-v4.1.9 NAS-mount prefix (the deployment's legacy NAS root, recorded
as the new `PHASE_TEAM_LEGACY_STRIP_ALIASES` default) returned
**HTTP 500 `misconfigured_path`** on every ACI-resolving endpoint
(`/api/images/{id}/aci`, `/api/scans/{id}`, `/api/map/layers/{id}`),
making the affected scans unbrowsable in Workbench + Map Mode. Live
audit found 3 such rows on the team-tier production DB (Mauchsberg sol
1810, Pafuri Gate sol 1798) that had been failing since the v4.1.9 R2
migration shipped.

`core/r2_keys.py` now accepts per-tier *legacy aliases* in addition to
the canonical strip prefix. Legacy aliases share the same R2 byte
layout under `sherloc-aci/<rel>/` (rclone preserves the post-prefix
tree), so the alias resolves to an identical key without code- or
data-layer changes elsewhere. Default carries the single
known-in-flight team-tier alias; deployments may add more via the
new `PHASE_TEAM_LEGACY_STRIP_ALIASES` / `PHASE_PUBLIC_LEGACY_STRIP_ALIASES`
env vars (colon-separated).

Tier-isolation invariants preserved — public-tier rejects team-tier
legacy aliases (and vice versa); path-traversal guard still applies
post-strip.

Operational pairing: the v4.1.17 deploy is gated on an `rclone` of the
2 missing NAS-side colorized variants — `sol_1810_colorized/` (Mauchsberg)
and `sol_1853_colorized/` (Côte d'Or) — into the team R2 bucket under
`sherloc-aci/loupe/`, plus `sol_1810/` (Mauchsberg base, also absent
from R2). Without those bytes, the code-side alias acceptance makes
Mauchsberg browsable (no 500) but the ACI still resolves 404; with the
bytes restored, Mauchsberg renders normally and the colorize button
enables for Côte d'Or.

### Added

- `TIER_TO_LEGACY_STRIP_ALIASES` table + `_resolve_strip_prefix` helper
  in `src/sherloc_pipeline/core/r2_keys.py`.
- `PHASE_TEAM_LEGACY_STRIP_ALIASES` + `PHASE_PUBLIC_LEGACY_STRIP_ALIASES`
  env vars (colon-separated path list).
- 4 new tests in `tests/unit/web/test_r2_reader.py` covering legacy-alias
  acceptance + tier-isolation + post-strip traversal-guard preservation.

### Changed

- `derive_r2_key()` and `derive_workspace_key()` now accept either the
  canonical strip prefix OR any per-tier legacy alias. Behavior for
  canonical-prefix paths is unchanged (1090/1093 of team-tier ACI rows).
- `pyproject.toml` version bumped 4.1.16 → 4.1.17.

### Investigation artifacts (not shipped)

- `_scratch/sessions/ui_regression_P1_evidence.md` — Playwright reproduction.
- `_scratch/sessions/ui_regression_P3_hypothesis.md` — root-cause analysis.
- `_scratch/sessions/ui_regression_P4_fix_proposals.md` — fix decision matrix.

## [4.1.9] - 2026-05-18

Backend fix for the last D12 blocker surfaced at end of m2020-phase
Session 93: `/api/map/layers/<scan_id>` returned **HTTP 400** for every
scanner-workspace scan on the v1.0-beta VPS deployment, because
`core/coordinates.py` did direct local-FS reads against the per-tier
strip-prefix root + `loupe/<sol>/<scan>/<workspace>/{spatial,loupe}.csv`,
which works on the legacy runtime but fails in the docker container that
has no local SHERLOC data mount (m2020-phase spec §3.9.6 — production is
pure-R2).

v4.1.9 extends the §3.9 R2-resolver pattern to cover Loupe-workspace
companion files via a new shared module `web/r2_reader`, adds the
`spatial.csv` + `loupe.csv` fetch path to `coordinates.py`, and refactors
`web/routes/images.py` to import the shared primitives (zero behavior
change for ACI fetches).

Spec amendment: m2020-phase
`docs/PHASE_PLATFORM_v1.0_SPEC-revised.md` §3.9.8 (Session 94 commit).

### Added

- `src/sherloc_pipeline/web/r2_reader.py`: shared R2-reader module.
  Public API: `get_r2_client_and_config`, `is_r2_mode`, `derive_r2_key`,
  `r2_get_bytes`, `r2_head_exists`, `find_colorized_key`,
  `colorized_variant_exists`, **`get_working_file(file_path, filename)`**
  (NEW — spec §3.9.8.2 companion-file fetch), `set_r2_client_for_tests`,
  `reset_r2_client_for_tests`. Tier→strip-prefix + tier→bucket constants
  preserved unchanged from v4.1.7. Tier isolation remains dual-enforced
  (code-side strip-prefix table + credential-side R2 bucket scoping).
- `core/coordinates.py:resolve_display_coordinates` and
  `_resolve_scanner_workspace`: new `workspace_reader` keyword arg
  (`Callable[[str, str], bytes] | None`). When provided, the resolver
  fetches `spatial.csv` + `loupe.csv` via the callable (production:
  `r2_reader.get_working_file`; tests: moto-backed mock), materializes
  the bytes through a `tempfile.TemporaryDirectory()`, then calls the
  unchanged FS-bound `load_spatial_table`. R2-404 from the reader maps
  to `CoordinatesUnavailableError` with an explicit "Loupe workspace
  files not found in R2 for scan `<id>`" message; 500/502/504 propagate
  unchanged.
- `web/routes/map.py:get_map_layers` + `start_map_fit`: branch on
  `r2_reader.is_r2_mode()` to inject `get_working_file` in production
  (PHASE_TIER + AWS_* env set) and `None` for legacy local FS reads in
  dev.
- `tests/unit/web/test_r2_reader.py` (NEW; 12 tests): moto-backed
  per-tier resolve + cross-tier 403→502 + missing 404 +
  misconfigured_path + traversal + timeout 504; `is_r2_mode` env-var
  branching.
- `tests/unit/core/test_coordinates.py` (NEW; 6 tests): R2-path happy +
  404 + 5xx-propagation + malformed-CSV + legacy-FS-path happy + missing.

### Changed

- `src/sherloc_pipeline/web/routes/images.py`: removed the inlined R2
  client + key-derivation + GET/HEAD machinery (lines 47-296 of v4.1.8)
  and imports the same primitives from `web.r2_reader`. Behavior is
  unchanged; all 19 existing `tests/unit/web/test_images.py` cases pass
  against the refactored module without modification.
- `tests/unit/web/test_images.py`: reaches into `web.r2_reader` instead
  of `web.routes.images` for the shared test-injection helpers
  (`set_r2_client_for_tests`, `reset_r2_client_for_tests`,
  `get_r2_client_and_config`, `derive_r2_key`). The route-level
  `_SCAN_ID_BANNED` regex stays in `images.py`.

### Migrated R2 path (full surface after v4.1.9)

Production VPS containers reach R2 from two code sites; the same
per-tier strip-prefix + bucket scoping applies to both:

| Caller | Endpoint | R2 key shape (team example) |
|---|---|---|
| `web/routes/images.py:get_aci_image` | `GET /api/images/<scan_id>/aci` | `phase-team/sherloc-aci/<rest>/img/<aci>.{PNG,IMG}` |
| `core/coordinates.py` (via `routes/map.py:get_map_layers` + `start_map_fit`) | `GET /api/map/layers/<scan_id>`, `POST /api/map/fit` | `phase-team/sherloc-aci/<rest>/{spatial.csv,loupe.csv}` |

`<rest>` derives from `Path(file_path).parent.parent.relative_to(strip_prefix)`
where `strip_prefix` is the per-tier ``TIER_TO_STRIP_PREFIX[tier]`` value
(distinct team / public roots; see ``web/r2_reader.py``). No new R2 prefix shape introduced; only a new filename within
the existing per-scan workspace directory.

### Test coverage

- 19/19 existing `test_images.py` cases pass (refactor verification).
- 22/22 `test_r2_reader.py` cases pass (12 baseline + 6 disallowed-
  filename parametrize per R1 F1 + 3 derive_workspace_key per R2 F4 +
  1 PHASE_TIER="" → True per R3 F5). is_r2_mode tests updated per R2 F3
  + R3 F5 — predicate now uses `"PHASE_TIER" in os.environ` (key
  presence, not value truthiness), so production misconfigurations
  including empty-string PHASE_TIER surface as tier_unset 500 instead
  of silent FS fallback.
- 7/7 new `test_coordinates.py` cases pass (R2 + FS branches +
  loupe.csv-404-names-loupe.csv per R3 F4-residual).
- 8/8 new `test_map_routes.py` cases pass — route-level 5xx propagation:
  /api/map/layers preserves R2 404→400 (via CoordinatesUnavailableError),
  502 upstream_credential_error, 504 upstream_timeout, 500
  misconfigured_path; PHASE_TIER-set + AWS-missing surfaces 500
  tier_unset (R2 F3); /api/map/fit propagates the same R2-mode failures
  (R2 F2 final close).
- Cumulative new + modified surface: 56 passed in 2.6s.
- Frontend `svelte-check` unchanged (no JS source changes in v4.1.9).

### Carryover to next release

- Frontend JS test infrastructure (vitest + Playwright) and automated
  coverage for the v4.1.8 auth helpers remain a waived manual gate.
  Will be added in a dedicated follow-up PR (operator-scheduled Session
  95). Same scope as the v4.1.8 CHANGELOG carryover.

## [4.1.8] - 2026-05-18

Frontend-only fix for two Auth0-related bugs surfaced at end of m2020-phase
Session 92 on `sherloc.m2020-phase.net` (blocks D12 beta team email):

- **Bug A — cross-tool SSO silent flow:** SHERLOC SPA now calls
  `getTokenSilently({ cacheMode: 'on' })` on mount, so users navigating
  from `m2020-phase.net` (apex dashboard) or `viewer.m2020-phase.net`
  inherit Auth0 session via hidden iframe without clicking "Log in".
  Failure paths (`login_required`, `consent_required`,
  `interaction_required`) are silenced; unexpected errors get sanitized
  console.warn (no token/URL leak).
- **Bug B — ACI image + map-layer fetches lacked Bearer auth:** the
  frontend previously used `new Image() + img.src = url` (browser-native,
  no `Authorization` header) and raw `fetch('/api/map/layers/...')`
  (bypassed the auth-attaching `fetchJson` wrapper). Both produced 401
  `authn failed reason=no_credential` under Auth0 mode (worked under
  legacy CF Zero Trust cookie auth). v4.1.8 introduces authenticated
  helpers `fetchAciImage()` (fetch → blob → decoded HTMLImageElement)
  and `getMapLayers()` (typed wrapper over `fetchJson`), and a typed
  `AuthRequiredError` for "Log in required" UI states.

Backend resolver (`src/sherloc_pipeline/web/routes/images.py`) is
**unchanged** from v4.1.7 — the bug is purely in frontend HTTP discipline.
Backend `tests/unit/web/test_images.py` still pass without modification.

(2 rounds Codex peer review; GO; all 10 findings accepted).

### Changed

- `src/sherloc_pipeline/web/frontend/src/lib/auth.ts`: add
  `bootstrapAuthReady` promise (auth-readiness gate for protected
  helpers; lets components await bootstrap settlement without polling
  `getSession()`). Add silent-SSO call in `buildAuth0Session()` after
  `createAuth0Client` returns and BEFORE the redirect-callback handler
  path. Narrow error catch (only expected Auth0 errors silenced).
- `src/sherloc_pipeline/web/frontend/src/lib/api.ts`: add
  `AuthRequiredError` class; add private `ensureAuthenticated()` gate;
  add `fetchAciImage(scanId, opts)` (replaces direct `<img src=>` usage)
  and `getMapLayers(scanId)` (typed, auth-attaching wrapper). Mark
  `getAciImageUrl()` `@deprecated` (kept for any unmigrated caller).
- `src/sherloc_pipeline/web/frontend/src/components/AciViewer.svelte`:
  `loadImage()` switches to `fetchAciImage`; preserves `loadGeneration`
  stale-load guard; handles `AuthRequiredError` → renders "Log in to
  view ACI image" placeholder.
- `src/sherloc_pipeline/web/frontend/src/lib/renderers/BaseImageRenderer.ts`:
  `loadImage(url)` replaced with synchronous `setImage(img)` (caller now
  owns the fetch).
- `src/sherloc_pipeline/web/frontend/src/components/map/MapMode.svelte`:
  swap `aciUrl: string` → `aciImage: HTMLImageElement | null`; both
  `/api/map/layers/...` raw fetches → `getMapLayers()`; add
  `mapLoadGeneration` stale-load guard; `mapAuthRequired` UI state.
- `src/sherloc_pipeline/web/frontend/src/components/map/MapCanvas.svelte`:
  accepts `aciImage` / `colorizedAciImage` / `aciLoading` props instead
  of URLs; `loadImage()` replaced with `applyActiveImage()` (sync).

### Test coverage — EXPLICIT WAIVER FOR v4.1.8

The peer-reviewed design memo (and Codex PR9 R1 F3) called for frontend
unit + Playwright E2E coverage of:

- `fetchAciImage()` Authorization header attachment + AuthRequiredError path
- `getMapLayers()` Authorization header attachment + AuthRequiredError path
- `bootstrapAuthReady` race prevention (no anonymous protected request before
  bootstrap settles)
- Silent-SSO `getTokenSilently()` invocation during `buildAuth0Session()`
- Canvas pixel-non-zero assertion after `fetchAciImage` blob revocation

The repo does not currently carry vitest or Playwright in the frontend
`package.json` (only `svelte-check`). The operator-approved scope decision
for v4.1.8 is to **WAIVE** JS test coverage for this hotfix, on the
grounds that D12 (operator beta team email) is blocked indefinitely
until ACI rendering works, and adding the test infrastructure (vitest +
playwright + their configs + the test specs) is comparable in size to
the fix itself. Acceptance gates instead are:

1. `svelte-check` typecheck: 0 new TS errors vs the v4.1.7 baseline (9
   pre-existing errors in `mapWebSocket.ts` + ingest helpers; not
   touched by this PR).
2. Backend `pytest tests/unit/web/test_images.py` — 19/19 pass
   (resolver unchanged from v4.1.7).
3. Operator manual release check on the live VPS post-deploy
   (expanded R4 to cover all migrated endpoints — Codex PR9 R4 F6):
   - Log into apex `m2020-phase.net` → navigate to `sherloc.m2020-phase.net`.
   - Verify top-right shows "Sign out" without click (Bug A: silent SSO).
   - Click into a known scan with ACI → verify ACI image renders in the
     Workbench panel (Bug B; exercises `/api/images/.../aci`,
     `/api/scans/<id>`, and the spectrum subset endpoint
     `/api/spectra/<id>/subset` via single-point and class-average
     spectrum modes).
   - Switch to Map mode → verify "Failed to load map layers" banner is
     absent, ACI base image renders, scan-point overlay visible (Bug B;
     exercises `/api/map/layers/<id>`).
   - **Select a non-trivial map display layer** (e.g., a domain that has
     `n_detections > 0`) → verify overlay values populate; exercises
     `/api/map/data/<id>?domain=...&value_type=snr` and the per-class
     variant.
   - **Start a SHERLOC fit job** on a domain → verify the fit progress
     panel transitions queued → running → complete with point counts
     incrementing; exercises `POST /api/map/fit` and the per-job
     WebSocket (`/api/ws/jobs/<id>`); job-status polling fallback at
     `/api/jobs/<id>` is exercised if WS reconnect is attempted.
4. Post-deploy container log audit — `docker logs` filtered for
   `authn failed reason=no_credential` shows no new occurrences for
   ANY of:
   - `/api/images/` (ACI)
   - `/api/map/layers/` (map metadata)
   - `/api/map/data/` (per-domain / per-class layer values)
   - `/api/map/fit` (fit-job create)
   - `/api/scans/` (scan info)
   - `/api/spectra/` (subset average)
   - `/api/jobs/` (job status poll)

JS test infrastructure addition (vitest config + playwright config +
the 6 test specs from the design memo §2.7 + §3.4) is logged as a
**v1.0.1 follow-up** in m2020-phase Session 93 closeout. Bug-class
regression risk is contained by the manual release check above:
failures of the kind Codex F3 worries about (Authorization header
not attached, race conditions, etc.) all surface as 401 in container
logs and visible broken ACI / "Failed to load map layers" in the UI.

## [4.1.7] - 2026-05-15

Implements the R2-aware SHERLOC ACI resolver mandated by m2020-phase
platform spec §3.9 (hierarchical-key model, amended Session 73). Closes
Track B2 + B3 + B4 from the m2020-phase deploy readiness plan — items
that were marked complete in the runbook but had never actually
shipped (Session 72 rehearsal NO-GO surfaced the gap). Produces image
`ghcr.io/archaeon-ai/sherloc-pipeline:v4.1.7` consumed by the
m2020-phase v1.0-beta deploy.

### Added

- `src/sherloc_pipeline/web/routes/images.py`: full rewrite of the ACI
  endpoint. Replaces local-FS reads with R2 GET via a boto3 S3 client
  cached at module level. Per-tier strip-prefix table (team data root
  → `phase-team`; public PDS data root → `phase-public`; defaults
  documented in the source via adjacent-literal concat and overridable
  via `PHASE_TEAM_STRIP_PREFIX` / `PHASE_PUBLIC_STRIP_PREFIX` env vars)
  derives the R2 key from the DB-stored `context_images.file_path`.
  Preserves the existing route surface
  (`colorized`, `enhanced`, `upscale` query params; VICAR `.IMG` →
  PNG in-process conversion via tempfile shim through the existing
  `read_aci_image` helper). New module-level helpers:
  `_get_r2_client_and_config`, `_derive_r2_key`, `_r2_get_bytes`,
  `_r2_head_exists`, `_find_colorized_key`, `colorized_variant_exists`
  (public predicate for `routes/map.py`).
- `boto3>=1.34.0` in the `[web]` optional-dependency extra.
- `moto[s3]>=5.0.0` in the `[dev]` extra for the new R2 test fixtures.
- `tests/unit/web/test_images.py`: full rewrite around moto-backed S3
  mock. New tests: team-tier resolve (200), public-tier VICAR convert,
  missing-object 404, cross-tier credential 403→502, R2 timeout →
  504 upstream_timeout, R2 non-timeout BotoCoreError → 500
  upstream_error, misconfigured-path 500 (DB tier ≠ file_path prefix),
  `pds:` LIDVID returns 500 misconfigured_path (unresolved on-demand
  ref is broken ingestion per spec §3.9.4), colorized variant via R2
  (SHA-asserted against base), colorized fall-back to base, resolver-
  config edge cases (PHASE_TIER unset, invalid tier, key derivation,
  path-traversal).
- `src/sherloc_pipeline/web/config_check.py`: new `_check_r2()` —
  enforces PHASE_TIER + AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY +
  AWS_ENDPOINT_URL when `SHERLOC_AUTH_MODE=auth0`. Fails container
  startup loud rather than at first ACI request. Defense-in-depth
  for the same condition the resolver maps to HTTP 500 "tier_unset"
  at request time.
- `tests/unit/web/test_config_check.py`: 3 new tests covering the R2
  env-var requirement under auth0 mode + the dev-mode bypass.
- `Dockerfile`: build-time smoke check extended with `import boto3`.

### Changed

- `src/sherloc_pipeline/web/routes/map.py`: colorized-variant
  availability now probes R2 (via `colorized_variant_exists`) instead
  of the local FS. Same API-stable predicate; backing store differs.
- `pyproject.toml` version: 4.1.6 → 4.1.7.
- `requirements-lock.txt` regenerated (boto3 1.43.9 + botocore +
  s3transfer + jmespath added; urllib3 bumped per boto3 transitive
  requirement).

### Spec compliance

Implements m2020-phase `docs/PHASE_PLATFORM_v1.0_SPEC-revised §3.9` as
amended Session 73 (hierarchical-key model). See §3.9.1 (resolver
inputs), §3.9.2 (env contract), §3.9.3 (tier → bucket mapping with
hierarchical-key derivation), §3.9.4 (failure modes), §3.9.5 (cross-
tier rejection rule, LOAD-BEARING), §3.9.6 (pure R2 at v1.0-beta, no
PHASE_LEGACY_FS_FALLBACK), §3.9.7 (out-of-scope clarifications).

## [4.1.6] - 2026-05-15

Track B5 — GHCR publish workflow + repo-move alignment. Produces
the `ghcr.io/archaeon-ai/sherloc-pipeline:v4.1.6` image consumed by
the m2020-phase v1.0-beta deploy (ARCHITECTURE_LOCKED §14 D3.2).

### Added

- `.github/workflows/publish.yml`: tag-triggered GHCR publish for
  `v*` tags. Builds the Dockerfile `runtime` stage on `linux/amd64`,
  logs into GHCR with `GITHUB_TOKEN`, and pushes
  `ghcr.io/archaeon-ai/sherloc-pipeline:<tag>`. Image visibility is
  private for v1.0-beta; the VPS pulls with a PAT scoped to
  `packages:read`. Supply-chain extras (cosign sign, SPDX SBOM, SLSA
  provenance) intentionally deferred to the v1.0.1 sunset on the new
  public `archaeon-ai/sherloc-pipeline` repo — see comment header
  in the workflow file.

### Fixed

- `Dockerfile` stage 2: align the pre-built `phase-platform-auth`
  wheel URL with the `pyproject.toml` pin so both reference
  `archaeon-ai/phase-platform-auth@v0.1.0`. The repo move from
  `kenwilliford/` was applied to `pyproject.toml` in `538c3c4` but
  missed the parallel string in the Dockerfile. GitHub's 12-month
  redirect was masking the drift; this commit closes it.

## [4.1.5] - 2026-05-08

### Fixed

- `core/data_ingestion.restructure_fluorescence_data` and
  `create_r123_spectrum` skipped past interleaved
  `R{1,2,3}_Channel*` header rows in raw Loupe `darkSubSpectra*.csv`,
  restoring `sherloc plot --domain fluor` and `--domain both` against
  R2/R3 from raw workspaces. The database ingest path (which has its
  own section-header guard) was unaffected. Cherry-picked from
  `main` `d883389`.

## [4.1.4] - 2026-05-07

### Fixed

- `docker-entrypoint.sh`: launch uvicorn with
  `--factory sherloc_pipeline.web.app:create_app` rather than
  importing `app` directly. The `runtime` stage's smoke check already
  imports `create_app`, but the entrypoint was still resolving
  `sherloc_pipeline.web.app:app`, which raised `AttributeError` at
  container start.

## [4.1.3] - 2026-05-07

### Fixed

- `Dockerfile` stage 3: drop `--no-index` from the final
  `pip install`. PEP 517 build for the direct-URL
  `phase-platform-auth` extra needs to fetch its build backend
  (hatchling) from PyPI; `--no-index` blocked that. `/wheels/`
  is still authoritative via `--find-links` for locked deps.

## [4.1.2] - 2026-05-07

### Fixed

- `Dockerfile` stage 3: install `git` in the runtime image too. pip
  resolves the `phase-platform-auth @ git+https://...` direct-URL
  specifier by cloning the URL even when a matching wheel exists in
  `/wheels/`; `--no-index` does not disable direct-URL fetches.
  Stage 2's pre-built wheel is kept as defense in depth. Sunset path:
  remove both git apt installs once `phase-platform-auth` is on PyPI.

## [4.1.1] - 2026-05-07

### Fixed

- `Dockerfile` stage 2: pre-build a wheel for the `[web]` extra's
  git-URL `phase-platform-auth` dep into `/wheels/` so stage 3 can
  resolve it offline alongside the other locked-dep wheels.

## [4.1.0] - 2026-05-06

PHASE Platform §2.6.1 validator B.0 switchover. The Auth0 JWT
validator now lives in
[`archaeon-ai/phase-platform-auth`](https://github.com/archaeon-ai/phase-platform-auth)
v0.1.0; SHERLOC consumes it as a runtime dependency rather than
shipping a parallel implementation. CFAccessValidator, DevValidator,
and the FastAPI dependency continue to live in this repo.

### Added

- New runtime dependency `phase-platform-auth >= 0.1.0, < 1` (web
  extra). Pinned to the v0.1.0 git tag until PyPI publish.

### Changed

- `sherloc_pipeline.web.auth.Auth0Validator` is now a re-export of
  `phase_platform_auth.Auth0Validator`. `TokenClaims`, `AuthError`,
  `JWKSUnavailableError`, and `build_www_authenticate` likewise route
  to the package. The SHERLOC import surface is unchanged.
- DevValidator synthetic claims now carry `phase:team-member` (the
  §2.6.1 namespace); the legacy `sherloc:internal` was retired with
  the rest of the Phase A path.
- `SHERLOC_AUTH0_IDENTITY_CLAIM_URI` is now mandatory in `auth0` mode;
  startup fails fast if unset. Better than a silent role-name mismatch
  at runtime.
- Default `WWW-Authenticate` realm is now `m2020-phase` (the §2.6.1
  contract literal). Override via `SHERLOC_AUTH_REALM`.

### Removed

- The `{role_claim_uri}/roles` Phase A backward-compat path. A SHERLOC
  instance running on the prior Auth0 tenant configuration without
  `SHERLOC_AUTH0_IDENTITY_CLAIM_URI` will refuse to start.
- `SHERLOC_AUTH0_ROLE_CLAIM_URI` env var (no longer read by the
  validator factory or the `/api/config` builder).
- Conformance test suite for the §2.6.1 Auth0 validator
  (`tests/unit/test_auth0_validator.py`) — replaced with a smoke test
  that confirms the import surface routes to the package. The
  conformance suite itself lives in the
  `phase-platform-auth` package.

## [4.0.0] - 2026-04-28

First public release. v4.0.0 supersedes the v3.0.0 stable release
previously published at the now-deleted `kenwilliford/sherloc-pipeline`
repository; the CLI surface (`full-pipeline`, `plot`, `apply-review`)
and Python API (`sherloc_pipeline.api.spectral`) of v3.0 are preserved.

### Security

- CF Access JWT signature validation now enforced on all authenticated
  routes. Validation covers signature (against the live JWKS), issuer
  (must equal `https://<SHERLOC_CF_TEAM_DOMAIN>`), audience, and
  expiry. The `Cf-Access-Authenticated-User-Email` convenience header
  is no longer trusted.
- CORS allowlist is now env-driven via `SHERLOC_CORS_ALLOWED_ORIGINS`;
  default is empty (no cross-origin requests).
- JWKS unavailability returns HTTP 503 (not 401), with a 24-hour grace
  window during which a stale cache is reused.
- Dev escape hatch: `SHERLOC_AUTH_MODE=dev` bypasses validation and
  logs a prominent startup warning.

### Added

- Web UI (FastAPI + Svelte) with averaged- and per-point spectral
  exploration, classification profiles, and user preferences.
- Map Mode: WebGL-rendered scan-point map with on-demand fitting,
  push-WebSocket job updates, and inline spectrum viewer.
- PDS4 ingestion (`sherloc pds-ingest`) for Mars 2020 SHERLOC archive
  data published through the PDS Geosciences Node.
- PIXL Pixlise ingestion (`sherloc pixl-ingest`) for cross-instrument
  context.
- Fluorescence fitting engine (`fit-fluor`, `core/fluor_fitting.py`)
  with agnostic AICc default and an optional hypothesis-driven
  strategy. Group assignment for Ce³⁺ doublet (anhydrite),
  Ce³⁺ phosphate, and silicate-defect bands.
- Unified peak persistence: `fitted_peaks.fit_modality` discriminates
  `minerals`, `organics`, `hydration`, and `fluorescence`. New
  `backfill`, `persist-peaks`, and `extract-training` CLI commands.
- Cross-modal annotation: fluorescence groups co-scored against Raman
  mineral assignments at the same scan point.
- R1 (523-channel) and R123 (2148-channel) spectrogram visualization
  pipelines.
- Grain segmentation and morphometry (optional SAM-based, behind the
  `[ml]` extra).
- Loupe-polynomial wavelength/wavenumber calibration; R123 stitching
  via Loupe overlap summation.
- Parallel per-point fitting for all four domains (configurable via
  `fitting.parallel_workers` and
  `fluorescence_fitting.parallel_workers`).

### Changed

- Database default location is now repository-relative (`./phase.db`,
  `./phase_pds.db`) rather than a hardcoded local path.
- Web UI configuration is now fully env-driven (`SHERLOC_DB`,
  `SHERLOC_ACCESS_MODE`, `SHERLOC_CORS_ALLOWED_ORIGINS`,
  `SHERLOC_CF_TEAM_DOMAIN`, `SHERLOC_CF_AUDIENCE`,
  `SHERLOC_AUTH_MODE`).
- Tests are installable on a fresh clone (no absolute paths).

### Removed

- Personal-infrastructure references (paths, hostnames) scrubbed from
  tracked content.
- AI-coordination tooling (`.ralph/`, `.beads/`, `.claude/`,
  `_internal/`, `CLAUDE.md`, agent-task scratch files) untracked.
- Experimental research scripts under `experiments/` are no longer
  shipped (kept locally by maintainers; not a public surface).

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
- PRD renamed from `docs/20251126_opus_PRD.md` to `docs/PRD.md`
- README updated with Python API section and examples
- New `docs/API.md` for API reference (T5.12)

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

