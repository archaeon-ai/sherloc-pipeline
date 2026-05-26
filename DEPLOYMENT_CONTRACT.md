# SHERLOC Pipeline Deployment Contract

> Authoritative surface for any consumer running
> `ghcr.io/archaeon-ai/sherloc-pipeline`. Tested by `tests/contract/`.
> Versioned with the image tag — a consumer who pins `v4.1.14` is
> pinning the exact contract surface documented here as of that tag.

## 1. What this document covers

This contract documents **what the SHERLOC backend image guarantees** to
a downstream consumer (e.g., the m2020-phase deployment, or any future
operator forking the public toolkit). It pins:

- Image identity (registry path, tag scheme, base image, runtime user).
- Container interface (ports, healthcheck endpoint, entrypoint, command
  dispatch).
- Volume requirements (mount points + permissions).
- Environment-variable surface (required, mode-driven, optional with
  defaults, retired).
- Public-mode database filename invariant (a safety rail that prevents
  internal-tier data from being served from a public-tier deployment).
- Migration head and boot sequence.

What it does **not** cover is listed in §9 — out-of-scope items the
consumer owns (network topology, TLS, secret-management mechanism,
auth-provider tenant config, performance, etc.).

Every machine-enforceable statement below has a corresponding test under
`tests/contract/`. New required vars, retired vars, or boot-sequence
changes MUST land alongside a test update — that is the deliberate
ratchet.

## 2. Image identity

| Element | Value | Source |
|---|---|---|
| Registry | `ghcr.io/archaeon-ai/sherloc-pipeline:<tag>` | `.github/workflows/publish.yml` |
| Tag scheme | Semver `v<major>.<minor>.<patch>` matching `pyproject.toml` `project.version` | Convention; v4.1.6 was the first GHCR publish |
| Visibility | Private during v1.0-beta; v1.0.1 sunset flips to public | `publish.yml` summary |
| Platforms | `linux/amd64,linux/arm64` (multi-arch added at v4.1.16 per FOUNDATION §3.5 step 6) | `publish.yml` |
| Base | `python:3.12.12-slim-bookworm` (runtime stage) | `Dockerfile` stage 3 |
| Runtime user | `sherloc` (uid 1000, gid 1000) | `Dockerfile` |

## 3. Container interface

| Element | Value | Required? |
|---|---|---|
| Internal port | `8000` (TCP) | yes |
| Healthcheck endpoint | `GET /api/health` (HTTP 200 on healthy) | yes |
| Healthcheck timing | `--interval=30s --timeout=5s --start-period=20s --retries=3` | recommended default; downstream may override |
| Entrypoint | `/usr/bin/tini -- /app/docker-entrypoint.sh` | yes |
| Default CMD | `["web"]` → `uvicorn sherloc_pipeline.web.app:create_app --factory --host 0.0.0.0 --port 8000` | yes |
| Alternate CMDs | `["cli", ...]` for the sherloc CLI; raw command pass-through | optional |

## 4. Volumes

| Mount point | Purpose | Required? | Permissions |
|---|---|---|---|
| `/data` | SQLite DB + any ephemeral state | yes | Must be writable by uid 1000 / gid 1000; bind-mount target must exist before first start |

The host-side path is **consumer-owned** (e.g., the m2020-phase deploy
uses `/var/lib/sherloc/internal` and `/var/lib/sherloc/public`). The
contract only specifies the container-internal path and the writer.

## 5. Environment variables

Authoritative source: `src/sherloc_pipeline/web/config_check.py`. The
validator runs at container boot (before `alembic upgrade head` and the
uvicorn launch) and exits 1 on any error.

### 5.1 Always required

| Variable | Constraint |
|---|---|
| `SHERLOC_DB` | Path; must be writable OR parent must exist and be writable. `:memory:` accepted (test only). |
| `PHASE_DATABASE_PATH` | Optional legacy alias. The entrypoint exports it from `SHERLOC_DB` when unset. If BOTH are set AND differ, `config_check` exits 1 with a `differ` error. The dangerous shape is "both set, pointing at different files" — `alembic/env.py` reads `PHASE_DATABASE_PATH` while the app reads `SHERLOC_DB`; a mismatch would silently migrate one DB while the API served another. |

The collapsing of `PHASE_DATABASE_PATH` to an optional alias landed in
v4.1.14 alongside the entrypoint export. Consumers may continue to set
both to the same value (the m2020-phase env templates do); the contract
only requires `SHERLOC_DB`.

### 5.2 Mode-driven

`SHERLOC_AUTH_MODE` is optional with default `cf-access`. Valid values:
`{auth0, cf-access, dev}`. Production m2020-phase deployments set `auth0`.

When `SHERLOC_AUTH_MODE=auth0`, these are required:

| Variable | Constraint |
|---|---|
| `SHERLOC_AUTH0_DOMAIN` | non-empty |
| `SHERLOC_AUTH0_AUDIENCE` | non-empty |
| `SHERLOC_AUTH0_SPA_CLIENT_ID` | non-empty (surfaced via `/api/config`) |
| `SHERLOC_AUTH0_IDENTITY_CLAIM_URI` | non-empty namespace URI (validated by `config_check` since v4.1.14) |
| `PHASE_TIER` | ∈ `{team, public}` |
| `AWS_ACCESS_KEY_ID` | non-empty |
| `AWS_SECRET_ACCESS_KEY` | non-empty |
| `AWS_ENDPOINT_URL` | non-empty URL |

When `SHERLOC_AUTH_MODE=cf-access` (legacy Cloudflare Access deployment):

| Variable | Constraint |
|---|---|
| `SHERLOC_CF_TEAM_DOMAIN` | non-empty |
| `SHERLOC_CF_AUDIENCE` | non-empty |

When `SHERLOC_AUTH_MODE=dev`: no further env requirements; localhost-only
mode, used for tests + the `contract-smoke` workflow.

### 5.3 Access mode

| Variable | Constraint |
|---|---|
| `SHERLOC_ACCESS_MODE` | ∈ `{internal, public}`. Default `internal`. Drives the role-per-API gate + the §6 public-mode DB invariant. |

### 5.4 Optional with documented defaults

| Variable | Default | Notes |
|---|---|---|
| `SHERLOC_AUTH0_JWKS_TTL_SECONDS` | 600 | JWKS cache TTL |
| `SHERLOC_AUTH0_JWKS_MAX_STALE_SECONDS` | 86400 | Stale-while-revalidate window |
| `SHERLOC_AUTH0_EXPECTED_AZP` | unset | Single-SPA azp pin; mutually exclusive with `_KNOWN_SPA_CLIENT_IDS` |
| `SHERLOC_AUTH0_KNOWN_SPA_CLIENT_IDS` | unset | CSV of allowed SPA client IDs |
| `SHERLOC_AUTH_REALM` | `m2020-phase` | WWW-Authenticate realm |
| `SHERLOC_LOG_LEVEL` | `INFO` | uvicorn + sherloc loggers |
| `SHERLOC_CORS_ALLOWED_ORIGINS` | unset | CSV of allowed Origins for `/api/*`. Empty string disables CORS; unset leaves it unconfigured. |
| `SHERLOC_FEATURE_PDS_BROWSER` | unset (= ENABLED) | Only literal `disabled` (case-insensitive) opts out |
| `AWS_REGION` | `auto` | Used by boto3 R2 client |
| `PHASE_TEAM_LEGACY_STRIP_ALIASES` | unset | Colon-separated (`$PATH`-style) list of legacy team-tier `file_path` prefixes to accept ALONGSIDE the canonical `PHASE_TEAM_STRIP_PREFIX`. Additive — does NOT replace the built-in production default carried in `src/sherloc_pipeline/core/r2_keys.py`. Alias values containing a literal `:` are unsupported. Trailing slash required to match real `file_path` rows. v4.1.17+. |
| `PHASE_PUBLIC_LEGACY_STRIP_ALIASES` | unset | Same shape and semantics as the team variant, for the public tier. Default has no built-in aliases. v4.1.17+. |

### 5.5 Retired — MUST NOT appear

| Variable | Removed in | Reason |
|---|---|---|
| (legacy identity-claim split var; see `src/sherloc_pipeline/web/auth.py`) | v4.1.0 / Phase B.0 | Single identity-claim URI under §2.6.1; legacy split path eliminated. |

The retired-var literal name is **deliberately omitted from this
document**. `tests/contract/test_env_template_contract.py` asserts the
literal does not appear in `deploy/env-templates/sherloc.env.example`;
re-introducing it in any tracked file (even as a comment) is a
contract violation.

## 6. Public-mode DB filename invariant

Hardcoded in `src/sherloc_pipeline/web/app.py`: if
`SHERLOC_ACCESS_MODE=public` AND the resolved DB path contains the
substring `phase.db` but NOT `phase_pds.db`, `create_app` raises
`ValueError` at startup. The filename naming convention is part of the
contract because it is the last-line defense against accidentally
serving the internal tier's Loupe data from a public endpoint.

| Tier | Required filename substring |
|---|---|
| internal | (no constraint; `/data/phase.db` by convention) |
| public | must contain `phase_pds.db` |

`:memory:` is accepted in public mode (the substring check matches
neither side); the contract pins this so a future tightening becomes a
deliberate decision rather than silent drift. See
`tests/contract/test_public_mode_unit.py`.

## 7. Migration head and boot sequence

| Element | Value |
|---|---|
| Current head | `412fc1e3ee92` (`add_user_sub_column_for_auth0_identity`) |
| Boot order | `python -m sherloc_pipeline.web.config_check` → `alembic upgrade head` → uvicorn |
| Contract guarantee | Single head (no fork); migrations idempotent |

A consumer who pins a SHERLOC tag pins a migration head. Downgrade across
major migrations is not part of the contract — the v1.0 line is
forward-only. The migration graph contains at least one merge migration
(`1ed5cea6c32d`) with a tuple `down_revision`; the contract test parses
revisions via AST + `ast.literal_eval` to handle tuple / list / str /
None uniformly.

## 8. Resource recommendations (advisory)

| Resource | Recommendation | Rationale |
|---|---|---|
| Memory limit | 1 GB per service | Empirical from CCX13 deploy; spikes during fitting + image conversion |
| CPU | 1.0–1.5 cores per service | uvicorn single worker; ThreadPoolExecutor for map fits |
| Restart policy | `unless-stopped` | Standard for long-running web service |

Not enforced by the test suite. Present as commented defaults in
`deploy/docker-compose.example.yml`.

## 9. What this contract does NOT govern

- Network topology (caddy / reverse-proxy choice; consumer-owned).
- TLS termination (consumer-owned).
- Backup schedule / paths (consumer-owned; the contract does not bundle
  a reference backup script — supply your own per host policy).
- Auth-provider tenant config (Auth0 Action source, JWKS endpoint shape)
  — that is the platform spec's job, not SHERLOC's contract.
- Orchestration (systemd vs Kubernetes vs raw `docker compose up` — all
  valid).
- Secret-management mechanism (Infisical, sops, k8s Secrets, plain env
  files — all valid).
- Behavioral guarantees beyond the deploy surface (auth correctness, R2
  resolver behavior, DB schema correctness, performance) — those are
  covered by `tests/unit/`, `tests/integration/`, and friends; the
  contract guards only what a deployer touches.

## 10. Versioning

The contract is versioned with the image tag. The
`tests/contract/test_image_identity.py` test asserts:

- `pyproject.toml::project.version` matches the latest `v*` git tag in
  PR mode (PASS when in-sync, `pytest.skip` when `project.version` is
  ahead of the latest tag — the operator-coordinated tag-push window,
  see §11.5 of the design — FAIL when `project.version` is behind).
- On tag push (`GITHUB_REF_TYPE=tag`), the test asserts strict equality
  `GITHUB_REF_NAME == f"v{project.version}"`.
- `.github/workflows/publish.yml` enforces the same strict equality as
  a pre-build step (before docker login / build / push) so a mistagged
  push aborts before the image is published. The publish gate does not
  depend on a successful `ci.yml` run — each workflow validates
  independently.
- `.github/workflows/publish.yml` `IMAGE_NAME` equals
  `ghcr.io/archaeon-ai/sherloc-pipeline`.
- `.github/workflows/publish.yml` `platforms:` line equals
  `linux/amd64,linux/arm64` (multi-arch added at v4.1.16 per
  FOUNDATION §3.5 step 6).

## 11. Two-tier deployment pattern

The `deploy/docker-compose.example.yml` ships a single-service shape.
The m2020-phase deploy runs **two** services from the same image with
different env files — internal-tier (team) and public-tier (public).
Per-tier env-var overrides:

| Variable | Internal tier | Public tier |
|---|---|---|
| `SHERLOC_ACCESS_MODE` | `internal` | `public` |
| `PHASE_TIER` | `team` | `public` |
| `SHERLOC_DB` | `/data/phase.db` | `/data/phase_pds.db` (MUST contain `phase_pds.db` per §6) |
| `PHASE_DATABASE_PATH` | unset (or `/data/phase.db`) | unset (or `/data/phase_pds.db`) |
| `AWS_ACCESS_KEY_ID` | team-tier R2 read credentials | public-tier R2 read credentials (R2-bucket-scoped) |
| `AWS_SECRET_ACCESS_KEY` | (same) | (same) |
| `AWS_ENDPOINT_URL` | same R2 account endpoint | same R2 account endpoint |

A two-tier compose looks like:

```yaml
services:
  sherloc-internal:
    image: ghcr.io/archaeon-ai/sherloc-pipeline:vX.Y.Z
    env_file: ./internal.env
    volumes: [./data/internal:/data]

  sherloc-public:
    image: ghcr.io/archaeon-ai/sherloc-pipeline:vX.Y.Z   # same tag in both services
    env_file: ./public.env
    volumes: [./data/public:/data]
```

Both services MUST pin the same tag — the contract surface is a property
of the image, not the env file. R2 credentials are scoped per bucket
(team-tier creds cannot read the public bucket and vice versa); this is
defense-in-depth alongside the §6 filename invariant.

## 12. Test suite

### 12.1 Static tests (run on every PR)

Located under `tests/contract/`; picked up by the normal pytest
invocation `pytest -m "not slow and not docker"` in `.github/workflows/ci.yml`.

- `test_env_contract.py` — `config_check` validator surface (§5).
- `test_dockerfile_contract.py` — image shape (§2, §3).
- `test_entrypoint_contract.py` — boot sequence (§7) + `PHASE_DATABASE_PATH`
  export line (§5.1).
- `test_compose_example.py` — sanitized example shape (parses, no
  operator-local strings).
- `test_env_template_contract.py` — sanitized env template (REQ set
  present, retired-var literal absent, secret-shaped values are
  placeholders).
- `test_alembic_contract.py` — single migration head (§7).
- `test_public_mode_unit.py` — `create_app` raises on public-mode DB
  filename mismatch (§6).
- `test_image_identity.py` — `pyproject.toml::version` matches latest
  tag + `publish.yml` constants (§10).

### 12.2 Live container smoke

Located at `tests/contract/test_container_smoke.py`. Double-marked
`@pytest.mark.docker` AND `@pytest.mark.slow` so the normal CI lane
filters it out. Opt-in via the dedicated path-filtered workflow at
`.github/workflows/contract-smoke.yml`.

Five cases:

- **A** (positive, dev mode internal tier): default boot reaches
  `/api/health` 200; `alembic current` (run via `docker exec`) matches
  `EXPECTED_HEAD`. Note: the `docker exec` invocation must pass
  `-e PHASE_DATABASE_PATH=<SHERLOC_DB value>` explicitly. `docker exec`
  starts a fresh exec process whose environment is the image config +
  the original `--env-file` plus any `-e` overrides; it does NOT
  inherit variables exported inside the already-running entrypoint
  process (no shell is involved). Operators running diagnostic
  `alembic current` on a live container hit the same constraint when
  only `SHERLOC_DB` is set in the env-file; the m2020-phase
  env-templates set both vars so the VPS containers do not see this
  issue.
- **B** (negative, missing `SHERLOC_DB`): container exits non-zero;
  stderr contains `missing required variable: SHERLOC_DB`.
- **C** (negative, public-mode DB invariant): `SHERLOC_ACCESS_MODE=public`
  with `SHERLOC_DB=/data/phase.db` — container exits non-zero with a
  `ValueError` referencing `phase_pds.db`. This is the load-bearing
  safety rail for the two-tier deploy.
- **D** (positive, public-mode compliant): `SHERLOC_ACCESS_MODE=public`
  with `SHERLOC_DB=/data/phase_pds.db` — container reaches health 200.
- **E** (negative, `PHASE_DATABASE_PATH` mismatch): `SHERLOC_DB` set,
  `PHASE_DATABASE_PATH` set differently — container exits non-zero with
  `differ` error.

Path filter on the workflow: changes to `Dockerfile`,
`docker-entrypoint.sh`, `deploy/**`,
`src/sherloc_pipeline/web/config_check.py`,
`src/sherloc_pipeline/web/app.py`, `alembic/versions/**`, or
`tests/contract/**` trigger the smoke. PRs that touch none of those
skip it.

### 12.3 How to add a contract change

1. Edit the relevant source-of-truth (Dockerfile, `config_check.py`,
   entrypoint, `app.py`, etc.).
2. Update this `DEPLOYMENT_CONTRACT.md`.
3. Update the matching test under `tests/contract/`.
4. Run `pytest tests/contract/ -m "not docker"` locally.
5. Run `pytest -m docker tests/contract/test_container_smoke.py` if
   Docker is available; otherwise rely on `contract-smoke.yml`.
6. Open a PR. The static tests run on every push; the smoke runs only
   if the path filter triggers.

## 13. References

- `tests/contract/` — the executable side of this contract.
- `src/sherloc_pipeline/web/config_check.py` — env validator.
- `src/sherloc_pipeline/web/app.py` — public-mode DB invariant.
- `Dockerfile` — image shape.
- `docker-entrypoint.sh` — boot sequence.
- `deploy/docker-compose.example.yml` — sanitized compose example.
- `deploy/env-templates/sherloc.env.example` — sanitized env template.
- `alembic/versions/` — migration graph.
- `.github/workflows/ci.yml` — static contract tests + the rest of the
  test suite.
- `.github/workflows/contract-smoke.yml` — path-filtered live smoke.
- `.github/workflows/publish.yml` — GHCR build/push.
