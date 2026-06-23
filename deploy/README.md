# `deploy/` — Public reference shapes

Reference materials for operators running `sherloc-pipeline` as a
self-hosted Docker service. The two files here are the canonical
*example* shapes — copy and tailor them to your environment.

| File | Purpose | Authoritative section |
|---|---|---|
| `docker-compose.example.yml` | Reference Compose stack (service + healthcheck + bind-mount layout) | [`DEPLOYMENT_CONTRACT.md` §11](../DEPLOYMENT_CONTRACT.md) |
| `env-templates/sherloc.env.example` | Required + optional environment variables with inline documentation | [`DEPLOYMENT_CONTRACT.md` §5](../DEPLOYMENT_CONTRACT.md) |

## What does NOT live here

- `Dockerfile`, `docker-compose.yml`, `docker-entrypoint.sh` — at repo
  root; they ship as part of the published GHCR image, not as
  separately-installed artefacts.
- Runtime environment files (`/etc/sherloc/*.env` or equivalent) —
  materialised on the deploy host from `env-templates/sherloc.env.example`
  plus operator-provided secrets. Do not commit per-deploy environment
  files to git.
- Operator-specific deployment glue (Auth0 Actions, host cron jobs,
  systemd units, backup scripts) — those are deployment-side concerns
  and belong in the consuming deployment's repository.
