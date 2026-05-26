# SHERLOC Pipeline — Quickstart

Two paths are documented below. **Path A** verifies your install with no external data. **Path B** runs the same server against an existing PHASE database (Loupe-derived or PDS-derived).

---

## Prerequisites

- Python 3.12+
- `git`, `npm` (for frontend builds — only required if you want to rebuild the UI; a pre-built `dist/` ships in the source tree)
- `sqlite3` (CLI useful for database inspection but not required at runtime)

```bash
git clone https://github.com/archaeon-ai/sherloc-pipeline.git
cd sherloc-pipeline
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

---

## Path A — Smoke test (no instrument data)

Verifies the install. Runs the test suite, initializes an empty schema, and starts the API server. The UI loads but shows no scans.

```bash
# Run unit + integration tests, skipping the slow golden baseline
pytest -m "not slow"

# Initialize an empty database at ./phase.db
SHERLOC_DB=./phase.db alembic upgrade head

# Start the API server (binds to 127.0.0.1:8000 by default)
SHERLOC_AUTH_MODE=dev SHERLOC_DB=./phase.db ./scripts/serve.sh
```

Then point a browser at `http://127.0.0.1:8000`. The Scan Browser will be empty; Processing Workbench, ACI Viewer, and PDS Browser will load with no data. Estimated time: ~5 minutes.

Hit `/api/health` to confirm liveness:

```bash
curl -fsS http://127.0.0.1:8000/api/health
```

---

## Path B — Run against an existing PHASE database

If you already have a `phase.db` (Loupe-derived) or `phase_pds.db` (PDS-derived):

```bash
# Make sure the schema is at the current migration head
SHERLOC_DB=/path/to/your/phase.db alembic upgrade head

# Start the server pointing at your DB
SHERLOC_AUTH_MODE=dev SHERLOC_DB=/path/to/your/phase.db ./scripts/serve.sh
```

The UI will load with your scans visible.

A public fixture database is **not** included in the initial release; the server requires either an existing PHASE database or a PDS ingestion run to produce one (see `docs/guides/PDS_INGESTION_GUIDE.md`).

---

## Production deployment

For deployments behind Cloudflare Access (or another reverse proxy that performs JWT-based authentication), see `SECURITY.md`. The required environment variables are:

| Variable | Purpose |
|----------|---------|
| `SHERLOC_DB` | Absolute or relative path to the PHASE SQLite database |
| `SHERLOC_CORS_ALLOWED_ORIGINS` | Comma-separated list of origins allowed for browser cross-origin requests; empty by default (no cross-origin requests) |
| `SHERLOC_CF_TEAM_DOMAIN` | Cloudflare Access team domain, e.g. `your-team.cloudflareaccess.com` (required unless `SHERLOC_AUTH_MODE=dev`) |
| `SHERLOC_CF_AUDIENCE` | Cloudflare Access app AUD tag (required unless `SHERLOC_AUTH_MODE=dev`) |
| `SHERLOC_AUTH_MODE` | Set to `dev` only on a developer workstation. Off by default. Bypasses CF Access JWT validation. |

The server binds to `127.0.0.1` by default and is intended to run behind a reverse proxy in production. Direct exposure on a public interface without an auth proxy is unsupported.
