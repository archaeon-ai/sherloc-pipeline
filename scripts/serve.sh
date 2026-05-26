#!/usr/bin/env bash
# scripts/serve.sh — Start the SHERLOC web app with uvicorn.
#
# Usage:
#   ./scripts/serve.sh
#
# Optional environment variables:
#   SHERLOC_DB                     Path to the SQLite database (default: ./phase.db)
#   SHERLOC_ACCESS_MODE            Access mode passed to the app factory:
#                                  "internal" or "public" (default: internal)
#   SHERLOC_CORS_ALLOWED_ORIGINS   Comma-separated CORS allowlist (default: empty,
#                                  i.e. no cross-origin requests)
#
# Examples:
#   # Use defaults (internal mode, ./phase.db, no CORS):
#   ./scripts/serve.sh
#
#   # Point at a specific database in public mode:
#   SHERLOC_DB=/path/to/phase_pds.db SHERLOC_ACCESS_MODE=public ./scripts/serve.sh
#
#   # Allow cross-origin requests from a deployed frontend:
#   SHERLOC_CORS_ALLOWED_ORIGINS=https://sherloc.example.com ./scripts/serve.sh
#
# Architecture:
#   uvicorn (port 8000) → FastAPI app (sherloc_pipeline.web.app:create_app)
#   Optionally expose via cloudflared tunnel — see scripts/tunnel.sh
#
# Notes:
#   - Single worker (uvicorn default). Only one pipeline operation runs at a time.
#   - Binds to 127.0.0.1 only; use tunnel.sh or a reverse proxy for external access.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Activate virtual environment
VENV="${REPO_ROOT}/.venv/bin/activate"
if [[ ! -f "${VENV}" ]]; then
    echo "ERROR: Virtual environment not found at ${VENV}" >&2
    echo "Run: python -m venv ${REPO_ROOT}/.venv && pip install -e ${REPO_ROOT}" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${VENV}"

# Apply defaults for optional env vars
export SHERLOC_DB="${SHERLOC_DB:-./phase.db}"
export SHERLOC_ACCESS_MODE="${SHERLOC_ACCESS_MODE:-internal}"

# Local dev convenience: default to dev auth mode so /api/* requests
# don't require a Cf-Access-Jwt-Assertion header (which only exists when
# requests come through a CF Access tunnel). Production deployments set
# SHERLOC_AUTH_MODE explicitly via env files (cf-access for legacy
# deployments, auth0 for VPS); dev mode is operator-trust per spec §13.5 and is
# bound to 127.0.0.1 by uvicorn below — never expose dev mode externally.
export SHERLOC_AUTH_MODE="${SHERLOC_AUTH_MODE:-dev}"

echo "Starting SHERLOC web app"
echo "  Database   : ${SHERLOC_DB}"
echo "  Access mode: ${SHERLOC_ACCESS_MODE}"
echo "  Listening  : http://127.0.0.1:8000"
echo ""

exec uvicorn \
    sherloc_pipeline.web.app:create_app \
    --factory \
    --host 127.0.0.1 \
    --port 8000
