#!/usr/bin/env bash
# scripts/serve-public.sh — Start the SHERLOC web app in public (PDS-only) mode.
#
# Usage:
#   ./scripts/serve-public.sh
#
# This is a convenience wrapper around serve.sh that configures:
#   - PDS-only database (phase_pds.db)
#   - Public access mode (no Loupe data)
#   - Port 8767 (distinct from internal on 8000)

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

export SHERLOC_DB="${SHERLOC_DB:-./phase_pds.db}"
export SHERLOC_ACCESS_MODE="public"

echo "Starting SHERLOC web app (PUBLIC MODE)"
echo "  Database   : ${SHERLOC_DB}"
echo "  Access mode: ${SHERLOC_ACCESS_MODE}"
echo "  Listening  : http://127.0.0.1:8767"
echo ""

exec uvicorn \
    sherloc_pipeline.web.app:create_app \
    --factory \
    --host 127.0.0.1 \
    --port 8767
