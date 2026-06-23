#!/bin/bash
set -euo pipefail

# Collapse the SHERLOC_DB / PHASE_DATABASE_PATH dual-variable surface to
# a single canonical name (SHERLOC_DB). alembic/env.py reads
# PHASE_DATABASE_PATH; the app reads SHERLOC_DB. Exporting here keeps
# the two in sync when consumers set only SHERLOC_DB. If they set both
# to different values, config_check (next step) fails fast with a
# `differ` error rather than silently migrating one DB while serving
# another. See DEPLOYMENT_CONTRACT.md §5.1.
export PHASE_DATABASE_PATH="${PHASE_DATABASE_PATH:-${SHERLOC_DB:-}}"

# 1. Validate required config (fails fast with clear messages)
python -m sherloc_pipeline.web.config_check

# 2. Run migrations (idempotent — Alembic stamps current head)
alembic upgrade head

# 3. Dispatch
case "$1" in
  web) exec uvicorn sherloc_pipeline.web.app:create_app --factory --host 0.0.0.0 --port 8000 ;;
  cli) shift; exec sherloc "$@" ;;
  *)   exec "$@" ;;
esac
