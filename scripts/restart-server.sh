#!/usr/bin/env bash
# scripts/restart-server.sh — Kill old server, start new one, restart tunnel.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${SHERLOC_PORT:-8766}"

# Kill existing server on the port
if pid=$(lsof -ti :"${PORT}" 2>/dev/null); then
    kill "$pid" 2>/dev/null || true
    sleep 1
fi

# Activate venv and start server in background
source "${REPO_ROOT}/.venv/bin/activate"
export SHERLOC_DB="${SHERLOC_DB:-./phase.db}"
export SHERLOC_ACCESS_MODE="${SHERLOC_ACCESS_MODE:-internal}"

nohup uvicorn sherloc_pipeline.web.app:create_app \
    --factory --host 127.0.0.1 --port "${PORT}" \
    > /tmp/sherloc-server.log 2>&1 &

sleep 2

# Restart tunnel so it reconnects to the new server
systemctl --user restart cloudflared 2>/dev/null || true

echo "Server running on port ${PORT} (PID $!)"
echo "Tunnel restarted"
