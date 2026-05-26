#!/usr/bin/env bash
# scripts/ingest_all_pds.sh — recipe combining PDS download + ingest.
#
# A thin bash wrapper around the `sherloc pds-download` and
# `sherloc pds-ingest` CLI subcommands. Generalised to use SHERLOC_HOME /
# SHERLOC_DB defaults; no hardcoded operator-local paths.
#
# Usage:
#   ./scripts/ingest_all_pds.sh --auto                 # all available sols
#   ./scripts/ingest_all_pds.sh --sol 921              # single sol
#   ./scripts/ingest_all_pds.sh --sol-range 800 1000   # inclusive range
#   PDS_DIR=/some/cache ./scripts/ingest_all_pds.sh --auto
#
# Optional environment variables:
#   PDS_DIR       PDS cache directory (default: ${SHERLOC_HOME:-.}/pds)
#   SHERLOC_HOME  Project root for default paths (defaults are CWD-relative)
#   SHERLOC_DB    Loupe DB used for target name cross-reference

set -euo pipefail

# Resolve defaults.
PDS_DIR_DEFAULT="${SHERLOC_HOME:-.}/pds"
PDS_DIR="${PDS_DIR:-$PDS_DIR_DEFAULT}"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 (--auto | --sol N | --sol-range FROM TO) [extra sherloc args]" >&2
    exit 64
fi

SCOPE_ARGS=()
case "${1:-}" in
    --auto)
        SCOPE_ARGS+=("--auto")
        shift
        ;;
    --sol)
        if [ "$#" -lt 2 ]; then
            echo "ERROR: --sol requires a sol number" >&2
            exit 64
        fi
        SCOPE_ARGS+=("--sol" "$2")
        shift 2
        ;;
    --sol-range)
        if [ "$#" -lt 3 ]; then
            echo "ERROR: --sol-range requires FROM and TO" >&2
            exit 64
        fi
        SCOPE_ARGS+=("--sol-range" "$2" "$3")
        shift 3
        ;;
    *)
        echo "ERROR: first argument must be --auto, --sol, or --sol-range" >&2
        exit 64
        ;;
esac

mkdir -p "$PDS_DIR"

echo "==> PDS download into ${PDS_DIR}"
sherloc pds-download "${SCOPE_ARGS[@]}" --output-dir "$PDS_DIR" "$@"

echo "==> PDS ingest from ${PDS_DIR}"
sherloc pds-ingest "${SCOPE_ARGS[@]}" --pds-dir "$PDS_DIR" "$@"

echo "==> Database stats"
sherloc db-stats || true

echo "==> Done."
