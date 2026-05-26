# PDS Ingestion Guide

## Purpose

The `pds-ingest` command downloads and ingests PDS4 SHERLOC spectral data from the PDS Geosciences Node (WUSTL) into a dedicated `phase_pds.db` SQLite database. This database is structurally isolated from the Loupe-based `phase.db` — same schema, separate file — enabling cross-source analysis via SQLite `ATTACH DATABASE`.

PDS ingestion supports:
- Laser-normalized spectra (RRS/RCS) with R1/R2/R3 regions
- Laser shot positions from RMO products
- ACI context image metadata (LIDVID association)
- Calibration data (RCC) and auxiliary products (RLI, RLS)
- Target name resolution via SCLK cross-reference to Loupe data

## Prerequisites

1. **Python environment** — Activate the project virtualenv:
   ```bash
   source .venv/bin/activate
   ```

2. **PDS data cached locally** — Sol data must be downloaded to the cache directory before ingestion. The default cache directory is `./pds/`, organized as `sol_SSSS/data_processed/`.

3. **Loupe database (optional)** — For target name resolution, the Loupe database (`./phase.db`) should be accessible. Without it, target names fall back to a curated mapping file or NULL.

## Basic Usage

### Ingest a single sol

```bash
sherloc pds-ingest --sol 921
```

Ingests all observations for Sol 921 from locally cached PDS data. Output shows per-observation results:

```
Sol 921: 5 observations ingested, 0 skipped, 1 failed (zpz)
  Points: 1498  Spectra: 4494  Context images: 5
```

### Ingest a range of sols

```bash
sherloc pds-ingest --sol-range 100 1000
```

Processes all locally cached sols in the inclusive range [100, 1000]. Sols without cached data are silently skipped.

### Ingest all cached sols

```bash
sherloc pds-ingest --auto
```

Discovers and ingests all sols in the PDS cache directory.

### Dry run (no database writes)

```bash
sherloc pds-ingest --sol 921 --dry-run
```

Parses and validates data without writing to the database. Displays a summary table showing observation count, scan types, and product counts. Useful for verifying data before committing to ingestion.

### Force re-ingestion

```bash
sherloc pds-ingest --sol 921 --force
```

Re-ingests all observations even if they already exist in the database. Existing records are cascade-deleted and replaced. Use this after code changes that affect how data is parsed or stored.

### Check for version updates

```bash
sherloc pds-ingest --sol 921 --check-updates
```

Read-only check that compares PDS product versions against the database. Reports which observations are new, current, or have updates available. Does not modify the database.

### Generate a JSON report

```bash
sherloc pds-ingest --sol 921 --report-json report.json
sherloc pds-ingest --sol 921 --report-json -   # stdout
```

Writes a structured JSON report with ingestion counts, timing, errors, and version updates.

### Show database statistics after ingestion

```bash
sherloc pds-ingest --sol 921 --stats
```

Displays total counts for sols, scans, scan points, spectra, and context images after ingestion completes.

## Configuration

PDS settings are defined in `src/sherloc_pipeline/config.yaml` under the `pds` section:

```yaml
pds:
  base_url: "https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_sherloc"
  cache_dir: "./pds"
  timeout_seconds: 60.0
  max_retries: 3
  backoff_factor: 2.0
```

| Setting | Default | Description |
|---------|---------|-------------|
| `base_url` | WUSTL PDS Geosciences Node | PDS archive base URL |
| `cache_dir` | `./pds` | Local directory for cached PDS downloads |
| `timeout_seconds` | `60.0` | HTTP request timeout |
| `max_retries` | `3` | Retry count for transient HTTP errors (5xx, 429) |
| `backoff_factor` | `2.0` | Exponential backoff multiplier (delays: 1s, 2s, 4s) |

### CLI path overrides

| Option | Default | Description |
|--------|---------|-------------|
| `--pds-dir` | `./pds` | PDS cache directory |
| `--pds-database` / `-d` | `./phase_pds.db` | PDS database path |
| `--loupe-database` | `./phase.db` | Loupe DB for target cross-reference |

## Sol 921 Example

Sol 921 (2023-09-23, L_s = 122.871) at the "Amherst Point" target contains 6 SCLK-grouped observations:

| SCLK | Type | Points | Products | Notes |
|------|------|--------|----------|-------|
| 748731011 | Calibration | 1 | RCS, RMO, RLI, RCC, RLS | AlGaN calibration |
| 748731413 | Detail | 100 | RRS, RMO, RLI, RLS, RM1-6 | |
| 748732975 | Detail | 100 | RRS, RMO, RLI, RLS, RM1-6 | |
| 748735042 | Survey | 1296 | RRS, RMO, RLI, RLS, RM1-6 | De-duped from 2592 raw positions |
| 748735903 | (zpz) | — | RLI, RLS only | zpz products filtered; no RRS/RCS |
| 748736149 | Calibration | 1 | RCS, RLI, RCC, RLS | No RMO (index fallback) |

After ingestion: 5 scans, 1498 scan points, 4494 spectra (3 regions per point), 5 context images.

```bash
# Full ingestion with stats and JSON report
sherloc pds-ingest --sol 921 --stats --report-json outputs/sol921_report.json
```

## Key Concepts

### Idempotency

Ingestion is idempotent by default. Running `pds-ingest --sol 921` twice produces the same database state. The second run skips all 5 observations (matched by PDS LID as `scan_id`).

### Version handling

PDS products are versioned (e.g., v1.0, v2.0). The ingestion service uses numeric tuple comparison — `(1, 10) > (1, 2)` — avoiding lexicographic bugs. When a newer version is detected, the old scan is cascade-deleted and the new version ingested automatically.

### Database isolation

PDS data lives in `phase_pds.db`, never in `phase.db`. The Loupe database is opened read-only for target name cross-reference. No writes are ever made to `phase.db`.

### Target name resolution

Target names are resolved via a three-tier strategy:

1. **SCLK cross-reference** — Match PDS SCLK to Loupe scans within ±3s (Pass 1) or ±5s (Pass 2). Nearest match wins; site_drive breaks ties.
2. **Curated mapping** — Optional JSON file at `$SHERLOC_HOME/configs/pds_target_mapping.json` with `{sol}_{sclk}` keys (CWD-relative when `SHERLOC_HOME` is unset). The repo does not ship a default copy; missing or malformed files are treated as an empty mapping. Operators can add a personal copy if they have curated entries.
3. **NULL fallback** — If no match, target is set to NULL with a log message for manual curation.

### zpz filtering

Products with "zpz" in the filename middle section are zero-point-zero calibration intermediates. They are filtered out during ingestion. If an observation has only zpz spectral products (no RRS/RCS), it is skipped with an error message.

## Troubleshooting

### "PDS cache directory not found"

The `--pds-dir` path does not exist. Verify the cache directory:
```bash
ls ./pds/
```

### "No locally cached sols found" or "No locally cached sols in range"

No `sol_NNNN` directories exist in the cache. Download data first using the PDS client.

### "No RRS/RCS spectral product"

The observation contains only zpz-filtered products with no usable spectral data. This is expected for zpz observations and reported as a non-fatal error.

### Target names are NULL

The Loupe database is either missing or doesn't contain matching SCLK data for the PDS observations. Ensure `--loupe-database` points to a valid `phase.db`, or create `$SHERLOC_HOME/configs/pds_target_mapping.json` with curated `{sol}_{sclk}` entries.

### "UNIQUE constraint failed: scans.scan_id"

A scan with the same PDS LID already exists. This shouldn't occur during normal operation (idempotency handles it). Use `--force` to re-ingest, or inspect the database for duplicate entries.

### Ingestion is slow

Each sol parses XML labels, CSV data files, and performs DB writes. For large-scale ingestion (`--auto` with many sols), expect several seconds per sol. The `--report-json` option records elapsed time for benchmarking.
