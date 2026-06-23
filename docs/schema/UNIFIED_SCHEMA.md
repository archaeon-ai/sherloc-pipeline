# Unified SHERLOC Data Schema

This document defines the unified data schema for SHERLOC spectroscopy data, supporting ingestion from both Loupe working directories and PDS4 archive products while enabling PHASE database storage and analysis workflows.

**Version:** 1.3
**Date:** 2026-04-29
**Status:** Implementation complete (v1.3 expands coverage to all 16 ORM tables and adds Units, Configuration Surface, and Example Queries appendices).

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Entity Relationship Diagram](#entity-relationship-diagram)
4. [Core Entities](#core-entities)
5. [Enums](#enums)
6. [Field Mappings](#field-mappings)
7. [Data Type Normalization](#data-type-normalization)
8. [Ingestion Strategy](#ingestion-strategy)
9. [Validation Rules](#validation-rules)
10. [Migration Path](#migration-path)
11. [Appendix A — Units & Conventions](#appendix-a--units--conventions)
12. [Appendix B — Configuration Surface](#appendix-b--configuration-surface)
13. [Appendix C — Example Queries](#appendix-c--example-queries)
14. [Appendix D — Field Inventory](#appendix-d--field-inventory)

---

## Overview

### Purpose

The unified schema serves as the canonical data model for SHERLOC spectroscopy data within the PHASE system. It:

1. **Unifies data sources** - Single schema for both Loupe and PDS4 inputs
2. **Preserves provenance** - Tracks data origin and processing history
3. **Enables analysis** - Structured for efficient querying and ML workflows
4. **Supports scalability** - Database-optimized design for 195+ sols

### Data Source Comparison

| Aspect | Loupe | PDS4 Archive |
|--------|-------|--------------|
| **Primary Use** | Interactive analysis | Archival storage |
| **Update Frequency** | Mission operations | Quarterly releases |
| **Data Completeness** | Selected workspaces | Complete archive |
| **Processing State** | Multiple levels in one workspace | Separate collections |
| **Metadata Format** | CSV key-value | XML labels |
| **Local Availability** | 195 sols, 19GB | Remote (or mirrored) |

### Schema Philosophy

The unified schema adopts a **Loupe-primary** approach because:
- Local data is in Loupe format
- Loupe format is more compact and analysis-ready
- PDS4 structure maps cleanly to Loupe concepts
- No data loss when ingesting either format

---

## Design Principles

### 1. No Data Loss

Every field from both Loupe and PDS4 formats must have a home in the unified schema, even if stored in auxiliary tables or JSON blobs.

### 2. Canonical Naming

Use descriptive, snake_case field names that are source-agnostic:
- `sol_number` instead of `SOL` or `sol`
- `spacecraft_clock` instead of `SCLK` or embedded in filenames

### 3. Typed Values

All fields have explicit types with validation:
- Temperatures as floats (strip units)
- Timestamps as ISO-8601 datetime
- Voltages as floats (strip units)

### 4. Referential Integrity

Foreign key relationships enforce data consistency:
- Spectra belong to ScanPoints
- ScanPoints belong to Scans
- Scans belong to Sols

### 5. Extensibility

Use JSON columns for:
- Source-specific metadata not in common fields
- Future fields without schema migrations
- Processing parameters and diagnostics

---

## Entity Relationship Diagram

The 16 ORM tables organize into three groups:

1. **Mission data hierarchy** — `Sol` → `Scan` → `ScanPoint` → `Spectrum` → `FittedPeak`, plus per-scan supporting tables (`InstrumentState`, `CCDConfiguration`, `ScannerCalibration`, `ContextImage`, `RegionOfInterest`, `Spectrogram`).
2. **Map Mode display cache** — `MapDisplayCoordinate` (per-scan-point ACI pixel cache).
3. **Application state** — `User`, `UserPreference`, `ClassificationProfile`, `MapFitCache`.

### Mission data hierarchy

```
                          +-------------------+
                          |       Sol         |
                          +-------------------+
                          | sol_number (PK)   |
                          | earth_date        |
                          | solar_longitude   |
                          +-------------------+
                                   |
                                   | 1:N
                                   v
                          +-----------------------------+
                          |            Scan             |
                          +-----------------------------+
                          | id (PK)                     |
                          | sol_number (FK)             |
                          | scan_name                   |
                          | target                      |
                          | scan_id                     |
                          | sclk_start, sclk_stop?      |
                          | n_points, n_channels        |
                          | target_type, scan_class     |
                          | parent_scan_id? (self-FK)   |
                          | source_scan_ids?            |
                          | processing_status?          |
                          | data_source, scan_type      |
                          +-----------------------------+
                            |   |   |    |   |    |    |
                            |   |   |    |   |    |    +-> InstrumentState (1:1)
                            |   |   |    |   |    +------> CCDConfiguration (1:1)
                            |   |   |    |   +-----------> ScannerCalibration (1:1)
                            |   |   |    +---------------> RegionOfInterest (1:N)
                            |   |   +--------------------> ContextImage (1:N)
                            |   +------------------------> Spectrogram (1:N)
                            |
                            | 1:N
                            v
                          +-----------------------+
                          |     ScanPoint         |
                          +-----------------------+
                          | id (PK)               |
                          | scan_id (FK)          |
                          | point_index           |
                          | azimuth_dn, elev_dn   |
                          | x_pixel, y_pixel      |
                          | coordinate_frame      |
                          | photodiode_mean/std   |
                          +-----------------------+
                                   |
                                   | 1:N
                                   v
                          +-----------------------+
                          |     Spectrum          |
                          +-----------------------+
                          | id (PK)               |
                          | scan_point_id (FK)    |
                          | region                |
                          | spectrum_type         |
                          | processing_level      |
                          | wavelength_source     |
                          | intensities (BLOB)    |
                          +-----------------------+
                                   |
                                   | 1:N
                                   v
                          +-----------------------+
                          |    FittedPeak         |
                          +-----------------------+
                          | id (PK)               |
                          | spectrum_id (FK)      |
                          | fit_modality          |
                          | center_cm1?, fwhm_cm1?|
                          | center_nm?, fwhm_nm?  |
                          | is_saturated?         |
                          | amplitude, snr        |
                          | mineral_assignment    |
                          +-----------------------+
```

### Map Mode display cache

```
+-----------------------+         +---------------------------+
|     ScanPoint         | 1:1     |  MapDisplayCoordinate     |
| (id PK)               |<--------| scan_point_id (PK, FK)    |
+-----------------------+         | aci_x, aci_y              |
                                  | transform_method          |
                                  +---------------------------+
```

### Application state

```
+-----------+   1:N   +---------------------+
|   User    |-------->|  UserPreference     |
| id (PK)   |         | (id, user_id, key,  |
| email     |         |  value, updated_at) |
+-----------+         +---------------------+
     |
     | 1:N
     v
+--------------------------+         +-----------------------+
| ClassificationProfile    |         |    MapFitCache        |
| id (PK), user_id (FK)    |         | id (PK)               |
| name, profile_json       |         | scan_id, domains      |
+--------------------------+         | profile_hash          |
                                     | user_id? (FK)         |
                                     | results_json          |
                                     | n_points              |
                                     | created_at, expires?  |
                                     +-----------------------+
```

---

## Core Entities

### Sol

Represents a Martian sol (day) of observations.

| Field | Type | Nullable | Description | Loupe Source | PDS4 Source |
|-------|------|----------|-------------|--------------|-------------|
| `sol_number` | INTEGER | NO (PK) | Mars sol number | Directory name `sol_XXXX` | `sol_number` in label |
| `earth_date` | DATE | YES | Corresponding Earth date | Derived from SCLK | `start_date_time` |
| `solar_longitude` | FLOAT | YES | Ls in degrees | Not stored | `solar_longitude` |
| `mission_phase` | TEXT | YES | Mission phase name | Not stored | `mission_phase_name` |
| `data_source` | TEXT | NO | 'loupe' or 'pds4' | Constant | Constant |
| `created_at` | DATETIME | NO | Record creation time | Generated | Generated |

### Scan

A complete spectroscopy scan of a target (equivalent to Loupe workspace or PDS4 product set).

| Field | Type | Nullable | Description | Loupe Source | PDS4 Source |
|-------|------|----------|-------------|--------------|-------------|
| `id` | UUID | NO (PK) | Unique identifier | Generated | Generated |
| `sol_number` | INTEGER | NO (FK) | Sol foreign key | Directory | Label |
| `scan_name` | VARCHAR(200) | NO | Human-readable workspace/scan name (renamed from `target_name` in migration `172d8c59b5c9`) | `human_readable_workspace` | Derived from products |
| `target` | VARCHAR(200) | YES | Geological target name (e.g., `Garde`, `Robine`, `AlGaN`) | `human_readable_workspace` parsed | From label / sequence |
| `scan_id` | VARCHAR(200) | NO | Original scan identifier | `original_data_file` | SCLK from filename |
| `sclk_start` | INTEGER | NO | Spacecraft clock start | From filename | `spacecraft_clock_start` |
| `sclk_stop` | INTEGER | YES | Spacecraft clock stop | Not stored | `spacecraft_clock_stop` |
| `n_points` | INTEGER | NO | Number of map points | `n_spectra` | RMO records |
| `n_channels` | INTEGER | NO | CCD channels (default 2148) | `n_channels` | Constant (2148) |
| `shots_per_point` | INTEGER | YES | Laser shots per point (NULL for PDS) | `shots_per_spec` | NULL (not in processed) |
| `laser_wavelength_nm` | FLOAT | NO | Laser wavelength (default 248.6) | `laser_wavelength` | Constant (248.6) |
| `processing_applied` | VARCHAR(100) | YES | Processing code | `specProcessingApplied` | `'laser_normalized'` |
| `data_source` | VARCHAR(20) | YES | `'loupe'` or `'pds4'` — data provenance | Constant `'loupe'` | Constant `'pds4'` |
| `scan_type` | VARCHAR(20) | YES | See `ScanType` enum below | Derived from name | Sequence code classification |
| `target_type` | VARCHAR(20) | NO | Auto-classified target category — see [`TargetType`](#targettype) below | `classify_target_type(target, scan_name)` | Same |
| `scan_class` | VARCHAR(20) | NO | Scan classification — see [`ScanClass`](#scanclass) below | `classify_scan_class(scan_name)` | Same |
| `parent_scan_id` | UUID | YES (FK self) | For `sub_scan` rows: parent primary scan; SET NULL on parent delete | NULL or set by classifier | NULL or set by classifier |
| `source_scan_ids` | JSON | YES | For `composite` rows: best-effort provenance list of contributing scan IDs | NULL | NULL |
| `processing_status` | TEXT | YES | Pipeline processing state (`pending`, `running`, `complete`, `error`) | Set by pipeline | Set by pipeline |
| `processed_at` | DATETIME | YES | Timestamp of last successful pipeline run | Set by pipeline | Set by pipeline |
| `processing_config_hash` | VARCHAR(64) | YES | SHA-256 of the config used for the most recent processing run | Set by pipeline | Set by pipeline |
| `processing_pipeline_version` | VARCHAR(20) | YES | Code-version tag (e.g., commit SHA) of the pipeline that ran | Set by pipeline | Set by pipeline |
| `processing_error` | TEXT | YES | Error message from the most recent failed run, if any | Set by pipeline | Set by pipeline |
| `site_drive` | VARCHAR(20) | YES | 7-digit Rover Motion Counter | NULL | `geom:Motion_Counter` site+drive |
| `sequence_id` | VARCHAR(20) | YES | SRLC sequence code | NULL | `msn_surface:Command_Execution` |
| `source_path` | TEXT | YES | Original file path | Workspace path | PDS4 LID |
| `loupe_metadata` | JSON | YES | Full loupe.csv as JSON | All fields | NULL |
| `pds4_metadata` | JSON | YES | PDS4 label extract | NULL | Selected fields |
| `created_at` | DATETIME | NO | Record creation time | Generated | Generated |
| `updated_at` | DATETIME | YES | Last modification time | Generated | Generated |

**Cross-field constraints** (`scan_class` invariants, enforced via CHECK constraint):

- `primary` rows: `parent_scan_id` and `source_scan_ids` must both be NULL.
- `sub_scan` rows: `parent_scan_id` must be set; `source_scan_ids` must be NULL.
- `composite` rows: `source_scan_ids` must be a non-empty JSON array; `parent_scan_id` must be NULL.

**Indexes:**

| Index Name | Columns | Purpose |
|------------|---------|---------|
| `ix_scans_sol_number` | `sol_number` | FK lookup |
| `ix_scans_scan_name` | `scan_name` | Name lookup |
| `ix_scans_target` | `target` | Target filter |
| `ix_scans_scan_id` | `scan_id` | Original-ID lookup |
| `ix_scans_sol_scan_name` | `(sol_number, scan_name)` | Compound lookup |
| `ix_scans_sclk` | `sclk_start` | SCLK range queries |
| `ix_scans_target_type` | `target_type` | Target-type filter |
| `ix_scans_scan_class` | `scan_class` | Scan-class filter |

### ScanPoint

A single measurement point within a scan.

| Field | Type | Nullable | Description | Loupe Source | PDS4 Source |
|-------|------|----------|-------------|--------------|-------------|
| `id` | UUID | NO (PK) | Unique identifier | Generated | Generated |
| `scan_id` | UUID | NO (FK) | Parent scan | Generated | Generated |
| `point_index` | INTEGER | NO | 0-based index | Row number | `Position_index` |
| `azimuth_dn` | INTEGER | YES | Scanner azimuth (DN) | spatial.csv `az` | RMO |
| `elevation_dn` | INTEGER | YES | Scanner elevation (DN) | spatial.csv `el` | RMO |
| `x_pixel` | FLOAT | YES | ACI X coordinate | spatial.csv `x` | RMO `x` |
| `y_pixel` | FLOAT | YES | ACI Y coordinate | spatial.csv `y` | RMO `y` |
| `coordinate_frame` | VARCHAR(30) | YES | See `CoordinateFrame` enum below | `'scanner_workspace'` | `'aci_pixel'` |
| `azimuth_error` | FLOAT | YES | Scanner az error | spatial.csv | Not available |
| `elevation_error` | FLOAT | YES | Scanner el error | spatial.csv | Not available |
| `photodiode_mean` | FLOAT | YES | Mean laser intensity | photodiodeRaw.csv mean | Not available |
| `photodiode_std` | FLOAT | YES | Laser intensity std | photodiodeRaw.csv std | Not available |

### Spectrum

A spectral measurement (one region, one processing level).

> **Spectral Region Reference:** See [`docs/schema/SPECTRAL_REGIONS.md`](SPECTRAL_REGIONS.md)
> for the canonical definition of spectral regions, wavelength calibration, and R123 stitching.

| Field | Type | Nullable | Description | Loupe Source | PDS4 Source |
|-------|------|----------|-------------|--------------|-------------|
| `id` | UUID | NO (PK) | Unique identifier | Generated | Generated |
| `scan_point_id` | UUID | NO (FK) | Parent point | Generated | Generated |
| `region` | TEXT | NO | 'R1', 'R2', 'R3', 'R123' — see [SPECTRAL_REGIONS.md](SPECTRAL_REGIONS.md) | From file structure | Product code |
| `spectrum_type` | TEXT | NO | 'active', 'dark', 'dark_subtracted' | Filename | Product code |
| `processing_level` | TEXT | NO | See ProcessingLevel enum | Processing suffix | Collection |
| `intensities` | BLOB | NO | Float32 array (compressed) | CSV values | CSV values |
| `wavelength_source` | VARCHAR(30) | YES | `'loupe_polynomial'` or `'pds_embedded'` | `'loupe_polynomial'` | `'pds_embedded'` |
| `wavelengths` | BLOB | YES | Float32 array (if custom) | Calculated | Calculated |
| `wavenumbers` | BLOB | YES | Float32 array (if custom) | Calculated | Calculated |

### InstrumentState

State-of-health telemetry for a scan.

| Field | Type | Nullable | Description | Loupe Source | PDS4 Source |
|-------|------|----------|-------------|--------------|-------------|
| `id` | UUID | NO (PK) | Unique identifier | Generated | Generated |
| `scan_id` | UUID | NO (FK) | Parent scan | Generated | Generated |
| `ccd_temp_c` | FLOAT | YES | CCD temperature | `SE_CCD_TEMP_STAT_REG` | EDR SOH |
| `pcb_temp_c` | FLOAT | YES | PCB temperature | `CNDH_PCB_TEMP_STAT_REG` | EDR SOH |
| `laser_prt1_c` | FLOAT | YES | Laser temp 1 | `SE_LASER_PRT1_STAT_REG` | EDR SOH |
| `laser_prt2_c` | FLOAT | YES | Laser temp 2 | `SE_LASER_PRT2_STAT_REG` | EDR SOH |
| `laser_shot_counter` | INTEGER | YES | Cumulative shots | `laser_shot_counter` | EDR SOH |
| `laser_misfire_counter` | INTEGER | YES | Cumulative misfires | `laser_misfire_counter` | EDR SOH |
| `arc_event_counter` | INTEGER | YES | Arc events | `arc_event_counter` | EDR SOH |
| `voltage_1_2v` | FLOAT | YES | 1.2V rail | `CNDH_1_2_V_STAT_REG` | EDR SOH |
| `voltage_3_3v` | FLOAT | YES | 3.3V rail | `CNDH_3_3_V_STAT_REG` | EDR SOH |
| `voltage_5v_dac` | FLOAT | YES | 5V DAC | `CNDH_5_V_DAC_STAT_REG` | EDR SOH |
| `voltage_5v_adc` | FLOAT | YES | 5V ADC | `CNDH_5_V_ADC_STAT_REG` | EDR SOH |
| `voltage_15v` | FLOAT | YES | +15V rail | `CNDH_15_V_STAT_REG` | EDR SOH |
| `voltage_neg_15v` | FLOAT | YES | -15V rail | `CNDH_NEG_15_V_STAT_REG` | EDR SOH |
| `laser_int_time_us` | INTEGER | YES | Integration time | `LASER_INT_TIME` | EDR |
| `laser_rep_rate_hz` | INTEGER | YES | Repetition rate | `LASER_REP_RATE` | EDR |
| `laser_current_a` | FLOAT | YES | Laser current | `LASER_CURRENT` | EDR |
| `full_telemetry` | JSON | YES | All SOH fields | loupe.csv JSON | EDR SOH JSON |

### CCDConfiguration

CCD timing and region configuration.

| Field | Type | Nullable | Description | Loupe Source | PDS4 Source |
|-------|------|----------|-------------|--------------|-------------|
| `id` | UUID | NO (PK) | Unique identifier | Generated | Generated |
| `scan_id` | UUID | NO (FK) | Parent scan | Generated | Generated |
| `region_enable` | INTEGER | YES | Enabled regions mask | `REGION_ENABLE` | EDR |
| `gain_2d` | INTEGER | YES | 2D gain setting | `CCD_GAIN_2D` | EDR |
| `mode_2d` | INTEGER | YES | 2D mode | `MODE_2D` | EDR |
| `vert_col1_low` | INTEGER | YES | Vertical column 1 low | `CCD_VERT_COL1_LOW` | EDR |
| `vert_col1_high` | INTEGER | YES | Vertical column 1 high | `CCD_VERT_COL1_HIGH` | EDR |
| `vert_col2_low` | INTEGER | YES | Vertical column 2 low | `CCD_VERT_COL2_LOW` | EDR |
| `vert_col2_high` | INTEGER | YES | Vertical column 2 high | `CCD_VERT_COL2_HIGH` | EDR |
| `vert_col3_low` | INTEGER | YES | Vertical column 3 low | `CCD_VERT_COL3_LOW` | EDR |
| `vert_col3_high` | INTEGER | YES | Vertical column 3 high | `CCD_VERT_COL3_HIGH` | EDR |
| `horz_clock_lim` | INTEGER | YES | Horizontal clock limit | `CCD_HORZ_CLOCK_LIM` | EDR |

### ScannerCalibration

Scanner coordinate calibration parameters.

| Field | Type | Nullable | Description | Loupe Source | PDS4 Source |
|-------|------|----------|-------------|--------------|-------------|
| `id` | UUID | NO (PK) | Unique identifier | Generated | Generated |
| `scan_id` | UUID | NO (FK) | Parent scan | Generated | Generated |
| `az_scale` | FLOAT | NO | Azimuth scale factor | `az_scale` | Default |
| `el_scale` | FLOAT | NO | Elevation scale factor | `el_scale` | Default |
| `laser_x` | INTEGER | NO | Laser center X pixel | `laser_x` | Default |
| `laser_y` | INTEGER | NO | Laser center Y pixel | `laser_y` | Default |
| `rotation_deg` | FLOAT | NO | Rotation angle | `rotation` | Default |

### ContextImage

Associated ACI or WATSON context images.

| Field | Type | Nullable | Description | Loupe Source | PDS4 Source |
|-------|------|----------|-------------|--------------|-------------|
| `id` | UUID | NO (PK) | Unique identifier | Generated | Generated |
| `scan_id` | UUID | NO (FK) | Parent scan | Generated | Generated |
| `image_type` | TEXT | NO | 'ACI' or 'WATSON' | From filename | From collection |
| `file_path` | TEXT | NO | Image file path | img/ directory | data_aci/watson |
| `pds_lidvid` | VARCHAR(200) | YES | Full PDS LIDVID for image product | NULL | `urn:nasa:pds:...::<ver>.0` |
| `product_id` | TEXT | YES | PDS product ID | `product_ID` | LID |
| `sclk` | BIGINT | YES | Acquisition SCLK | From filename | From label |
| `pixel_scale_um` | FLOAT | YES | um/pixel | `pixel_scale` | From label |
| `working_distance_cm` | FLOAT | YES | Working distance | `range` | From label |
| `motor_position` | INTEGER | YES | Focus motor pos | `motor_pos` | From label |
| `exposure_time_ms` | FLOAT | YES | Exposure time | `exp_time` | From label |
| `led_illumination` | BOOLEAN | YES | LED on/off | `led_flag` | From label |
| `width_px` | INTEGER | YES | Image width | Derived | From label |
| `height_px` | INTEGER | YES | Image height | Derived | From label |

**VICAR metadata columns** (added in migration `0385ab87eb83`):

| Field | Type | Nullable | Description | Loupe Source | PDS4 Source |
|-------|------|----------|-------------|--------------|-------------|
| `file_format` | VARCHAR(10) | YES | VICAR FORMAT field | VICAR header | NULL |
| `camera_id` | VARCHAR(10) | YES | Camera identifier (ACI/WATSON) | VICAR header | NULL |
| `sol_number` | INTEGER | YES | Mars sol from VICAR label | `PLANET_DAY_NUMBER` | From label |
| `sclk_start` | BIGINT | YES | Spacecraft clock start | `SPACECRAFT_CLOCK_START_COUNT` | From label |
| `sclk_stop` | BIGINT | YES | Spacecraft clock stop | `SPACECRAFT_CLOCK_STOP_COUNT` | From label |
| `sequence_id` | VARCHAR(50) | YES | Observation sequence code | `SEQUENCE_ID` | NULL |
| `image_time` | DATETIME | YES | Image acquisition time | `IMAGE_TIME` | `start_date_time` |
| `focus_mode` | VARCHAR(20) | YES | Autofocus mode | `FOCUS_MODE` | NULL |
| `focus_position_count` | INTEGER | YES | Focus motor step count | `FOCUS_POSITION_COUNT` | NULL |
| `local_mean_solar_time` | VARCHAR(50) | YES | Local Mars time | `LOCAL_MEAN_SOLAR_TIME` | NULL |
| `rover_motion_counter` | TEXT | YES | Site-drive position code | `ROVER_MOTION_COUNTER` | `geom:Motion_Counter` |
| `vicar_metadata` | JSON | YES | Full VICAR label as JSON | All parsed fields | NULL |
| `updated_at` | DATETIME | YES | Last modification time | Generated | Generated |

### RegionOfInterest

User-defined regions of interest (Loupe-specific).

| Field | Type | Nullable | Description | Loupe Source | PDS4 Source |
|-------|------|----------|-------------|--------------|-------------|
| `id` | UUID | NO (PK) | Unique identifier | Generated | N/A |
| `scan_id` | UUID | NO (FK) | Parent scan | Generated | N/A |
| `name` | TEXT | NO | ROI display name | ROI name | N/A |
| `color_hex` | TEXT | NO | Color code | Color hex | N/A |
| `point_indices` | JSON | NO | List of point indices | Point list | N/A |

### FittedPeak

Peak fitting results (analysis output). Supports four fitting domains via the `fit_modality` discriminator: Raman peaks (minerals, organics, hydration) use wavenumber fields (`center_cm1`, `fwhm_cm1`), while fluorescence peaks use wavelength fields (`center_nm`, `fwhm_nm`).

| Field | Type | Nullable | Description | Source |
|-------|------|----------|-------------|--------|
| `id` | UUID | NO (PK) | Unique identifier | Generated |
| `spectrum_id` | UUID | NO (FK) | Parent spectrum | Generated |
| `fit_modality` | STRING(20) | NO | Domain discriminator — see [FitModality](#fitmodality) | Analysis |
| `peak_type` | TEXT | NO | 'gaussian', 'lorentzian', 'voigt' | Fitting |
| `center_cm1` | FLOAT | YES | Peak center in wavenumber (cm⁻¹). Required for Raman domains, NULL for fluorescence | Fitting |
| `center_uncertainty` | FLOAT | YES | Center uncertainty (cm⁻¹) | Fitting |
| `center_nm` | FLOAT | YES | Peak center in wavelength (nm). Required for fluorescence, NULL for Raman | Fitting |
| `amplitude` | FLOAT | NO | Peak amplitude (counts) | Fitting |
| `amplitude_uncertainty` | FLOAT | YES | Amplitude uncertainty | Fitting |
| `fwhm_cm1` | FLOAT | YES | Full width half max (cm⁻¹). Required for Raman, NULL for fluorescence | Fitting |
| `fwhm_uncertainty` | FLOAT | YES | FWHM uncertainty (cm⁻¹) | Fitting |
| `fwhm_nm` | FLOAT | YES | Full width half max (nm). Required for fluorescence, NULL for Raman | Fitting |
| `is_saturated` | BOOLEAN | YES | CCD saturation flag (three-tier). Only used for fluorescence peaks | Fitting |
| `area` | FLOAT | YES | Integrated area | Fitting |
| `snr` | FLOAT | YES | Signal to noise ratio | Fitting |
| `fit_quality` | FLOAT | YES | Goodness of fit (R²) | Fitting |
| `mineral_assignment` | TEXT | YES | Feature label — see [Cross-Domain Assignment Semantics](#cross-domain-assignment-semantics) | Analysis |
| `assignment_confidence` | FLOAT | YES | Assignment confidence (currently NULL; reserved for future ML classifiers) | Analysis |

**Indexes:**

| Index Name | Columns | Purpose |
|------------|---------|---------|
| `ix_fitted_peaks_spectrum_id` | `spectrum_id` | FK lookup |
| `ix_fitted_peaks_center_cm1` | `center_cm1` | Raman wavenumber range queries |
| `ix_fitted_peaks_center_nm` | `center_nm` | Fluorescence wavelength range queries |
| `ix_fitted_peaks_fit_modality` | `fit_modality` | Domain filtering |
| `ix_fitted_peaks_mineral_assignment` | `mineral_assignment` | Feature label lookup |
| `ix_fitted_peaks_modality_assignment` | `(fit_modality, mineral_assignment)` | Composite: co-occurrence queries across domains |

**Triggers:**

| Trigger | Event | Constraint |
|---------|-------|------------|
| `check_fit_modality_insert` | BEFORE INSERT | `fit_modality` must be one of: `minerals`, `organics`, `hydration`, `fluorescence` |
| `check_fit_modality_update` | BEFORE UPDATE | Same constraint on UPDATE of `fit_modality` |

**Domain consistency rules** (enforced at application layer via Pydantic `validate_domain_fields`):

- Raman domains (`minerals`, `organics`, `hydration`): `center_cm1` and `fwhm_cm1` required; `center_nm`, `fwhm_nm`, `is_saturated` must be NULL
- Fluorescence domain: `center_nm` and `fwhm_nm` required; `center_cm1`, `fwhm_cm1` must be NULL

### Spectrogram

Cached 2D heatmap of spectral intensity across the points of a scan, used for fast Map Mode rendering. Configuration is stored as JSON; the intensity matrix is zlib-compressed float32 (same recipe as `Spectrum.intensities`). Added in migration `87cb884d3399`.

| Field | Type | Nullable | Description | Source |
|-------|------|----------|-------------|--------|
| `id` | UUID | NO (PK) | Unique identifier | Generated |
| `scan_id` | UUID | NO (FK) | Parent scan | Generated |
| `region` | VARCHAR(10) | NO | Spectral region (`R1`, `R2`, `R3`, `R123`) — see [SPECTRAL_REGIONS.md](SPECTRAL_REGIONS.md) | Set on creation |
| `processing_level` | VARCHAR(20) | NO | Processing level applied (see [`ProcessingLevel`](#processinglevel)) | Set on creation |
| `config` | JSON | NO | Visualization config (see [SpectrogramConfig](#spectrogramconfig)) | Set on creation |
| `intensity_matrix` | BLOB | NO | Compressed 2D intensity array (zlib-compressed float32) | Computed |
| `n_points` | INTEGER | NO | Number of scan points (rows) | Computed |
| `n_channels` | INTEGER | NO | Number of spectral channels (columns) | Computed |
| `wavenumber_min` | FLOAT | NO | Minimum wavenumber (cm⁻¹) — for R1; for R2/R3 the wavelength bound is encoded here | Computed |
| `wavenumber_max` | FLOAT | NO | Maximum wavenumber (cm⁻¹) | Computed |
| `wavenumbers` | BLOB | YES | Optional non-uniform wavenumber grid (compressed float32) | Computed |
| `point_labels` | JSON | YES | Optional point label annotations | Computed |
| `point_indices` | JSON | YES | Optional included-point indices (subset rendering) | Computed |
| `intensity_min` | FLOAT | YES | Minimum intensity value across the matrix | Computed |
| `intensity_max` | FLOAT | YES | Maximum intensity value across the matrix | Computed |
| `title` | VARCHAR(500) | YES | Display title | Set on creation |
| `created_at` | DATETIME | NO | Record creation time | Generated |
| `updated_at` | DATETIME | YES | Last modification time | Generated |

**Indexes:**

| Index Name | Columns | Purpose |
|------------|---------|---------|
| `ix_spectrograms_scan_id` | `scan_id` | FK lookup |
| `ix_spectrograms_scan_region` | `(scan_id, region)` | Compound: per-scan-per-region rendering |

#### SpectrogramConfig

Embedded JSON structure stored in `Spectrogram.config`. Defined by the `SpectrogramConfig` Pydantic model (`src/sherloc_pipeline/models/spectrogram.py`).

```json
{
  "colormap": "viridis",
  "normalization": "percentile",
  "percentile_low": 1.0,
  "percentile_high": 99.0,
  "interpolation": "none",
  "show_colorbar": true,
  "figure_size": [12.0, 8.0]
}
```

### MapDisplayCoordinate

Cache of resolved ACI-pixel coordinates for Map Mode point overlay. One row per scan point. Populated lazily on first access by `core.coordinates.resolve_display_coordinates()`, then reused without re-reading workspace files.

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `scan_point_id` | UUID | NO (PK, FK → scan_points) | Scan point this coordinate belongs to |
| `aci_x` | FLOAT | NO | Resolved X pixel coordinate in ACI image frame |
| `aci_y` | FLOAT | NO | Resolved Y pixel coordinate in ACI image frame |
| `transform_method` | VARCHAR(30) | NO | `'identity'` (already in `aci_pixel` frame) or `'scanner_calibration'` (transformed via `load_spatial_table()` + Loupe polynomial calibration) |
| `computed_at` | DATETIME | NO | Cache write time |

CASCADE on `scan_point_id` delete.

### User

User identity for preference and profile persistence. Application-state table (no Loupe/PDS source).

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `id` | INTEGER | NO (PK, autoincrement) | Synthetic user ID |
| `email` | TEXT | NO (UNIQUE) | User email — uniqueness key |
| `display_name` | TEXT | YES | Optional display name |
| `created_at` | TEXT | NO | ISO-8601 timestamp (server default) |
| `last_seen_at` | TEXT | NO | ISO-8601 timestamp (server default; updated by app code) |

### UserPreference

Key-value preferences scoped to a user. Values are JSON-encoded strings.

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `id` | INTEGER | NO (PK, autoincrement) | Synthetic preference ID |
| `user_id` | INTEGER | NO (FK → users.id) | Owning user |
| `key` | TEXT | NO | Preference key |
| `value` | TEXT | NO | JSON-encoded preference value |
| `updated_at` | TEXT | NO | ISO-8601 timestamp (server default) |

**Constraints:** `UNIQUE(user_id, key)` — `uq_user_preferences_user_key`.

### ClassificationProfile

Custom peak-classification profiles owned by a user. The full profile (peak families, wavenumber/wavelength bands, thresholds) is serialized as JSON.

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `id` | TEXT | NO (PK) | Profile UUID |
| `user_id` | INTEGER | NO (FK → users.id) | Owning user |
| `name` | TEXT | NO | Display name |
| `profile_json` | TEXT | NO | Full `ClassificationProfile` as JSON |
| `created_at` | TEXT | NO | ISO-8601 timestamp |
| `updated_at` | TEXT | NO | ISO-8601 timestamp |

### MapFitCache

Ephemeral per-scan map-fit results cache. One row per (scan, domain set, point subset, profile) tuple. Used by Map Mode to avoid recomputing fits when the user re-opens a scan with the same configuration.

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `id` | TEXT | NO (PK) | Cache row UUID |
| `scan_id` | TEXT | NO | Scan being cached (string-typed; not a declared FK) |
| `domains` | TEXT | NO | JSON array of fit domains, sorted (e.g., `["minerals","fluorescence"]`) |
| `point_subset` | TEXT | YES | JSON array of point indices, or NULL meaning "all points" |
| `profile_hash` | TEXT | NO | SHA-256 of the `ClassificationProfile` JSON used for the fit |
| `profile_name` | TEXT | YES | Human-readable profile name (denormalized for display) |
| `user_id` | INTEGER | YES (FK → users.id) | User who triggered the fit (NULL for anonymous/auto-save) |
| `results_json` | TEXT | NO | Full per-point fit results as JSON |
| `n_points` | INTEGER | NO | Number of points cached |
| `n_detections_json` | TEXT | NO | JSON object with per-domain detection counts |
| `created_at` | TEXT | NO | ISO-8601 creation timestamp |
| `expires_at` | TEXT | YES | Optional ISO-8601 expiration timestamp; reserved for future TTL enforcement (no current cleanup process reads this column) |

---

## Enums

### ScanType

Classifies observations by scan type. Replaces fragile name-pattern matching with a first-class enum.

| Value | Description | Loupe Derivation | PDS Derivation |
|-------|-------------|------------------|----------------|
| `detail` | Detail scan (typically 100 points) | n_points ≤ 200 | Sequence code + n_spectra ≤ 200 |
| `survey` | Survey scan (typically 1296 points) | n_points > 200 | Sequence code + n_spectra > 200 |
| `calibration` | Calibration scan (AlGaN, dark current) | Name contains 'AlGaN'/'algan' | Sequence code `srlc10000`/`srlc16000` |

### DataSource

Provenance tracking for data origin.

| Value | Description |
|-------|-------------|
| `loupe` | Ingested from Loupe working directories |
| `pds4` | Ingested from PDS4 archive products |

### TargetType

Classification of scan targets by purpose. Replaces fragile target-name pattern matching with a first-class column. Auto-classified by `classify_target_type(target, scan_name)` in `src/sherloc_pipeline/models/spectra.py` on insert.

| Value | Description | Classification Rule |
|-------|-------------|---------------------|
| `mars_target` | Mars surface science target (e.g., `Amherst_Point`, `Garde`) | Default — anything not engineering or calibration |
| `cal_target` | Calibration target (e.g., AlGaN, Teflon, external calibration) | Known cal targets in target field, or scan name starts with `algan` |
| `engineering` | Engineering / housekeeping (e.g., conjunction, arm stowed) | NULL/empty target, known engineering targets, scan_names starting with `power_` or containing `laser_disabled` |

### ScanClass

Distinguishes primary observations from sub-scans (detail variants) and composites (aggregate products). Auto-classified by `classify_scan_class(scan_name)` on insert. Used together with `parent_scan_id` and `source_scan_ids` to model derivation lineage.

| Value | Description | Classification Rule | Linkage |
|-------|-------------|---------------------|---------|
| `primary` | Standalone scan | Default | `parent_scan_id` and `source_scan_ids` both NULL |
| `sub_scan` | Detail-scan variant tied to a parent | `scan_name` ends in `a`, `b`, or `c` after a digit/underscore (e.g., `detail_1a`) | `parent_scan_id` set; `source_scan_ids` NULL |
| `composite` | Aggregate product derived from multiple scans | `scan_name` contains `_all`, `_median`, or `_sum_active` | `source_scan_ids` non-empty JSON array; `parent_scan_id` NULL |

A cross-field CHECK constraint on `scans` enforces these linkage rules.

### CoordinateFrame

Coordinate system for scan point positions.

| Value | Description | Typical Range |
|-------|-------------|---------------|
| `scanner_workspace` | Loupe scanner workspace coordinates (relative) | ±0.5 (detail), ±2.5 (survey) |
| `aci_pixel` | ACI image pixel coordinates (absolute) | 700–900 |

### SpectrumType

Type of spectral measurement. Extended to support PDS processed products.

| Value | Description | Source |
|-------|-------------|--------|
| `active` | Laser-illuminated spectrum | Loupe |
| `dark` | Dark frame (no laser) | Loupe |
| `dark_subtracted` | Active minus dark | Loupe |
| `laser_normalized` | Laser-power-normalized spectrum | PDS processed |

### ProcessingLevel

Processing level of spectral data.

| Value | Description |
|-------|-------------|
| `raw` | Original CCD counts |
| `calibrated` | Wavelength/wavenumber calibrated |
| `normalized` | Laser-power normalized (PDS processed) |
| `derived` | Higher-level derived products |

### FitModality

Domain discriminator for fitted peaks. Determines which position/width fields are populated and what assignment logic applies.

| Value | Description | Position Field | Width Field | Assignment Function |
|-------|-------------|----------------|-------------|---------------------|
| `minerals` | Raman mineral peak fitting (R1 region) | `center_cm1` | `fwhm_cm1` | `assign_min_id()` |
| `organics` | Raman organic band fitting (R1 region) | `center_cm1` | `fwhm_cm1` | `classify_organic_band()` |
| `hydration` | Raman hydration band fitting (R1 region) | `center_cm1` | `fwhm_cm1` | `classify_hydration_band()` |
| `fluorescence` | Fluorescence emission fitting (R2/R3 regions) | `center_nm` | `fwhm_nm` | `assign_fluor_group()` |

Enforced by SQLite triggers (`check_fit_modality_insert`, `check_fit_modality_update`).

### Cross-Domain Assignment Semantics

The `mineral_assignment` column stores feature labels whose meaning varies by `fit_modality`:

| fit_modality | mineral_assignment Values | Assignment Logic |
|--------------|--------------------------|------------------|
| `minerals` | Mineral feature names (e.g., `sulf1_v1`, `carb_v1`, `oliv_v1`) | `assign_min_id()`: wavenumber band lookup from `MINERAL_BANDS` table |
| `organics` | `D_band` (1250-1450 cm⁻¹), `G_band` (1500-1700 cm⁻¹), `unidentified` | `classify_organic_band()`: center position range |
| `hydration` | `OH_stretch` (3000-4000 cm⁻¹), `H2O_bend` (1500-1700 cm⁻¹), `unidentified` | `classify_hydration_band()`: center position range |
| `fluorescence` | Group labels: `group1a` (300-307 nm), `group1b` (322-329 nm), `group2` (335-350 nm), `group3` (270-295 nm), `unidentified` | `assign_fluor_group()`: center wavelength range |

The composite index `ix_fitted_peaks_modality_assignment` enables efficient cross-domain queries such as co-occurrence analysis (e.g., "find scan points with both Ca-sulfate Raman peaks and Ce³⁺ fluorescence emission").

---

## Field Mappings

### Loupe to Unified Schema

#### loupe.csv Fields

| Loupe Field | Unified Field | Table | Notes |
|-------------|---------------|-------|-------|
| `original_data_file` | `scan_id` | Scan | |
| `human_readable_workspace` | `scan_name` | Scan | (column was renamed from `target_name` in `172d8c59b5c9`) |
| (parsed from name) | `target` | Scan | Geological target name |
| `n_spectra` | `n_points` | Scan | |
| `n_channels` | `n_channels` | Scan | Should be 2148 |
| `laser_wavelength` | `laser_wavelength_nm` | Scan | |
| `shots_per_spec` | `shots_per_point` | Scan | |
| `az_scale` | `az_scale` | ScannerCalibration | |
| `el_scale` | `el_scale` | ScannerCalibration | |
| `laser_x` | `laser_x` | ScannerCalibration | |
| `laser_y` | `laser_y` | ScannerCalibration | |
| `rotation` | `rotation_deg` | ScannerCalibration | |
| `specProcessingApplied` | `processing_applied` | Scan | |
| `CNDH_PCB_TEMP_STAT_REG` | `pcb_temp_c` | InstrumentState | Parse float from string |
| `SE_CCD_TEMP_STAT_REG` | `ccd_temp_c` | InstrumentState | Parse float from string |
| `laser_shot_counter` | `laser_shot_counter` | InstrumentState | |
| `LASER_INT_TIME` | `laser_int_time_us` | InstrumentState | Parse int |
| `LASER_REP_RATE` | `laser_rep_rate_hz` | InstrumentState | Parse int |
| All fields | `loupe_metadata` | Scan | Store as JSON |

#### spatial.csv Fields

| Loupe Field | Unified Field | Table |
|-------------|---------------|-------|
| `az` | `azimuth_dn` | ScanPoint |
| `el` | `elevation_dn` | ScanPoint |
| `x` | `x_pixel` | ScanPoint |
| `y` | `y_pixel` | ScanPoint |
| `azimuth error` | `azimuth_error` | ScanPoint |
| `elevation error` | `elevation_error` | ScanPoint |

### PDS4 to Unified Schema

#### Label Time Coordinates

| PDS4 Field | Unified Field | Table |
|------------|---------------|-------|
| `sol_number` | `sol_number` | Sol |
| `start_date_time` | `earth_date` | Sol |
| `solar_longitude` | `solar_longitude` | Sol |
| `spacecraft_clock_start` | `sclk_start` | Scan |
| `spacecraft_clock_stop` | `sclk_stop` | Scan |
| `local_mean_solar_time` | `pds4_metadata.lmst` | Scan (JSON) |

#### RMO Product Fields

| PDS4 Field | Unified Field | Table |
|------------|---------------|-------|
| `Position_index` | `point_index` | ScanPoint |
| `Image_name` | Related via scan | ContextImage |
| `x` | `x_pixel` | ScanPoint |
| `y` | `y_pixel` | ScanPoint |
| `Spectral_intensity_N` | Computed from spectra | - |

#### Product Type Mapping

| PDS4 Product | Unified Equivalent |
|--------------|-------------------|
| `data_raw` EDR | InstrumentState, CCDConfiguration |
| `data_intermediate` RAC | Spectrum (processing_level='calibrated') |
| `data_processed` RCS | Spectrum (processing_level='derived') |
| `data_processed` RMO | ScanPoint (positions) |
| `data_aci` | ContextImage (image_type='ACI') |
| `data_watson` | ContextImage (image_type='WATSON') |

---

## Data Type Normalization

### Temperature Parsing

Loupe stores temperatures as strings with units:
```python
def parse_temperature(value: str) -> float | None:
    """Parse '37.200 C' -> 37.2"""
    if value in ('N/A', 'None', ''):
        return None
    return float(value.split()[0])
```

### Voltage Parsing

Loupe stores voltages as strings with units:
```python
def parse_voltage(value: str) -> float | None:
    """Parse '1.204 V' -> 1.204"""
    if value in ('N/A', 'None', ''):
        return None
    return float(value.split()[0])
```

### Time Parsing

Loupe stores times as strings:
```python
def parse_time(value: str) -> int | None:
    """Parse '20 us' -> 20 or '80 Hz' -> 80"""
    if value in ('N/A', 'None', ''):
        return None
    return int(value.split()[0])
```

### SCLK Extraction

From Loupe filenames:
```python
def extract_sclk(original_data_file: str) -> int:
    """Extract SCLK from 'SrlcSpecSpecSohRaw_0672194998-62417-1'"""
    parts = original_data_file.split('_')[1].split('-')
    return int(parts[0])
```

### Spectral Array Storage

Intensities stored as compressed binary:
```python
import numpy as np
import zlib

def compress_spectrum(intensities: np.ndarray) -> bytes:
    """Compress float32 array for storage."""
    return zlib.compress(intensities.astype(np.float32).tobytes())

def decompress_spectrum(data: bytes, n_channels: int = 2148) -> np.ndarray:
    """Decompress to float32 array."""
    return np.frombuffer(zlib.decompress(data), dtype=np.float32)
```

---

## Ingestion Strategy

### Phase 1: Loupe Ingestion (Primary)

1. **Scan sols directory** for `sol_XXXX` directories
2. **Parse session file** (`.lpe`) or discover workspaces
3. **For each workspace**:
   - Read `loupe.csv` -> Scan, InstrumentState, CCDConfiguration, ScannerCalibration
   - Read `spatial.csv` -> ScanPoints
   - Read `activeSpectra.csv`, `darkSpectra.csv`, `darkSubSpectra.csv` -> Spectra
   - Read `photodiodeRaw.csv` -> Update ScanPoints
   - Read `roi.csv` -> RegionOfInterest
   - Read `img/` -> ContextImages

### Phase 2: PDS4 Ingestion (Optional/Supplementary)

1. **Download/mirror** specific collections if needed
2. **Parse inventory** CSV for product list
3. **For each product**:
   - Parse XML label -> Sol, Scan metadata
   - Parse CSV data -> Spectra, ScanPoints
   - Link to existing Loupe data via SCLK matching

### Deduplication

When ingesting both sources:
1. Match by `sclk_start` (unique per observation)
2. Prefer Loupe data (more complete for analysis)
3. Merge PDS4-only fields into existing records
4. Flag conflicts in `data_source` field

---

## Validation Rules

### Cross-Field Validation

```python
def validate_scan(scan: Scan) -> List[str]:
    """Validate scan consistency."""
    errors = []

    # Check spectrum count matches point count
    if scan.n_points != len(scan.scan_points):
        errors.append(f"n_points ({scan.n_points}) != scan_points count")

    # Check region coverage
    regions = {s.region for p in scan.scan_points for s in p.spectra}
    if 'R1' not in regions:
        errors.append("Missing R1 region spectra")

    # Check scan_class invariants
    if scan.scan_class == "sub_scan" and scan.parent_scan_id is None:
        errors.append("sub_scan rows must have parent_scan_id set")
    if scan.scan_class == "composite" and not scan.source_scan_ids:
        errors.append("composite rows must have non-empty source_scan_ids")
    if scan.scan_class == "primary" and (scan.parent_scan_id or scan.source_scan_ids):
        errors.append("primary rows must have parent_scan_id and source_scan_ids both NULL")

    return errors
```

### Required Fields

| Entity | Required Fields |
|--------|-----------------|
| Sol | `sol_number` |
| Scan | `sol_number`, `scan_name`, `scan_id`, `sclk_start`, `n_points`, `target_type`, `scan_class` |
| ScanPoint | `scan_id`, `point_index` |
| Spectrum | `scan_point_id`, `region`, `spectrum_type`, `processing_level`, `intensities` |

### Value Constraints

| Field | Constraint |
|-------|------------|
| `sol_number` | >= 0 |
| `n_points` | > 0, typically 100 |
| `n_channels` | == 2148 |
| `shots_per_point` | > 0 when present; NULL for PDS (not in processed products) |
| `region` | in {'R1', 'R2', 'R3', 'R123'} — see [SPECTRAL_REGIONS.md](SPECTRAL_REGIONS.md) |
| `spectrum_type` | in {'active', 'dark', 'dark_subtracted', 'laser_normalized'} |
| `scan_type` | in {'detail', 'survey', 'calibration'} — see `ScanType` enum |
| `coordinate_frame` | in {'scanner_workspace', 'aci_pixel'} — see `CoordinateFrame` enum |
| `data_source` | in {'loupe', 'pds4'} — see `DataSource` enum |
| `wavelength_source` | in {'loupe_polynomial', 'pds_embedded'} |

---

## Migration Path

### From Current Pipeline

The existing pipeline uses:
- Direct CSV parsing in processing code
- No persistent storage
- Per-request data loading

### To PHASE Database

1. **Create SQLite database** at `./phase.db`
2. **Run initial ingestion** of all Loupe data
3. **Update pipeline** to query database instead of parsing files
4. **Add incremental ingestion** for new sols

### Schema Evolution

Use Alembic for migrations (batch mode for SQLite). Current head: `87cb884d3399` (11 migrations).

```python
# Example migration
def upgrade():
    op.add_column('scan', sa.Column('mission_phase', sa.Text()))
```

Recent migrations:
- `6cff8ea9cb06`: Add fit_modality, center_nm, fwhm_nm, is_saturated; relax cm1 nullability; remove duplicate indexes; add domain triggers

---

## Appendix A — Units & Conventions

### A.1 Spectroscopic units

| Quantity | Unit | Description | Typical range |
|----------|------|-------------|---------------|
| Wavenumber | cm⁻¹ | Raman shift from laser line (R1) | ~640 – 4200 (usable R1 window) |
| Wavelength | nm | Fluorescence emission (R2/R3) | 282 – 357 |
| Intensity | counts | Detector signal (16-bit DN) | 0 – 65535 |
| FWHM (Raman) | cm⁻¹ | Spectral peak width | 5 – 100 |
| FWHM (fluorescence) | nm | Emission band width | 1 – 30 |
| SNR | ratio | Peak-to-noise (Raman acceptance ≥ 3.0) | 1 – 1000 |

For canonical region definitions, channel mappings, and the Loupe polynomial calibration see [`SPECTRAL_REGIONS.md`](SPECTRAL_REGIONS.md).

### A.2 Spatial units

| Quantity | Unit | Description | Typical range |
|----------|------|-------------|---------------|
| Pixel coordinates | pixels | ACI image frame | 0 – 1648 × 0 – 1200 |
| Physical scale | μm/pixel | ACI projection | 10.1 (constant) |
| Scanner position | DN | Motor drive units | ±32768 |
| Field of view | mm | ACI physical area | 16.6 × 12.1 |

### A.3 Temporal units

| Quantity | Unit | Description |
|----------|------|-------------|
| SCLK | spacecraft-clock ticks | Mars 2020 spacecraft clock; stored as INTEGER |
| Sol | days since landing | Mars mission day; primary key on `sols` |
| Earth date | UTC date | Corresponding terrestrial date |
| LMST | HH:MM:SS string | Local Mean Solar Time at the rover |
| Solar longitude (Ls) | degrees | Mars seasonal angle, 0 – 360 |

### A.4 Conventions

- **UUIDs.** Application-generated identifiers use RFC 4122 v4 (random) and are stored as 36-character strings (e.g., `550e8400-e29b-41d4-a716-446655440000`).
- **Timestamps.** `created_at` / `updated_at` columns on the mission-data tables use timezone-aware `DateTime(timezone=True)`; tables in the application-state group store ISO-8601 strings with millisecond precision (`strftime('%Y-%m-%dT%H:%M:%fZ', 'now')`).
- **JSON columns.** UTF-8, RFC 7159, no trailing whitespace. Used for `loupe_metadata`, `pds4_metadata`, `vicar_metadata`, `full_telemetry`, `point_indices`, `source_scan_ids`, `config` (Spectrogram), and the `*_json` columns on the application-state tables.
- **BLOB compression.** Spectral arrays (`Spectrum.intensities/wavelengths/wavenumbers`, `Spectrogram.intensity_matrix`, `Spectrogram.wavenumbers`) use the recipe in §"Spectral Array Storage" above: `zlib.compress(arr.astype(np.float32).tobytes())` round-tripped via `np.frombuffer(zlib.decompress(blob), dtype=np.float32)`.

---

## Appendix B — Configuration Surface

This appendix points at where pipeline configuration actually lives in the codebase. There is no centralized JSON-Schema validator: configuration is split across a top-level YAML file and per-component pydantic dataclasses, and changes are governed by code review rather than schema enforcement.

### B.1 Top-level config

`config.yaml` at the repo root. Sections include `spectral_regions`, `fitting`, `fluorescence_fitting`, and per-domain parameter blocks. Many runtime knobs (e.g., `fitting.parallel_workers`, `fluorescence_fitting.parallel_workers`, R1 cutoff_channel) are read from this file via the loader in `src/sherloc_pipeline/core/utils.py`.

### B.2 Preprocessing parameters

Defined as `@dataclass` in `src/sherloc_pipeline/core/preprocessing.py`:

```python
@dataclass
class DespikeParams:
    window_size: int = 7              # rolling-median window
    zscore_threshold: float = 6.0     # MAD-based outlier threshold
    max_iterations: int = 1
    interpolation_method: str = "linear"
    run_length_max: int = 2
    laser_window: Tuple[float, float] = (600.0, 700.0)
    sulfate_center_window: Tuple[float, float] = (1014.0, 1020.0)
    sulfate_guard_enable: bool = True
    sulfate_guard_search: Tuple[float, float] = (990.0, 1050.0)
    sulfate_guard_min_prominence: float = 100.0
    sulfate_guard_min_halfwidth: float = 15.0
    sulfate_guard_max_halfwidth: float = 25.0
```

The cosmic-ray detection algorithm (rolling-median + MAD with sulfate-band guard) is provenance-tagged in the source as domain-expert work, not AI-tuned.

### B.3 Baseline parameters

Defined in `src/sherloc_pipeline/core/baseline.py`:

```python
@dataclass
class BaselineParams:
    lam: float = 1e6           # asPLS smoothness
    asymmetric_coef: float = 0.01
    iters: int = 10
    diff_order: int = 2
    tol: float = 1e-3
```

The R1 baseline path uses asPLS (`pybaselines.Baseline.aspls`); a `"poly"` linear fallback is available via `fit_baseline_window(method="poly")` for protected windows. asPLS hyperparameters and protected mineral windows are domain-expert calibrated.

### B.4 Fluorescence-fitting parameters

See `src/sherloc_pipeline/core/fluor_fitting.py` and the spec at `docs/specs/FLUORESCENCE_FITTING_SPEC.md`. Strategy is `"agnostic"` by default (1..max_peaks unconstrained Gaussians, AICc-selected; SNR ≥ 20, min peak separation 15 nm). The hypothesis-driven path produces constrained candidate models (M0–M7); falls back to agnostic if R² < 0.7. Group labels (`group1a`, `group1b`, `group2`, `group3`, `unidentified`) are post-hoc via `assign_fluor_group()`.

### B.5 ML / clustering parameters

Algorithm classes in `src/sherloc_pipeline/ml/clustering.py` — `KMeansClusterer`, `DBSCANClusterer`, `HierarchicalClusterer`, all inheriting from `BaseClusterer`. Hyperparameters are constructor-injected; results are returned in `ClusteringResult` with `labels`, `n_clusters`, `cluster_sizes`, optional `centroids` / `inertia` / `silhouette_score`, and a free-form `metadata` dict.

### B.6 VICAR header keys parsed into `context_images.vicar_metadata`

`src/sherloc_pipeline/vision/img_reader.py:parse_vicar_label()` extracts the keys below from the raw VICAR label of each ACI .IMG file and stores them in the `vicar_metadata` JSON column. This list reflects what the parser actually captures, not the broader VICAR-format standard.

| Key | Type | Group |
|-----|------|-------|
| `LBLSIZE`, `FORMAT`, `NL`, `NS`, `NB`, `RECSIZE`, `ORG` | int / string | Core |
| `INSTRUMENT_ID`, `INSTRUMENT_NAME`, `PRODUCT_ID`, `SEQUENCE_ID`, `IMAGE_ID` | string | Identification |
| `IMAGE_TIME`, `START_TIME`, `STOP_TIME` | string (ISO-8601) | Timing |
| `PLANET_DAY_NUMBER` | int | Timing (sol) |
| `SOLAR_LONGITUDE` | float | Timing |
| `SPACECRAFT_CLOCK_START_COUNT`, `LOCAL_MEAN_SOLAR_TIME`, `LOCAL_TRUE_SOLAR_TIME` | string | Timing |
| `FRAME_TYPE`, `GEOMETRY_PROJECTION_TYPE`, `DATA_PRODUCT_COMPRESSION_TYPE` | string | Image properties |
| `MISSION_NAME`, `TARGET_NAME` | string | Mission |

PDS3/ODL labels (used for processed ortho products) are parsed line-oriented and yield the same `vicar_metadata` dict shape with whichever keys the label declares; see `parse_pds3_label()`.

---

## Appendix C — Example Queries

Convenience patterns against the schema. All examples use the unified table names defined above and assume `region='R1'` for Raman work (substitute `'R2'` / `'R3'` for fluorescence).

### C.1 Retrieve all R1 dark-subtracted spectra for a given target

```sql
SELECT s.id, s.processing_level, sp.point_index
FROM spectra s
JOIN scan_points sp ON s.scan_point_id = sp.id
JOIN scans      sc ON sp.scan_id       = sc.id
WHERE sc.target = 'Garde'
  AND s.region        = 'R1'
  AND s.spectrum_type = 'dark_subtracted';
```

### C.2 Find scans with a linked ACI image

```sql
SELECT sc.scan_name, sc.target, sc.sol_number, ci.file_path
FROM scans          sc
JOIN context_images ci ON sc.id = ci.scan_id
WHERE ci.image_type = 'ACI'
ORDER BY sc.sol_number DESC;
```

### C.3 Count detected mineral peaks per assignment

```sql
SELECT mineral_assignment, COUNT(*) AS n_peaks
FROM fitted_peaks
WHERE fit_modality = 'minerals'
  AND mineral_assignment IS NOT NULL
GROUP BY mineral_assignment
ORDER BY n_peaks DESC;
```

### C.4 High-quality Raman mineral peaks

```sql
SELECT spectrum_id, center_cm1, amplitude, fwhm_cm1, snr, mineral_assignment
FROM fitted_peaks
WHERE fit_modality   = 'minerals'
  AND snr            >= 5.0
  AND fit_quality    >= 0.8
  AND mineral_assignment IS NOT NULL
ORDER BY snr DESC;
```

For fluorescence, swap `fit_modality = 'minerals'` for `'fluorescence'` and reference `center_nm` / `fwhm_nm` instead of the `cm1` columns.

---

## Appendix D — Field Inventory

Total: **16 ORM tables** across three groups.

**Mission data hierarchy (11 tables):**

| Entity | Field count |
|--------|-------------|
| Sol | 7 |
| Scan | 27 (incl. processing-state, classification, and lineage columns) |
| ScanPoint | 13 |
| Spectrum | 10 |
| InstrumentState | 21 |
| CCDConfiguration | 13 |
| ScannerCalibration | 8 |
| ContextImage | 28 (incl. VICAR metadata block) |
| RegionOfInterest | 7 |
| FittedPeak | 19 |
| Spectrogram | 18 |

**Map Mode display cache (1 table):** `MapDisplayCoordinate` — 5 fields.

**Application state (4 tables):** `User` (5), `UserPreference` (5), `ClassificationProfile` (6), `MapFitCache` (12).

**Provenance:** all Loupe fields preserved (68 in loupe.csv + image attributes + per-region spectra + spatial); all PDS4 fields supported via explicit columns on `ContextImage`/`Scan` plus the `pds4_metadata` JSON catch-all.
