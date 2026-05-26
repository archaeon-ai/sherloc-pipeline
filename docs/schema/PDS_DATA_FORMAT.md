# PDS4 Archive Data Format Specification

This document describes the data formats and structures used by the Planetary Data System (PDS4) archive for SHERLOC Raman and fluorescence spectroscopy data from NASA's Mars 2020 Perseverance rover mission.

**Version:** 2.0
**Date:** 2026-01-26
**Source:** [PDS Geosciences Node - Mars 2020 SHERLOC Archive](https://pds-geosciences.wustl.edu/missions/mars2020/sherloc.htm)
**Research basis:** Ralph loop research phases 1-5 (`.ralph/research/`)

---

## Table of Contents

1. [Overview](#overview)
2. [Archive Structure](#archive-structure)
3. [Processing Levels](#processing-levels)
4. [File Naming Conventions](#file-naming-conventions)
5. [PDS4 Label Structure](#pds4-label-structure)
6. [Product Types](#product-types)
7. [Data Collections](#data-collections)
8. [Field Definitions](#field-definitions)
9. [Mapping to Loupe Format](#mapping-to-loupe-format)
10. [Sol 921 Validation Examples](#sol-921-validation-examples)
11. [References](#references)

---

## Overview

The PDS4 SHERLOC archive is the official NASA data repository for spectroscopy data from the SHERLOC (Scanning Habitable Environments with Raman and Luminescence for Organics and Chemicals) instrument on the Mars 2020 Perseverance rover.

### Archive Identification

| Attribute | Value |
|-----------|-------|
| **Logical Identifier** | `urn:nasa:pds:mars2020_sherloc` |
| **Information Model** | PDS4_PDS_1G00 (v1.16.0.0) |
| **Bundle Type** | Archive |
| **DOI** | 10.17189/1522643 |
| **Primary Author** | Beegle, L. |
| **Data Coverage** | Sol 4 onwards (February 2021 - present) |
| **Processing Software** | iSDS SHERLOC PGE v1.6.8 |

### Key Characteristics

- **Laser Wavelength:** 248.5794 nm (deep UV)
- **Spectrometer:** Deep UV Resonance Raman and Fluorescence
- **CCD Channels:** 2148 per region (full detector)
- **Spectral Regions:** R1 (Raman, 250-282 nm), R2 (Fluorescence, 282-338 nm), R3 (Fluorescence, 338-357 nm)
- **Science Domain:** Geosciences

---

## Archive Structure

### Bundle Organization

```
urn-nasa-pds-mars2020_sherloc/
|-- bundle_sherloc.xml           # Bundle metadata (7.6 KB)
|-- readme.txt                   # Documentation (3.0 KB)
|-- urn-nasa-pds-mars2020_sherloc.md5  # Checksums (7.4 MB)
|
|-- data_raw/                    # EDR products (152 sol dirs)
|   |-- collection_data_raw.xml
|   |-- collection_data_raw_inventory.csv  (1.3 MB)
|   |-- sol_XXXXX/               # ~53 CSV+XML pairs per sol (typical)
|
|-- data_intermediate/           # Partially processed (RAC, RRS variants)
|   |-- collection_data_intermediate.xml
|   |-- collection_data_intermediate_inventory.csv
|   |-- sol_XXXXX/               # ~25 CSV+XML pairs per sol (typical)
|
|-- data_processed/              # RDR products (105 sol dirs)
|   |-- collection_data_processed.xml
|   |-- collection_data_processed_inventory.csv  (1.2 MB)
|   |-- sol_XXXXX/               # ~40 CSV+XML pairs per sol (typical)
|
|-- data_aci/                    # ACI context images (inventory only; images at CIS Node)
|-- data_watson/                 # WATSON images (~193K products)
|-- document/                    # SIS documents, user guide, release notes
```

**Sol range:** sol_00004 through sol_01613 (data_raw: 152 sols, data_processed: 105 sols)

### Image Archive Locations

ACI and WATSON images are hosted at the **CIS (Cartography and Imaging Sciences) Node** at JPL, not the Geosciences Node:

| Image Type | Bundle | URL |
|-----------|--------|-----|
| ACI context | `mars2020_imgops` / `data_aci_imgops` | `planetarydata.jpl.nasa.gov/img/data/mars2020/mars2020_imgops/data_aci_imgops/` |
| WATSON | `mars2020_sherloc` / `data_watson` + `mars2020_imgops` / `data_watson_imgops` | Geosciences + CIS Node |

The Geosciences Node hosts ACI/WATSON inventory CSVs but the actual image files are at the CIS Node.

---

## Processing Levels

### Processing Pipeline

```
data_raw (EDR)  -->  data_intermediate  -->  data_processed (RDR)
  EXH, ECH              RAC, RCA/RCB          RCS (calibration only)
  ECA, ECB              RRA/RRB               RRS (Mars surface)
  EPA, ESP              RRS (intermediate)     RMO, RCC, RLI, RLS, RM1-6
  ERA, ERB
  ERP (photodiode)
```

### CODMAC Levels

| PDS4 Processing Level | CODMAC | Collection | Description |
|-----------------------|--------|------------|-------------|
| Raw | 1-2 | data_raw | Unprocessed instrument telemetry |
| Partially Processed | 2-3 | data_intermediate | Initial calibrations applied |
| Calibrated | 3-4 | data_processed | Radiometrically calibrated |
| Derived | 4-5 | data_processed | Higher-order products |

### EDR vs RDR

| Type | Full Name | Level | Description |
|------|-----------|-------|-------------|
| **EDR** | Experiment Data Record | Raw/Partially Processed | Initial data with minimal processing |
| **RDR** | Reduced Data Record | Calibrated/Derived | Science-ready processed data |

---

## File Naming Conventions

### Processed/RDR Products (lowercase)

```
ss__SSSS_CCCCCCCCCC_NNNxxx__DDDDDDDsrlcQQQQQ_WNNNsssssVV.csv
     |        |       |  |      |       |      |    |    |
     sol     SCLK   obs product site/ sequence grid suffix version
                     id  type   drive  code    pos
```

| Component | Position | Description | Example |
|-----------|----------|-------------|---------|
| `ss` | 1-2 | SHERLOC Spectrometer prefix | `ss` |
| Sol | 5-8 | Sol number (4 digits, zero-padded) | `0921` |
| SCLK | 10-19 | Spacecraft clock (10 digits) | `0748731413` |
| Observation ID | 21-23 | 3-digit sequence identifier | `045` |
| Product type | 24-26 | Product type code | `rrs`, `rmo`, `rcc` |
| Site/drive | 29-35 | 7-digit Rover Motion Counter | `0450000` |
| Sequence code | 36-44 | SRLC sequence code | `srlc11374` |
| Grid positions | After `w` or `b` | Grid position count (NOT spectrum count) | `w104`, `b108` |
| Processing suffix | Before VV | Processing flags | `cgnj`, `zpzj` |
| Version | Last 2 digits | Processing version | `01`, `02` |

**CAUTION:** The `wNNN` or `bNNN` suffix encodes grid positions, NOT spectrum count. A `w108` file may contain 1296 spectra (108 columns x 12 rows, fine scan mode). Always count CSV data rows or check the XML `records` element for actual spectrum count.

#### Concrete Examples (Sol 921)

| Filename | Product | Obs | Points | Spectra |
|----------|---------|-----|--------|---------|
| `ss__0921_0748731413_045rrs__0450000srlc11374_104cgnj01.csv` | RRS | Detail scan | w104 | 100 |
| `ss__0921_0748735042_800rrs__0450000srlc11420_108cgnj01.csv` | RRS | Fine scan | w108 | 1296 |
| `ss__0921_0748735903_665rrs__0450000srlc11420_108zpzj01.csv` | RRS | Summary | b108zpz | 2 |
| `ss__0921_0748731011_645rcs__0450000srlc10000_0cgnj01.csv` | RCS | Calibration | w0 | 1 |
| `ss__0921_0748731413_045rmo__0450000srlc11374_104cgnj01.csv` | RMO | Positions | — | 100 |
| `ss__0921_0748731011_645rcc__0450000srlc10000_0cgnj02.csv` | RCC | Cal. fits | — | 47 |

### Raw/EDR Products (UPPERCASE)

```
SS__<SOL>_<SCLK>_<INST_CODE>__<PRODUCT_ID>_<VER>_____J<VV>.CSV
```

| Component | Description | Example |
|-----------|-------------|---------|
| `INST_CODE` | Instrument code (6 chars) | `028ECH`, `056EPA` |
| `PRODUCT_ID` | Product identifier including SRLC code | `0010052SRLC10002` |
| `J<VV>` | Processing version | `J05`, `J10` |

### Processing Suffix Decode

| Suffix | Meaning | Notes |
|--------|---------|-------|
| `cgnj` | Calibrated, gain-corrected, normalized | Standard processed (most products) |
| `cgzj` | Calibrated, gain-corrected, zeroed | Variant processing |
| `zpzj` | Summary/derived processing | Produces "PROCESS DATA SPECTRA" tables; only 2 records |
| `cg_j` | Calibrated, gain-corrected (no normalization) | Intermediate product |

### Sequence Code Classification

| SRLC Range | Type | Description |
|-----------|------|-------------|
| `SRLC0xxxx` | WATSON | WATSON camera sequences |
| `SRLC10000` | Calibration | Calibration target measurement |
| `SRLC1xxxx` (1xxxx > 10000) | Mars surface | Detail/survey/fine spectral scans |
| `SRLC16000` | Internal cal | AlGaN internal calibration |

---

## PDS4 Label Structure

Each data file has an accompanying XML label following PDS4 standards. Below are the fields actually observed in Sol 921 product labels.

### Core Label Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">

  <!-- Identification -->
  <Identification_Area>
    <logical_identifier>urn:nasa:pds:mars2020_sherloc:data_processed:ss__0921_...</logical_identifier>
    <version_id>1.0</version_id>
    <product_class>Product_Observational</product_class>
    <information_model_version>1.16.0.0</information_model_version>
  </Identification_Area>

  <!-- Observation Context -->
  <Observation_Area>
    <Time_Coordinates>
      <start_date_time>2023-09-23T09:09:06.711Z</start_date_time>
      <stop_date_time>2023-09-23T09:30:12.935Z</stop_date_time>
      <local_mean_solar_time>20:31:00.656</local_mean_solar_time>
      <local_true_solar_time>20:55:31</local_true_solar_time>
      <solar_longitude unit="deg">122.873</solar_longitude>
    </Time_Coordinates>

    <Primary_Result_Summary>
      <purpose>Science</purpose>
      <processing_level>Calibrated</processing_level>
      <Science_Facets>
        <wavelength_range>Ultraviolet</wavelength_range>
        <domain>Surface</domain>
        <discipline_name>Geosciences</discipline_name>
      </Science_Facets>
    </Primary_Result_Summary>

    <Investigation_Area>
      <name>Mars2020</name>
      <type>Mission</type>
    </Investigation_Area>

    <Observing_System>
      <Observing_System_Component>
        <name>Mars 2020</name>
        <type>Host</type>
      </Observing_System_Component>
      <Observing_System_Component>
        <name>Sherloc Deep Uv Resonance Raman And Flourescence Spectrometer</name>
        <type>Instrument</type>
      </Observing_System_Component>
    </Observing_System>

    <Target_Identification>
      <name>Mars</name>
      <type>Planet</type>
    </Target_Identification>
  </Observation_Area>

  <!-- Mission-Specific Parameters -->
  <Mission_Area>
    <Mars2020_Parameters>
      <sol_number>921</sol_number>
      <spacecraft_clock_start>748731411.515</spacecraft_clock_start>
      <spacecraft_clock_stop>748733482.117</spacecraft_clock_stop>
      <spacecraft_clock_partition>1</spacecraft_clock_partition>
      <mission_phase_name>Surface Mission</mission_phase_name>
      <active_flight_computer>A</active_flight_computer>
      <sequence_id>srlc11374</sequence_id>
      <product_completion_status>COMPLETE_CHECKSUM_PASS</product_completion_status>
    </Mars2020_Parameters>
  </Mission_Area>

  <!-- Geometry (Telemetry) -->
  <Geometry>
    <!-- RSM (Remote Sensing Mast) angles -->
    <RSM_AZIMUTH_FINAL-RESOLVER>3.15163 rad</RSM_AZIMUTH_FINAL-RESOLVER>
    <RSM_ELEVATION_FINAL-RESOLVER>0.748024 rad</RSM_ELEVATION_FINAL-RESOLVER>
    <!-- Rover Motion Counter -->
    <SITE_INDEX>45</SITE_INDEX>
    <DRIVE_INDEX>0</DRIVE_INDEX>
    <POSE_INDEX>146</POSE_INDEX>
    <!-- Rover frame quaternion -->
    <ROVER_MECH_FRAME>(0.72, -0.001, 0.002, -0.69)</ROVER_MECH_FRAME>
  </Geometry>

  <!-- Processing Provenance -->
  <Processing_Information>
    <Software_Program_Name>iSDS SHERLOC PGE SOFTWARE</Software_Program_Name>
    <Software_Version_Id>v1.6.8</Software_Version_Id>
    <creation_date_time>2024-02-13</creation_date_time>
  </Processing_Information>

  <!-- Source Product Chain -->
  <Source_Product_Internal>
    <lidvid_reference>urn:nasa:pds:mars2020_sherloc:data_intermediate:...</lidvid_reference>
    <reference_type>data_to_partially_processed_source_product</reference_type>
  </Source_Product_Internal>
  <Source_Product_Internal>
    <lidvid_reference>urn:nasa:pds:mars2020_sherloc:data_raw:...</lidvid_reference>
    <reference_type>data_to_raw_source_product</reference_type>
  </Source_Product_Internal>

  <!-- Data File Area -->
  <File_Area_Observational>
    <File>
      <file_name>data_file.csv</file_name>
    </File>
    <!-- Table definitions with byte offsets -->
    <Table_Delimited>
      <local_identifier>laser-normalized-spectra-region-1</local_identifier>
      <records>100</records>
      <!-- ... column definitions ... -->
    </Table_Delimited>
  </File_Area_Observational>

</Product_Observational>
```

### Time Coordinates

| Element | Format | Example | Description |
|---------|--------|---------|-------------|
| `start_date_time` | ISO-8601 | `2023-09-23T09:09:06.711Z` | UTC observation start |
| `stop_date_time` | ISO-8601 | `2023-09-23T09:30:12.935Z` | UTC observation end |
| `local_mean_solar_time` | `HH:MM:SS.sss` | `20:31:00.656` | Mars local mean solar time |
| `local_true_solar_time` | `HH:MM:SS` | `20:55:31` | Mars local true solar time |
| `solar_longitude` | degrees | `122.873` | Ls (seasonal indicator) |

### Mission Parameters

| Element | Example | Description |
|---------|---------|-------------|
| `sol_number` | `921` | Martian sol (day) number |
| `spacecraft_clock_start` | `748731411.515` | SCLK with fractional seconds |
| `spacecraft_clock_stop` | `748733482.117` | SCLK end (fills DB NULL) |
| `mission_phase_name` | `Surface Mission` | Mission phase |
| `active_flight_computer` | `A` | A or B computer |
| `sequence_id` | `srlc11374` | Command sequence code |
| `product_completion_status` | `COMPLETE_CHECKSUM_PASS` | Quality flag |

### Target Identification

**Critical:** PDS4 `Target_Identification` is always `name=Mars`, `type=Planet`. Geological target names (e.g., "Amherst Point") are **NOT** present in any PDS4 field or product. Target names originate exclusively from Loupe's `human_readable_workspace` field.

---

## Product Types

### Raw Data Products (data_raw)

#### EDR Product Types

| Code | Name | Description | Typical CSV Size |
|------|------|-------------|-----------------|
| EXH | EDR Examination Header | Command/engineering header | ~3 KB |
| ECH | EDR Calibration Header | Calibration header | ~12 KB |
| ECA | EDR Calibration A | Calibration data CCD-A | ~139 KB |
| ECB | EDR Calibration B | Calibration data CCD-B | ~139 KB |
| EPA | EDR Primary A | Primary measurement data (14 tables, 78+ fields) | ~17-166 KB |
| ESP | EDR Secondary Processed | Engineering/secondary data | ~6-48 KB |
| ERA | EDR Raw A | Raw spectral data CCD-A | ~3 MB (104 pts) |
| ERB | EDR Raw B | Raw spectral data CCD-B | ~3 MB (104 pts) |
| ERP | EDR Raw Photodiode | Per-shot raw photodiode data | Variable |

#### Key EPA Tables (Raw EDR — Not in Processed Products)

The EPA product contains engineering data organized in 14 tables:

| Table | Records | Key Fields |
|-------|---------|------------|
| SRLCSPECDEFAULT | 1 | Default scan parameters |
| SRLCSPECARGS1 | 1 | Scan arguments |
| SRLCSPECPARAM3 | 1 | Scanner origin (ORIGIN_AZ, ORIGIN_EL), step sizes |
| SRLCSPECPARAM4 | 1 | CCD timing, laser parameters |
| COLLECT SOH | 3 | 12 temperatures, power switches, heater status |
| SE COLLECT SOH | 3 | Science electronics SOH (CCD temp, voltages) |
| CONFIG LASER TIMING | 1 | INTEGRATION_TIME, LASER_REP_RATE, SHOTS_PER_SPECTRA |
| CONFIG CCD REGIONS | 1 | SKIP/SUM rows, GAIN, MODE |
| CONFIG CCD HORZ TIMING | 1 | Readout clock timing |
| CONFIG CCD VERT TIMING | 1 | Vertical timing |
| LASER PHOTODIODE DATA | N | Per-shot photodiode (full, not averaged) |

**Important:** Scanner origin (ORIGIN_AZ, ORIGIN_EL), step sizes, shots-per-spectra, all temperature/voltage telemetry, and CCD configuration are **only available in raw EDR**, not in processed products.

### Intermediate Data Products (data_intermediate)

| Code | Name | Description | Typical CSV Size |
|------|------|-------------|-----------------|
| RAC | Reduced Accumulated Counts | Partially processed accumulated counts | ~440 bytes |
| RCA/RCB | Reduced Calibration A/B | Intermediate calibration spectra | ~153 KB |
| RRA/RRB | Reduced Raw A/B | Intermediate raw spectra | ~5.6 MB |
| RRS | Reduced Raw Spectra (intermediate) | Intermediate spectra with partial calibration | ~5-6 MB |

### Processed Data Products (data_processed)

#### RRS — Reduced Raw Spectra (Mars Surface — PRIMARY)

Full spectral data for Mars surface observations. Contains laser-normalized spectra.

**RRS and RCS have identical CSV data format.** The distinction is only provenance:
- **RRS**: Generated for Mars surface targets (SRLC1xxxx sequences, except SRLC10000 and SRLC16000)
- **RCS**: Generated for calibration targets only (SRLC10000 sequence)

Both contain laser-normalized spectra organized in 4 concatenated CSV tables.

| Table | Local ID | Records | Columns | Description |
|-------|----------|---------|---------|-------------|
| 1 | `laser-normalized-spectra-region-1` | N_spectra | 2148 | Laser-normalized R1 spectra |
| 2 | `laser-normalized-spectra-region-2` | N_spectra | 2148 | Laser-normalized R2 spectra |
| 3 | `laser-normalized-spectra-region-3` | N_spectra | 2148 | Laser-normalized R3 spectra |
| 4 | `wavelength-nm` | 1 | 2148 | Wavelength values per CCD channel (nm) |

Tables are concatenated in a single CSV file, with byte offsets defined in the XML label.

**Spectrum count varies by observation mode:**

| Mode | Filename Suffix | Grid Positions | Actual Spectra | Explanation |
|------|----------------|----------------|----------------|-------------|
| Detail scan | `w104` | 104 | 100 | ~1 shot/position, some dropped |
| Fine scan | `w108` | 108 | 1296 | 108 columns x 12 rows |
| Summary | `b108zpz` | 108 | 2 | Derived summary product |
| Calibration | `w0` | 1 | 1 | Single calibration point |

**zpz summary products:** Uses table name "PROCESS DATA SPECTRA REGION N" instead of "LASER-NORMALIZED SPECTRA REGION N". Only 2 records. These are post-processed summaries (likely mean + uncertainty), not raw observations.

#### RCS — Reduced Calibrated Spectra (Calibration — same format as RRS)

Identical 4-table CSV format as RRS. Only generated for calibration observations (SRLC10000).

| Characteristic | Value |
|----------------|-------|
| Format | CSV (comma-delimited), same as RRS |
| Typical Size | ~194 KB |
| Records | Typically 1 (single calibration point) |

#### RMO — Reduced Measurement Overview

Contains laser shot positions and integrated spectral intensities.

**Table 1: Laser Shot Positions** (N_positions records)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `Image_name` | ASCII_String | — | Associated ACI image filename (definitive link) |
| `Position_index` | ASCII_Integer | — | Laser shot number (0-based) |
| `x` | ASCII_Real | pixels | Sample coordinate on ACI image |
| `y` | ASCII_Real | pixels | Line coordinate on ACI image |

**Table 2: Wavelength Regions** (6 records)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `Column_index` | ASCII_Integer | — | Wavelength region index (0-5) |
| `Wavelength_start` | ASCII_Real | nm | Region start boundary |
| `Wavelength_stop` | ASCII_Real | nm | Region end boundary |

**Table 3: Spectral Intensity** (N_intensities records)

| Column | Type | Description |
|--------|------|-------------|
| `Position_index` | ASCII_Integer | Laser shot number |
| `Spectral_intensity_0` through `_5` | ASCII_Real | Integrated intensity per band |

**Position/Intensity ratio:** For fine-mode scans, RMO has 2x more position rows than intensity rows. Sol 921 survey (obs 800): 2592 positions for 1296 intensities. Each position appears twice — once per ACI reference image (pre-scan and post-scan). The x/y coordinates are identical across the two references.

**ACI association:** The `Image_name` field is the **definitive link** between spectral observations and ACI context images. Detail scans reference 1 ACI image (~100s before scan). Survey scans reference 2 ACI images (pre-scan + post-scan).

#### RCC — Reduced Calibrated Compact (Calibration Drift)

Multi-sol calibration fit parameters tracking wavelength drift.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `SOL` | ASCII_String | `98` | Sol number |
| `SCLK` | ASCII_String | `0675636651_555` | Spacecraft clock with fractional |
| `laser_reflection_peak_location` | ASCII_Real | `252.937` | Laser peak wavelength (nm) |
| `laser_reflection_FWHM` | ASCII_Real | `0.178` | Laser peak width (nm) |
| `AlGaN_275_peak_location` | ASCII_Real | `276.792` | AlGaN calibration peak (nm) |
| `AlGaN_275_FWHM` | ASCII_Real | `9.368` | AlGaN peak width (nm) |

Typically ~47 records spanning multiple sols. Provides wavelength calibration drift history. Sol 921 RCC shows 1.1 nm total AlGaN drift over 823 sols (rate: -0.32 nm per 1000 sols).

#### RLI — Reduced Laser Intensity (Photodiode)

Average photodiode intensity per laser shot position.

| Column | Type | Description |
|--------|------|-------------|
| `avg_photodiode` | ASCII_Real | Average photodiode ADC counts per shot |

One row per laser shot position (e.g., 100 rows for a 100-point detail scan). Values typically range 82-94 ADC counts. This is the processed equivalent of the raw EDR `LASER PHOTODIODE DATA` table.

**Note:** This product was previously documented as "Reduced Line Information (wavelength list)". The actual content is photodiode intensity mapping, as confirmed by parsing Sol 921 data.

#### RLS — Reduced Laser Shot (Cross-Reference)

Shot-level cross-reference between spectral data, positions, and ACI images.

| Column | Type | Description |
|--------|------|-------------|
| `number` | ASCII_Integer | Shot sequence number |
| `spec_name` | ASCII_String | Associated spectral product filename |
| `image_name` | ASCII_String | Associated ACI image filename |
| `samp` | ASCII_Real | Sample (x) coordinate on ACI image |
| `line` | ASCII_Real | Line (y) coordinate on ACI image |

One row per laser shot. RLS `samp`/`line` exactly match RMO `x`/`y` (both are ACI pixel coordinates). The `spec_name` field provides a direct shot-to-spectrum-to-image triple cross-reference.

#### RM1-RM6 — Reduced Band Intensity

Single-column CSVs with integrated spectral intensity per wavelength band (6 bands defined by RMO WAVELENGTH_REGIONS). One value per laser shot position.

---

## Data Collections

### Collection Inventory

Each collection includes:
- `collection_data_*.xml` — Collection metadata
- `collection_data_*_inventory.csv` — Product inventory

### Inventory Format

```csv
P,urn:nasa:pds:mars2020_sherloc:data_processed:ss__0921_0748731413_045rrs__0450000srlc11374_104cgnj01
P,urn:nasa:pds:mars2020_sherloc:data_processed:ss__0921_0748731413_045rmo__0450000srlc11374_104cgnj01
```

| Field | Description |
|-------|-------------|
| Member Status | `P` (Primary) or `S` (Secondary) |
| LIDVID_LID | Product logical identifier |

### Collection Statistics (as of Release 14)

| Collection | Records | Sol Range | Description |
|------------|---------|-----------|-------------|
| data_raw | 12,965 | sol_00004 - sol_01613 | Raw EDR products |
| data_intermediate | 7,875 | — | Intermediate products |
| data_processed | 10,781 | — | Final RDR products |
| data_watson | ~193,158 | — | WATSON images |

---

## Field Definitions

### Geometry and Articulation (from XML Labels)

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| RSM_AZIMUTH_FINAL-RESOLVER | Float | rad | Remote Sensing Mast azimuth angle |
| RSM_ELEVATION_FINAL-RESOLVER | Float | rad | RSM elevation angle |
| RSM_AZIMUTH_FINAL-ENCODER | Float | rad | RSM azimuth (encoder) |
| RSM_ELEVATION_FINAL-ENCODER | Float | rad | RSM elevation (encoder) |
| SITE_INDEX | Integer | — | Rover Motion Counter: site |
| DRIVE_INDEX | Integer | — | Rover Motion Counter: drive |
| POSE_INDEX | Integer | — | Rover Motion Counter: pose |
| ROVER_MECH_FRAME | Quaternion | — | Rover orientation (x, y, z, w) |

### Processing Metadata (from XML Labels)

| Field | Example | Description |
|-------|---------|-------------|
| `Software_Program_Name` | `iSDS SHERLOC PGE SOFTWARE` | Processing software |
| `Software_Version_Id` | `v1.6.8` | Software version |
| `Processing_Level` | `Calibrated` / `Derived` | CODMAC level |
| `product_completion_status` | `COMPLETE_CHECKSUM_PASS` | Processing status |
| `creation_date_time` | `2024-02-13` | Product creation date |

### Source Product References

Processed products reference source EDR and intermediate products:

```xml
<Source_Product_Internal>
  <lidvid_reference>urn:nasa:pds:mars2020_sherloc:data_intermediate:...</lidvid_reference>
  <reference_type>data_to_partially_processed_source_product</reference_type>
</Source_Product_Internal>
<Source_Product_Internal>
  <lidvid_reference>urn:nasa:pds:mars2020_sherloc:data_raw:...</lidvid_reference>
  <reference_type>data_to_raw_source_product</reference_type>
</Source_Product_Internal>
```

---

## Mapping to Loupe Format

### Conceptual Equivalence

| PDS4 Concept | Loupe Concept | Notes |
|--------------|---------------|-------|
| Sol directory | Sol directory | Direct |
| Product set (by SCLK) | Workspace | Loupe workspace = one SCLK observation |
| RRS/RCS table | darkSubSpectra.csv (laser-normalized) | Same processing level |
| RMO Position table | spatial.csv | **Different coordinate frame** |
| RMO Intensity table | Derived from spectra | — |
| RLI photodiode | photodiode.csv | PDS has mean only, Loupe has mean + std |
| XML label metadata | loupe.csv + SOFF XML | — |

### Key Differences (Research-Validated)

| Aspect | PDS4 Archive | Loupe Format |
|--------|--------------|--------------|
| **Spectral data type** | Laser-normalized only | Active, dark, dark-subtracted (3 types) |
| **Channels per region** | 2148 (full CCD) | 465-508 (wavelength-filtered subset) |
| **Spatial coordinates** | ACI pixel coords (e.g., 757-846) | Scanner workspace coords (e.g., -0.4 to 0.5) |
| **Coordinate frame** | ACI image corner origin | Scanner center origin |
| **Target name** | "Mars" (planet-level only) | Geological name (e.g., "Amherst Point") |
| **SCLK precision** | Fractional (748731411.515) | Integer-truncated (748731411) |
| **SCLK offset** | 1-2s higher than Loupe | — |
| **Scanner DN values** | NOT in any product | azimuth_dn, elevation_dn in spatial.csv |
| **Scanner errors** | NOT in any product | azimuth_error, elevation_error |
| **Photodiode std** | NOT in any product | photodiode_std in spatial.csv |
| **Instrument state** | Raw EDR only (not processed) | loupe.csv (55+ fields) |
| **ROIs** | NOT in PDS | roi.csv (analyst-defined) |
| **File organization** | One file per product type | All data in working directory |
| **Processing state** | Separate collections | `specProcessingApplied` field |
| **Naming** | Product-type codes | Human-readable names |

### Spatial Coordinate Mapping

**PDS and Loupe use completely different spatial reference frames:**

| Property | PDS (RMO/RLS) | Loupe (scan_points) |
|----------|---------------|---------------------|
| Coordinate type | ACI image pixels | Scanner workspace units |
| x range (detail) | 757-846 | -0.4 to 0.5 |
| y range (detail) | 613-702 | -0.4 to 0.5 |
| x range (survey) | varies | -2.5 to 2.5 |
| Origin | ACI image corner (0,0) | Workspace center (~0,0) |
| Units | pixels (1024x1024 ACI) | Scanner-relative (arbitrary) |
| Resolution | ~10.1 um/pixel (ACI) | Proportional to scan area |

**Note:** Loupe scanner coordinates are NOT normalized to a fixed range. Detail scans use approximately +/-0.5, but survey scans extend to +/-2.5. The range scales with the physical scan area.

### Spectral Region Mapping

> **Canonical Reference:** See [`SPECTRAL_REGIONS.md`](SPECTRAL_REGIONS.md) for the definitive specification of R1, R2, R3, and R123 regions, including channel ranges, wavelength boundaries, and the R123 stitching algorithm.

| PDS4 Product Region | SHERLOC Region | Wavelength Range | Channels (Usable) |
|---------------------|----------------|------------------|--------------------|
| Region 1 (Raman) | R1 | 250-282 nm | 52-574 (523 ch) |
| Region 2 (Fluorescence) | R2 | 282-337.8 nm | — |
| Region 3 (Fluorescence) | R3 | 337.8-357.4 nm | — |
| Full spectrum | R123 | 250-357.4 nm (stitched) | 2148 total |

### Spectral Data Structure Comparison

**PDS4 RRS/RCS:**
- 4 tables concatenated in a single CSV: wavelength (1 row) + R1/R2/R3 spectra (N rows each)
- Byte offsets defined in XML label separate the tables
- Each region has 2148 columns (full CCD)
- Column headers: `R1_Channel`, `R2_Channel`, `R3_Channel`, `Channel`
- All spectra are laser-normalized

**Loupe:**
- Separate CSVs per region per type (activeSpectra1.csv, darkSubSpectra1.csv, etc.)
- Rows = spectra, columns = channels (wavelength-filtered subset only)
- Wavelength derived from calibration polynomial (see [SPECTRAL_REGIONS.md](SPECTRAL_REGIONS.md) Section 3)

### Wavelength Calibration Comparison (Sol 921 Validated)

PDS wavelength arrays were compared against Loupe V5.1.5a polynomial coefficients:

| Region | Channels | Max Difference | Notes |
|--------|----------|---------------|-------|
| Raman (0-499) | 500 | < 0.001 nm | Effectively identical |
| Fluorescence (500-2147) | 1648 | ~0.001 nm | Very similar |
| **Channel 500 (boundary)** | **1** | **0.393 nm** | Known discontinuity |

At the Raman/Fluorescence boundary (channel 500):
- PDS: 277.348 nm (continuous calibration bridging boundary)
- Loupe: 277.741 nm (hard polynomial switch at `cutoff_channel=500`)
- This is the **documented wavelength calibration discontinuity** from PDS Release Notes

**All observations within a sol share the same wavelength array.** PDS wavelength calibration is fixed per processing version, not per-observation.

### Data Fields Not in Current DB Schema (PDS-Only)

| PDS Field | Product | Potential Value |
|-----------|---------|----------------|
| `start_date_time` / `stop_date_time` (UTC) | XML label | Earth-readable timestamps |
| `local_true_solar_time` | XML label | True solar time |
| `active_flight_computer` | XML label | A/B computer tracking |
| `logical_identifier` (LIDVID) | XML label | PDS product traceability |
| `version_id` | XML label | Product version |
| `sequence_id` | XML label | Command sequence code |
| `software_version` | XML label | Processing pipeline version |
| RSM AZIMUTH/ELEVATION | XML geometry | Arm pointing angles (rad) |
| SITE/DRIVE/POSE indices | XML geometry | Rover Motion Counter |
| Rover frame quaternion | XML geometry | Full rover pose |
| WAVELENGTH_REGIONS boundaries | RMO CSV | 6 spectral band definitions |
| Spectral_intensity_0-5 | RMO CSV | Per-band integrated intensity |
| AlGaN calibration fits | RCC CSV | Wavelength drift tracking |
| avg_photodiode per shot | RLI CSV | Laser intensity normalization |
| spec_name cross-reference | RLS CSV | Shot-to-spectrum mapping |

---

## Sol 921 Validation Examples

Sol 921 (Amherst Point target area) was downloaded and cross-referenced against Loupe-ingested DB records during research Phase 3.

### Observation Structure

Sol 921 contains **6 observations**:

| # | SCLK | Obs ID | Type | Points (filename) | Spectra (actual) | Sequence |
|---|------|--------|------|-------------------|------------------|----------|
| 1 | 0748731011 | 645 | Calibration | w0 | 1 | SRLC10000 |
| 2 | 0748731413 | 045 | Detail scan | w104 | 100 | SRLC11374 |
| 3 | 0748732975 | 435 | Detail scan | w104 | 100 | SRLC11373 |
| 4 | 0748735042 | 800 | Fine scan | w108 | 1296 | SRLC11420 |
| 5 | 0748735903 | 665 | Summary | b108zpz | 2 | SRLC11420 |
| 6 | 0748736149 | 380 | AlGaN internal | — | 1 | SRLC16000 |

**Key findings:**
- Mars surface scans produce **RRS** (not RCS). Only calibration observations get RCS.
- Observation 665 is a derived summary of observation 800 (same sequence code SRLC11420, different suffix `zpz`).
- Observation 380 (AlGaN internal) does not produce RMO (no spatial coordinates for internal calibration).

### Downloaded Products (134 files, 106 MB)

| Collection | Files Downloaded | Completeness | Size |
|-----------|-----------------|-------------|------|
| data_processed | 104 (complete) | 100% | 76 MB |
| data_raw | 20 (representative) | ~15% | 7.9 MB |
| data_intermediate | 10 (representative) | ~14% | 12 MB |
| ACI images | Inventory only | — | 10.4 MB (inventory) |

### Cross-Reference Results

All 5 Loupe scans matched to PDS observations via SCLK (1-2 second consistent offset):

| Loupe Scan | Loupe SCLK | PDS SCLK | Delta | Points Match |
|-----------|-----------|---------|-------|-------------|
| AlGaN_1 | 748731010 | 748731011 | +1s | 1 = 1 |
| detail_1 | 748731411 | 748731413 | +2s | 100 = 100 |
| detail_2 | 748732974 | 748732975 | +1s | 100 = 100 |
| survey_1296 | 748735041 | 748735042 | +1s | 1296 = 1296 |
| AlGaN_2 | 748736148 | 748736149 | +1s | 1 = 1 |

**Unmatched:** PDS observation 665 (zpz summary) has no Loupe counterpart.

### Product File Sizes

| Product | Obs 045 (detail, 100 pts) | Obs 800 (fine, 1296 pts) | Obs 665 (summary, 2 pts) |
|---------|--------------------------|--------------------------|---------------------------|
| RRS | 5.2 MB | 62 MB | 222 KB |
| RMO | 15 KB | 280 KB | 204 KB |
| RLI | ~56 B | — | — |
| RM1-6 | ~62 B each | ~62 B each | — |

### ACI Image References (from RMO)

| Spectral Obs | ACI Image_name | ACI SCLK | Timing |
|-------------|----------------|----------|--------|
| 645 (cal) | SC0_0921_0748731023_488ECM_N0450000SRLC10000_0000LUJ01.IMG | 748731023 | +12s before scan |
| 045 (detail) | SC3_0921_0748731308_359ECM_N0450000SRLC11374_0000LMJ01.IMG | 748731308 | ~105s before scan |
| 435 (detail) | SC3_0921_0748732871_148ECM_N0450000SRLC11373_0000LMJ01.IMG | 748732871 | ~104s before scan |
| 800 (survey) | Pre: SC3_0921_0748735026_996ECM + Post: SC3_0921_0748735954_949ECM | 735026 / 735954 | -16s / +912s |

ACI filename prefix classification: `SC0` = calibration context, `SC1` = autofocus, `SC3` = science context.

---

## References

### Primary Documentation

1. **PDS SHERLOC Archive Page:** https://pds-geosciences.wustl.edu/missions/mars2020/sherloc.htm
2. **Bundle SIS:** `sherloc_bundle_sis.pdf`
3. **EDR SIS:** `sherloc_edr_sis.pdf` (20.9 MB)
4. **User Guide:** `sherloc_user_guide.pdf` (2.5 MB)
5. **Release Notes:** `sherloc_release_notes.txt`

### Instrument References

6. **Bhartia, R. et al. (2021):** "Perseverance's Scanning Habitable Environments with Raman and Luminescence for Organics and Chemicals (SHERLOC) Investigation," Space Science Reviews, 217, 58. doi:10.1007/s11214-021-00812-z

### PDS4 Standards

8. **PDS4 Information Model:** v1.16.0.0 (1G00)
9. **PDS4 Data Dictionary:** https://pds.nasa.gov/datastandards/
10. **Mars Analyst's Notebook:** https://an.rsl.wustl.edu/

### Data Access

11. **Direct Archive:** https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_sherloc/
12. **Imaging Node (CIS):** https://pds-imaging.jpl.nasa.gov/volumes/mars2020.html
13. **CIS ACI Node:** https://planetarydata.jpl.nasa.gov/img/data/mars2020/mars2020_imgops/data_aci_imgops/

### Related Project Documents

14. **PDS Ingestion Spec:** [`docs/specs/PDS_INGESTION_SPEC.md`](../specs/PDS_INGESTION_SPEC.md)
15. **Spectral Regions:** [`docs/schema/SPECTRAL_REGIONS.md`](SPECTRAL_REGIONS.md)
16. **Unified Schema:** [`docs/schema/UNIFIED_SCHEMA.md`](UNIFIED_SCHEMA.md)

---

## Appendix A: Complete Product Type Codes

### Raw/EDR Instrument Codes

| Code | Description |
|------|-------------|
| `028ECH` | Engineering/Command Header |
| `027ECH` | Engineering variant |
| `029ECH` | Engineering variant |
| `060ECH` | Engineering variant |
| `056ECA` | Calibration data A |
| `056ECB` | Calibration data B |
| `055ECA` | Secondary data A |
| `055ECB` | Secondary data B |
| `056EPA` | Primary measurement A |
| `056ESP` | Secondary processed |
| `055EPA` | Secondary measurement A |
| `055ESP` | Secondary processed |

### Processed/RDR Product Codes

| Code | Full Name | Content | Ingestion Priority |
|------|-----------|---------|-------------------|
| **rrs** | Reduced Raw Spectra | Laser-normalized spectra (Mars surface) | **PRIMARY** |
| **rcs** | Reduced Calibrated Spectra | Laser-normalized spectra (calibration only) | **PRIMARY** |
| **rmo** | Reduced Measurement Overview | Position + band intensity | **PRIMARY** |
| **rli** | Reduced Laser Intensity | Photodiode intensity per shot | Secondary |
| **rcc** | Reduced Calibrated Compact | Calibration drift (AlGaN fits) | Secondary |
| **rls** | Reduced Laser Shot | Shot-to-spectrum-to-image cross-reference | Secondary |
| rm1-rm6 | Reduced Band Intensity 1-6 | Per-band integrated intensity | Deferred |
| rac | Reduced Accumulated Counts | Intermediate accumulated | Not needed |

---

## Appendix B: Processing Version Codes

| Code | Description |
|------|-------------|
| `J01` | Processing version 1 |
| `J02` | Processing version 2 |
| `J03` | Processing version 3 |
| `J05` | Processing version 5 |
| `J10` | Processing version 10 |

---

## Appendix C: Known Issues and Anomalies

### Wavelength Calibration Discontinuity

A 0.393 nm discontinuity exists at SCCD channel 500 (Raman/Fluorescence polynomial boundary). This is an artifact of the dual-polynomial calibration:
- Channels 0-499: Raman polynomial coefficients `[-7.85e-06, 6.524e-02, 2.467e+02]`
- Channels 500-2147: Fluorescence polynomial coefficients `[-5.657e-06, 6.336e-02, 2.475e+02]`

PDS wavelength arrays show a smooth bridge at this boundary; Loupe polynomial has a hard switch. The difference is only at channel 500.

### Wavelength Calibration Drift

RCC data shows measurable wavelength drift: AlGaN 275 nm peak drifted 1.1 nm over 823 sols (-0.32 nm per 1000 sols). This drift is monitored but NOT corrected in per-observation wavelength arrays. For precise peak position work, this corresponds to ~5-20 cm^-1 wavenumber shift at typical Raman positions.

### Dust Cover Malfunction (Sol 1024)

An instrumentation malfunction on Sol 1024 affected dust cover operations, halting rock measurements through Sol 1139.

### DO_AREA Measurement Type

Sol 1426 required format updates to support the DO_AREA measurement type, delaying inclusion until Release 14.

### Early Sol Restrictions

Sols 4 and 11 used cruise flight software, restricted to 1000-pulse measurements.

### RRS File Size Variation

RRS file sizes vary dramatically even for similar point counts: from 222 KB (665, 108 pts, zpz) to 62 MB (800, 108 pts, cgnj). Processing suffix codes control which calibration variants are included.

---

## Appendix D: Release Schedule

| Release | Date | Sol Range | Notes |
|---------|------|-----------|-------|
| 1 | 2021-06 | Initial | Launch |
| 3+ | — | — | Wavelength calibration discontinuity introduced |
| 10 | — | — | Dust cover anomaly documented |
| 14 | 2025-12-04 | 1500-1619 | Latest (as of doc creation). DO_AREA support added |

Releases occur approximately quarterly, with data typically available 2-3 months after acquisition.

---

## Appendix E: ACI Image Naming and Association

### ACI Filename Convention

```
SC{T}_{SOL}_{SCLK}_{NNN}{PROC}_{FLAGS}_{VV}.IMG
  |                        |
  Type (0=cal, 1=AF, 3=science)   Processing level
```

| Processing Code | Description | Recommended Use |
|----------------|-------------|----------------|
| edr | Raw CFA (Bayer pattern) | Not useful for context |
| ecm | CFA-interpolated (color mosaic) | **Primary for visual context** |
| fdr | Flat-field corrected | Quantitative |
| rad | Radiometric calibration | Scientific analysis |
| raf | Float-precision radiometric | Scientific analysis |
| ras | Subframe radiometric | If subframing used |
| rzs | Subframe float | If subframing used |

Each acquisition produces 2 variants (normal `n` + thumbnail `t`) x 7 processing levels = 14 products.

### Image_name to LIDVID Construction

```
RMO Image_name: SC3_0921_0748731308_359ECM_N0450000SRLC11374_0000LMJ01.IMG
                 --> lowercase, strip version suffix "01" and ".IMG"
Product ID:     sc3_0921_0748731308_359ecm_n0450000srlc11374_0000lmj
                 --> prefix with bundle/collection
Full LIDVID:    urn:nasa:pds:mars2020_imgops:data_aci_imgops:sc3_0921_0748731308_359ecm_n0450000srlc11374_0000lmj::1.0
```

### WATSON Filename Convention

```
SIF_{SOL}_{SCLK}_{NNN}{PROC}_{FLAGS}SRLC0{NNNN}_{PARAMS}.IMG
^^^                                      ^
WATSON (SHERLOC Imager Focusable)       SRLC0 = WATSON sequence
```

WATSON uses `SRLC0xxxx` (digit after SRLC is `0`), while spectroscopy uses `SRLC1xxxx`. Both share the same site/drive code (7-digit RMC), enabling cross-instrument association.

WATSON association is heuristic (no direct cross-reference in spectral labels); see [`docs/specs/PDS_INGESTION_SPEC.md`](../specs/PDS_INGESTION_SPEC.md) for strategy details.
