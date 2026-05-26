# Loupe Data Format Specification

This document describes the data formats and structures used by the Loupe visualization tool for SHERLOC Raman and fluorescence spectroscopy data on NASA's Mars 2020 Perseverance rover mission.

**Version:** 1.0
**Date:** 2026-01-24

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [File Types](#file-types)
4. [CSV Data Files](#csv-data-files)
5. [SOFF XML Manifest](#soff-xml-manifest)
6. [Image Files](#image-files)
7. [Session Files](#session-files)
8. [Key Classes and Models](#key-classes-and-models)
9. [Metadata Fields Reference](#metadata-fields-reference)
10. [Wavelength Calibration](#wavelength-calibration)
11. [Spatial Coordinate System](#spatial-coordinate-system)

---

## Overview

Loupe is a modular, interactive environment for visualizing, processing, and analyzing SHERLOC Raman and fluorescence spectroscopy data. It uses the SPAR (SOFF PDS4 Array Reader) library to read data files described by SOFF (Spectrometer Open File Format) manifest files.

### Key Characteristics

- **Laser Wavelength:** 248.5794 nm (deep UV)
- **CCD Channels:** 2148 per spectrum
- **CCD Regions:** 3 regions (R1, R2, R3) covering different wavelength ranges
- **Spectral Range:** Raman shifts from ~0 to ~4000+ cm^-1
- **Map Points:** Typically 100 points per map, but variable
- **Shots per Point:** Variable (10, 100, 300 common values)

---

## Directory Structure

### Sol-Level Organization

```
./data/loupe/
|-- sol_XXXX/                     # Sol number (Mars day)
|   |-- Sol_XXXX_*.lpe            # Session file linking workspaces
|   |-- <target_name>/            # Target/sample subdirectory
|   |   |-- <rawDP>_Loupe_working/
|   |   |   |-- <rawDP>_soff.xml  # SOFF manifest
|   |   |   |-- loupe.csv         # Workspace metadata
|   |   |   |-- roi.csv           # Region of interest definitions
|   |   |   |-- activeSpectra.csv # Active (laser-on) spectra
|   |   |   |-- darkSpectra.csv   # Dark spectra
|   |   |   |-- darkSubSpectra.csv# Dark-subtracted spectra
|   |   |   |-- photodiodeRaw.csv # Laser photodiode readings
|   |   |   |-- spatial.csv       # Scanner positions
|   |   |   |-- img/              # Context images
|   |   |   |   |-- *.PNG         # ACI/WATSON images
|   |   |   |   |-- *.CSV         # Image attributes
```

### File Naming Convention

**Raw Data Product Names:**
```
SrlcSpecSpecSohRaw_<SCLK>-<SEQ>-<VER>
```
- `Srlc`: SHERLOC instrument prefix
- `SpecSpecSohRaw`: Spectroscopy State-of-Health Raw
- `SCLK`: Spacecraft clock time (e.g., `0672194998`)
- `SEQ`: Sequence number (e.g., `62417`)
- `VER`: Version number (e.g., `1`)

**Working Directory:**
```
<rawDP>_Loupe_working/
```

---

## File Types

| Extension | Description | Format |
|-----------|-------------|--------|
| `.csv` | Tabular data (spectra, positions, metadata) | CSV with headers |
| `.xml` | SOFF manifest file | PDS4 XML |
| `.lpe` | Loupe session file | CSV |
| `.PNG` | Context images (ACI/WATSON) | PNG image |
| `.IMG` | Raw PDS image with VICAR header | Binary |
| `.LBL` | ODL label file for images | Text |
| `.dat` / `.emd` | Raw data product files | Binary |

---

## CSV Data Files

### loupe.csv - Workspace Metadata

Key-value pairs describing the workspace configuration and instrument state.

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `original_data_file` | string | Source raw data product name | `SrlcSpecSpecSohRaw_0672194998-62417-1` |
| `human_readable_workspace` | string | User-friendly workspace name | `Teflon` |
| `n_spectra` | integer | Number of spectral points in map | `100` |
| `n_channels` | integer | CCD channels per spectrum | `2148` |
| `laser_wavelength` | float | Laser wavelength (nm) | `248.6` |
| `shots_per_spec` | integer | Laser shots per spectrum | `10`, `100`, `300` |
| `az_scale` | float | Azimuth scale factor | `0.62815` |
| `el_scale` | float | Elevation scale factor | `0.42244` |
| `laser_x` | integer | Laser X position on ACI image | `809` |
| `laser_y` | integer | Laser Y position on ACI image | `664` |
| `rotation` | float | Rotation angle (degrees) | `20.67936` |
| `specProcessingApplied` | string | Processing status | `None`, `N`, `B`, `C`, `NB`, etc. |

#### Instrument State of Health (SOH) Fields

**COLLECT_SOH Group:**
| Field | Type | Description |
|-------|------|-------------|
| `CNDH_PCB_TEMP_STAT_REG` | string | PCB temperature | `37.200 C` |
| `CNDH_1_2_V_STAT_REG` | string | 1.2V rail status | `1.204 V` |
| `CNDH_5_V_DAC_STAT_REG` | string | 5V DAC status | `4.990 V` |
| `CNDH_3_3_V_STAT_REG` | string | 3.3V rail status | `3.341 V` |
| `CNDH_5_V_ADC_STAT_REG` | string | 5V ADC status | `5.000 V` |
| `CNDH_NEG_15_V_STAT_REG` | string | -15V rail status | `-15.414 V` |
| `CNDH_15_V_STAT_REG` | string | +15V rail status | `14.930 V` |
| `CNDH_1_5_V_STAT_REG` | string | 1.5V rail status | `1.494 V` |
| `laser_shot_counter` | integer | Total laser shots fired | `98970` |
| `laser_misfire_counter` | integer | Laser misfires | `23` |
| `arc_event_counter` | integer | Arc events detected | `100` |

**SE_COLLECT_SOH Group (Spectrometer Electronics):**
| Field | Type | Description |
|-------|------|-------------|
| `SE_CCD_ID_STAT_REG` | string | CCD ID status | `1.747 V` |
| `SE_CCD_TEMP_STAT_REG` | string | CCD temperature | `-4.279 C` |
| `SE_PCB_TEMP_STAT_REG` | string | SE PCB temperature | `5.609 C` |
| `SE_V_1_5_STAT_REG` | string | SE 1.5V status | `1.490 V` |
| `SE_LASER_PRT2_STAT_REG` | string | Laser PRT2 temp | `-6.146 C` |
| `SE_LASER_PRT1_STAT_REG` | string | Laser PRT1 temp | `-6.981 C` |
| `SE_LPS_PRT1_STAT_REG` | string | LPS PRT1 temp | `-0.771 C` |
| `SE_TPRB_HOUSING_PRT_STAT_REG` | string | Housing temp | `3.293 C` |
| `SE_LPS_PRT2_STAT_REG` | string | LPS PRT2 temp | `18.831 C` |
| `SE_SPARE1_PRT_STAT_REG` | string | Spare PRT temp | `158.879 C` |

**CONFIG_CCD_VERT_TIMING:**
| Field | Type | Description |
|-------|------|-------------|
| `CCD_VERT_COL1_LOW` | integer | CCD vertical column 1 low | `179` |
| `CCD_VERT_COL1_HIGH` | integer | CCD vertical column 1 high | `447` |
| `CCD_VERT_COL2_LOW` | integer | CCD vertical column 2 low | `179` |
| `CCD_VERT_COL2_HIGH` | integer | CCD vertical column 2 high | `N/A` |
| `CCD_VERT_COL3_LOW` | integer | CCD vertical column 3 low | `1` |
| `CCD_VERT_COL3_HIGH` | integer | CCD vertical column 3 high | `269` |

**CONFIG_CCD_HORZ_TIMING:**
| Field | Type | Description |
|-------|------|-------------|
| `CCD_HORZ_CLOCK_LIM` | integer | Horizontal clock limit | `74` |
| `CCD_HORZ_R1_CLOCK_HIGH` | integer | R1 clock high | `0` |
| `CCD_HORZ_R1_CLOCK_Low` | integer | R1 clock low | `66` |
| `CCD_HORZ_R2_CLOCK_HIGH` | integer | R2 clock high | `62` |
| `CCD_HORZ_R2_CLOCK_Low` | integer | R2 clock low | `72` |
| `CCD_HORZ_R3_CLOCK_HIGH` | integer | R3 clock high | `68` |
| `CCD_HORZ_R3_CLOCK_Low` | integer | R3 clock low | `31` |

**CONFIG_CCD_REGIONS:**
| Field | Type | Description |
|-------|------|-------------|
| `CCD_GAIN_2D` | integer | 2D gain setting | `0` |
| `MODE_2D` | integer | 2D mode | `0` |
| `REGION_ENABLE` | integer | Enabled regions bitmask | `7` (all three) |
| `HORZ_ENABLE` | integer | Horizontal enable | `0` |
| `GAIN_ENABLE` | integer | Gain enable bitmask | `7` |
| `SKIP_1` through `SKIP_5` | integer | Skip rows per region | varies |
| `SUM_1` through `SUM_5` | integer | Sum rows per region | varies |
| `LAST_SKIP` | integer | Final skip | `0` |

**CONFIG_LASER_TIMING:**
| Field | Type | Description |
|-------|------|-------------|
| `LASER_INT_TIME` | string | Laser integration time | `20 us` |
| `LASER_REP_RATE` | string | Laser repetition rate | `80 Hz` |
| `LASER_ON_TIME` | string | Laser on time | `625000 (DN)` |
| `PULSE_WIDTH` | string | Pulse width | `40 us` |
| `LASER_CURRENT` | string | Laser current | `20 A` or `25 A` |
| `LASER_SHOTS` | integer | Shots per point | `10`, `100`, `300` |

---

### activeSpectra.csv / darkSpectra.csv / darkSubSpectra.csv

Spectral intensity data organized by CCD region.

**Structure:**
```
<header row with column names>
<R1 data rows (n_spectra rows, 2148 columns)>
<header row for R2>
<R2 data rows (n_spectra rows, 2148 columns)>
<header row for R3>
<R3 data rows (n_spectra rows, 2148 columns)>
```

**Column Headers:**
- R1: `R1_Channel0, R1_Channel1, ... R1_Channel2147`
- R2: `R2_Channel0, R2_Channel1, ... R2_Channel2147`
- R3: `R3_Channel0, R3_Channel1, ... R3_Channel2147`

**Data Types:** ASCII_Real (floating point values)

**Records:** n_spectra rows per region (typically 100)

**Files:**
- `activeSpectra.csv`: Laser-on spectra
- `darkSpectra.csv`: Laser-off (dark) spectra
- `darkSubSpectra.csv`: Active minus dark spectra

---

### spatial.csv

Scanner position and error data.

**Structure (4 tables concatenated):**

**Table 1: Az/El Position**
```csv
az,el
7080,4840
6664,3154
...
```

**Table 2: X/Y Position**
```csv
x,y
<calculated from az/el>
```

**Table 3: Scanner Error**
```csv
azimuth error,elevation error
```

**Table 4: Scanner Current**
```csv
sum current,difference current
```

---

### photodiodeRaw.csv

Laser photodiode intensity readings per shot.

**Structure:**
```csv
shot_number_0,shot_number_1,...,shot_number_N
80,77,71,64,67,72,77,80,83,86
88,84,80,77,72,63,58,64,73,79
...
```

**Columns:** N = shots_per_spec (e.g., 10, 100, or 300)
**Rows:** n_spectra (one row per map point)
**Values:** ASCII_Real (photodiode intensity in DN)

---

### roi.csv - Region of Interest Definitions

Custom format for defining spectral regions of interest.

**Structure:**
```
<ROI_name>
<color_hex>
<point_index_0>
<point_index_1>
...
ENDROI
<next_ROI_name>
...
```

**Example:**
```csv
Full Map
#ffffff
0
1
2
...
99
ENDROI
```

**Fields:**
- ROI name: Human-readable name
- Color: Hex color code (e.g., `#ffffff` for white)
- Point indices: Zero-based indices into the spectrum arrays
- `ENDROI`: Delimiter marking end of ROI

---

## SOFF XML Manifest

The SOFF (Spectrometer Open File Format) XML file provides PDS4-compliant metadata describing all data tables.

### Key Elements

**Identification Area:**
```xml
<Identification_Area>
  <logical_identifier>test_id</logical_identifier>
  <version_id>1.0</version_id>
  <title>this is a test</title>
  <information_model_version>1.11.1.0</information_model_version>
  <product_class>Product_Observational</product_class>
</Identification_Area>
```

**File Area Observational:**
For each data file, defines:
- File name
- Table structure (delimited or binary)
- Local identifier
- Byte offset
- Record count
- Field definitions

**Table Types:**

| Local Identifier | Description | Dimensions |
|-----------------|-------------|------------|
| `active_R1`, `active_R2`, `active_R3` | Active spectra | Points x Channels |
| `dark_R1`, `dark_R2`, `dark_R3` | Dark spectra | Points x Channels |
| `darkSub_R1`, `darkSub_R2`, `darkSub_R3` | Dark-subtracted | Points x Channels |
| `photodiode_all` | Photodiode data | Points x Shots |
| `spatial_az_el` | Az/El positions | Points x 2 |
| `spatial_x_y` | X/Y positions | Points x 2 |
| `scanner_err` | Scanner error | Points x 2 |
| `scanner_current` | Scanner current | Points x 2 |
| `LoupeTable` | Workspace metadata | Rows x 2 |
| `ACIAttributes` | ACI image metadata | Rows x 2 |
| `WATSONAttributes` | WATSON image metadata | Rows x 2 |

**SOFF Dimensions:**
```xml
<soff_Dimensions>
  <soff_Dimension>
    <soff_class>Points</soff_class>
    <elements>100</elements>
    <local_identifier>Points</local_identifier>
  </soff_Dimension>
  <soff_Dimension>
    <soff_class>Channels</soff_class>
    <elements>2148</elements>
    <local_identifier>Channels</local_identifier>
  </soff_Dimension>
  <soff_Dimension>
    <soff_class>Shots</soff_class>
    <elements>10</elements>
    <local_identifier>Shots</local_identifier>
  </soff_Dimension>
</soff_Dimensions>
```

---

## Image Files

### ACI (Autofocus Context Imager)

**Filename Pattern:**
```
SC3_<SOL>_<SCLK>_<PRODUCT>_<SEQ>_<VER>LUJ01.PNG
```

Example: `SC3_0059_0672194984_480ECM_N0032046SRLC15090_0000LUJ01.PNG`

**Characteristics:**
- Grayscale images
- Fixed pixel scale: ~10.1 um/pixel at nominal working distance
- Co-registered with spectral data via laser center position

### WATSON (Wide Angle Topographic Sensor for Operations and eNgineering)

**Filename Pattern:**
```
SI*_<SOL>_<SCLK>_<PRODUCT>_<SEQ>_<VER>LUJ01.PNG
```

**Characteristics:**
- Color (RGB) images
- Variable pixel scale (depends on focus position)
- Used for broader context

### Image Attributes CSV

| Field | Type | Description |
|-------|------|-------------|
| `pixel_scale` | float | um/pixel |
| `range` | float | Working distance (cm) |
| `CDPID` | integer | Context Data Product ID |
| `motor_pos` | integer | Focus motor position |
| `exp_time` | float | Exposure time (ms) |
| `product_ID` | string | Product identifier |
| `led_flag` | string | LED illumination flag |

---

## Session Files

### .lpe - Loupe Session File

Links multiple workspaces into a single session.

**Format:** CSV with header
```csv
workspaceDictName,workspaceHumanReadableName,soffPath
SrlcSpecSpecSohRaw_0672191108-30092-1,AlGaN_335,algan_335/.../...soff.xml
SrlcSpecSpecSohRaw_0672191331-30623-1,maze_1,maze_1/.../...soff.xml
```

---

## Key Classes and Models

### workspaceClass

Main data container for a Loupe workspace.

**Key Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `nSpectra` | int | Number of spectra in map |
| `nChannels` | int | CCD channels (2148) |
| `nShots` | int | Laser shots per point |
| `laserWavelength` | float | Laser wavelength (nm) |
| `wavelength` | list[float] | Wavelength array |
| `wavenumber` | list[float] | Raman shift array (cm^-1) |
| `activeSpectraR1/R2/R3` | dict[str, DataFrame] | Active spectra by processing level |
| `darkSpectraR1/R2/R3` | dict[str, DataFrame] | Dark spectra |
| `darkSubSpectraR1/R2/R3` | dict[str, DataFrame] | Dark-subtracted spectra |
| `photodiodeAll` | DataFrame | Photodiode intensities |
| `scannerTable` | DataFrame | Az/El positions |
| `xyTable` | DataFrame | X/Y positions |

**Processing Level Keys:**
- `None`: Unprocessed
- `N`: Normalized
- `B`: Baselined
- `C`: Cosmic-ray removed
- Combinations: `NB`, `NC`, `BN`, `BC`, `CN`, `CB`, `NBC`, `NCB`, etc.

### roiClass

Region of interest definition.

| Attribute | Type | Description |
|-----------|------|-------------|
| `humanReadableName` | str | Display name |
| `dictName` | str | Dictionary key |
| `color` | str | Hex color |
| `specIndexList` | list[int] | Point indices |

### SoffClassTable

SOFF table descriptor.

| Attribute | Type | Description |
|-----------|------|-------------|
| `filename` | str | Data file name |
| `lid` | str | Local identifier |
| `byteOffset` | int | Byte offset in file |
| `records` | int | Number of records |
| `fieldsMain` | int | Fields per record |
| `groups` | int | Groups per record |
| `fieldNames` | list[str] | Field names |
| `dataType` | list[str] | Data types |

### SoffClassDimension

SOFF dimension descriptor.

| Attribute | Type | Description |
|-----------|------|-------------|
| `soffClass` | str | Dimension class (Points, Channels, Shots) |
| `comment` | str | Description |
| `elements` | int | Number of elements |
| `lid` | str | Local identifier |

---

## Wavelength Calibration

> **Canonical Reference:** See [`docs/schema/SPECTRAL_REGIONS.md`](SPECTRAL_REGIONS.md)
> for the definitive wavelength calibration specification, including polynomial coefficients,
> R1/R2/R3 region boundaries, and code patterns. The information below is a summary.

### Default Calibration Polynomial

Segmented polynomial for wavelength calibration:

**Raman Region (channels 0-500):**
```python
popt_R = [-7.85000e-06, 6.52400e-02, 2.46690e+02]
wavelength = np.polyval(popt_R, channel)
```

**Fluorescence Region (channels 501+):**
```python
popt_F = [-5.65724e-06, 6.33627e-02, 2.47474e+02]
wavelength = np.polyval(popt_F, channel)
```

### Raman Shift Conversion

```python
laser_wavelength = 248.5794  # nm
wavenumber = (10**7) * ((1/laser_wavelength) - (1/wavelength))  # cm^-1
```

### CCD Region Boundaries

> For precise channel-to-wavelength mappings and the R123 stitching algorithm, see
> [`SPECTRAL_REGIONS.md`](SPECTRAL_REGIONS.md) Section 2 and Section 4.

The CCD has three regions with overlapping wavelength coverage:

| Region | Channel Range | Primary Raman Coverage |
|--------|--------------|----------------------|
| R1 | 0-565 | ~0 - ~4000 cm^-1 |
| R2 | ~565-1650 | ~600 - ~1700 cm^-1 |
| R3 | ~1650-2148 | ~1650 - ~4000 cm^-1 |

**Overlap Handling:**
- R1 + R2 overlap: channels 565-690
- R2 + R3 overlap: channels 1663-1696

---

## Spatial Coordinate System

### Scanner Coordinates

**Raw coordinates:** Azimuth (az) and Elevation (el) in digital numbers (DN)

**Scale factors (default):**
- `az_scale`: 0.628154699
- `el_scale`: 0.422441487

### Pixel Coordinates

Conversion from scanner coordinates to ACI image pixels:

```python
laser_center = (809, 664)  # pixels
rotation = 20.6793583  # degrees
```

### ACI/WATSON Working Distance

From motor position to working distance (cm):

**ACI:**
```python
working_distance = (0.005 * motor_pos) - 20.34
```

**WATSON:**
```python
# Complex polynomial - see ACI_WATSON_calc.py
working_distance = 1 / (
    (1091060 / motor_pos) +
    (-332.921) +
    (0.0382592 * motor_pos) +
    (-0.00000196922 * motor_pos**2) +
    (0.0000000000384562 * motor_pos**3)
)
```

---

## Data Quality Flags

### Processing Applied Codes

| Code | Meaning |
|------|---------|
| `None` | No processing |
| `N` | Laser intensity normalized |
| `B` | Baseline corrected |
| `C` | Cosmic rays removed |

### Composite Codes

Processing can be applied in different orders, resulting in codes like:
- `NB`: Normalized then baselined
- `NBC`: Normalized, baselined, cosmic-ray removed
- etc.

---

## References

1. **SHERLOC Instrument:** Bhartia et al. (2021). "Perseverance's SHERLOC Investigation." Space Science Reviews, 217(4).
2. **Cosmic Ray Removal:** Uckert et al. (2019). "A Semi-autonomous Method to Detect Cosmic Rays." Applied Spectroscopy 73.9.
3. **SOFF Format:** Uckert & Deen (2020). "Spectrometer Open File Format." AGU Fall Meeting.
4. **SPAR Library:** NASA-AMMOS/SPAR (https://github.com/NASA-AMMOS/SPAR)

---

## Appendix: Complete Field Inventory

### loupe.csv Fields (68 fields)

1. `original_data_file`
2. `human_readable_workspace`
3. `n_spectra`
4. `n_channels`
5. `laser_wavelength`
6. `shots_per_spec`
7. `az_scale`
8. `el_scale`
9. `laser_x`
10. `laser_y`
11. `rotation`
12. `specProcessingApplied`
13. `CNDH_PCB_TEMP_STAT_REG`
14. `CNDH_1_2_V_STAT_REG`
15. `CNDH_5_V_DAC_STAT_REG`
16. `CNDH_3_3_V_STAT_REG`
17. `CNDH_5_V_ADC_STAT_REG`
18. `CNDH_NEG_15_V_STAT_REG`
19. `CNDH_15_V_STAT_REG`
20. `CNDH_1_5_V_STAT_REG`
21. `laser_shot_counter`
22. `laser_misfire_counter`
23. `arc_event_counter`
24. `SE_CCD_ID_STAT_REG`
25. `SE_CCD_TEMP_STAT_REG`
26. `SE_PCB_TEMP_STAT_REG`
27. `SE_V_1_5_STAT_REG`
28. `SE_LASER_PRT2_STAT_REG`
29. `SE_LASER_PRT1_STAT_REG`
30. `SE_LPS_PRT1_STAT_REG`
31. `SE_TPRB_HOUSING_PRT_STAT_REG`
32. `SE_LPS_PRT2_STAT_REG`
33. `SE_SPARE1_PRT_STAT_REG`
34. `CCD_VERT_COL1_LOW`
35. `CCD_VERT_COL1_HIGH`
36. `CCD_VERT_COL2_LOW`
37. `CCD_VERT_COL2_HIGH`
38. `CCD_VERT_COL3_LOW`
39. `CCD_VERT_COL3_HIGH`
40. `CCD_HORZ_CLOCK_LIM`
41. `CCD_HORZ_R1_CLOCK_HIGH`
42. `CCD_HORZ_R1_CLOCK_Low`
43. `CCD_HORZ_R2_CLOCK_HIGH`
44. `CCD_HORZ_R2_CLOCK_Low`
45. `CCD_HORZ_R3_CLOCK_HIGH`
46. `CCD_HORZ_R3_CLOCK_Low`
47. `CCD_GAIN_2D`
48. `MODE_2D`
49. `REGION_ENABLE`
50. `HORZ_ENABLE`
51. `GAIN_ENABLE`
52. `SKIP_1`
53. `SUM_1`
54. `SKIP_2`
55. `SUM_2`
56. `SKIP_3`
57. `SUM_3`
58. `SKIP_4`
59. `SUM_4`
60. `SKIP_5`
61. `SUM_5`
62. `LAST_SKIP`
63. `LASER_INT_TIME`
64. `LASER_REP_RATE`
65. `LASER_ON_TIME`
66. `PULSE_WIDTH`
67. `LASER_CURRENT`
68. `LASER_SHOTS`

### Image Attribute Fields (7 fields)

1. `pixel_scale`
2. `range`
3. `CDPID`
4. `motor_pos`
5. `exp_time`
6. `product_ID`
7. `led_flag`
