# SHERLOC Spectral Regions Reference

**Version:** 1.0.0
**Date:** 2026-01-25
**Status:** Canonical Reference (Single Source of Truth)

This document defines the authoritative specification for SHERLOC spectral regions. All other documentation should reference this file for region definitions.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Region Definitions](#2-region-definitions)
3. [Wavelength Calibration](#3-wavelength-calibration)
4. [R123 Stitching Algorithm](#4-r123-stitching-algorithm)
5. [Database Schema](#5-database-schema)
6. [Code Patterns](#6-code-patterns)
7. [Validation Targets](#7-validation-targets)

---

## 1. Overview

SHERLOC (Scanning Habitable Environments with Raman and Luminescence for Organics and Chemicals) uses a deep-UV laser (248.5794 nm) to excite samples and collects both Raman scattering and fluorescence emission on a 2148-channel CCD detector.

The CCD is logically divided into three regions based on wavelength/spectral domain:

| Region | Domain | Primary Use |
|--------|--------|-------------|
| **R1** | Raman | Mineral identification via vibrational modes |
| **R2** | Fluorescence | Aromatic/organic detection, rare earth fluorescence |
| **R3** | Fluorescence | Extended fluorescence range |
| **R123** | Combined | Cross-modal pattern discovery |

---

## 2. Region Definitions

### 2.1 Summary Table

| Region | Wavelength (nm) | Channels | Channel Count | Wavenumber (cm⁻¹) | Domain |
|--------|-----------------|----------|---------------|-------------------|--------|
| **R1** | 250.0 - 282.0 | 52 - 574 | 523 | ~238 - 4765 | Raman |
| **R2** | 282.0 - 337.8 | 690 - 1668 | 979 | N/A | Fluorescence |
| **R3** | 337.8 - 357.4 | 1690 - 2147 | 458 | N/A | Fluorescence |
| **R123** | 250.0 - 357.4 | 0 - 2147 | 2148 (stitched) | Mixed | Combined |

### 2.2 R1 Raman Region (Primary Science Region)

The R1 region captures Raman-scattered photons, providing molecular vibrational information.

```
Wavelength Range:  250.0 - 282.0 nm
Channel Range:     52 - 574 (after wavelength filtering)
Channel Count:     523 channels
Wavenumber Range:  ~238 - 4765 cm⁻¹
Usable Range:      ~640 - 4200 cm⁻¹ (below 640 is laser line region)
```

**Key Raman Bands (Mars Minerals):**

| Mineral Class | Wavenumber (cm⁻¹) | Example Targets |
|---------------|-------------------|-----------------|
| Sulfate ν1 | 1000-1018 | Uganik Island, Dragons Egg Rock |
| Sulfate ν3 | 1130-1160 | Steamboat Mountain |
| Carbonate ν1 | 1085 | Amherst Point, Bills Bay |
| Olivine doublet | 820, 850 | Lake Haiyaha |
| Phosphate ν1 | 960 | TBD |

### 2.3 R2 Fluorescence Region

The R2 region captures near-UV fluorescence emission.

```
Wavelength Range:  282.0 - 337.8 nm
Channel Range:     690 - 1668 (approximate)
Channel Count:     ~979 channels
```

**Key Fluorescence Features:**
- Ce³⁺ doublet at 304/325 nm (anhydrite)
- Aromatic organic signatures
- Polycyclic aromatic hydrocarbons (PAHs)

### 2.4 R3 Fluorescence Region

The R3 region captures extended fluorescence emission.

```
Wavelength Range:  337.8 - 357.4 nm
Channel Range:     1690 - 2147 (approximate)
Channel Count:     ~458 channels
```

**Key Fluorescence Features:**
- Ce³⁺ at 340 nm (apatite)
- Extended organic fluorescence

### 2.5 R123 Combined Spectrum

The R123 spectrum is a **stitched** combination of all three regions with overlap summation.

```
Wavelength Range:  250.0 - 357.4 nm (full detector range)
Channel Count:     2148 channels (stitched)
```

**Use Cases:**
- Cross-modal Raman-Fluorescence correlation
- Detecting mineral phases with both Raman and fluorescence signatures
- Example: Anhydrite shows BOTH 1018 cm⁻¹ Raman AND 304/325 nm fluorescence

---

## 3. Wavelength Calibration

### 3.1 Polynomial Coefficients (Loupe V5.1.5a)

The wavelength calibration uses a **segmented polynomial** with different coefficients for Raman and Fluorescence regions.

```yaml
# config.yaml calibration section
wavelength_calibration:
  raman_coefficients: [-7.85000e-06, 6.52400e-02, 2.46690e+02]
  fluorescence_coefficients: [-5.65724e-06, 6.33627e-02, 2.47474e+02]
  cutoff_channel: 500  # Polynomial switch point, NOT region boundary!
  laser_wavelength: 248.5794
```

### 3.2 Calibration Algorithm

```python
import numpy as np

def calculate_loupe_wavelength_wavenumber(n_channels: int = 2148) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate wavelength and wavenumber arrays using Loupe polynomial calibration.

    NEVER use np.linspace() for wavenumber - always use this function!

    Returns:
        wavelength: Array of wavelengths in nm (2148 values)
        wavenumber: Array of Raman shifts in cm⁻¹ (2148 values)
    """
    # Loupe V5.1.5a coefficients
    raman_coefficients = [-7.85000e-06, 6.52400e-02, 2.46690e+02]
    fluorescence_coefficients = [-5.65724e-06, 6.33627e-02, 2.47474e+02]
    cutoff_channel = 500
    laser_wavelength = 248.5794

    channels = np.arange(n_channels)
    wavelength = np.zeros(n_channels)

    # Apply segmented polynomial
    raman_mask = channels <= cutoff_channel
    wavelength[raman_mask] = np.polyval(raman_coefficients, channels[raman_mask])
    wavelength[~raman_mask] = np.polyval(fluorescence_coefficients, channels[~raman_mask])

    # Convert to wavenumber (Raman shift)
    wavenumber = 1e7 * (1/laser_wavelength - 1/wavelength)

    return wavelength, wavenumber
```

### 3.3 Critical Warning

> **NEVER use `np.linspace()` for wavenumber axes!**
>
> The wavenumber scale is non-linear due to the λ⁻¹ relationship. Sprint 3 ML models
> may have used linear wavenumber scales, leading to incorrect peak positions.
> Always use the polynomial calibration above.

### 3.4 R1 Extraction Pattern

```python
def extract_r1_region(full_spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract R1 Raman region from full 2148-channel spectrum.

    Uses wavelength filtering (250-282 nm), NOT raw channel slicing.

    Args:
        full_spectrum: 2148-channel intensity array

    Returns:
        r1_intensities: 523-channel R1 intensities
        r1_wavenumber: 523-value wavenumber array (cm⁻¹)
    """
    wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(2148)

    # Wavelength filter (NOT channel slice!)
    r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)

    r1_intensities = full_spectrum[r1_mask]  # 523 values
    r1_wavenumber = wavenumber[r1_mask]      # 523 values

    return r1_intensities, r1_wavenumber
```

---

## 4. R123 Stitching Algorithm

### 4.1 Overview

When SHERLOC acquires data with multiple regions enabled, the R1, R2, and R3 spectra overlap at their boundaries. The R123 stitched spectrum combines these with **overlap summation** (not averaging).

### 4.2 Overlap Regions

Based on Loupe `file_IO.py:714-739`:

| Channel Range | Width | Source Regions | Operation |
|---------------|-------|----------------|-----------|
| 0 - 564 | 565 | R1 only | Copy |
| 565 - 689 | 125 | R1 + R2 | **Sum** |
| 690 - 1667 | 978 | R2 only | Copy |
| 1668 - 1689 | 22 | R2 + R3 | **Sum** |
| 1690 - 2147 | 458 | R3 only | Copy |

### 4.3 Stitching Implementation

```python
def stitch_r123_spectrum(r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> np.ndarray:
    """
    Stitch R1, R2, R3 spectra into combined R123 spectrum.

    Matches Loupe algorithm with overlap summation.

    Args:
        r1: R1 spectrum (2148 channels, R1 data in channels 0-574)
        r2: R2 spectrum (2148 channels, R2 data in channels 565-1668)
        r3: R3 spectrum (2148 channels, R3 data in channels 1668-2147)

    Returns:
        r123: Stitched 2148-channel spectrum
    """
    r123 = np.zeros(2148, dtype=np.float64)

    # Region 1: R1 only (channels 0-564)
    r123[0:565] = r1[0:565]

    # Overlap 1: R1 + R2 (channels 565-689)
    r123[565:690] = r1[565:690] + r2[565:690]

    # Region 2: R2 only (channels 690-1667)
    r123[690:1668] = r2[690:1668]

    # Overlap 2: R2 + R3 (channels 1668-1689)
    r123[1668:1690] = r2[1668:1690] + r3[1668:1690]

    # Region 3: R3 only (channels 1690-2147)
    r123[1690:2148] = r3[1690:2148]

    return r123
```

### 4.4 Validation

R123 stitching should be validated against Loupe-generated R123 spectra:

```python
def validate_r123_stitching(loupe_r123: np.ndarray, computed_r123: np.ndarray) -> bool:
    """Validate stitching matches Loupe output within tolerance."""
    return np.allclose(loupe_r123, computed_r123, rtol=1e-5, atol=1e-10)
```

---

## 5. Database Schema

### 5.1 Valid Region Values

The `spectra.region` column accepts these values:

| Value | Meaning | Current Status |
|-------|---------|----------------|
| `R1` | R1 Raman region (523 channels after wavelength filter) | **Active** — all 524,181 spectra |
| `R2` | R2 Fluorescence region | Planned |
| `R3` | R3 Fluorescence region | Planned |
| `R123` | Stitched full spectrum (overlap summation) | Planned (requires R2+R3 ingestion) |

**Constraint enforcement:** Valid values enforced at application level via the `SpectralRegion`
enum in `src/sherloc_pipeline/models/spectra.py`. SQLite does not support `ALTER TABLE ADD CONSTRAINT`,
so the recommended CHECK constraint (Section 5.2) is documented for future table rebuilds.

### 5.2 Recommended Constraint

```sql
-- Add CHECK constraint to enforce valid region values
ALTER TABLE spectra ADD CONSTRAINT chk_spectra_region
CHECK (region IN ('R1', 'R2', 'R3', 'R123'));
```

### 5.3 Spectrum Storage

| Region | Stored Channels | intensities BLOB Size |
|--------|-----------------|----------------------|
| R1 | 2148 (full CCD)* | ~8.5 KB compressed |
| R2 | 2148 (full CCD)* | ~8.5 KB compressed |
| R3 | 2148 (full CCD)* | ~8.5 KB compressed |
| R123 | 2148 (stitched) | ~8.5 KB compressed |

*Full CCD stored; extract meaningful channels via wavelength filtering at query time.

---

## 6. Code Patterns

### 6.1 Correct R1 Data Loading

```python
from sherloc_pipeline.core.normalization import calculate_loupe_wavelength_wavenumber

def load_r1_spectra_for_ml(db_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load R1 spectra from database with proper wavenumber calibration.

    Returns:
        spectra: (N, 523) array of R1 intensities
        wavenumber: (523,) array of wavenumber values
    """
    # Get calibrated wavelength/wavenumber
    wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(2148)
    r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)
    r1_wavenumber = wavenumber[r1_mask]  # 523 values

    # Query database
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT s.intensities
        FROM spectra s
        JOIN scan_points sp ON s.scan_point_id = sp.id
        JOIN scans sc ON sp.scan_id = sc.id
        WHERE s.spectrum_type = 'dark_subtracted'
          AND sc.target IS NOT NULL
    """)

    spectra_list = []
    for row in cursor:
        full_spectrum = decompress_spectrum(row[0])  # 2148 values
        r1_spectrum = full_spectrum[r1_mask]         # 523 values
        spectra_list.append(r1_spectrum)

    return np.array(spectra_list), r1_wavenumber
```

### 6.2 Anti-Patterns (DO NOT USE)

```python
# WRONG: Linear wavenumber scale
wavenumber = np.linspace(200, 4000, 2148)  # NEVER DO THIS!

# WRONG: Raw channel slicing for R1
r1_spectrum = full_spectrum[:501]  # cutoff_channel is NOT region boundary!

# WRONG: Assuming region="R123" means stitched data
# Historical data labeled R123 is actually R1-only (see Section 7)
```

---

## 7. Validation Targets

### 7.1 Known Mineral Targets for R1 Validation

| Target | Expected Peaks (cm⁻¹) | Mineral | Notes |
|--------|----------------------|---------|-------|
| Uganik Island | 1018, 1130 | Anhydrite | + Fluorescence at 304/325 nm |
| Dragons Egg Rock | 1000, 1018 | Sulfate | Strong sulfate signal |
| Steamboat Mountain | 1018, 1130 | Sulfate | |
| Amherst Point | 1085 | Carbonate | |
| Bills Bay | 1085 | Carbonate | |
| Lake Haiyaha | 820, 850 | Olivine | Doublet pattern |
| AlGaN (calibration) | 575, 1435 | N/A | Calibration target |
| Teflon (calibration) | 732, 1218, 1382 | PTFE | Calibration target |

### 7.2 Cross-Modal Validation (R123)

| Target | Raman Feature | Fluorescence Feature | Interpretation |
|--------|---------------|----------------------|----------------|
| Uganik Island | 1018 cm⁻¹ (ν1 sulfate) | 304/325 nm doublet | Ce³⁺ in anhydrite |
| TBD | 960 cm⁻¹ (ν1 phosphate) | 340 nm | Ce³⁺ in apatite |
| Lake Haiyaha | 820/850 cm⁻¹ (olivine) | None | Control (no REE) |

---

## Appendix A: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                 SHERLOC SPECTRAL REGIONS                        │
├─────────────────────────────────────────────────────────────────┤
│ R1 (Raman):     250-282 nm  │  523 ch  │  ~640-4200 cm⁻¹       │
│ R2 (Fluor 1):   282-338 nm  │  979 ch  │  Fluorescence         │
│ R3 (Fluor 2):   338-357 nm  │  458 ch  │  Fluorescence         │
│ R123 (Stitched): Full range │ 2148 ch  │  Overlap summation    │
├─────────────────────────────────────────────────────────────────┤
│ CRITICAL: cutoff_channel=500 is POLYNOMIAL SWITCH not R1 bound │
│ ALWAYS: Use wavelength filter (250-282 nm) for R1 extraction   │
│ NEVER:  Use np.linspace() for wavenumber axes                  │
├─────────────────────────────────────────────────────────────────┤
│ Calibration: raman_coef = [-7.85e-06, 6.524e-02, 2.4669e+02]   │
│              laser_wavelength = 248.5794 nm                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix B: Cross-References

This document is the canonical reference. The following documents should reference this file:

| Document | Section to Update |
|----------|-------------------|
| `docs/schema/UNIFIED_SCHEMA.md` | Spectrum entity, region field |
| `docs/schema/LOUPE_DATA_FORMAT.md` | Wavelength Calibration section |
| `docs/schema/PDS_DATA_FORMAT.md` | Product type mapping |
| `CLAUDE.md` | Critical Spectral Calibration section |
| `src/sherloc_pipeline/models/spectra.py` | SpectralRegion docstring |

---

*Document Version: 1.0.0*
*Created: 2026-01-25*
*Author: Sprint 4 Schema Foundation*
