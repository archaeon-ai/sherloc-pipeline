# SHERLOC Pipeline Python API

**Module:** `sherloc_pipeline.api.spectral`

The Python API provides notebook-friendly functions for SHERLOC spectral analysis, enabling Jupyter workflows without CLI interaction.

---

## Quick Start

```python
from sherloc_pipeline.api.spectral import (
    process_scan_average,
    process_point,
    process_subset_average,
    load_point_spectrum,
    load_reference_spectrum,
    plot_spectrum,
    plot_overlay,
)

# Process averaged spectrum with fitting
df, fit = process_scan_average(
    sol="0921", target="Amherst_Point", scan="detail_1",
    background="fs", baseline=True, fit=True
)

# Generate a plot
fig = plot_spectrum(df, fit_result=fit, xlim=(700, 1200))
fig.savefig("spectrum.png", dpi=300)
```

---

## Processing Functions

### `process_scan_average()`

Process averaged spectrum from Loupe data (all points).

```python
def process_scan_average(
    sol: str,
    target: str,
    scan: str,
    *,
    avg_method: Literal["mean", "median", "trim-mean"] = "trim-mean",
    trim_pct: float = 2.0,
    background: Optional[Literal["as", "fs"]] = "fs",
    bgscale: Union[float, Literal["auto"]] = "auto",
    baseline: bool = True,
    fit: bool = False,
    fit_range: Optional[Tuple[float, float]] = None,
    single_peak_center: Optional[float] = None,
    n_peaks: Optional[int] = None,
    data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional[FitResult]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sol` | str | required | Sol number (e.g., "0921") |
| `target` | str | required | Target name (e.g., "Amherst_Point") |
| `scan` | str | required | Scan identifier (e.g., "detail_1") |
| `avg_method` | str | "trim-mean" | Averaging method: "mean", "median", "trim-mean" |
| `trim_pct` | float | 2.0 | Trim percentage for trim-mean (0-50) |
| `background` | str/None | "fs" | Background type: "as", "fs", or None |
| `bgscale` | float/"auto" | "auto" | Scale factor or "auto" for PPP-based |
| `baseline` | bool | True | Apply asPLS baseline correction |
| `fit` | bool | False | Apply Gaussian peak fitting |
| `fit_range` | tuple | None | Fit range in cm⁻¹ (min, max) |
| `single_peak_center` | float | None | Fit single peak at position |
| `n_peaks` | int | None | Maximum peaks to fit |
| `min_snr` | float | None | Override minimum SNR threshold (default: 3.0) |
| `fwhm_min` | float | None | Override minimum FWHM in cm⁻¹ (default: 30) |
| `fwhm_max` | float | None | Override maximum FWHM in cm⁻¹ (default: 90) |
| `data_dir` | Path | None | Override data directory |
| `results_dir` | Path | None | Override results directory |

**Returns:** `Tuple[DataFrame, Optional[FitResult]]`

- DataFrame with columns: `raman_shift`, `intensity`
- FitResult if `fit=True`, else None

**Example:**

```python
# Basic processing with defaults (trim-mean, FS background, baseline)
df, _ = process_scan_average("0921", "Amherst_Point", "detail_1")

# Full processing with fitting
df, fit = process_scan_average(
    sol="0921", target="Amherst_Point", scan="detail_1",
    background="fs", baseline=True, fit=True
)
if fit:
    for peak in fit.peaks:
        print(f"Peak at {peak.m_cm1:.1f} cm⁻¹, FWHM={peak.fwhm:.1f}")

# Single-peak fitting for carbonate
df, fit = process_scan_average(
    sol="0921", target="Amherst_Point", scan="detail_1",
    fit=True, fit_range=(1000, 1200), single_peak_center=1090
)
```

---

### `process_point()`

Process a single point from Loupe data.

```python
def process_point(
    sol: str,
    target: str,
    scan: str,
    point: int,
    *,
    background: Optional[Literal["as", "fs"]] = None,
    bgscale: Union[float, Literal["auto"]] = "auto",
    baseline: bool = False,
    fit: bool = False,
    fit_range: Optional[Tuple[float, float]] = None,
    single_peak_center: Optional[float] = None,
    n_peaks: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional[FitResult]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sol` | str | required | Sol number |
| `target` | str | required | Target name |
| `scan` | str | required | Scan identifier |
| `point` | int | required | Point index (0-based) |
| `background` | str/None | None | Background type: "as", "fs", or None |
| `bgscale` | float/"auto" | "auto" | Scale factor |
| `baseline` | bool | False | Apply baseline correction |
| `fit` | bool | False | Apply Gaussian fitting |
| `fit_range` | tuple | None | Fit range in cm⁻¹ |
| `single_peak_center` | float | None | Single peak center |
| `n_peaks` | int | None | Maximum peaks |
| `min_snr` | float | None | Override minimum SNR threshold (default: 3.0) |
| `fwhm_min` | float | None | Override minimum FWHM in cm⁻¹ (default: 30) |
| `fwhm_max` | float | None | Override maximum FWHM in cm⁻¹ (default: 90) |
| `data_dir` | Path | None | Override data directory |

**Returns:** `Tuple[DataFrame, Optional[FitResult]]`

**Example:**

```python
# Process point 91 with background and baseline
df, fit = process_point(
    sol="0921", target="Amherst_Point", scan="detail_1",
    point=91, background="fs", baseline=True, fit=True
)
```

---

### `process_subset_average()`

Process averaged spectrum from a subset of points.

```python
def process_subset_average(
    sol: str,
    target: str,
    scan: str,
    points: List[int],
    *,
    avg_method: Literal["mean", "median", "trim-mean"] = "trim-mean",
    trim_pct: float = 2.0,
    background: Optional[Literal["as", "fs"]] = "fs",
    bgscale: Union[float, Literal["auto"]] = "auto",
    baseline: bool = True,
    fit: bool = False,
    fit_range: Optional[Tuple[float, float]] = None,
    single_peak_center: Optional[float] = None,
    n_peaks: Optional[int] = None,
    data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional[FitResult]]
```

**Parameters:**

Same as `process_scan_average()` plus:

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | List[int] | List of point indices to average (0-based, min 2 points) |

**Example:**

```python
# Average specific points (like ad-hoc label averaging)
df, fit = process_subset_average(
    sol="0921", target="Amherst_Point", scan="detail_1",
    points=[21, 41, 49, 71, 86, 87, 88, 90, 91, 92, 98],
    background="fs", baseline=True, fit=True
)
```

---

## Loading Functions

### `load_point_spectrum()`

Load a single point spectrum from existing pipeline outputs.

```python
def load_point_spectrum(
    sol: str,
    target: str,
    scan: str,
    point: int,
    level: Literal[
        "normalized",
        "normalized_baselined",
        "normalized_despiked_baselined"
    ] = "normalized_despiked_baselined",
    results_dir: Optional[Path] = None,
) -> pd.DataFrame
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sol` | str | required | Sol number |
| `target` | str | required | Target name |
| `scan` | str | required | Scan identifier |
| `point` | int | required | Point index (0-based) |
| `level` | str | "normalized_despiked_baselined" | Processing level |
| `results_dir` | Path | None | Override results directory |

**Processing levels:**
- `"normalized"`: Laser-normalized only
- `"normalized_baselined"`: Normalized + baseline corrected
- `"normalized_despiked_baselined"`: Normalized + despiked + baseline corrected

**Returns:** DataFrame with `raman_shift` and `intensity` columns

**Example:**

```python
# Load fully processed point
df = load_point_spectrum(
    sol="0921", target="Amherst_Point", scan="detail_1",
    point=91, level="normalized_despiked_baselined"
)
```

---

### `load_reference_spectrum()`

Load a reference mineral spectrum.

```python
def load_reference_spectrum(
    mineral: str,
    reference_dir: Optional[Path] = None,
) -> pd.DataFrame
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mineral` | str | required | Mineral name (case-insensitive partial match) |
| `reference_dir` | Path | None | Override reference directory |

**Returns:** DataFrame with `raman_shift` and `intensity` columns

**Example:**

```python
# Load forsterite reference
ref_df = load_reference_spectrum("forsterite")

# Case-insensitive partial matching
ref_df = load_reference_spectrum("Forst")  # Also works
```

---

## Plotting Functions

### `plot_spectrum()`

Generate a single-spectrum plot.

```python
def plot_spectrum(
    df: pd.DataFrame,
    *,
    color: Optional[str] = None,
    linewidth: float = 1.0,
    linestyle: str = "-",
    fit_result: Optional[FitResult] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    xlabel: str = "Raman Shift (cm⁻¹)",
    ylabel: str = "Intensity (a.u.)",
    figsize: Tuple[float, float] = (10, 6),
) -> Figure
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | required | Spectrum data with `raman_shift` and `intensity` |
| `color` | str | None | Line color (default: matplotlib default) |
| `linewidth` | float | 1.0 | Line width |
| `linestyle` | str | "-" | Line style: "-", "--", "-.", ":" |
| `fit_result` | FitResult | None | Optional fitting results to overlay |
| `xlim` | tuple | None | X-axis limits (min, max) |
| `ylim` | tuple | None | Y-axis limits (min, max) |
| `title` | str | None | Plot title |
| `xlabel` | str | "Raman Shift (cm⁻¹)" | X-axis label |
| `ylabel` | str | "Intensity (a.u.)" | Y-axis label |
| `figsize` | tuple | (10, 6) | Figure size in inches |

**Returns:** matplotlib Figure

**Example:**

```python
df, fit = process_scan_average(
    sol="0921", target="Amherst_Point", scan="detail_1",
    fit=True
)

fig = plot_spectrum(
    df,
    fit_result=fit,
    xlim=(700, 1200),
    title="Amherst Point - R1 Region"
)
fig.savefig("spectrum.png", dpi=300)
```

---

### `plot_overlay()`

Generate a multi-spectrum overlay plot.

```python
def plot_overlay(
    spectra: List[Dict[str, Any]],
    *,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    scale_to_peak: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    xlabel: str = "Raman Shift (cm⁻¹)",
    ylabel: str = "Intensity (a.u.)",
    figsize: Tuple[float, float] = (10, 6),
    legend_loc: str = "best",
) -> Figure
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spectra` | List[dict] | required | List of spectrum dictionaries |
| `xlim` | tuple | None | X-axis limits |
| `ylim` | tuple | None | Y-axis limits |
| `scale_to_peak` | tuple | None | Range (min, max) for peak normalization |
| `title` | str | None | Plot title |
| `xlabel` | str | "Raman Shift (cm⁻¹)" | X-axis label |
| `ylabel` | str | "Intensity (a.u.)" | Y-axis label |
| `figsize` | tuple | (10, 6) | Figure size |
| `legend_loc` | str | "best" | Legend location |

**Spectrum dictionary format:**

```python
{
    "df": DataFrame,           # Required: spectrum data
    "label": str,              # Optional: legend label
    "color": str,              # Optional: line color
    "linewidth": float,        # Optional: line width (default: 1.0)
    "linestyle": str,          # Optional: line style (default: "-")
}
```

**Returns:** matplotlib Figure

**Example:**

```python
# Load Mars spectrum and reference
mars_df, _ = process_scan_average(
    sol="0921", target="Amherst_Point", scan="detail_1"
)
ref_df = load_reference_spectrum("forsterite")

# Create overlay with peak normalization
fig = plot_overlay(
    spectra=[
        {"df": mars_df, "label": "Mars (Amherst Point)", "color": "blue"},
        {"df": ref_df, "label": "Forsterite reference", 
         "color": "green", "linestyle": "--"},
    ],
    xlim=(700, 1200),
    scale_to_peak=(820, 870),  # Normalize to olivine doublet
    title="Mars vs Forsterite Comparison"
)
fig.savefig("comparison.png", dpi=300)
```

---

## Error Handling

All functions may raise:

- `ValueError`: Invalid parameters (e.g., point out of range, invalid method)
- `SpectralPlotError`: Processing failures (file not found, parsing errors)
- `FileNotFoundError`: Missing data or reference files

**Example:**

```python
from sherloc_pipeline.services.spectral import SpectralPlotError

try:
    df, fit = process_scan_average(
        sol="9999", target="Nonexistent", scan="detail_1"
    )
except SpectralPlotError as e:
    print(f"Processing failed: {e}")
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

---

## FitResult Structure

When `fit=True`, returns a `FitResult` object:

```python
@dataclass
class FitResult:
    peaks: List[PeakFit]      # Fitted peaks
    r2: float                 # R² goodness of fit
    rss: float                # Residual sum of squares
    dof: int                  # Degrees of freedom
    warnings: List[str]       # Any fitting warnings

@dataclass
class PeakFit:
    m_cm1: float              # Peak center / mean (cm⁻¹)
    a: float                  # Amplitude (peak height)
    fwhm: float               # Full width at half maximum (cm⁻¹)
    sigma: float              # Standard deviation corresponding to FWHM
    area: float               # Area under the Gaussian
    snr: float                # Signal-to-noise ratio estimate
    pass_snr: bool            # Met the SNR threshold
    pass_fwhm: bool           # Met the FWHM bounds
    pass_r2: bool             # Met the R² threshold
    sharpness_ratio: float    # data_at_center / amplitude (>>1 → cosmic ray)
    pass_sharpness: bool      # False if sharpness_ratio exceeds threshold
```

**Example:**

```python
df, fit = process_scan_average(
    sol="0921", target="Amherst_Point", scan="detail_1",
    fit=True
)

if fit:
    print(f"R² = {fit.r2:.4f}")
    print(f"Found {len(fit.peaks)} peaks:")
    for peak in fit.peaks:
        accepted = peak.pass_snr and peak.pass_fwhm and peak.pass_r2 and peak.pass_sharpness
        status = "✓" if accepted else "✗"
        print(f"  {peak.m_cm1:.1f} cm⁻¹, FWHM={peak.fwhm:.1f}, A={peak.a:.1f} {status}")
```

---

## Web HTTP API — Scan Browser query parameters

`GET /api/scans` supports the Scan Browser's range filtering, sortable
columns, and pagination. All parameters are optional:

| Parameter | Meaning |
|---|---|
| `sol` | Exact sol match (retained for backward compatibility) |
| `sol_from`, `sol_to` | Inclusive lower and upper sol bounds |
| `target` | Case-insensitive substring match; spaces and underscores are equivalent |
| `scan_class`, `scan_type`, `processing_status` | Existing categorical filters |
| `sort_by` | `sol` or `target` |
| `sort_order` | `asc` or `desc` (defaults to `asc`) |
| `offset`, `limit` | Pagination offset and page size |

A reversed sol range, unknown sort field, or unknown sort direction returns
HTTP 400. Sorting includes stable sol/name tie-breakers so rows do not move
between paginated requests.

---

## Web HTTP API — Spectra despike method selector

The web API (`sherloc_pipeline.web`) exposes a despike **method selector** on
the three spectra read endpoints. Two strategies are available:

- **`ml`** — a **stored-mask** cosmic-ray despike. The serving host never
  runs model inference: it looks up the cosmic-ray masks the pipeline already
  persisted (`cosmic_ray_masks` table) and applies the same interpolation
  replacement the pipeline uses. See [`METHODS.md`](METHODS.md) §2.2.1 for the
  detector and its coverage limits.
- **`modz`** — the legacy rolling-median **modified-z-score** despike
  (`core/preprocessing.py::despike_r1_spectrum`, the same algorithm behind
  `POST /api/process/despike`), computed live on the host. **R1 only.**

### Parameter

| Endpoint | How to pass the selector |
|---|---|
| `GET /api/spectra/{scan_id}/average` | `?despike_method=none\|ml\|modz` (and legacy `?despike=true`) |
| `GET /api/spectra/{scan_id}/point/{idx}` | `?despike_method=none\|ml\|modz` (and legacy `?despike=true`) |
| `POST /api/spectra/{scan_id}/subset` | `despike_method: "none"\|"ml"\|"modz"` JSON body field (and legacy `despike: true`) |

- **Allowed values:** `none`, `ml`, `modz` (case-sensitive). Any other value
  returns **422**.
- **Precedence (backward-compatible):** an explicit `despike_method` wins.
  When `despike_method` is absent, the legacy `despike` bool maps
  `false → none`, `true → ml` — the exact v4.2.x stored-mask behavior. The
  `despike` bool keeps working forever (see [`INVARIANTS.md`](INVARIANTS.md)).

### `ml` semantics (unchanged from v4.2.x)

Masks attach to the **DARK_SUBTRACTED** spectrum row, so `ml` is a no-op for
representations that carry no stored mask (e.g. PDS `laser_normalized`
spectra) — the response simply reports `despike_applied: false`. Despiking is
applied per point **before** any averaging. The response `despike_method`
returns the **precise stored provenance string** (e.g. `ml_v1.3_tau_matched`),
not the coarse request enum.

### `modz` semantics (new)

`modz` runs the rolling-median modified-z-score despike **on the served
array**:

- The **average** / **subset** endpoints despike the *averaged* spectrum
  (not each constituent). The **point** endpoint despikes the single point's
  served spectrum.
- This is **display-level** behavior that intentionally matches the legacy
  Workbench client-side modz step. It **differs** from the CLI pipeline's
  per-spectrum, pre-fit modz, which despikes each raw point spectrum before
  averaging and fitting.
- **R1 only.** For any non-R1 region (`R2`, `R3`, or the `R123` composite)
  the array is served **non-despiked** with `despike_applied: false` and
  `despike_missing_regions` naming the un-covered region(s) (`R123` lists
  `["R1","R2","R3"]`). Windowed modz inside a stitched composite is not
  attempted.
- On the R1 path: `despike_applied: true`, `despike_method: "modz"`,
  `n_masked_channels` = number of spike channels replaced,
  `despike_missing_regions: []`, and `despike_params_used` populated with the
  config-default modz parameters used. Because `modz` is a live compute, it
  works on scans that have **no** stored masks.

### Additive response fields

All fields are optional with defaults, so existing clients are unaffected and
`schema_version` stays `1.0.0`:

| Field | Type | Meaning |
|---|---|---|
| `despike_applied` | bool | `true` iff a despike was applied to this view |
| `despike_method` | str \| null | precise provenance label of the applied despike: the stored mask's label for `ml` (e.g. `ml_v1.3_tau_matched`), or the literal `"modz"` |
| `n_masked_channels` | int | channels replaced in the returned array |
| `masked_channels` | int[] | *(point endpoint, `ml` only)* the applied **absolute CCD channel** indices, `0..2147` (empty for `modz`, which reports a count) |
| `masked_positions` | int[] | *(all three endpoints, `ml` only)* positions **into the served arrays** where the stored-mask despike replaced channels — empty for `none`/`modz` |
| `despike_missing_regions` | str[] | regions that could not be despiked — `ml` composite contributors lacking a stored mask, or the non-R1 region(s) for `modz` |
| `n_uncovered_contributor_channels` | int | *(`ml` composite views)* channels carrying ≥1 uncovered contributor; `0` for single-region and `modz` views |
| `despike_params_used` | object \| null | *(`modz` R1 path only)* the modz parameters used (config defaults); `null` on the `ml`/`none` paths |

#### `masked_channels` vs `masked_positions` — two coordinate systems

These two `ml`-only fields describe the **same** replaced channels in
**different** coordinate systems, and are not interchangeable:

- **`masked_channels`** (point endpoint only) — **absolute** CCD channel
  indices in `0..2147`. The original v4.2.x field, retained unchanged for
  back-compat.
- **`masked_positions`** (average, point, subset — added in issue #8) —
  **0-based positions into the served `wavenumber`/`intensity` arrays** (length
  == `n_channels`). A marker at `masked_positions[k]` lands on
  `wavenumber[masked_positions[k]]`. This is what the Workbench uses to render
  the red-triangle spike markers in ML mode.

  For a **single region** the served array is the region's wavelength
  selection, so an absolute channel maps to its row within that selection
  (e.g. R1 begins at absolute channel 52, so absolute channel 205 is served
  position 153). For the **`R123`** composite the served array is the full
  2148-channel stitch (a position-preserving copy/sum at each channel index),
  so served position equals the absolute channel and
  `masked_positions == masked_channels`. The average/subset endpoints report
  the **union** of replaced positions over the contributing points.

### `ml` `R123` semantics

- **No stored mask → communicated state.** A spectrum without a stored
  mask renders non-despiked with `despike_applied: false` (no error, no
  inference attempt).
- **R123 is constituent-first.** Each region's stored mask is applied to
  its DARK_SUBTRACTED constituent **before** the overlap summation — a mask
  is never applied to an already-summed value.
- **All-or-none composite availability.** If any contributing region of an
  R123 view lacks a stored mask, the whole composite renders non-despiked
  with `despike_applied: false` and `despike_missing_regions` naming the
  missing regions. A partially despiked composite is never labeled
  applied.
- **Coverage disclosure.** Composite views report
  `n_uncovered_contributor_channels` — the count of summed-output channels
  carrying at least one contributor outside its certified detection window
  (these retain legacy never-screened behavior). With the certified
  windows this is **207 of 2148** for the R123 summation view; single-region
  views report `0`. The value is derived from the frozen manifest windows
  plus the construction segment map, not hardcoded.

### Scan-detail ML availability — `ml_mask_count`

`GET /api/scans/{scan_id}` carries an additive field `ml_mask_count`
(`int`): the count of `cosmic_ray_masks` rows stored for the scan. The
frontend gates the despike selector's **ML** option on `ml_mask_count > 0`.
It is `0` for a mask-less scan, and `0` (never a 500) against a pre-migration
database with no `cosmic_ray_masks` table — mirroring the tolerance of the
stored-mask read path.

`POST /api/process/despike` (the interactive modz endpoint) is unchanged and
does not gain a method selector.

---

## Web HTTP API — Known-noisy channel annotation (issue #9)

`GET /api/spectra/badpix?region=R1|R2|R3|R123`

Returns the curated list of **known-noisy detector channels** for a region
view, as an analyst-visible annotation layer. This is a **static** list — it
takes **no scan id**, touches **no database**, and has **no interaction with
despiking whatsoever**. It is an annotation surface only: it never masks,
replaces, or alters any spectral value. Some channels can carry both real
mineral signal **and** intrinsic detector noise; cosmic-ray despiking must
never touch a real spectral band, yet the analyst can still surface
"known-noisy channel" on demand from this independent annotation layer.

`region` defaults to `R1`; an invalid region returns **400** (matching the
spectra endpoints' invalid-region contract). `schema_version` stays `1.0.0`
(additive endpoint).

### Response

```json
{
  "schema_version": "1.0.0",
  "region": "R1",
  "n_channels": 523,
  "badpix": [
    { "position": 2, "channel": 54, "tier": 1, "source": "dark_veto" }
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `region` | str | the requested region view |
| `n_channels` | int | length of the served array for that region |
| `badpix[].position` | int | **0-based position into the served `wavenumber`/`intensity` arrays** for this region — the same coordinate system as `masked_positions`, so a marker at `position` lands on `wavenumber[position]`. For `R123` the served array is the full 2148-channel stitch, so `position == channel` |
| `badpix[].channel` | int | absolute CCD channel index, `0..2147` |
| `badpix[].tier` | int | `1` = CR-confusable elevated-noise channel; `2` = stable hot pixel that cancels in dark subtraction |
| `badpix[].source` | str | attribution: `dark_veto` (dark-plane-confirmed defect), `jb25` (published mission bad-pixel table), or `both` |

Channels outside the requested region's served selection are omitted (e.g. an
R2 channel is not returned for a `region=R1` view). The list is sorted by
served position. The frontend fetches it once per region view (cached
client-side) and renders it as a separate, default-OFF overlay with a distinct
glyph (hollow blue diamonds) from the red-triangle ML/modz spike markers.

---

## See Also

- [README.md](../README.md) - CLI usage and examples
- [METHODS.md](METHODS.md) - Cosmic-ray detection (ML, v1.1) and coverage limits
- [notebooks/spectral_analysis_example.ipynb](../notebooks/spectral_analysis_example.ipynb) - Complete notebook examples
- [CHANGELOG.md](../CHANGELOG.md) - Version history
