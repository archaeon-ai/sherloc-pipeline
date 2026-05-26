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
    r_squared: float          # R² goodness of fit
    rss: float                # Residual sum of squares
    dof: int                  # Degrees of freedom
    warnings: List[str]       # Any fitting warnings

@dataclass
class PeakFit:
    m_cm1: float              # Peak center (cm⁻¹)
    fwhm: float               # Full width at half maximum
    amplitude: float          # Peak height
    is_accepted: bool         # Passed quality criteria
    reject_reason: str        # Rejection reason if not accepted
```

**Example:**

```python
df, fit = process_scan_average(
    sol="0921", target="Amherst_Point", scan="detail_1",
    fit=True
)

if fit:
    print(f"R² = {fit.r_squared:.4f}")
    print(f"Found {len(fit.peaks)} peaks:")
    for peak in fit.peaks:
        status = "✓" if peak.is_accepted else f"✗ ({peak.reject_reason})"
        print(f"  {peak.m_cm1:.1f} cm⁻¹, FWHM={peak.fwhm:.1f} {status}")
```

---

## See Also

- [README.md](../README.md) - CLI usage and examples
- [notebooks/spectral_analysis_example.ipynb](../notebooks/spectral_analysis_example.ipynb) - Complete notebook examples
- [CHANGELOG.md](../CHANGELOG.md) - Version history

