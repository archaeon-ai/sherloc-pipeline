# Spectral Averaging Methods

> How the pipeline computes averaged spectra from per-point measurements, including
> cosmic-ray rejection via trimmed mean.

## Overview

SHERLOC acquires spectra at multiple scan points per target. Averaged spectra (used
for preprocessing, plotting, and quality assessment) are computed from these per-point
measurements. The pipeline supports three averaging methods:

| Method | CLI flag | Description |
|--------|----------|-------------|
| **Mean** | `--avg mean` | Arithmetic mean across points |
| **Median** | `--avg median` | Median across points |
| **Trimmed mean** | `--avg trim-mean` (default) | Robust mean with tail trimming for cosmic-ray rejection |

## Trimmed Mean (Default)

The default averaging method is `trim-mean` with `trim_pct=2` (2% per tail), using
`scipy.stats.trim_mean(data, proportiontocut)`.

### How scipy.stats.trim_mean works

Given `n` data points and `proportiontocut = p`:

1. Compute `m = int(p * n)` — number of points to remove from **each** tail
2. Sort the data
3. Remove the lowest `m` and highest `m` values
4. Return the mean of the remaining `n - 2m` values

This is the same per-tail convention used by Loupe (verified against Loupe V5.1.7).

### The small-scan problem

For the default `proportiontocut = 0.02`:

| n_points | m = int(0.02 * n) | Effect |
|----------|-------------------|--------|
| 100 | 2 | 2 points trimmed per tail — effective CR rejection |
| 50 | 1 | 1 point trimmed per tail — minimal but sufficient |
| 25 | **0** | **No trimming — cosmic rays pass through** |
| 10 | **0** | **No trimming** |

Line scans (25 points), some HDR scans (33-34 points), and other small-n scans get
**zero trimming** with the fixed 2% proportion.

### Dynamic trim proportion

The pipeline uses `resolve_trim_proportion()` to guarantee at least 1 point is trimmed
from each tail, regardless of scan size:

```python
def resolve_trim_proportion(n_points: int, baseline_pct: float = 0.02) -> float:
    if n_points < 3 or baseline_pct <= 0.0:
        return baseline_pct  # can't meaningfully trim, or explicit no-trim request
    return max(baseline_pct, (1.0 + 1e-9) / n_points)
```

When `baseline_pct = 0.0` (explicit no-trim request via `--trim-pct 0`), the function
returns 0.0 unchanged — the dynamic floor only applies to positive baselines.

The result for all scan sizes:

| n_points | Baseline (0.02) | Effective | m per tail | Behavior |
|----------|----------------|-----------|------------|----------|
| 100 | 0.02 | 0.02 | 2 | Unchanged from baseline |
| 50 | 0.02 | 0.02 | 1 | Unchanged (baseline already sufficient) |
| 25 | 0.02 | ~0.04 | 1 | **Fixed**: 1 point trimmed per tail |
| 10 | 0.02 | ~0.10 | 1 | **Fixed**: 1 point trimmed per tail |
| 3 | 0.02 | ~0.33 | 1 | Takes median (1 trimmed from each side of 3) |
| 2 | 0.02 | 0.02 (m=0) | 0 | Can't trim — plain mean (only sensible option) |
| 1 | 0.02 | 0.02 (m=0) | 0 | Single point — no averaging |

**Threshold:** For `baseline_pct = 0.02`, scans with >= 51 points are unaffected.
Only scans with < 51 points get dynamic adjustment.

### IEEE 754 epsilon

The formula uses `(1.0 + 1e-9) / n` rather than `1.0 / n` because naive division
fails for 82 out of 998 integer values in [3, 1000] due to floating-point truncation.
For example, `int(1.0/49 * 49) = 0` because `1.0/49 * 49` evaluates to
`0.9999999999999999` in IEEE 754 double precision. The epsilon ensures the product
always exceeds 1.0.

## Output labeling

### Filenames

Output filenames use `2p_trim_mean` as a **semantic label** for the method, not a
literal encoding of the scipy parameter. This is consistent regardless of whether
dynamic adjustment was applied:

```
0921_Amherst_Point_detail_1_R1_raw-n_2p_trim_mean_bkgsub_baselined.csv
0921_Amherst_Point_detail_1_R1_avg-2p_trim_mean_fs_baselined_fit.png
```

### Plot titles

Plot titles include the method label:

```
sol 0921 Amherst_Point detail_1 R1 avg 2p_trim_mean fs baselined
```

### JSON metadata

The exported JSON metadata includes the full averaging details, including the
**effective** proportion when dynamic adjustment was applied:

```json
{
  "averaging": {
    "method": "trim-mean",
    "trim_pct": 2.0,
    "effective_pct_per_tail": 4.0,
    "n_points_averaged": 25,
    "m_trimmed_per_tail": 1
  }
}
```

For scans where no dynamic adjustment was needed (n >= 51 with default baseline):

```json
{
  "averaging": {
    "method": "trim-mean",
    "trim_pct": 2.0,
    "effective_pct_per_tail": 2.0,
    "n_points_averaged": 100,
    "m_trimmed_per_tail": 2
  }
}
```

### Log messages

When dynamic adjustment occurs, an INFO-level log message is emitted:

```
Trim mean: dynamic adjustment for 25 points (baseline 2.0% → effective 4.0% per tail)
```

## Configuration

The baseline proportion is configurable in `config.yaml`:

```yaml
preprocessing:
  trim_mean_baseline_pct: 0.02   # per-tail proportion for scipy.stats.trim_mean
```

## Where trim mean is computed

| Location | Context |
|----------|---------|
| `core/data_ingestion.py` `calculate_average_spectrum()` | Initial per-scan averaging during ingestion (handles NaN per-row) |
| `services/spectral.py` `_compute_average()` | Raman averaged/subset spectrum for plotting |
| `services/spectral.py` `_compute_fluor_average()` | Fluorescence averaged/subset spectrum for plotting |

All three call sites use `resolve_trim_proportion()` from `core/utils.py`.

## Affected scans in the database

| n_points | Count | Scan type | Status |
|----------|-------|-----------|--------|
| 25 | 18 | Line scans | **Previously zero-trimming; now fixed** |
| 33 | 16 | HDR scans | **Previously zero-trimming; now fixed** |
| 34 | 8 | HDR scans | **Previously zero-trimming; now fixed** |
| 50 | 137 | Various | Unaffected (m=1 at baseline) |
| 100 | ~600 | Detail scans | Unaffected (m=2 at baseline) |

Engineering scans (1 point each, n=473) are never averaged in the pipeline.
