# Methods & Reproducibility

**SHERLOC Pipeline Analysis Framework**
**Version:** 5.0.0
**Date:** 2026-01-25
**Pipeline SHA:** `git log -1 --format="%H"`

> **Freshness note.** Dataset counts and experimental result tables in this document describe an earlier codebase and database state (cut-off 2026-01). Algorithm parameters and citation entries were re-audited against `src/sherloc_pipeline/config.yaml` and Crossref. For current database statistics, run `sherloc db-stats`. A refresh of the experimental sections is planned but not blocking.

---

## Table of Contents

1. [Data Provenance](#1-data-provenance)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)
3. [Peak Fitting Methods](#3-peak-fitting-methods)
4. [Machine Learning Methods](#4-machine-learning-methods)
5. [Vision Processing Methods](#5-vision-processing-methods)
6. [Statistical Analysis](#6-statistical-analysis)
7. [Software Environment](#7-software-environment)
10. [Reproducibility Guidelines](#10-reproducibility-guidelines)
11. [References](#11-references)

---

## 1. Data Provenance

### 1.1 Dataset Overview

The SHERLOC (Scanning Habitable Environments with Raman & Luminescence for Organics & Chemicals) instrument aboard the Mars 2020 Perseverance rover has collected spectroscopic data from Martian surface materials in Jezero Crater since Sol 58.

**Dataset Statistics (snapshot, 2026-01):**

The values below describe the dataset as it existed at the 2026-01 cut-off and are the basis for the experimental results discussed later in this document. Run `sherloc db-stats` (or `sqlite3 ./phase.db ".tables"`) for current counts on the live database.

- **Total R1 Spectra:** 524,181
- **Total Scans:** 981
- **Total Sols:** 152
- **Mars Target Scans:** 648 (66.1%)
- **Calibration Scans:** 333 (33.9%)
- **Detail Scans (100 points):** 344
- **Database Size:** 2.22 GB
- **Collection Period:** Sol 58 - Sol 1266

### 1.2 SHERLOC Instrument Specifications

Drawn from Bhartia et al. (2021) and the codebase's Loupe-derived calibration in `src/sherloc_pipeline/config.yaml`. See `docs/schema/SPECTRAL_REGIONS.md` for the authoritative wavelength↔wavenumber mapping.

| Parameter | Specification |
|-----------|---------------|
| **Laser** | Pulsed deep-UV neon-copper, 248.6 nm (40 μs pulses) |
| **Spectral Regions** | R1 (Raman), R2/R3 (Fluorescence) |
| **R1 Range** | 250-282 nm; usable Raman shift ~640-4200 cm⁻¹ |
| **R2 Range** | 282-337.8 nm |
| **R3 Range** | 337.8-357.4 nm |
| **Spot Size** | ~100 μm diameter |
| **Working Distance** | ~48 mm (Bhartia et al. 2021) |

### 1.3 Data Sources and Processing Levels

**Primary Data Source:** SHERLOC Loupe v5.1.5 format files [@Bhartia2021; @Loupe2022]
- **Format:** Proprietary binary with embedded metadata
- **Processing Level:** EDR (Experimental Data Record) to RDR (Reduced Data Record)
- **Calibration:** Dark subtraction, flat-field correction applied

**Scan Types:**
1. **Point Scans:** Single-point measurements (1-10 points)
2. **Line Scans:** Linear transects (10-50 points)
3. **Detail Scans:** High-resolution grids (10×10 = 100 points)
4. **Calibration Scans:** Standard materials (AlGaN, Teflon, Polycarbonate)

**Acquisition Parameters:**
- **Pulses per Point (PPP):** 25-1000 (median: 100)
- **Integration Time:** Variable (5-60 seconds per point)
- **Laser Energy:** ~15-50 μJ per pulse
- **Atmospheric Pressure:** ~6-8 mbar (CO₂ dominated)

### 1.4 Target Context Integration

**Target Field Enhancement:**
- **Geological Context:** Added target field to 967/981 scans (98.6%)
- **Target Categories:** Rock outcrops, regolith, drill tailings, calibration materials
- **Spatial Context:** ACI imagery linked via SCLK timestamps (±5-60s tolerance)

---

## 2. Preprocessing Pipeline

The preprocessing pipeline transforms raw Loupe format data into analysis-ready spectral arrays through a series of validated processing steps.

### 2.1 Data Ingestion and Validation

**Format Parsing:**
```python
# Loupe format parsing with validation
def parse_loupe_scan(file_path: Path) -> List[Spectrum]:
    """Parse Loupe binary format to Pydantic Spectrum models."""
    # Binary structure parsing with CRC validation
    # Metadata extraction (sol, target, coordinates)
    # Spectral array reconstruction
```

**Quality Control Checks:**
- Header CRC validation
- Spectral array completeness
- Wavelength calibration verification
- Metadata consistency checks

### 2.2 Cosmic Ray and Spike Removal

**Algorithm:** Robust rolling-median residual with MAD-derived threshold [@Whitaker2018]

**Parameters (`DespikeParams`):**
```python
window_size: int = 7              # Rolling window size
zscore_threshold: float = 6.0     # Robust z-score cutoff
max_iterations: int = 1           # Despiking passes
interpolation_method: str = "linear"
run_length_max: int = 2           # Maximum consecutive spikes
```

**Protected Regions:**
- **Laser line:** 600-700 cm⁻¹ (excluded from despiking)
- **Sulfate guard:** 990-1050 cm⁻¹ (conditional protection for genuine peaks)

**Implementation:**
```python
def despike_r1_spectrum(intensity_series: pd.Series,
                       params: DespikeParams) -> pd.Series:
    """Remove cosmic ray spikes using robust statistics."""
    # Compute rolling median and MAD
    median = intensity_series.rolling(window=params.window_size,
                                    center=True).median()
    residual = intensity_series - median
    mad = np.median(np.abs(residual - np.median(residual)))

    # Flag outliers exceeding z-score threshold
    threshold = params.zscore_threshold * 1.4826 * mad
    spikes = np.abs(residual) > threshold

    # Apply run-length and exclusion constraints
    # Interpolate flagged spikes
```

### 2.2.1 Cosmic-Ray Detection and Removal — ML Detector (v1.3)

**Overview.** Starting with pipeline v4.2.0, the default despike method is `ml`
(`despike_method ∈ {ml, modz, none}`). The ML detector replaces the legacy modz
algorithm as the default because, on held-out test data, it has a roughly 5×
lower false-flag (false-positive) rate and better cosmic-ray recall at every
amplitude, while leaving genuine mineral and organic bands intact. Detection runs
at processing time, producing per-spectrum channel masks that are persisted in the
database; the web server applies the stored masks by simple interpolation and
never runs model inference.

**Install.** The ONNX runtime is an optional dependency in the `[ml-despike]`
extra:

```bash
pip install 'sherloc-pipeline[ml-despike]'
```

`none` and `modz` paths require no extra and never import `onnxruntime`.

#### Input contract

The detector's input is the **raw paired ACTIVE and DARK
planes**, 2148 channels × 3 regions (R1/R2/R3), in raw DN — as read from the
Loupe workspace files `activeSpectra.csv` and `darkSpectra.csv`. Per-frame
robust normalization happens inside featurization, not upstream. Inputs must be
1-D float arrays of length 2148.

#### Featurization

Reference implementation: `src/sherloc_pipeline/ml_despike/featurize.py`.

Per plane, over the detection window `[lo, hi)` for the frame's
region:

```
med   = median(plane[lo:hi])
mad   = median(|plane[lo:hi] − med|)
scale = 1.4826 · mad + 1.0
```

The 8 input channels (float32, shape `(8, 2148)`):

| Channel | Formula |
|---------|---------|
| `x0` | `(active − med_a) / scale_a` |
| `x1` | `(dark − med_d) / scale_d` |
| `x2..x4` | Region one-hot broadcast (R1/R2/R3) |
| `x5` | `log10(scale_a) / 4.0` |
| `x6` | `log10(scale_d) / 4.0` |
| `x7` | `log10(1 + |median(active[lo:hi])|) / 4.0` |

`featurize_batch` stacks per-frame features to shape `(N, 8, 2148)`.

#### Frozen operating point

The operating point is frozen in-code in
`src/sherloc_pipeline/ml_despike/manifest.py` and is not user-tunable. The
thresholds are rate-matched per region and fixed for the released v1.3 model
(provenance label `ml_v1.3_tau_matched`).

**Per-region decision thresholds (taus):**

| Region | tau (full precision) |
|--------|----------------------|
| R1 | `0.29882812500038747` |
| R2 | `0.2656250000008831` |
| R3 | `0.2656250000008831` |

R2 and R3 share the fluorescence tau.

**Detection windows `[lo, hi)`:**

| Region | Window |
|--------|--------|
| R1 | `(52, 575)` |
| R2 | `(575, 1677)` |
| R3 | `(1677, 2140)` |

A channel is flagged when `p > tau[region]` (strict) inside the window, where
`p = sigmoid(logits)` is computed in float64
(`1 / (1 + exp(−logits.astype(float64)))`). Flags are returned as absolute
channel indices (0–2147).

#### Artifact identity

The fp32 ONNX deployment artifact is the sole artifact executed by the
pipeline:

| Item | Value |
|------|-------|
| Filename | `v1_stageB_v13c.onnx` |
| ONNX sha256 | `9668a0b2ca257ce333d57e3f76598dda8cb5c1839e2fde6bd955086d959be0ba` |
| Source checkpoint sha256 | `a77cd435d65631a8728c9d39c01c31dd30805ac37062b8c48937be6fb3594881` |
| Fetch URL | `https://github.com/archaeon-ai/sherloc-pipeline/releases/download/model-cr-despike-1.3/v1_stageB_v13c.onnx` |
| Cache path | `~/.cache/sherloc-pipeline/ml_despike/v1_stageB_v13c.onnx` |

The source checkpoint sha256 is recorded for provenance only; the checkpoint is
never fetched or loaded by the pipeline. The fetch is sha256-pinned: the digest
is verified before the file reaches its loadable cache name; mismatches are
quarantined (renamed to `<name>.corrupt-<ts>`), never loaded.

#### Provenance semantics

Every run records the following provenance, in pipeline metadata,
`ServiceResult.metadata["despike"]`, and the `cr_masks.json` run artifact:

```json
{
  "method": "ml_v1.3_tau_matched",
  "model_sha256": "9668a0b2ca257ce333d57e3f76598dda8cb5c1839e2fde6bd955086d959be0ba",
  "tau": {
    "R1": 0.29882812500038747,
    "R2": 0.2656250000008831,
    "R3": 0.2656250000008831
  },
  "ort_version": "<installed onnxruntime version>"
}
```

The installed `onnxruntime` version is recorded (not the reference version) so
any flag discrepancy under a later runtime is attributable. Masks persisted to
the `cosmic_ray_masks` database table carry the same provenance trio
(`method`, `model_sha256`, `tau`).

#### Reference configuration vs installed configuration

The reference runtime configuration is **onnxruntime 1.26.0,
CPUExecutionProvider, fp32, opset 18**. Later 1.x onnxruntime versions are
operationally acceptable but not the reference configuration; the installed version is recorded in
run provenance as noted above. Changing the reference-configuration line in this
document requires a recorded local parity re-run (zero symmetric difference on
the golden batch via `scripts/verify_ml_despike_parity.py`) — this is an
integration parity check, not a model re-evaluation.

#### Coverage limits

The ML detector flags channels only within its detection windows
(§ Frozen operating point above). In combined representations the uncovered
contributions — channels where a contributing region's detection window does not
include the channel — retain exactly the legacy never-screened behavior. (The
legacy modz method never despiked the fluorescence or R123 representations at
all, so every covered channel is a strict improvement.)

Coverage for the Loupe R123 summation view (derived from
`core/r123_stitching.py` segment map and `DEFAULT_MANIFEST.region_windows`):

| Channels | Contributors | Covered | Uncovered |
|----------|-------------|---------|-----------|
| `[0, 52)` | {R1} | — | R1 (below window) |
| `[52, 565)` | {R1} | R1 | — |
| `[565, 575)` | {R1, R2} | R1 | R2 (below window) |
| `[575, 690)` | {R1, R2} | R2 | R1 (above window) |
| `[690, 1668)` | {R2} | R2 | — |
| `[1668, 1677)` | {R2, R3} | R2 | R3 (below window) |
| `[1677, 1690)` | {R2, R3} | R3 | R2 (above window) |
| `[1690, 2140)` | {R3} | R3 | — |
| `[2140, 2148)` | {R3} | — | R3 (above window) |

Derived uncovered-contributor counts: **207 of 2148** channels carry at least
one uncovered contributor in the R123 summation view (147 overlap channels + 60
edge channels). The fluorescence full-plane sum (R2 + R3 across all 2148
channels) carries an uncovered contributor at **every** channel (2148).

Note that `services/fitting.py` and `services/map_fitting.py` R123 consumers
do **not** receive ML-despiked data in v1.

#### Characterized behavior summary

Two science-favorable performance trades are characterized on held-out test data:

- **High pulses-per-point, fluorescence:** at high PPP the ML detector trades
  some fluorescence weak-mid recall for a 2–5× lower false-alarm rate relative
  to the legacy method.
- **R1 weak-dip sparing:** the detector exhibits conservative sparing of weak
  dip features in R1.

Genuine mineral and organic bands were left intact on the held-out test data, and
the false-flag (false-positive) rate is roughly 5× lower than the legacy method.

#### Adaptation guidance for power users

To run the detector standalone on custom active/dark plane pairs:

```python
from sherloc_pipeline.ml_despike import (
    MLCRDetector,
    DEFAULT_MANIFEST,
    resolve_artifact,
    featurize_batch,
)

# Resolves and digest-verifies the cached artifact (fetches if absent).
detector = MLCRDetector()

# actives, darks: lists of 1-D numpy arrays, length 2148, raw DN.
# regions: list of "R1", "R2", or "R3" labels, one per frame.
masks = detector.detect(actives, darks, regions)
# masks: list of int64 arrays — absolute channel indices (0–2147) per frame.
```

The `detect` method signature is:

```python
def detect(
    self,
    actives: Sequence[np.ndarray],   # raw ACTIVE planes, 1-D length-2148
    darks: Sequence[np.ndarray],     # raw DARK planes, parallel to actives
    regions: Sequence[str],          # "R1" / "R2" / "R3", one per frame
) -> List[np.ndarray]:              # sorted absolute channel indices per frame
```

`MLCRDetector.__init__` accepts optional arguments:

```python
MLCRDetector(
    manifest=DEFAULT_MANIFEST,   # frozen identity; use DEFAULT_MANIFEST
    artifact_path=None,          # explicit artifact path (digest-verified); default fetches/caches
    intra_op_threads=2,          # ORT intra-op thread count; reference config is 2
)
```

The `featurize_batch` function is available for inspection or testing without
invoking inference:

```python
features = featurize_batch(actives, darks, regions)
# features: (N, 8, 2148) float32
```

Returned masks are absolute channel indices (0–2147) on each region's
2148-channel plane. The `none` and `modz` methods require no extra and produce
no masks from this package.

### 2.2.2 Known-Noisy Channel Annotation (distinct from cosmic-ray despiking)

Some individual detector channels are intrinsically noisier than their
neighbors — a fixed property of the CCD, separate from the transient
cosmic-ray hits that despiking targets. The pipeline ships a curated table of
these **known-noisy channels** (`data/badpix_channels.csv`) and exposes them as
an analyst-visible **annotation layer**, served by `GET /api/spectra/badpix`.
This layer is **completely separate from cosmic-ray despiking**: it never
masks, replaces, interpolates, or alters any spectral value — it only marks
where the known-noisy channels fall, on demand, when the analyst opts in.

The distinction matters because a channel can be *both* real mineral signal and
intrinsically noisy. The clearest case is the carbonate ν₁ apex near
1086.7 cm⁻¹: the spectral feature there is genuine carbonate signal, yet that
same channel is also a known-noisy channel. Cosmic-ray despiking must **never**
touch it — masking it would erase real mineral information — so the despike
detector is deliberately built to leave such channels alone. The annotation
layer fills the complementary need: the analyst can still toggle on a
"known-noisy channel" marker to keep that channel's noise character in view
while interpreting the carbonate band, without any risk of the value being
modified.

The curated table carries two tiers. **Tier 1** channels are elevated-noise
channels whose excursions a cosmic-ray detector could plausibly confuse for
real hits (the operationally relevant set for analysis caution). **Tier 2**
channels are stable hot pixels that cancel out under dark-frame subtraction;
they are retained for raw-plane analyses. Each entry also records its
attribution source — a detector characterization over the mission spectral
corpus, the published mission bad-pixel table (Jakubek et al. 2025), or both.

---

### 2.3 Baseline Correction

**Primary Method:** Adaptive smoothness penalized least squares (aspls) [@Zhang2020], as implemented by `pybaselines.Baseline.aspls` [@PyBaselines2022].

**Parameters (`BaselineParams`, defaults from `src/sherloc_pipeline/config.yaml`):**
```python
lam: float = 1.0e6         # Smoothness parameter (higher = smoother)
asymmetric_coef: float = 0.01  # Asymmetry coefficient
iters: int = 10            # Maximum iterations
diff_order: int = 2        # Difference operator order
tol: float = 0.001         # Convergence tolerance
```

The aspls algorithm adapts the smoothness penalty along the spectrum, allowing flatter regions to receive stronger smoothing than peak-rich regions. R1 fits also apply soft "keep windows" (configurable in `config.yaml`) that down-weight the baseline within ranges where genuine peaks are expected, reducing under-fit at strong features.

### 2.4 Normalization Strategies

**Vector Normalization (Default):**
```python
def l2_normalize(spectrum: np.ndarray) -> np.ndarray:
    """L2 normalization to unit vector length."""
    return spectrum / np.linalg.norm(spectrum)
```

**Alternative Normalization Methods:**
- **Min-Max:** Scale to [0, 1] range
- **Standard:** Zero mean, unit variance
- **Robust:** Median centering, MAD scaling
- **Peak:** Normalize to highest peak intensity

---

## 3. Peak Fitting Methods

### 3.1 Multi-Gaussian Fitting

**Model:** Sum of independent Gaussian peaks
```python
def gaussian(x: np.ndarray, center: float, amplitude: float,
             fwhm: float) -> np.ndarray:
    """Single Gaussian peak model."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

def multi_gaussian(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Sum of Gaussians: params = [m1,a1,f1, m2,a2,f2, ...]"""
    y = np.zeros_like(x)
    for i in range(0, params.size, 3):
        center, amplitude, fwhm = params[i:i+3]
        y += gaussian(x, center, amplitude, fwhm)
    return y
```

### 3.2 Peak Detection and Initial Estimates

**Peak Detection:** SciPy `find_peaks` with adaptive thresholds
```python
def detect_peaks(spectrum: np.ndarray, wavenumber: np.ndarray,
                prominence_factor: float = 0.1) -> List[int]:
    """Detect peak candidates using adaptive prominence."""
    baseline_noise = np.std(spectrum[:50])  # Estimate from edges
    min_prominence = max(prominence_factor * np.max(spectrum),
                        3 * baseline_noise)

    peaks, properties = find_peaks(spectrum,
                                  prominence=min_prominence,
                                  width=2,  # Minimum width in channels
                                  distance=5)  # Minimum separation
    return peaks
```

### 3.3 Nonlinear Optimization

**Method:** Trust Region Reflective (scipy.optimize.least_squares)
```python
def fit_peaks(wavenumber: np.ndarray, intensity: np.ndarray,
              initial_params: np.ndarray) -> FitResult:
    """Fit multiple Gaussian peaks to spectrum."""

    def residuals(params):
        model = multi_gaussian(wavenumber, params)
        return intensity - model

    # Parameter bounds: [center_min, amplitude_min, fwhm_min, ...]
    bounds = build_parameter_bounds(initial_params, wavenumber)

    result = least_squares(residuals, initial_params,
                          bounds=bounds, method='trf',
                          ftol=1e-12, xtol=1e-12, gtol=1e-12)

    return build_fit_result(result, wavenumber, intensity)
```

### 3.4 Model Selection and Quality Metrics

Two model-selection strategies are supported for choosing the number of Gaussian components: AICc (the default in the bundled `config.yaml`) and a sequential F-test. Both operate on the same candidate seeds and the same nonlinear least-squares fit; they differ only in the rule used to stop adding peaks. The active strategy is set by `fitting.model_selection` in `config.yaml` (`"aicc"` or `"ftest"`).

**Sequential F-test:**

Adds peaks one at a time starting from the null model (y=0). After each addition, computes the F-statistic for the nested-model comparison:

```
F = ((RSS_n − RSS_{n+1}) / Δk) / (RSS_{n+1} / dof_{n+1})
```

with Δk=3 (each Gaussian adds m, a, fwhm) and dof = N − 3·peaks. The F-distribution gives a p-value; if `p < ftest_alpha` (default 0.01) the new peak is retained and the search continues. **The first non-significant addition halts the search** (greedy stop). Strict statistical control over the false-positive rate per addition; assumes Gaussian residuals.

```python
def f_test_pvalue(rss_reduced: float, rss_full: float,
                  dof_reduced: int, dof_full: int,
                  delta_params: int) -> float:
    """p-value for nested-model F-test (Δk=3 per Gaussian)."""
    f_stat = ((rss_reduced - rss_full) / delta_params) / (rss_full / dof_full)
    return 1.0 - f_dist.cdf(f_stat, delta_params, dof_full)
```

**Corrected Akaike Information Criterion (AICc, default):**

Fits every candidate peak count from `aicc_min` (default 1) to `aicc_max` and picks the one minimizing AICc. No threshold; balances goodness-of-fit against parameter count via the +2k penalty term plus a small-sample correction.

```python
def compute_aicc(n_samples: int, rss: float, num_params: int) -> float:
    """Compute AICc for model selection."""
    aic = n_samples * np.log(rss / n_samples) + 2 * num_params
    correction = 2 * num_params * (num_params + 1) / (n_samples - num_params - 1)
    return aic + correction
```

For fluorescence fitting (`domain="fluorescence"`), AICc is used regardless of UI selection — see §3.5 of `docs/specs/FLUORESCENCE_FITTING_SPEC.md`.

**Quality Thresholds (defaults from `src/sherloc_pipeline/config.yaml`, minerals domain):**
- **R²:** ≥ 0.25 (`r_squared_min`)
- **SNR:** ≥ 3.0 (`min_snr`)
- **FWHM:** initial-search lower bound 22 cm⁻¹ (`fit_fwhm_min_initial_cm1`), upper bound 90 cm⁻¹ (`fwhm_max_cm1`); post-fit filter ≥ 30 cm⁻¹ (`filter_fwhm_min_cm1`); reviewable/persist eligibility ≥ 25 cm⁻¹ (`reviewable_fwhm_min_cm1`)

Domain overrides (organics, hydration, fluorescence) are defined in the same `fitting:` block of `config.yaml` and may relax or tighten these defaults. The fluorescence `posthoc_filters` block sets `r2_min: 0.0` and `fwhm_min_cm1: 0.0` — see §3.5 of `docs/specs/FLUORESCENCE_FITTING_SPEC.md` for the rationale.

**References:**
- F-test for nested models: standard derivation; implementation in `core/fitting.py:_f_test_pvalue`.
- AICc: Burnham & Anderson (2002), *Model Selection and Multimodel Inference*; implementation in `core/fitting.py:_compute_aicc`.

---

## 4. Machine Learning Methods

### 4.1 Clustering Analysis

#### 4.1.1 K-Means Clustering

**Implementation:** scikit-learn KMeans with Lloyd's algorithm
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def cluster_spectra(spectra: np.ndarray, n_clusters: int) -> ClusteringResult:
    """Perform K-means clustering on spectral data."""
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(spectra)

    # Optional dimensionality reduction
    pca = PCA(n_components=50, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters,
                   random_state=42,
                   n_init=10,
                   max_iter=300,
                   tol=1e-4)
    labels = kmeans.fit_predict(X_reduced)

    # Quality metrics
    silhouette = silhouette_score(X_reduced, labels)

    return ClusteringResult(labels=labels,
                          silhouette_score=silhouette,
                          cluster_centers=kmeans.cluster_centers_)
```

**Hyperparameters (optimal):**
- **n_clusters:** 5 (silhouette-optimized)
- **init:** 'k-means++'
- **n_init:** 10
- **random_state:** 42 (reproducibility)
- **algorithm:** 'lloyd'

#### 4.1.2 DBSCAN (Density-Based Clustering)

**Implementation:** Scikit-learn DBSCAN with epsilon-neighborhood
```python
from sklearn.cluster import DBSCAN

def dbscan_cluster(spectra: np.ndarray, eps: float,
                   min_samples: int) -> ClusteringResult:
    """Density-based clustering for outlier detection."""
    dbscan = DBSCAN(eps=eps,
                   min_samples=min_samples,
                   metric='euclidean',
                   n_jobs=-1)
    labels = dbscan.fit_predict(spectra)

    # Separate core samples from noise (-1 labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    return ClusteringResult(labels=labels,
                          n_clusters=n_clusters,
                          noise_points=n_noise)
```

**Hyperparameters:**
- **eps:** 0.1284 (optimized via k-distance plot)
- **min_samples:** 5 (2D rule of thumb: 2×dimensions)
- **metric:** 'euclidean'

### 4.2 Classification Methods

#### 4.2.1 Calibration vs. Mars Classification

**Best Model:** Gradient Boosting Classifier (96.95% accuracy)
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def train_calibration_classifier(spectra: np.ndarray,
                                labels: np.ndarray) -> GradientBoostingClassifier:
    """Train binary classifier for calibration vs. Mars scans."""
    # Preprocessing pipeline
    scaler = StandardScaler()
    pca = PCA(n_components=50, random_state=42)

    X_scaled = scaler.fit_transform(spectra)
    X_reduced = pca.fit_transform(X_scaled)

    # Gradient boosting classifier
    gb_classifier = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10
    )

    # Cross-validation
    cv_scores = cross_val_score(gb_classifier, X_reduced, labels,
                               cv=5, scoring='accuracy')

    gb_classifier.fit(X_reduced, labels)
    return gb_classifier
```

**Model Comparison Results:**
| Algorithm | Test Accuracy | CV Mean ± Std | Precision | Recall | F1 Score |
|-----------|---------------|---------------|-----------|--------|----------|
| **GradientBoosting** | **96.95%** | **96.17 ± 1.2%** | **0.980** | **0.962** | **0.971** |
| RandomForest | 96.45% | 95.41 ± 1.8% | 0.980 | 0.952 | 0.966 |
| LogisticRegression | 91.37% | 93.62 ± 2.1% | 0.885 | 0.962 | 0.922 |
| SVM (RBF) | 90.86% | 91.71 ± 2.4% | 0.884 | 0.952 | 0.917 |

### 4.3 Dimensionality Reduction

#### 4.3.1 Principal Component Analysis (PCA)

**Implementation:** Incremental PCA for large datasets
```python
from sklearn.decomposition import PCA, IncrementalPCA

def apply_pca(spectra: np.ndarray, n_components: int = 50) -> tuple:
    """Apply PCA with explained variance analysis."""

    # For large datasets, use incremental PCA
    if spectra.shape[0] > 10000:
        pca = IncrementalPCA(n_components=n_components)
        batch_size = 1000
        for i in range(0, spectra.shape[0], batch_size):
            batch = spectra[i:i+batch_size]
            pca.partial_fit(batch)
        X_reduced = pca.transform(spectra)
    else:
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(spectra)

    return X_reduced, pca.explained_variance_ratio_
```

**Results:**
- **Components:** 50
- **Explained Variance:** 17.7% (indicating high spectral dimensionality)
- **Cumulative Variance:** First 10 components explain 9.2%

#### 4.3.2 UMAP (Uniform Manifold Approximation)

**Implementation:** Three-method approach for comprehensive analysis
```python
import umap
from sklearn.model_selection import train_test_split

# Method 1: Standard UMAP (CPU)
def standard_umap(spectra: np.ndarray) -> np.ndarray:
    """Standard UMAP dimensionality reduction."""
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric='euclidean',
        random_state=42,
        n_jobs=-1
    )
    return reducer.fit_transform(spectra)

# Method 2: GPU UMAP (cuML/RAPIDS)
def gpu_umap(spectra: np.ndarray) -> np.ndarray:
    """GPU-accelerated UMAP using cuML."""
    from cuml.manifold import UMAP as cuUMAP

    reducer = cuUMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )
    return reducer.fit_transform(spectra)

# Method 3: Parametric UMAP (hybrid training)
def parametric_umap(spectra: np.ndarray, sample_size: int = 50000) -> tuple:
    """Parametric UMAP with neural network encoder."""
    from umap.parametric_umap import ParametricUMAP
    import tensorflow as tf

    # Stratified + diversity sampling for training
    train_spectra = hybrid_sample(spectra, sample_size)

    # Neural network architecture
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)  # 2D embedding
    ])

    reducer = ParametricUMAP(
        encoder=encoder,
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )

    # Train on subset, apply to full dataset
    reducer.fit(train_spectra)
    full_embedding = reducer.transform(spectra)

    return full_embedding, reducer
```

**UMAP Strategy:**
- **Standard UMAP:** Full 524K dataset, baseline 30-60 min
- **GPU UMAP:** Full 524K dataset, ~5-10 min with RTX 3090 Ti
- **Parametric UMAP:** 50K hybrid sample training, project full 524K

---

## 5. Vision Processing Methods

### 5.1 ACI Image Processing

#### 5.1.1 Image Format Support

**VICAR Format Parser:**
```python
def read_vicar_image(file_path: Path) -> tuple[np.ndarray, dict]:
    """Read VICAR format ACI images with metadata extraction."""
    with open(file_path, 'rb') as f:
        # Parse VICAR label
        label = parse_vicar_label(f)

        # Extract image dimensions and data type
        width = label['NS']  # Number of samples
        height = label['NL']  # Number of lines
        data_type = label['FORMAT']

        # Read binary image data
        f.seek(label['LBLSIZE'])
        image_data = np.frombuffer(f.read(), dtype=data_type)
        image = image_data.reshape(height, width)

    return image, label
```

**ACI Specifications:**
- **Sensor:** 1648 × 1200 pixels
- **Resolution:** 10.1 μm/pixel
- **Field of View:** 16.6 × 12.1 mm
- **Format:** 8-bit grayscale, uncompressed
- **Metadata:** VICAR/PDS3 labels with acquisition parameters

### 5.2 Grain Segmentation

#### 5.2.1 Segment Anything Model (SAM)

**Primary Model:** SAM ViT-B (Vision Transformer Base)
```python
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def setup_sam_segmenter() -> SamAutomaticMaskGenerator:
    """Initialize SAM ViT-B for grain segmentation."""
    sam_checkpoint = "models/sam_vit_b_01ec64.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,      # Grid density for prompt points
        pred_iou_thresh=0.86,    # Quality threshold
        stability_score_thresh=0.92,  # Stability threshold
        min_mask_region_area=100,     # ~10 μm² minimum area
        box_nms_thresh=0.7,     # Non-maximum suppression
        crop_n_layers=0,        # No crop/zoom layers
    )

    return mask_generator
```

**Performance Comparison (RTX 3090 Ti):**
| Model | Masks/Image | Time (s) | GPU Memory | Quality |
|-------|-------------|----------|------------|---------|
| **SAM ViT-B** | **54** | **2.2** | **2.7 GB** | **Best** |
| MobileSAM | 30 | 2.4 | 2.7 GB | Good |
| Watershed | 1140 | 0.4 | 0 GB | Baseline |

#### 5.2.2 Watershed Fallback

**Traditional Computer Vision:** Watershed segmentation for compatibility
```python
from skimage.segmentation import watershed
from skimage.feature import peak_local_maxima
from scipy.ndimage import distance_transform_edt

def watershed_segmentation(image: np.ndarray) -> np.ndarray:
    """Watershed-based grain segmentation fallback."""
    # Preprocessing
    blurred = gaussian_filter(image, sigma=1.0)

    # Distance transform for marker generation
    binary = blurred > threshold_otsu(blurred)
    distance = distance_transform_edt(binary)

    # Find local maxima as seeds
    local_maxima = peak_local_maxima(distance,
                                   min_distance=10,
                                   threshold_abs=5)
    markers = np.zeros_like(distance, dtype=int)
    markers[tuple(local_maxima.T)] = np.arange(1, len(local_maxima) + 1)

    # Watershed segmentation
    labels = watershed(-distance, markers, mask=binary)

    return labels
```

### 5.3 Grain Morphometry

#### 5.3.1 Size Distribution Analysis

**Wentworth Size Classification:**
```python
from enum import Enum

class SizeClass(Enum):
    """Wentworth grain size classification for geological materials."""
    VERY_COARSE_SAND = (1000, 2000, "Very coarse sand")
    COARSE_SAND = (500, 1000, "Coarse sand")
    MEDIUM_SAND = (250, 500, "Medium sand")
    FINE_SAND = (125, 250, "Fine sand")
    VERY_FINE_SAND = (62.5, 125, "Very fine sand")
    COARSE_SILT = (31.25, 62.5, "Coarse silt")

def classify_grain_size(equivalent_diameter_um: float) -> SizeClass:
    """Classify grain size according to Wentworth scale."""
    for size_class in SizeClass:
        min_size, max_size, _ = size_class.value
        if min_size <= equivalent_diameter_um <= max_size:
            return size_class
    return SizeClass.VERY_FINE_SAND  # Default for smaller grains
```

#### 5.3.2 Shape Analysis

**Morphometric Parameters:**
```python
from skimage.measure import regionprops

def compute_grain_morphometry(mask: np.ndarray,
                            pixel_scale_um: float = 10.1) -> dict:
    """Compute comprehensive grain morphometry metrics."""
    props = regionprops(mask.astype(int))[0]

    # Size metrics
    area_pixels = props.area
    area_um2 = area_pixels * (pixel_scale_um ** 2)
    equivalent_diameter_um = 2 * np.sqrt(area_um2 / np.pi)

    # Shape metrics
    perimeter_pixels = props.perimeter
    perimeter_um = perimeter_pixels * pixel_scale_um

    # Circularity (4π×Area/Perimeter²)
    circularity = 4 * np.pi * area_pixels / (perimeter_pixels ** 2)

    # Aspect ratio (major/minor axis length)
    aspect_ratio = props.major_axis_length / props.minor_axis_length

    # Centroid coordinates
    centroid_y, centroid_x = props.centroid

    return {
        'area_um2': area_um2,
        'equivalent_diameter_um': equivalent_diameter_um,
        'perimeter_um': perimeter_um,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'centroid_x': centroid_x * pixel_scale_um,
        'centroid_y': centroid_y * pixel_scale_um,
        'major_axis_um': props.major_axis_length * pixel_scale_um,
        'minor_axis_um': props.minor_axis_length * pixel_scale_um
    }
```

**Morphometry Results (14,852 grains):**
| Metric | Mean | Median | Std Dev |
|--------|------|--------|---------|
| Equivalent Diameter (μm) | 870.5 | 599.8 | 1161.3 |
| Circularity | 0.480 | 0.480 | 0.192 |
| Aspect Ratio | 4.02 | 1.52 | 8.64 |

---

## 6. Statistical Analysis

### 6.1 Quality Metrics

#### 6.1.1 Clustering Validation

**Silhouette Analysis:**
```python
from sklearn.metrics import silhouette_score, silhouette_samples

def evaluate_clustering_quality(X: np.ndarray, labels: np.ndarray) -> dict:
    """Comprehensive clustering quality assessment."""
    # Global silhouette score
    global_silhouette = silhouette_score(X, labels)

    # Per-sample silhouette coefficients
    sample_silhouettes = silhouette_samples(X, labels)

    # Per-cluster analysis
    cluster_scores = {}
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # Skip noise points in DBSCAN
            continue
        mask = labels == cluster_id
        cluster_silhouettes = sample_silhouettes[mask]
        cluster_scores[cluster_id] = {
            'mean_silhouette': np.mean(cluster_silhouettes),
            'std_silhouette': np.std(cluster_silhouettes),
            'size': np.sum(mask)
        }

    return {
        'global_silhouette': global_silhouette,
        'cluster_scores': cluster_scores,
        'sample_silhouettes': sample_silhouettes
    }
```

#### 6.1.2 Classification Metrics

**Cross-Validation Strategy:**
```python
from sklearn.model_selection import StratifiedKFold, cross_validate

def evaluate_classifier(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Comprehensive classifier evaluation with cross-validation."""

    # Stratified 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(model, X, y, cv=cv,
                               scoring=scoring_metrics,
                               return_train_score=True)

    # Aggregate statistics
    results = {}
    for metric in scoring_metrics:
        test_scores = cv_results[f'test_{metric}']
        results[metric] = {
            'mean': np.mean(test_scores),
            'std': np.std(test_scores),
            'scores': test_scores
        }

    return results
```

### 6.2 Statistical Tests

#### 6.2.1 Distribution Comparisons

**Kolmogorov-Smirnov Tests:**
```python
from scipy.stats import ks_2samp, mannwhitneyu

def compare_spectral_distributions(group1: np.ndarray,
                                 group2: np.ndarray) -> dict:
    """Statistical comparison of spectral intensity distributions."""

    # Kolmogorov-Smirnov test for distribution similarity
    ks_statistic, ks_pvalue = ks_2samp(group1.flatten(),
                                      group2.flatten())

    # Mann-Whitney U test for median differences
    mw_statistic, mw_pvalue = mannwhitneyu(group1.flatten(),
                                          group2.flatten(),
                                          alternative='two-sided')

    return {
        'ks_test': {'statistic': ks_statistic, 'p_value': ks_pvalue},
        'mw_test': {'statistic': mw_statistic, 'p_value': mw_pvalue},
        'effect_size': np.median(group1) - np.median(group2)
    }
```

---

## 7. Software Environment

### 7.1 Core Dependencies

**Python Environment:**
```yaml
Python: ">=3.9"
Primary Dependencies:
  numpy: ">=1.20.0"      # Numerical computing
  pandas: ">=1.3.0"      # Data manipulation
  matplotlib: ">=3.5.0"  # Plotting and visualization
  scipy: ">=1.8.0"       # Scientific computing
  scikit-learn: ">=1.0.0" # Machine learning
  scikit-image: ">=0.19.0" # Image processing

Spectroscopy-Specific:
  pybaselines: ">=1.0.0" # Baseline correction

Deep Learning (Optional):
  torch: ">=2.0.0"       # PyTorch for SAM
  torchvision: ">=0.15.0"
  segment-anything: ">=1.0" # Meta SAM models

GPU Acceleration (Optional):
  cuml: ">=22.10.0"      # RAPIDS ML
  cudf: ">=22.10.0"      # RAPIDS DataFrames

Database:
  sqlalchemy: ">=2.0.0" # ORM and database abstraction
  alembic: ">=1.13.0"   # Database migrations

API Framework:
  pydantic: ">=1.10.0"  # Data validation
  typer: ">=0.9.0"      # CLI framework
  rich: ">=12.0.0"      # Terminal formatting
```

### 7.2 Version Control and Reproducibility

**Git SHA Tracking:**
```python
def get_code_sha() -> str:
    """Get current git commit SHA for reproducibility."""
    import subprocess
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                              capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"

# Embed in output metadata
metadata = {
    "schema_version": "1.0.0",
    "created_at": datetime.utcnow().isoformat(),
    "code_sha": get_code_sha(),
    "random_seed": 42,
    "python_version": sys.version,
    "numpy_version": np.__version__,
    "sklearn_version": sklearn.__version__
}
```

### 7.3 Configuration Management

**Centralized Configuration:**
```yaml
# config.yaml structure (excerpt; see src/sherloc_pipeline/config.yaml for full schema)
preprocessing:
  baseline_method: "aspls"
  despike:
    window_size: 7
    zscore_threshold: 6.0
    max_iterations: 1
  baseline:
    lam: 1.0e6
    asymmetric_coef: 0.01
    iters: 10
    diff_order: 2
    tol: 0.001

machine_learning:
  clustering:
    default_algorithm: "kmeans"
    kmeans:
      n_clusters: 5
      n_init: 10
      random_state: 42
    dbscan:
      eps: 0.1284
      min_samples: 5

  feature_extraction:
    pca_components: 50
    normalization: "l2"

vision:
  sam:
    model_type: "vit_b"
    points_per_side: 32
    pred_iou_thresh: 0.86
    stability_score_thresh: 0.92
    min_mask_region_area: 100
```

---

## 10. Reproducibility Guidelines

### 10.1 Random Seed Management

**Deterministic Results:**
```python
import numpy as np
import random
from sklearn.utils import check_random_state

# Global random seed for reproducibility
RANDOM_SEED = 42

def set_random_seeds(seed: int = RANDOM_SEED):
    """Set all random seeds for reproducible results."""
    np.random.seed(seed)
    random.seed(seed)

    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# Apply consistently across all analysis
set_random_seeds(42)
```

### 10.2 Output Metadata Standards

**Mandatory Metadata for All Outputs:**
```python
@dataclass
class AnalysisMetadata:
    """Standardized metadata for all analysis outputs."""
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    code_sha: str = field(default_factory=get_code_sha)
    random_seed: int = 42
    python_version: str = field(default_factory=lambda: sys.version)

    # Library versions
    numpy_version: str = field(default_factory=lambda: np.__version__)
    pandas_version: str = field(default_factory=lambda: pd.__version__)
    sklearn_version: str = field(default_factory=lambda: sklearn.__version__)

    # Hardware context
    cpu_info: str = field(default_factory=get_cpu_info)
    gpu_info: str = field(default_factory=get_gpu_info)
    total_memory_gb: float = field(default_factory=get_memory_info)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
```

### 10.3 Data Versioning

**Dataset Versioning Strategy:**
```python
def compute_dataset_hash(spectra_paths: List[Path]) -> str:
    """Compute reproducible hash for dataset version tracking."""
    import hashlib

    hasher = hashlib.sha256()

    # Sort paths for deterministic ordering
    sorted_paths = sorted(spectra_paths)

    for path in sorted_paths:
        # Hash file path and modification time
        hasher.update(str(path).encode())
        hasher.update(str(path.stat().st_mtime).encode())

    return hasher.hexdigest()[:16]  # First 16 characters
```

---

## 11. References

### 11.1 Primary Literature

```bibtex
@article{Bhartia2021,
  title={Perseverance's Scanning Habitable Environments with Raman and Luminescence for Organics and Chemicals (SHERLOC) Investigation},
  author={Bhartia, Rohit and Beegle, Luther W. and DeFlores, Lauren and Abbey, William and Razzell Hollis, Joseph and Uckert, Kyle and others},
  journal={Space Science Reviews},
  volume={217},
  pages={58},
  year={2021},
  publisher={Springer},
  doi={10.1007/s11214-021-00812-z}
}

@article{Zhang2020,
  title={Baseline correction for infrared spectra using adaptive smoothness parameter penalized least squares method},
  author={Zhang, Feng and Tang, Xiaojun and Tong, Angxin and Wang, Bin and Wang, Jingwei},
  journal={Spectroscopy Letters},
  volume={53},
  number={3},
  pages={222--233},
  year={2020},
  publisher={Taylor \& Francis},
  doi={10.1080/00387010.2020.1730908}
}

@article{Whitaker2018,
  title={A simple algorithm for despiking Raman spectra},
  author={Whitaker, Darren A and Hayes, Kevin},
  journal={Chemometrics and Intelligent Laboratory Systems},
  volume={179},
  pages={82--84},
  year={2018},
  publisher={Elsevier},
  doi={10.1016/j.chemolab.2018.06.009}
}

@article{Kirillov2023,
  title={Segment anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}

@article{McInnes2018,
  title={UMAP: Uniform manifold approximation and projection},
  author={McInnes, Leland and Healy, John and Melville, James},
  journal={arXiv preprint arXiv:1802.03426},
  year={2018}
}

@article{Pedregosa2011,
  title={Scikit-learn: Machine learning in Python},
  author={Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
```

### 11.2 Software and Data References

```bibtex
@software{Williford2024,
  author={Williford, Kenneth H.},
  title={SHERLOC Pipeline: Mars 2020 Raman/Fluorescence Data Processing},
  version={3.0.0},
  year={2024},
  url={https://github.com/archaeon-ai/sherloc-pipeline}
}

@software{Loupe2022,
  author={Uckert, Kyle},
  title={nasa/Loupe: LoupeV5.1.5},
  year={2022},
  publisher={Zenodo},
  doi={10.5281/zenodo.7062998},
  url={https://zenodo.org/records/7062998}
}

@software{PyBaselines2022,
  author={Erb, Donald},
  title={pybaselines: A Python library of algorithms for the baseline correction of experimental data},
  year={2022},
  url={https://github.com/derb12/pybaselines},
  doi={10.5281/zenodo.5608581}
}

@software{RAPIDS2023,
  author={{RAPIDS Development Team}},
  title={RAPIDS: GPU-Accelerated Data Science},
  version={22.10.0},
  year={2023},
  url={https://rapids.ai}
}
```

### 11.3 Standards and Conventions

```bibtex
@techreport{Wentworth1922,
  title={A scale of grade and class terms for clastic sediments},
  author={Wentworth, Chester K},
  journal={The Journal of Geology},
  volume={30},
  number={5},
  pages={377--392},
  year={1922},
  publisher={University of Chicago Press}
}

@article{Rousseeuw1987,
  title={Silhouettes: a graphical aid to the interpretation and validation of cluster analysis},
  author={Rousseeuw, Peter J},
  journal={Journal of Computational and Applied Mathematics},
  volume={20},
  pages={53--65},
  year={1987},
  publisher={Elsevier}
}

@book{Hastie2009,
  title={The elements of statistical learning: data mining, inference, and prediction},
  author={Hastie, Trevor and Tibshirani, Robert and Friedman, Jerome},
  year={2009},
  publisher={Springer},
  edition={2nd},
  doi={10.1007/978-0-387-84858-7}
}
```

---

## Appendix A: Algorithm Flowcharts

### A.1 Preprocessing Pipeline

```mermaid
graph TD
    A[Raw Loupe Data] --> B[Format Parsing & Validation]
    B --> C[Cosmic Ray Removal]
    C --> D[Baseline Correction]
    D --> E[Normalization]
    E --> F[Quality Control Check]
    F --> G[Analysis-Ready Spectra]

    C --> H[Protected Regions Check]
    H --> C

    F --> I[Failed QC]
    I --> J[Flag for Manual Review]
```

### A.2 Machine Learning Workflow

```mermaid
graph TD
    A[Preprocessed Spectra] --> B[Feature Extraction]
    B --> C[Dimensionality Reduction]
    C --> D[Model Selection]

    D --> E[K-Means Clustering]
    D --> F[DBSCAN Clustering]
    D --> G[Classification]

    E --> H[Silhouette Analysis]
    F --> I[Density Analysis]
    G --> J[Cross-Validation]

    H --> K[Model Validation]
    I --> K
    J --> K

    K --> L[Results & Metadata]
```

---

*Document Version: 1.0.0*
*Generated: 2026-01-25*
*Pipeline Version: 3.0.0*
*Git SHA: [Embedded at runtime]*