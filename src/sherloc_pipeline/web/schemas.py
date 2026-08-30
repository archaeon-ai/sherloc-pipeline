"""Pydantic request/response models (DTOs) for the SHERLOC Web API.

All API responses include ``schema_version`` at the top level.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

API_SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Shared / reusable schemas
# ---------------------------------------------------------------------------


class BaselineParamsSchema(BaseModel):
    """Baseline correction parameters."""

    method: str = Field(default="aspls", description="Baseline algorithm.")
    lam: float = Field(default=1_000_000.0, description="Smoothness parameter.")
    max_iter: int = Field(default=10, description="Maximum iterations.")
    wavenumber_range: Optional[List[float]] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Optional baseline fit/subtraction ROI [lo, hi] in cm-1.",
    )

    @field_validator("wavenumber_range")
    @classmethod
    def validate_wavenumber_range(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is not None and value[0] >= value[1]:
            raise ValueError("wavenumber_range must be in ascending order")
        return value


class FitParamsSchema(BaseModel):
    """Peak fitting parameters."""

    domain: Literal["minerals", "organics", "hydration", "fluorescence"] = Field(
        default="minerals", description="Fitting domain."
    )
    wavenumber_range: List[float] = Field(
        default=[700.0, 1200.0], description="Fit ROI [lo, hi] in cm-1."
    )
    max_peaks: int = Field(default=5, ge=1, le=20, description="Max Gaussian components.")
    min_snr: float = Field(default=3.0, ge=0, description="Minimum peak SNR.")
    fwhm_bounds: List[float] = Field(
        default=[22.0, 90.0], description="FWHM bounds [min, max] in cm-1."
    )
    model_selection: Literal["f-test", "aicc"] = Field(
        default="aicc", description="Model selection method."
    )


class ProcessingParamsSchema(BaseModel):
    """Full processing chain parameters."""

    baseline: Optional[BaselineParamsSchema] = None
    fitting: Optional[FitParamsSchema] = None



# ---------------------------------------------------------------------------
# Scan DTOs
# ---------------------------------------------------------------------------


class ScanListItem(BaseModel):
    id: str
    sol_number: int
    target: Optional[str] = None
    scan_name: str
    scan_id: str
    n_points: int
    scan_class: str
    scan_type: Optional[str] = None
    target_type: str
    processing_status: Optional[str] = None
    processed_at: Optional[str] = None
    processing_pipeline_version: Optional[str] = None


class ScanListResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    scans: List[ScanListItem]
    total: int
    offset: int
    limit: int
    # Populated only when the response represents an unfiltered empty DB
    # (no scans visible to the active access mode and no filters applied).
    # The frontend uses this as a signal to render an onboarding panel
    # rather than the standard "no scans matched" empty-table message.
    # Per spec §12.2.
    message: Optional[str] = None


class ScanDetail(BaseModel):
    id: str
    sol_number: int
    target: Optional[str] = None
    scan_name: str
    scan_id: str
    n_points: int
    n_channels: int
    shots_per_point: Optional[int] = None
    laser_wavelength_nm: float
    scan_class: str
    scan_type: Optional[str] = None
    target_type: str
    data_source: Optional[str] = None
    site_drive: Optional[str] = None
    sequence_id: Optional[str] = None
    parent_scan_id: Optional[str] = None
    source_scan_ids: Optional[Any] = None
    processing_status: Optional[str] = None
    processed_at: Optional[str] = None
    processing_pipeline_version: Optional[str] = None
    processing_config_hash: Optional[str] = None
    processing_error: Optional[str] = None
    sclk_start: Optional[int] = None
    sclk_stop: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    # True iff the R2 backend has a sol_NNNN_colorized/ sibling for the
    # scan's ACI context image. The Workbench AciViewer uses this to
    # gate the "Colorized" toggle, replacing the prior silent-fallback
    # behaviour where the button appeared for every scan but did
    # nothing when the variant was missing (170 of 205 historical sols
    # lack a colorized variant). Default False so an absent
    # context_images row or any R2-side failure surfaces as "no
    # variant" rather than a broken-button UX.
    colorized_aci_available: bool = False
    # Count of stored cosmic-ray mask rows for this scan (issue #6). The
    # frontend gates the despike "ML" selector option on
    # ``ml_mask_count > 0``. Defaults to 0 so a mask-less scan or a
    # pre-migration database (no ``cosmic_ray_masks`` table) surfaces as
    # "ML unavailable" rather than erroring — mirroring the stored-mask
    # read path's tolerance.
    ml_mask_count: int = 0


class ScanDetailResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    scan: ScanDetail


# ---------------------------------------------------------------------------
# Points DTO
# ---------------------------------------------------------------------------


class PointItem(BaseModel):
    id: str
    point_index: int
    x_pixel: Optional[float] = None
    y_pixel: Optional[float] = None
    x_aci_pixel: Optional[float] = None
    y_aci_pixel: Optional[float] = None
    # ACI pixel coords against the colorized variant (sol_NNNN_colorized/),
    # populated only when a colorized ACI exists for the scan (issue #8).
    # The colorized image is a pure crop of grayscale, so these differ from
    # x_aci_pixel/y_aci_pixel by the crop origin; the Workbench overlay uses
    # them when the user toggles Colorized so points stay registered.
    x_aci_pixel_colorized: Optional[float] = None
    y_aci_pixel_colorized: Optional[float] = None
    azimuth_dn: Optional[int] = None
    elevation_dn: Optional[int] = None
    azimuth_error: Optional[float] = None
    elevation_error: Optional[float] = None
    photodiode_mean: Optional[float] = None
    photodiode_std: Optional[float] = None
    coordinate_frame: Optional[str] = None


class PointsResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    scan_id: str
    points: List[PointItem]
    n_points: int


# ---------------------------------------------------------------------------
# Spectra DTOs
# ---------------------------------------------------------------------------


class ProvenanceInfo(BaseModel):
    calibration_version: str = "loupe_v5.1.5a"
    wavenumber_unit: str = "cm-1"
    intensity_unit: str = "counts"
    averaging_method: Optional[str] = None
    wavelength_filter: Optional[Dict[str, float]] = None


# Despike request enum shared by the spectra endpoints (issue #6 method
# selector). The request enum is COARSE — ``none | ml | modz`` — and selects
# *which* despike strategy to run; it is distinct from the response
# ``despike_method`` field, which is the PRECISE provenance label of the
# despike that was actually applied (e.g. the stored ML mask's
# ``ml_v1.3_tau_matched``, or the literal ``"modz"`` for the live path).
DespikeMethodLiteral = Literal["none", "ml", "modz"]


# Additive despike fields shared by the spectra responses (spec §4.6;
# issue #6 method selector). All optional with defaults: a
# client that never sends ``despike``/``despike_method`` sees the
# pre-integration response contract, so API_SCHEMA_VERSION stays "1.0.0"
# (additive optional fields do not bump the version, which flags breaking
# shape changes only).
#
# - ``despike_applied``  — true iff a despike was applied to this view.
#   For the stored-mask (``ml``) composite (R123) views this is all-or-none:
#   a missing constituent mask yields ``false`` with the missing regions
#   named. For the live ``modz`` path, true only on the R1 single-region
#   path.
# - ``despike_method``   — PRECISE provenance label of the applied despike,
#   else null. For ``ml`` this is the stored mask's provenance string; for
#   ``modz`` it is the literal ``"modz"``.
# - ``n_masked_channels``— count of channels replaced in the returned array.
# - ``masked_positions`` — indices INTO THE SERVED wavenumber/intensity arrays
#   where the stored-mask (``ml``) despike replaced channels (issue #8, for the
#   Workbench red-triangle markers). Populated on the ``ml`` path only; empty
#   for ``none``/``modz`` (the Workbench computes its own full-length modz mask
#   client-side). DISTINCT from the point endpoint's ``masked_channels``, which
#   are *absolute* CCD channel indices. The two coordinate systems:
#     - ``masked_channels``  (point, ``ml``) — absolute 0..2147 CCD channels.
#     - ``masked_positions`` (all three endpoints, ``ml``) — 0-based positions
#       into the served array (length == ``n_channels``), so a marker at
#       ``masked_positions[k]`` lands on ``wavenumber[masked_positions[k]]``.
#   For a single region the served array is the region's wavelength selection,
#   so the absolute channel maps to its row in that selection. For the ``R123``
#   composite the served array is the full 2148-channel stitch (a
#   position-preserving copy/sum at each channel index), so served position
#   equals the absolute channel and ``masked_positions == masked_channels``.
# - ``despike_missing_regions`` — regions that could not be despiked. For
#   ``ml`` composite views: contributing regions lacking a stored mask. For
#   ``modz``: the requested non-R1 region(s) (modz is R1-only; empty when
#   applied or single-region R1).
# - ``n_uncovered_contributor_channels`` — composite ``ml`` views only:
#   static per-construction count of channels carrying ≥1 uncovered
#   contributor (§3.3); 0 for single-region and modz views.
# - ``despike_params_used`` — modz path only: the modz parameters used
#   (config defaults). Null on the ml/none paths.


class AverageSpectrumResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    scan_id: str
    region: str
    n_points_averaged: int
    effective_trim_pct_per_tail: float
    m_trimmed_per_tail: int
    baseline_corrected: bool
    laser_normalized: bool = False
    wavenumber: List[float]
    wavelength: Optional[List[float]] = None
    intensity: List[float]
    n_channels: int
    provenance: ProvenanceInfo
    despike_applied: bool = False
    despike_method: Optional[str] = None
    n_masked_channels: int = 0
    # ml-path-only positions INTO the served arrays (issue #8). See the block
    # comment above for the masked_channels-vs-masked_positions coordinate
    # systems. Empty on none/modz (Workbench computes its own modz mask).
    masked_positions: List[int] = Field(default_factory=list)
    despike_missing_regions: List[str] = Field(default_factory=list)
    n_uncovered_contributor_channels: int = 0
    despike_params_used: Optional[Dict[str, Any]] = None


class PointSpectrumResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    scan_id: str
    point_index: int
    region: str
    spectrum_type: str
    wavenumber: List[float]
    wavelength: Optional[List[float]] = None
    intensity: List[float]
    n_channels: int
    photodiode_mean: Optional[float] = None
    provenance: ProvenanceInfo
    despike_applied: bool = False
    despike_method: Optional[str] = None
    n_masked_channels: int = 0
    # Absolute CCD channel indices (0..2147) replaced on the ml path — the
    # original, back-compat field (empty for modz, which reports only a count).
    masked_channels: List[int] = Field(default_factory=list)
    # Positions INTO the served arrays where the ml despike replaced channels
    # (issue #8). See the block comment above for how this differs from the
    # absolute masked_channels. Empty on none/modz.
    masked_positions: List[int] = Field(default_factory=list)
    despike_missing_regions: List[str] = Field(default_factory=list)
    n_uncovered_contributor_channels: int = 0
    despike_params_used: Optional[Dict[str, Any]] = None


class SubsetRequest(BaseModel):
    point_indices: List[int]
    region: str = "R1"
    averaging_method: str = "trim_mean"
    trim_pct: Optional[float] = None
    despike: bool = False
    # Coarse despike strategy selector (issue #6). When set, wins over the
    # legacy ``despike`` bool. Pydantic rejects any other value with 422.
    despike_method: Optional[DespikeMethodLiteral] = None

    @field_validator("point_indices")
    @classmethod
    def non_empty(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("point_indices must be non-empty")
        return v


class SubsetResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    scan_id: str
    region: str
    n_points_averaged: int
    point_indices: List[int]
    effective_trim_pct_per_tail: float
    m_trimmed_per_tail: int
    wavenumber: List[float]
    wavelength: Optional[List[float]] = None
    intensity: List[float]
    n_channels: int
    provenance: ProvenanceInfo
    despike_applied: bool = False
    despike_method: Optional[str] = None
    n_masked_channels: int = 0
    # ml-path-only positions INTO the served arrays (issue #8); see the block
    # comment above. Empty on none/modz.
    masked_positions: List[int] = Field(default_factory=list)
    despike_missing_regions: List[str] = Field(default_factory=list)
    n_uncovered_contributor_channels: int = 0
    despike_params_used: Optional[Dict[str, Any]] = None


# Known-noisy detector-channel annotation layer (issue #9). A STATIC list
# served by ``GET /api/spectra/badpix?region=...``; the asset never changes at
# runtime and there is no DB involvement. This is an ANNOTATION surface,
# entirely separate from despiking — nothing here masks or alters spectral
# values. Additive endpoint, so API_SCHEMA_VERSION stays "1.0.0".
class BadpixChannelItem(BaseModel):
    #: 0-based index INTO the served wavenumber/intensity array for the
    #: requested region view (the same coordinate system as
    #: ``masked_positions`` — so a marker at ``position`` lands on
    #: ``wavenumber[position]``). For R123 the served array is the full
    #: 2148-channel stitch, so ``position == channel``.
    position: int
    #: Absolute 0-based CCD channel index (0..2147).
    channel: int
    #: 1 = CR-confusable elevated-noise; 2 = stable hot (cancels in dark sub).
    tier: int
    #: Attribution: ``dark_veto`` | ``jb25`` | ``both``.
    source: str


class BadpixResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    region: str
    n_channels: int
    badpix: List[BadpixChannelItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Processing DTOs
# ---------------------------------------------------------------------------


class BaselineRequest(BaseModel):
    wavenumber: List[float]
    intensity: List[float]
    params: Optional[BaselineParamsSchema] = None


class BaselineResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    raw: List[float]
    baseline: List[float]
    corrected: List[float]
    wavenumber: List[float]
    params_used: BaselineParamsSchema


class DespikeParamsSchema(BaseModel):
    """Despiking parameters exposed to the web API."""

    window_size: int = Field(default=7, ge=3, le=15, description="Rolling window size (odd).")
    zscore_threshold: float = Field(
        default=6.0, ge=2.0, le=10.0, description="Robust z-score threshold."
    )
    max_iterations: int = Field(default=1, ge=1, le=5, description="Despiking passes.")
    sulfate_guard: bool = Field(default=True, description="Enable sulfate guard protection.")

    @field_validator("window_size")
    @classmethod
    def must_be_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("window_size must be odd")
        return v


class DespikeRequest(BaseModel):
    wavenumber: List[float]
    intensity: List[float]
    params: Optional[DespikeParamsSchema] = None


class DespikeResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    despiked: List[float]
    spike_mask: List[bool]
    n_spikes: int
    params_used: DespikeParamsSchema


class BackgroundRequest(BaseModel):
    wavenumber: List[float]
    intensity: List[float]
    bg_type: Literal["as", "fs"]
    scale: Any = Field(default="auto", description="Scale factor or 'auto'.")
    scan_ppp: int = Field(default=900, ge=1, description="Scan pulses per point.")

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v: Any) -> Any:
        if isinstance(v, str):
            if v != "auto":
                raise ValueError("scale must be 'auto' or a numeric value")
            return v
        if isinstance(v, (int, float)):
            if v <= 0:
                raise ValueError("scale must be positive")
            return float(v)
        raise ValueError("scale must be 'auto' or a numeric value")


class BackgroundResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    subtracted: List[float]
    background_scaled: List[float]
    scale_used: float
    bg_type: Literal["as", "fs"]


class FitRequest(BaseModel):
    wavenumber: List[float]
    intensity: List[float]
    params: Optional[ProcessingParamsSchema] = None
    # Optional scan context. When present, the per-peak quality classifier
    # downgrades calibration / engineering scans to "review" regardless of
    # fit quality (no Mars-target ground truth to validate against). Values
    # match models/spectra.py:TargetType: "mars_target" | "cal_target" |
    # "engineering". Absent → assume mars_target for backwards compatibility
    # with callers that don't yet pass scan context.
    target_type: Optional[str] = None


class PeakDTO(BaseModel):
    center_cm1: Optional[float] = None
    center_uncertainty: Optional[float] = None
    amplitude: float
    amplitude_uncertainty: Optional[float] = None
    fwhm_cm1: Optional[float] = None
    fwhm_uncertainty: Optional[float] = None
    area: Optional[float] = None
    snr: Optional[float] = None
    fit_quality: Optional[float] = None
    mineral_assignment: Optional[str] = None
    assignment_confidence: Optional[float] = None
    fit_modality: str = "minerals"
    sharpness_ratio: Optional[float] = None
    pass_sharpness: Optional[bool] = None
    # Tri-state display label set by services.quality.classify_fit_quality.
    # Workbench Peak Results "Quality" column renders this. None only when a
    # caller bypasses classifier population (no current production path).
    quality: Optional[Literal["pass", "review", "fail"]] = None


class FitProvenance(BaseModel):
    params_used: ProcessingParamsSchema
    calibration_version: str = "loupe_v5.1.5a"
    config_hash: Optional[str] = None
    wavenumber_unit: str = "cm-1"
    intensity_unit: str = "counts"


class FitResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    peaks: List[PeakDTO]
    n_peaks: int
    r_squared: Optional[float] = None
    model_selection_method: str = "aicc"
    residual: List[float]
    baseline: List[float]
    corrected: List[float]
    wavenumber: List[float]
    provenance: FitProvenance


# ---------------------------------------------------------------------------
# Config DTO
# ---------------------------------------------------------------------------


class AuthConfig(BaseModel):
    """Auth metadata exposed to the SPA at runtime via /api/config (§13.4).

    Permits the same SPA build to bootstrap against any auth backend
    without rebuild. Auth0 fields are populated only when
    ``auth_mode == "auth0"`` so the SPA can branch cleanly.
    """

    auth_mode: Literal["auth0", "cf-access", "dev"]
    auth0_domain: Optional[str] = None
    auth0_client_id: Optional[str] = None
    auth0_audience: Optional[str] = None
    role_claim_uri: Optional[str] = None


class FeaturesConfig(BaseModel):
    """Runtime feature-flag block exposed to the SPA via /api/config.

    Lets a single image swap features on/off via env vars without rebuild.
    Each flag defaults to enabled so dev/legacy deploys behave as before;
    production env templates opt features out explicitly.
    """

    pds_browser: bool = True


class ConfigResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    config_hash: Optional[str] = None
    fitting: Dict[str, Any]
    fluorescence_fitting: Dict[str, Any]
    preprocessing: Dict[str, Any]
    calibration: Dict[str, Any]
    auth: AuthConfig
    features: FeaturesConfig = FeaturesConfig()


# ---------------------------------------------------------------------------
# Health DTO
# ---------------------------------------------------------------------------


class HealthCheck(BaseModel):
    status: str
    path: Optional[str] = None
    n_scans: Optional[int] = None
    n_spectra: Optional[int] = None
    error: Optional[str] = None
    config_hash: Optional[str] = None
    source: Optional[str] = None
    running: Optional[int] = None
    queued: Optional[int] = None
    max_depth: Optional[int] = None
    n_unprocessed: Optional[int] = None
    n_failed: Optional[int] = None


class HealthResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    status: str
    timestamp: str
    pipeline_version: str
    checks: Dict[str, HealthCheck]


# ---------------------------------------------------------------------------
# PDS DTOs
# ---------------------------------------------------------------------------


class PDSSolInfo(BaseModel):
    sol: int
    n_scans: Optional[int] = None
    data_volume_mb: Optional[float] = None
    pds_url: Optional[str] = None


class PDSAvailableResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    sols: List[PDSSolInfo]
    total: int
    already_ingested: List[int]


class PDSDownloadRequest(BaseModel):
    sol: int
    force_reingest: bool = False


class PDSDownloadResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    job_id: str
    status: str
    queue_position: int
    sol: int
    submitter_token: str
    created_at: str


class PDSCatalogSolInfo(BaseModel):
    sol: int


class PDSCatalogResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    available_sols: List[PDSCatalogSolInfo]
    total_available: int
    already_ingested: List[int]


# ---------------------------------------------------------------------------
# Access Mode DTO
# ---------------------------------------------------------------------------


class AccessModeResponse(BaseModel):
    schema_version: str = API_SCHEMA_VERSION
    access_mode: Literal["internal", "public"]


# ---------------------------------------------------------------------------
# Map Mode schemas
# ---------------------------------------------------------------------------


class MapFitRequest(BaseModel):
    """Request to start map-mode fitting."""

    scan_id: str
    domains: List[str]  # ["minerals", "organics", "hydration", "fluorescence"]
    point_indices: Optional[List[int]] = None  # null = all points


class MapFitResponse(BaseModel):
    """Response from POST /api/map/fit."""

    schema_version: str = API_SCHEMA_VERSION
    job_id: str
    n_points: int
    ws_url: str


class MapPointDTO(BaseModel):
    """A single point in the map point set."""

    index: int
    x: float
    y: float


class MapVoronoiDTO(BaseModel):
    """Voronoi geometry for rendering."""

    vertices: List[List[float]]
    regions: List[List[int]]
    boundary: List[List[float]]
    edge_mask: List[bool]


class MapLayerInfoDTO(BaseModel):
    """Available layers for a domain."""

    n_detections: int
    classes: List[str]


class MapCachedResultDTO(BaseModel):
    """A cached fit result."""

    cache_id: str
    domains: List[str]
    profile_name: Optional[str]
    profile_hash: str
    created_at: str
    n_points: int
    n_detections: Dict[str, int]


class MapLayersResponse(BaseModel):
    """Response from GET /api/map/layers/{scan_id}."""

    schema_version: str = API_SCHEMA_VERSION
    scan_id: str
    coordinate_source: str
    base_images: List[Dict[str, Any]]
    point_set: Dict[str, Any]  # {points: MapPointDTO[], voronoi: MapVoronoiDTO | null}
    # Point set resolved against the colorized ACI variant, present only when
    # a colorized ACI exists (issue #8). Same shape as point_set; the frontend
    # selects it when the colorized base image is active so the overlay stays
    # registered on the cropped colorized image. Absent (None) → no colorized
    # variant, frontend keeps using point_set.
    point_set_colorized: Optional[Dict[str, Any]] = None
    available_layers: Dict[str, Dict[str, Any]]
    cached_results: List[MapCachedResultDTO]


class MapDataPointDTO(BaseModel):
    """A single point's scalar value."""

    index: int
    value: Optional[float]
    status: str  # "measured" | "below_threshold" | "missing" | "masked"
    assignment: Optional[str] = None
    center_cm1: Optional[float] = None
    fwhm_cm1: Optional[float] = None
    center_nm: Optional[float] = None


class MapDataResponse(BaseModel):
    """Response from GET /api/map/data/{scan_id}."""

    schema_version: str = API_SCHEMA_VERSION
    scan_id: str
    domain: str
    value_type: str
    cache_id: Optional[str]
    points: List[MapDataPointDTO]


class MapJobStatusResponse(BaseModel):
    """Response from GET /api/map/jobs/{job_id}.

    The REST polling fallback carries the same liveness signals the fit
    WebSocket sends in its heartbeat frame. Collapsing ``queued`` into
    ``running`` and dropping the queue position left a client without a
    WebSocket unable to tell "waiting behind another scan" from "frozen"
    — the exact condition this endpoint exists to report.
    """

    schema_version: str = API_SCHEMA_VERSION
    job_id: str
    status: str  # "queued" | "running" | "complete" | "failed" | "cancelled"
    fitted: int
    total: int
    results_available: bool
    # Active jobs submitted ahead of this one on the single-threaded map
    # executor. 0 for a job that is running or already terminal.
    queue_position: int = 0
    # True when a running job has produced no output for longer than the
    # server's stall threshold. Advisory: the job is not killed.
    stalled: bool = False
    since_last_message_s: Optional[float] = None
    elapsed_s: Optional[float] = None
    # Per-point results the server still holds for this job, retrievable
    # from GET /api/map/jobs/{job_id}/results. Non-zero means a client
    # whose stream dropped can recover the frames it missed.
    results_retained: int = 0
    # False while the fitting thread may still add a point to that store.
    # A terminal status is not the same signal: a cancel is recorded when
    # it is requested, but the fitting loop only notices between points,
    # so the point it was on is retained afterwards. A client that reads
    # the results in that window loses a finished measurement (issue #6).
    results_final: bool = True


class MapJobFitDomainDTO(BaseModel):
    """One domain's fit outcome for a single point, as streamed."""

    status: str  # "measured" | "below_threshold" | "missing"
    peaks: List[Dict[str, Any]] = Field(default_factory=list)


class MapJobFitPointDTO(BaseModel):
    """A retained per-point fit result, shaped like the WS ``point_fitted`` frame."""

    point_index: int
    x: float
    y: float
    results: Dict[str, MapJobFitDomainDTO] = Field(default_factory=dict)


class MapJobResultsResponse(BaseModel):
    """Response from GET /api/map/jobs/{job_id}/results.

    Map fitting streams its results and never writes them to
    ``fitted_peaks``, so results missed while a client's WebSocket was down
    exist nowhere else — reloading the map from the database would show an
    earlier pipeline run's peaks instead. The server retains this job's own
    per-point results for its lifetime in the registry so a reconnecting or
    polling client can fetch exactly what it missed.
    """

    schema_version: str = API_SCHEMA_VERSION
    job_id: str
    status: str  # "queued" | "running" | "complete" | "failed" | "cancelled"
    fitted: int
    total: int
    # True when the job outgrew the retention ceiling and some points were
    # dropped, so a client knows the recovered map is incomplete.
    truncated: bool = False
    points: List[MapJobFitPointDTO] = Field(default_factory=list)
