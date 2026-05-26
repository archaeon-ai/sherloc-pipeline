"""Stateless processing endpoints: baseline, fit, despike, and background."""

import hashlib
import json
import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from scipy import interpolate

from sherloc_pipeline.core.baseline import BaselineParams, fit_baseline
from sherloc_pipeline.core.fitting import fit_spectrum, multi_gaussian
from sherloc_pipeline.core.fluor_fitting import fit_fluorescence_spectrum
from sherloc_pipeline.core.fluor_id import assign_fluor_group
from sherloc_pipeline.core.mineral_id import (
    assign_min_id,
    classify_hydration_band,
    classify_organic_band,
    load_mineral_rules,
)
from sherloc_pipeline.core.preprocessing import DespikeParams, despike_r1_spectrum
from sherloc_pipeline.services.quality import classify_fit_quality
from sherloc_pipeline.web.adapters import numpy_to_list
from sherloc_pipeline.web.schemas import (
    BackgroundRequest,
    BackgroundResponse,
    BaselineParamsSchema,
    BaselineRequest,
    BaselineResponse,
    DespikeParamsSchema,
    DespikeRequest,
    DespikeResponse,
    FitParamsSchema,
    FitProvenance,
    FitRequest,
    FitResponse,
    PeakDTO,
    ProcessingParamsSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["processing"])


def _baseline_params_from_schema(schema: BaselineParamsSchema) -> BaselineParams:
    """Convert schema to core BaselineParams."""
    return BaselineParams(lam=schema.lam, iters=schema.max_iter)


@router.post("/process/baseline", response_model=BaselineResponse)
def process_baseline(request: Request, body: BaselineRequest) -> BaselineResponse:
    """Apply baseline correction to a provided spectrum."""
    wn = body.wavenumber
    intensity = body.intensity

    if len(wn) != len(intensity):
        raise HTTPException(status_code=400, detail="wavenumber and intensity must have same length")
    if len(wn) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 data points")

    # Check monotonicity
    for i in range(1, len(wn)):
        if wn[i] <= wn[i - 1]:
            raise HTTPException(status_code=400, detail="wavenumber must be monotonically increasing")

    params_schema = body.params or BaselineParamsSchema()
    if params_schema.method != "aspls":
        raise HTTPException(status_code=400, detail=f"Unsupported baseline method: {params_schema.method}")

    params = _baseline_params_from_schema(params_schema)

    x = np.array(wn, dtype=np.float64)
    y = np.array(intensity, dtype=np.float64)
    series = pd.Series(y, index=x)

    try:
        corrected_series, baseline_series = fit_baseline(series, params)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Baseline computation failed: {exc}")

    return BaselineResponse(
        raw=body.intensity,
        baseline=numpy_to_list(baseline_series.values),
        corrected=numpy_to_list(corrected_series.values),
        wavenumber=body.wavenumber,
        params_used=params_schema,
    )


@router.post("/process/fit", response_model=FitResponse)
def process_fit(request: Request, body: FitRequest) -> FitResponse:
    """Run peak fitting on a provided spectrum."""
    wn = body.wavenumber
    intensity = body.intensity

    if len(wn) != len(intensity):
        raise HTTPException(status_code=400, detail="wavenumber and intensity must have same length")
    if len(wn) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 data points")

    config = request.app.state.config
    params = body.params or ProcessingParamsSchema()
    fit_params_schema = params.fitting or FitParamsSchema()

    x = np.array(wn, dtype=np.float64)
    y = np.array(intensity, dtype=np.float64)

    is_fluor = fit_params_schema.domain == "fluorescence"

    # Baseline correction (skip for fluorescence — data is already intensity)
    if is_fluor or params.baseline is None:
        corrected = y.copy()
        baseline = np.zeros_like(y)
    else:
        bl_params_schema = params.baseline or BaselineParamsSchema()
        bl_params = _baseline_params_from_schema(bl_params_schema)
        series = pd.Series(y, index=x)
        try:
            corrected_series, baseline_series = fit_baseline(series, bl_params)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Baseline failed: {exc}")
        corrected = corrected_series.values
        baseline = baseline_series.values

    domain = fit_params_schema.domain
    peaks: list[PeakDTO] = []

    if is_fluor:
        # Fluorescence: use fit_fluorescence_spectrum (differential evolution +
        # agnostic AICc, doublet-aware). The Raman fit_spectrum() does not work
        # in nm space and was producing single-peak fits across the Berry Hollow
        # Ce3+ doublet (group1a ~304 nm + group1b ~326 nm).
        fluor_cfg_obj = (
            config.get("fluorescence_fitting", {})
            if isinstance(config, dict)
            else getattr(config, "fluorescence_fitting", {})
        )
        fluor_cfg = (
            dict(fluor_cfg_obj)
            if isinstance(fluor_cfg_obj, dict)
            else {k: getattr(fluor_cfg_obj, k) for k in dir(fluor_cfg_obj) if not k.startswith("_")}
        )

        fit_range = tuple(fit_params_schema.wavenumber_range)
        fwhm_range = tuple(fit_params_schema.fwhm_bounds)

        try:
            fluor_result = fit_fluorescence_spectrum(
                x,
                corrected,
                fit_range=fit_range,
                position_bounds=tuple(fluor_cfg.get("position_bounds", [270.0, 357.0])),
                fwhm_range=fwhm_range,
                min_peak_separation=float(fluor_cfg.get("min_peak_separation", 15.0)),
                max_peaks=int(fit_params_schema.max_peaks),
                snr_threshold=float(fit_params_schema.min_snr),
                min_fwhm_nm=float(fluor_cfg.get("min_fwhm_nm", 8.0)),
                noise_window=tuple(fluor_cfg.get("noise_window", [261.5, 262.3])),
                saturation_threshold=float(fluor_cfg.get("saturation_threshold", 60000.0)),
                saturation_channel_limit=int(fluor_cfg.get("saturation_channel_limit", 5)),
                overlap_exclusion=tuple(fluor_cfg.get("overlap_exclusion", [337.4, 338.4])),
                strategy=str(fluor_cfg.get("strategy", "agnostic")),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Fluorescence fitting failed: {exc}")

        for fp in fluor_result.peaks:
            group = assign_fluor_group(fp.center_nm)
            peaks.append(
                PeakDTO(
                    center_cm1=fp.center_nm,
                    amplitude=fp.amplitude,
                    fwhm_cm1=fp.fwhm_nm,
                    area=fp.area,
                    snr=fp.snr,
                    fit_quality=fluor_result.r2,
                    mineral_assignment=group if group != "unidentified" else None,
                    fit_modality="fluorescence",
                    quality=classify_fit_quality(
                        r_squared=fluor_result.r2,
                        fwhm=fp.fwhm_nm,
                        modality="fluorescence",
                        target_type=body.target_type,
                    ),
                )
            )

        if fluor_result.peaks:
            params_array: list[float] = []
            for fp in fluor_result.peaks:
                params_array.extend([fp.center_nm, fp.amplitude, fp.fwhm_nm])
            y_model_full = multi_gaussian(x, np.array(params_array))
        else:
            y_model_full = np.zeros_like(x)

        r_squared = fluor_result.r2
        model_selection_method = "aicc"
    else:
        # Raman / organics / hydration: use fit_spectrum (cm⁻¹ space)
        fit_cfg = dict(config.fitting)
        fit_cfg["r1_fit_range"] = fit_params_schema.wavenumber_range
        fit_cfg["max_peaks"] = fit_params_schema.max_peaks
        fit_cfg["min_snr"] = fit_params_schema.min_snr
        fit_cfg["fit_fwhm_min_initial_cm1"] = fit_params_schema.fwhm_bounds[0]
        fit_cfg["fwhm_max_cm1"] = fit_params_schema.fwhm_bounds[1]
        # Same posthoc_filters.sharpness_max value the fitter uses for its
        # own pass_sharpness gate (core/fitting.py:270). Reading it here
        # keeps the classifier's cosmic-ray fail threshold in sync if the
        # operator ever retunes the config.
        sharpness_max = float(
            fit_cfg.get("posthoc_filters", {}).get("sharpness_max", 3.0)
        )
        if fit_params_schema.model_selection == "f-test":
            fit_cfg["parsimony"] = {"model_selection": "ftest", "ftest_alpha": 0.01}
        else:
            fit_cfg["parsimony"] = {"model_selection": "aicc"}

        try:
            fit_result, y_model_full = fit_spectrum(x, corrected, fit_cfg)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Fitting failed: {exc}")

        if domain == "minerals":
            fitting_cfg = config.fitting if not isinstance(config.fitting, dict) else config.fitting
            inline_rules = fitting_cfg.get("mineral_rules") if isinstance(fitting_cfg, dict) else getattr(fitting_cfg, "mineral_rules", None)
            mineral_rules = load_mineral_rules(inline_rules=inline_rules)
            _assign = lambda center: assign_min_id(center, mineral_rules)
        elif domain == "organics":
            _assign = classify_organic_band
        elif domain == "hydration":
            _assign = classify_hydration_band
        else:
            _assign = lambda _: None

        for p in fit_result.peaks:
            center = p.m_cm1
            assignment = _assign(center) if center is not None else None
            sharpness_ratio = getattr(p, "sharpness_ratio", None)
            peaks.append(
                PeakDTO(
                    center_cm1=center,
                    amplitude=p.a,
                    fwhm_cm1=p.fwhm,
                    area=p.area,
                    snr=p.snr,
                    fit_quality=fit_result.r2,
                    mineral_assignment=assignment,
                    fit_modality=domain,
                    sharpness_ratio=sharpness_ratio,
                    pass_sharpness=getattr(p, "pass_sharpness", None),
                    quality=classify_fit_quality(
                        r_squared=fit_result.r2,
                        fwhm=p.fwhm,
                        modality=domain,
                        target_type=body.target_type,
                        sharpness_ratio=sharpness_ratio,
                        sharpness_max=sharpness_max,
                    ),
                )
            )

        r_squared = fit_result.r2
        model_selection_method = fit_params_schema.model_selection

    if y_model_full is not None and len(y_model_full) > 0:
        residual = corrected - y_model_full[: len(corrected)]
    else:
        residual = np.zeros_like(corrected)

    fitting_json = json.dumps(config.fitting, sort_keys=True, default=str)
    config_hash = f"sha256:{hashlib.sha256(fitting_json.encode()).hexdigest()[:12]}"

    return FitResponse(
        peaks=peaks,
        n_peaks=len(peaks),
        r_squared=r_squared,
        model_selection_method=model_selection_method,
        residual=numpy_to_list(residual),
        baseline=numpy_to_list(baseline),
        corrected=numpy_to_list(corrected),
        wavenumber=body.wavenumber,
        provenance=FitProvenance(
            params_used=params,
            config_hash=config_hash,
        ),
    )


@router.post("/process/despike", response_model=DespikeResponse)
def process_despike(request: Request, body: DespikeRequest) -> DespikeResponse:
    """Apply despiking (cosmic ray removal) to a provided spectrum."""
    wn = body.wavenumber
    intensity = body.intensity

    if len(wn) != len(intensity):
        raise HTTPException(status_code=400, detail="wavenumber and intensity must have same length")
    if len(wn) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 data points")

    params_schema = body.params or DespikeParamsSchema()

    # Convert schema to core DespikeParams
    core_params = DespikeParams(
        window_size=params_schema.window_size,
        zscore_threshold=params_schema.zscore_threshold,
        max_iterations=params_schema.max_iterations,
        sulfate_guard_enable=params_schema.sulfate_guard,
    )

    x = np.array(wn, dtype=np.float64)
    y = np.array(intensity, dtype=np.float64)
    series = pd.Series(y, index=x)

    try:
        despiked_series, spike_mask_series = despike_r1_spectrum(
            series, core_params, raman_shift=x
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Despiking failed: {exc}")

    spike_mask = spike_mask_series.values.astype(bool)

    return DespikeResponse(
        despiked=numpy_to_list(despiked_series.values),
        spike_mask=[bool(v) for v in spike_mask],
        n_spikes=int(spike_mask.sum()),
        params_used=params_schema,
    )


# Whitelisted background filename pattern. Operator-trusted config values
# only — but constrain anyway so a typo / mis-edit cannot escape the
# package-data directory (e.g., absolute path, parent-dir traversal).
_BACKGROUND_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.csv$")


def _load_background_standalone(
    config,
    bg_type: str,
) -> pd.DataFrame:
    """Load a background spectrum without requiring a full SpectralPlotService.

    Resolution order:

    1. Package-data (``sherloc_pipeline.data.background``) — primary
       source for the v1.0-beta containerized deployment. Tier-agnostic;
       same bytes serve ``PHASE_TIER=team`` and ``PHASE_TIER=public``.
       Loaded via :mod:`importlib.resources` so editable installs +
       wheel installs both resolve.
    2. Legacy FS search (``./data/background/`` + ``../background/``)
       — legacy local dev worktree and
       dev runtimes where the operator may override with locally
       customized references.

    Args:
        config: Application config object (must have ``preprocessing`` dict).
        bg_type: Background type - "as" or "fs".

    Returns:
        DataFrame with columns ``raman_shift`` and ``intensity``.

    Raises:
        FileNotFoundError: if the requested background file is not in
            package-data AND not at any legacy FS location.
    """
    from pathlib import Path

    default_filenames = {
        "as": "Arm_Stowed_post-anomaly_900ppp_trimmed_mean_1266.csv",
        "fs": "Fused_Silica_Corning7980_Air_Subtracted-Bandwidth-35_SB-Pitt.csv",
    }
    default_column_mappings = {
        "as": {"raman_shift": "raman_shift", "intensity": "intensity"},
        "fs": {"raman_shift": "Raman shift (cm-1)", "intensity": "Intensity"},
    }

    # Read from config
    preprocessing = config.preprocessing if isinstance(config.preprocessing, dict) else {}
    bg_config = (
        preprocessing
        .get("background_subtraction", {})
        .get("backgrounds", {})
        .get(bg_type, {})
    )

    bg_filename = bg_config.get("file", default_filenames[bg_type])
    col_config = bg_config.get("columns", {})
    col_map = {
        "raman_shift": col_config.get("raman_shift", default_column_mappings[bg_type]["raman_shift"]),
        "intensity": col_config.get("intensity", default_column_mappings[bg_type]["intensity"]),
    }

    if not _BACKGROUND_FILENAME_RE.match(bg_filename):
        raise FileNotFoundError(
            f"Background filename rejected (must match "
            f"{_BACKGROUND_FILENAME_RE.pattern!r}): {bg_filename!r}"
        )

    bg_df = _read_packaged_background_csv(bg_filename)
    if bg_df is None:
        # Legacy FS search retained for local dev worktree where the
        # operator may symlink alternate references.
        data_root = Path("./data")
        possible_paths = [
            data_root / "background" / bg_filename,
            data_root.parent / "background" / bg_filename,
        ]
        bg_path = next((p for p in possible_paths if p.exists()), None)
        if bg_path is None:
            raise FileNotFoundError(
                f"Background file not found in package-data or legacy "
                f"FS: {bg_filename}. FS searched: "
                f"{[str(p) for p in possible_paths]}"
            )
        bg_df = pd.read_csv(bg_path)

    return pd.DataFrame({
        "raman_shift": bg_df[col_map["raman_shift"]].values,
        "intensity": bg_df[col_map["intensity"]].values,
    })


def _read_packaged_background_csv(filename: str) -> Optional[pd.DataFrame]:
    """Return DataFrame from ``sherloc_pipeline.data.background.<filename>`` or None.

    None signals "not shipped with this install" so the caller can fall
    back to legacy FS search. Any non-missing failure (decode error,
    permission error) propagates so the operator sees the root cause.
    """
    try:
        from importlib import resources
    except ImportError:  # pragma: no cover — 3.12 always has it
        return None
    pkg_files = resources.files("sherloc_pipeline.data.background")
    target = pkg_files / filename
    if not target.is_file():
        return None
    with resources.as_file(target) as path:
        return pd.read_csv(path)


@router.post("/process/background", response_model=BackgroundResponse)
def process_background(request: Request, body: BackgroundRequest) -> BackgroundResponse:
    """Apply scaled background subtraction to a provided spectrum."""
    wn = body.wavenumber
    intensity = body.intensity

    if len(wn) != len(intensity):
        raise HTTPException(status_code=400, detail="wavenumber and intensity must have same length")
    if len(wn) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 data points")

    # Resolve scale factor
    if body.scale == "auto":
        scale = body.scan_ppp / 900.0
    else:
        scale = float(body.scale)

    config = request.app.state.config

    try:
        bg_df = _load_background_standalone(config, body.bg_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load background: {exc}")

    # Interpolate background to match input wavenumber grid
    x = np.array(wn, dtype=np.float64)
    y = np.array(intensity, dtype=np.float64)

    bg_interp = interpolate.interp1d(
        bg_df["raman_shift"].values,
        bg_df["intensity"].values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    bg_interpolated = bg_interp(x)
    scaled_bg = bg_interpolated * scale
    subtracted = y - scaled_bg

    return BackgroundResponse(
        subtracted=numpy_to_list(subtracted),
        background_scaled=numpy_to_list(scaled_bg),
        scale_used=scale,
        bg_type=body.bg_type,
    )
