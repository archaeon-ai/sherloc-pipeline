"""
Pipeline orchestration service for SHERLOC pipeline.

This module encapsulates the legacy Typer `full_pipeline` command behaviour in a
dedicated service class so that the CLI becomes a thin wrapper while the service
coordinates preprocessing, fitting, review aggregation, and spatial overlays.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from rich.console import Console

from .base import ServiceResult
from .errors import PipelineRunError, SherlocServiceError, enrich
from .fitting import FittingService
from .preprocessing import PreprocessingService
from .review import ReviewService
from .spatial import SpatialService
from .runtime import RuntimeContext
from .metadata import RunMetadata, StageMetadata
from ..core.data_ingestion import DataIngestion

logger = logging.getLogger(__name__)

DEFAULT_STAGE_WINDOW: Dict[str, object] = {
    "from_stage": "preprocess",
    "to_stage": "summary",
    "allow_partial": False,
}


def archive_existing_results(base: Path) -> Optional[Path]:
    """Archive existing results directory if present.

    Returns the archived path if a directory was moved, else None.
    """
    try:
        if not base.exists():
            return None
        parent = base.parent
        stem = base.name
        suffix_index = 0
        while True:
            suffix = f"{stem}_archive_{suffix_index}" if suffix_index > 0 else f"{stem}_archive"
            candidate = parent / suffix
            if not candidate.exists():
                break
            suffix_index += 1
        import shutil

        shutil.move(str(base), str(candidate))
        return candidate
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.warning("failed to archive existing results at %s: %s", base, exc)
        return None


def _load_fit_counts(base: Path, sol: str, target: str, scan: str) -> Dict[str, int]:
    """Return per-modality accepted peak counts from scan results."""
    minerals_csv = base / "minerals_fit" / f"{sol}_{target}_{scan}_R1_accepted_peaks.csv"
    organics_csv = base / "organics_fit" / f"{sol}_{target}_{scan}_R1_organics_accepted_peaks.csv"
    hydration_csv = base / "hydration_fit" / f"{sol}_{target}_{scan}_R1_hydration_accepted_peaks.csv"

    def _count_rows(path: Path) -> int:
        try:
            if not path.exists():
                return 0
            import pandas as pd

            return int(pd.read_csv(path).shape[0])
        except Exception:
            return 0

    return {
        "minerals": _count_rows(minerals_csv),
        "organics": _count_rows(organics_csv),
        "hydration": _count_rows(hydration_csv),
    }


def print_scan_fit_summary(
    console: Console,
    ingestion: DataIngestion,
    *,
    sol: str,
    target: str,
    scan: str,
) -> Dict[str, int]:
    """Print and return accepted peak counts for a scan."""
    try:
        base = ingestion.get_results_path(target=target, sol=sol, scan=scan)
    except Exception:
        return {}

    counts = _load_fit_counts(base, sol, target, scan)
    if counts:
        try:
            console.print(
                f"[bold]Fitting summary ({sol} {target} {scan}):[/bold] "
                f"minerals={counts['minerals']}, "
                f"organics={counts['organics']}, "
                f"hydration={counts['hydration']}"
            )
        except Exception:
            pass
    return counts


# ---------------------------------------------------------------------------
# Post-hoc analysis functions (standalone, not PipelineService methods)
# ---------------------------------------------------------------------------

# Silicate hump detection range (cm⁻¹)
_SILICATE_HUMP_LO = 950.0
_SILICATE_HUMP_HI = 1060.0
_SILICATE_HUMP_SINGLE_FWHM_MIN = 70.0
_SILICATE_HUMP_ENVELOPE_SPAN_MIN = 80.0
_SILICATE_HUMP_NARROW_EXCLUDE = 40.0


def detect_silicate_hump(peaks: List[Dict]) -> Optional[Dict]:
    """Detect broad silicate hump from fitted mineral peaks.

    Post-hoc detection — does NOT modify fitting results.

    Criteria:
    - One or more peaks in 950-1060 cm-1 range
    - At least one peak with FWHM >= 70 cm-1 (near ceiling)
      OR two+ peaks in the range whose combined envelope spans >= 80 cm-1
    - Not a single narrow pyroxene or sulfate peak (FWHM < 40 excludes)

    Args:
        peaks: List of peak dicts with keys ``center_cm1`` and ``fwhm_cm1``.

    Returns:
        Dict with detection result, or None if not detected.
    """
    if not peaks:
        return None

    in_range = [
        p for p in peaks
        if p.get("center_cm1") is not None
        and p.get("fwhm_cm1") is not None
        and _SILICATE_HUMP_LO <= p["center_cm1"] <= _SILICATE_HUMP_HI
    ]

    if not in_range:
        return None

    # Filter out single narrow peaks (< 40 cm-1) — those are pyroxene/sulfate
    broad_enough = [p for p in in_range if p["fwhm_cm1"] >= _SILICATE_HUMP_NARROW_EXCLUDE]
    if not broad_enough:
        return None

    # Check criterion: single peak with FWHM >= 70
    has_single_broad = any(
        p["fwhm_cm1"] >= _SILICATE_HUMP_SINGLE_FWHM_MIN for p in broad_enough
    )

    # Check criterion: two+ peaks whose combined envelope spans >= 80 cm-1
    has_wide_envelope = False
    if len(broad_enough) >= 2:
        lo_edge = min(p["center_cm1"] - p["fwhm_cm1"] / 2 for p in broad_enough)
        hi_edge = max(p["center_cm1"] + p["fwhm_cm1"] / 2 for p in broad_enough)
        if (hi_edge - lo_edge) >= _SILICATE_HUMP_ENVELOPE_SPAN_MIN:
            has_wide_envelope = True

    if not has_single_broad and not has_wide_envelope:
        return None

    centers = [p["center_cm1"] for p in broad_enough]
    fwhms = [p["fwhm_cm1"] for p in broad_enough]
    return {
        "detected": True,
        "contributing_peaks": broad_enough,
        "mean_center_cm1": sum(centers) / len(centers),
        "mean_fwhm_cm1": sum(fwhms) / len(fwhms),
        "interpretation": "Broad silicate feature — likely feldspar, pyroxene, or mixture",
    }


# Co-occurrence pattern definitions
_CO_OCC_PATTERNS = [
    {
        "pattern": "Ce3+-bearing Ca-sulfate",
        "raman_labels": {"sulf1_v1"},
        "fluor_groups": {"group1a", "group1b"},
        "raman_feature": "sulf1_v1",
        "fluorescence_feature": "group1a + group1b",
    },
    {
        "pattern": "Ce3+-bearing phosphate",
        "raman_labels": {"phosphate"},
        "fluor_groups": {"group2"},
        "raman_feature": "phosphate",
        "fluorescence_feature": "group2",
    },
    {
        "pattern": "Silicate defect luminescence",
        "raman_labels": {"pyroxene", "1050", "silicate_hump"},
        "fluor_groups": {"group3"},
        "raman_feature": "pyroxene / 1050 / silicate_hump",
        "fluorescence_feature": "group3",
    },
]


def aggregate_co_occurrences(
    raman_peaks: List[Dict],
    fluor_peaks: List[Dict],
) -> List[Dict]:
    """Aggregate per-point co-occurrence patterns into scan-level summary.

    Args:
        raman_peaks: List of Raman peak dicts with ``point_idx`` and
            ``mineral_assignment`` keys.
        fluor_peaks: List of fluorescence peak dicts with ``point_idx`` and
            ``fluor_group`` keys.

    Returns:
        List of confirmed co-occurrence patterns with point counts.
    """
    # Build per-point sets
    raman_by_point: Dict[int, Set[str]] = defaultdict(set)
    fluor_by_point: Dict[int, Set[str]] = defaultdict(set)

    for p in raman_peaks:
        pidx = p.get("point_idx")
        assignment = p.get("mineral_assignment")
        if pidx is not None and assignment:
            raman_by_point[pidx].add(assignment)

    for p in fluor_peaks:
        pidx = p.get("point_idx")
        group = p.get("fluor_group")
        if pidx is not None and group:
            fluor_by_point[pidx].add(group)

    all_points = set(raman_by_point.keys()) | set(fluor_by_point.keys())

    results: List[Dict] = []

    for pattern_def in _CO_OCC_PATTERNS:
        confirmed_points: List[int] = []
        raman_only_points: List[int] = []
        fluor_only_points: List[int] = []
        confidence_sum = 0.0
        n_conf = 0

        for pidx in sorted(all_points):
            has_raman_match = bool(raman_by_point.get(pidx, set()) & pattern_def["raman_labels"])
            has_fluor_match = bool(fluor_by_point.get(pidx, set()) & pattern_def["fluor_groups"])

            if has_raman_match and has_fluor_match:
                confirmed_points.append(pidx)
                confidence_sum += 1.3
                n_conf += 1
            elif has_raman_match:
                raman_only_points.append(pidx)
            elif has_fluor_match:
                fluor_only_points.append(pidx)

        if not confirmed_points and not raman_only_points and not fluor_only_points:
            continue

        results.append({
            "pattern": pattern_def["pattern"],
            "raman_feature": pattern_def["raman_feature"],
            "fluorescence_feature": pattern_def["fluorescence_feature"],
            "n_points_confirmed": len(confirmed_points),
            "n_points_raman_only": len(raman_only_points),
            "n_points_fluor_only": len(fluor_only_points),
            "point_indices": confirmed_points,
            "mean_confidence": confidence_sum / n_conf if n_conf > 0 else 0.0,
        })

    return results


# Fluorescence group display labels
_FLUOR_GROUP_LABELS = {
    "group1a": "Ce3+ in anhydrite (short-lambda)",
    "group1b": "Ce3+ in anhydrite (long-lambda)",
    "group2": "Ce3+ in phosphate",
    "group3": "Silicate defect luminescence",
}


def _classify_confidence(max_snr: float, n_points: int) -> str:
    """Classify detection confidence as high/moderate/low."""
    if max_snr >= 10.0 and n_points >= 5:
        return "high"
    if max_snr >= 5.0 or n_points >= 3:
        return "moderate"
    return "low"


def summarize_findings(
    raman_peaks: List[Dict],
    fluor_peaks: List[Dict],
    n_total_points: int,
) -> Dict[str, Any]:
    """Generate structured scientific summary from pipeline results.

    Deterministic, rule-based summary — not LLM-generated.

    Args:
        raman_peaks: List of Raman peak dicts with keys including
            ``mineral_assignment``, ``point_idx``, ``snr``, ``center_cm1``,
            ``fwhm_cm1``, ``fit_modality``.
        fluor_peaks: List of fluorescence peak dicts with keys including
            ``fluor_group``, ``point_idx``.
        n_total_points: Total scan points for detection-rate context.

    Returns:
        Structured summary dict.
    """
    # --- Minerals ---
    mineral_peaks = [
        p for p in raman_peaks
        if p.get("fit_modality", "minerals") == "minerals"
        and p.get("mineral_assignment")
    ]
    minerals_by_assignment: Dict[str, List[Dict]] = defaultdict(list)
    for p in mineral_peaks:
        minerals_by_assignment[p["mineral_assignment"]].append(p)

    minerals_detected: List[Dict] = []
    for assignment, pks in sorted(minerals_by_assignment.items()):
        point_indices = {p["point_idx"] for p in pks if p.get("point_idx") is not None}
        snrs = [p["snr"] for p in pks if p.get("snr") is not None]
        centers = [p["center_cm1"] for p in pks if p.get("center_cm1") is not None]
        max_snr = max(snrs) if snrs else 0.0
        n_pts = len(point_indices)
        minerals_detected.append({
            "assignment": assignment,
            "n_points": n_pts,
            "max_snr": max_snr,
            "confidence": _classify_confidence(max_snr, n_pts),
            "mean_center_cm1": sum(centers) / len(centers) if centers else 0.0,
        })

    # --- Silicate hump ---
    silicate_hump = detect_silicate_hump(mineral_peaks)

    # --- Fluorescence groups ---
    fluor_by_group: Dict[str, int] = defaultdict(int)
    for p in fluor_peaks:
        group = p.get("fluor_group")
        if group:
            fluor_by_group[group] += 1

    fluorescence_groups: List[Dict] = []
    for group, count in sorted(fluor_by_group.items()):
        fluorescence_groups.append({
            "group": group,
            "label": _FLUOR_GROUP_LABELS.get(group, "Unidentified"),
            "n_peaks": count,
        })

    # --- Co-occurrences ---
    co_occurrences = aggregate_co_occurrences(raman_peaks, fluor_peaks)

    # --- Organics / Hydration flags ---
    organics_detected = any(
        p.get("fit_modality") == "organics" for p in raman_peaks
    )
    hydration_detected = any(
        p.get("fit_modality") == "hydration" for p in raman_peaks
    )

    # --- Narrative ---
    narrative = _build_narrative(
        minerals_detected, silicate_hump, co_occurrences,
        organics_detected, hydration_detected, n_total_points,
    )

    return {
        "minerals_detected": minerals_detected,
        "silicate_hump": silicate_hump,
        "fluorescence_groups": fluorescence_groups,
        "co_occurrences": co_occurrences,
        "organics_detected": organics_detected,
        "hydration_detected": hydration_detected,
        "narrative": narrative,
    }


def _build_narrative(
    minerals: List[Dict],
    silicate_hump: Optional[Dict],
    co_occurrences: List[Dict],
    organics_detected: bool,
    hydration_detected: bool,
    n_total_points: int,
) -> str:
    """Build a deterministic template-based narrative summary."""
    parts: List[str] = []

    if not minerals and not co_occurrences:
        parts.append("No mineral or co-occurrence detections in this scan.")
    else:
        # Dominant mineral (most points)
        if minerals:
            dominant = max(minerals, key=lambda m: m["n_points"])
            parts.append(
                f"Dominated by {dominant['assignment']} "
                f"({dominant['n_points']}/{n_total_points} points, "
                f"max SNR {dominant['max_snr']:.1f})."
            )
            other = [m for m in minerals if m["assignment"] != dominant["assignment"]]
            if other:
                names = ", ".join(m["assignment"] for m in other)
                parts.append(f"Also detected: {names}.")

        if silicate_hump:
            parts.append(
                f"Broad silicate hump detected "
                f"(center ~{silicate_hump['mean_center_cm1']:.0f} cm-1)."
            )

        # Co-occurrences
        for co in co_occurrences:
            if co["n_points_confirmed"] > 0:
                parts.append(
                    f"{co['pattern']} confirmed at "
                    f"{co['n_points_confirmed']} points."
                )

    if organics_detected:
        parts.append("Organic features (D/G bands) detected.")
    if hydration_detected:
        parts.append("Hydration (OH stretch) detected.")

    return " ".join(parts)


class PipelineService:
    """Service that orchestrates the SHERLOC full pipeline workflow."""

    def __init__(
        self,
        console: Optional[Console] = None,
        logger_: Optional[logging.Logger] = None,
        context: Optional[RuntimeContext] = None,
    ) -> None:
        self.console = console if console is not None else Console()
        self.logger = logger_ if logger_ is not None else logging.getLogger(__name__)
        self.context = context if context is not None else RuntimeContext.bootstrap()

    # Public API -----------------------------------------------------------------

    def run_full_pipeline(
        self,
        *,
        sol: str,
        target: str,
        scan: str,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        stage_window: Optional[Dict[str, object]] = None,
    ) -> ServiceResult:
        """Run the entire legacy CLI full pipeline for a single scan."""
        from sherloc_pipeline.core.data_ingestion import normalize_target_name
        target = normalize_target_name(target)

        run_context = self.context.with_overrides(
            data_dir=data_dir,
            results_dir=results_dir,
        )
        cfg = run_context.config
        resolved_data_dir = run_context.data_root
        resolved_results_dir = run_context.results_root

        ingestion = DataIngestion(
            base_data_dir=resolved_data_dir,
            results_dir=resolved_results_dir,
            sol=sol,
            target=target,
            scan=scan,
        )
        results_path = ingestion.get_results_path(target=target, sol=sol, scan=scan)

        archived_path = archive_existing_results(results_path)
        if archived_path is not None:
            self.console.print(f"[yellow]Archived existing results to {archived_path}[/yellow]")

        self._configure_logging(cfg)

        artifacts: List[Path] = []
        warnings: List[str] = []
        metadata: Dict[str, Any] = {
            "sol": sol,
            "target": target,
            "scan": scan,
            "results_path": str(results_path),
            "run_id": run_context.run_id,
        }
        stage_window_payload = dict(stage_window or DEFAULT_STAGE_WINDOW)
        metadata["requested_stage_range"] = stage_window_payload
        if archived_path is not None:
            metadata["archived_results_path"] = str(archived_path)

        config_hash = getattr(run_context, "config_hash", "")
        metadata["config_hash"] = config_hash
        metadata["data_root"] = str(resolved_data_dir)
        metadata["results_root"] = str(resolved_results_dir)

        run_started_at = datetime.now(timezone.utc)
        run_start_perf = time.perf_counter()
        run_metadata = RunMetadata(
            run_id=run_context.run_id,
            config_hash=config_hash,
            sol=sol,
            target=target,
            scan=scan,
            results_path=str(results_path),
            data_root=str(resolved_data_dir),
            results_root=str(resolved_results_dir),
        )
        run_metadata.start_time = run_started_at.isoformat()
        run_metadata.extra["requested_stage_range"] = stage_window_payload
        if archived_path is not None:
            run_metadata.archived_results_path = str(archived_path)

        @contextmanager
        def capture_stage(stage_name: str):
            stage_start_wall = datetime.now(timezone.utc)
            stage_start_perf = time.perf_counter()
            stage_artifacts: List[Path] = []
            stage_warnings: List[str] = []
            stage_extra: Dict[str, Any] = {}
            try:
                yield stage_artifacts, stage_warnings, stage_extra
            finally:
                stage_end_wall = datetime.now(timezone.utc)
                duration = time.perf_counter() - stage_start_perf
                stage_metadata = StageMetadata(
                    name=stage_name,
                    start_time=stage_start_wall.isoformat(),
                    end_time=stage_end_wall.isoformat(),
                    duration_s=duration,
                    artifacts=[str(artifact) for artifact in stage_artifacts],
                    warnings=list(stage_warnings),
                    extra=dict(stage_extra),
                )
                run_metadata.add_stage(stage_metadata)

        # Step 1/7 - Preprocessing ------------------------------------------------
        self.console.print("[bold cyan]Step 1/7: Preprocessing[/bold cyan]")
        preprocessing_service = PreprocessingService(console=self.console, context=run_context)
        generate_plots = bool(cfg.output.get("create_plots", True))
        with capture_stage("preprocessing") as (stage_artifacts, stage_warnings, stage_extra):
            try:
                pre_result = preprocessing_service.run_scan(
                    sol=sol,
                    target=target,
                    scan=scan,
                    data_dir=resolved_data_dir,
                    results_dir=resolved_results_dir,
                    generate_plots=generate_plots,
                    verbose=False,
                    despike_r1=True,
                    despike_window=None,
                    despike_threshold=None,
                    despike_max_iter=None,
                    despike_example_point=50,
                    baseline_r1=True,
                    baseline_lam=None,
                    baseline_asym=None,
                    baseline_iters=None,
                    background_file=None,
                    bkg_scale_method="ppp",
                    bkg_laser_roi=None,
                )
            except Exception as exc:
                stage_warnings.append(str(exc))
                stage_extra["exception"] = str(exc)
                self._raise_stage_error("Preprocessing", exc, sol, target, scan)
            else:
                artifacts.extend(pre_result.artifacts)
                warnings.extend(pre_result.warnings)
                stage_artifacts.extend(pre_result.artifacts)
                stage_warnings.extend(pre_result.warnings)
                if pre_result.metadata:
                    stage_extra["metadata"] = pre_result.metadata
                metadata["preprocessing"] = pre_result.metadata

        # Step 2/7 - Raman Fitting (minerals, organics, hydration) ----------------
        self.console.print("[bold cyan]Step 2/7: Raman Fitting (minerals, organics, hydration)[/bold cyan]")
        fitting_service = FittingService(console=self.console, context=run_context)
        fitting_metadata: Dict[str, Any] = {}
        with capture_stage("fitting_modality") as (stage_artifacts, stage_warnings, stage_extra):
            try:
                minerals_result = fitting_service.fit_minerals(
                    sol=sol,
                    target=target,
                    scan=scan,
                    data_dir=resolved_data_dir,
                    results_dir=resolved_results_dir,
                    region="R1",
                    verbose=False,
                )
            except Exception as exc:
                stage_warnings.append(str(exc))
                stage_extra["minerals_error"] = str(exc)
                self._raise_stage_error("Fitting (minerals)", exc, sol, target, scan)
            else:
                artifacts.extend(minerals_result.artifacts)
                warnings.extend(minerals_result.warnings)
                stage_artifacts.extend(minerals_result.artifacts)
                stage_warnings.extend(minerals_result.warnings)
                if minerals_result.metadata:
                    fitting_metadata["minerals"] = minerals_result.metadata
                    stage_extra["minerals"] = minerals_result.metadata

            try:
                organics_result = fitting_service.fit_organics(
                    sol=sol,
                    target=target,
                    scan=scan,
                    data_dir=resolved_data_dir,
                    results_dir=resolved_results_dir,
                    verbose=False,
                )
            except Exception as exc:
                stage_warnings.append(str(exc))
                stage_extra["organics_error"] = str(exc)
                self._raise_stage_error("Fitting (organics)", exc, sol, target, scan)
            else:
                artifacts.extend(organics_result.artifacts)
                warnings.extend(organics_result.warnings)
                stage_artifacts.extend(organics_result.artifacts)
                stage_warnings.extend(organics_result.warnings)
                if organics_result.metadata:
                    fitting_metadata["organics"] = organics_result.metadata
                    stage_extra["organics"] = organics_result.metadata

            try:
                hydration_result = fitting_service.fit_hydration(
                    sol=sol,
                    target=target,
                    scan=scan,
                    data_dir=resolved_data_dir,
                    results_dir=resolved_results_dir,
                    verbose=False,
                )
            except Exception as exc:
                stage_warnings.append(str(exc))
                stage_extra["hydration_error"] = str(exc)
                self._raise_stage_error("Fitting (hydration)", exc, sol, target, scan)
            else:
                artifacts.extend(hydration_result.artifacts)
                warnings.extend(hydration_result.warnings)
                stage_artifacts.extend(hydration_result.artifacts)
                stage_warnings.extend(hydration_result.warnings)
                if hydration_result.metadata:
                    fitting_metadata["hydration"] = hydration_result.metadata
                    stage_extra["hydration"] = hydration_result.metadata

        # Step 3/7 - Fluorescence Fitting + Persistence (non-fatal) ---------------
        db_path_str = os.environ.get("SHERLOC_DB_PATH", "./phase.db")
        db_path = Path(db_path_str)
        fluor_metadata: Dict[str, Any] = {}
        if db_path.exists():
            self.console.print("[bold cyan]Step 3/7: Fluorescence Fitting + Persistence[/bold cyan]")
            fitting_service._database_path = db_path
            with capture_stage("fluorescence_fitting") as (stage_artifacts, stage_warnings, stage_extra):
                try:
                    fluor_result = fitting_service.fit_fluorescence(
                        sol=sol,
                        target=target,
                        scan=scan,
                        data_dir=resolved_data_dir,
                        results_dir=resolved_results_dir,
                    )
                except Exception as exc:
                    self.logger.warning("Fluorescence fitting skipped: %s", exc)
                    warning = f"Fluorescence fitting skipped: {exc}"
                    warnings.append(warning)
                    stage_warnings.append(warning)
                    stage_extra["exception"] = str(exc)
                else:
                    if fluor_result.warnings:
                        warnings.extend(fluor_result.warnings)
                        stage_warnings.extend(fluor_result.warnings)
                    fluor_metadata = fluor_result.metadata or {}
                    stage_extra.update(fluor_metadata)
                    peak_count = fluor_metadata.get("peaks_inserted", 0)
                    self.console.print(
                        f"  [green]Fluorescence: fitted and persisted {peak_count} peaks[/green]"
                    )
            metadata["fluorescence"] = fluor_metadata
        else:
            self.logger.debug("Fluorescence fitting skipped: database not found at %s", db_path)

        # Step 4/7 - Raman Peak Persistence (non-fatal) --------------------------
        persistence_metadata: Dict[str, Any] = {}
        if db_path.exists():
            self.console.print("[bold cyan]Step 4/7: Raman Peak Persistence[/bold cyan]")
            fitting_service._database_path = db_path
            with capture_stage("raman_persistence") as (stage_artifacts, stage_warnings, stage_extra):
                for domain in ["minerals", "organics", "hydration"]:
                    try:
                        persist_result = fitting_service.persist_raman_peaks(
                            sol=sol,
                            target=target,
                            scan=scan,
                            domain=domain,
                            data_dir=resolved_data_dir,
                            results_dir=resolved_results_dir,
                        )
                    except Exception as exc:
                        self.logger.warning("Raman persistence (%s) skipped: %s", domain, exc)
                        warning = f"Raman persistence ({domain}) skipped: {exc}"
                        warnings.append(warning)
                        stage_warnings.append(warning)
                        stage_extra[f"{domain}_error"] = str(exc)
                    else:
                        if persist_result.warnings:
                            warnings.extend(persist_result.warnings)
                            stage_warnings.extend(persist_result.warnings)
                        domain_meta = persist_result.metadata or {}
                        persistence_metadata[domain] = domain_meta
                        stage_extra[domain] = domain_meta
                        peak_count = domain_meta.get("peaks_inserted", 0)
                        self.console.print(
                            f"  [green]{domain}: persisted {peak_count} peaks[/green]"
                        )
            fitting_metadata["persistence"] = persistence_metadata
        else:
            self.logger.debug("Raman persistence skipped: database not found at %s", db_path)

        metadata["fitting"] = fitting_metadata

        # Review aggregation (non-fatal) -----------------------------------------
        review_service = ReviewService(console=self.console, context=run_context)
        with capture_stage("review") as (stage_artifacts, stage_warnings, stage_extra):
            try:
                review_result = review_service.write_unified_tables(
                    sol=sol,
                    target=target,
                    scan=scan,
                    data_dir=resolved_data_dir,
                    results_dir=resolved_results_dir,
                )
            except SherlocServiceError as exc:
                self.logger.warning("failed to write unified accepted-peaks tables: %s", exc)
                warning = f"Failed to write unified accepted-peaks tables: {exc}"
                warnings.append(warning)
                stage_warnings.append(warning)
                stage_extra["exception"] = str(exc)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("unexpected error updating review tables: %s", exc)
                warning = f"Unexpected error updating review tables: {exc}"
                warnings.append(warning)
                stage_warnings.append(warning)
                stage_extra["exception"] = str(exc)
            else:
                artifacts.extend(review_result.artifacts)
                warnings.extend(review_result.warnings)
                stage_artifacts.extend(review_result.artifacts)
                stage_warnings.extend(review_result.warnings)
                if review_result.metadata:
                    stage_extra["metadata"] = review_result.metadata
                metadata["review"] = review_result.metadata

        # Step 5/7 - Fitting (averages) ------------------------------------------
        self.console.print("[bold cyan]Step 5/7: Fitting (averages)[/bold cyan]")
        with capture_stage("fitting_averages") as (stage_artifacts, stage_warnings, stage_extra):
            try:
                averages_result = fitting_service.fit_averages(
                    sol=sol,
                    target=target,
                    scan=scan,
                    data_dir=resolved_data_dir,
                    results_dir=resolved_results_dir,
                    region="R1",
                    verbose=False,
                )
            except Exception as exc:
                stage_warnings.append(str(exc))
                stage_extra["exception"] = str(exc)
                self._raise_stage_error("Fitting (averages)", exc, sol, target, scan)
            else:
                artifacts.extend(averages_result.artifacts)
                warnings.extend(averages_result.warnings)
                stage_artifacts.extend(averages_result.artifacts)
                stage_warnings.extend(averages_result.warnings)
                if averages_result.metadata:
                    stage_extra["metadata"] = averages_result.metadata
                metadata["averages"] = averages_result.metadata

        # Step 6/7 - Spatial overlays --------------------------------------------
        self.console.print("[bold cyan]Step 6/7: Spatial overlays[/bold cyan]")
        spatial_service = SpatialService(console=self.console, context=run_context)
        layers = self._resolve_layers(cfg)
        upscale = self._resolve_upscale(cfg)
        with capture_stage("spatial_overlays") as (stage_artifacts, stage_warnings, stage_extra):
            try:
                spatial_result = spatial_service.render_overlay(
                    sol=sol,
                    target=target,
                    scan=scan,
                    layers=layers,
                    upscale=upscale,
                    save_debug=True,
                    use_reviewed=False,
                    data_dir=resolved_data_dir,
                    results_dir=resolved_results_dir,
                )
            except Exception as exc:
                stage_warnings.append(str(exc))
                stage_extra["exception"] = str(exc)
                self._raise_stage_error("Spatial overlays", exc, sol, target, scan)
            else:
                artifacts.extend(spatial_result.artifacts)
                warnings.extend(spatial_result.warnings)
                stage_artifacts.extend(spatial_result.artifacts)
                stage_warnings.extend(spatial_result.warnings)
                if spatial_result.metadata:
                    stage_extra["metadata"] = spatial_result.metadata
                metadata["spatial"] = spatial_result.metadata

        # Step 7/7 - Summary ------------------------------------------------------
        self.console.print("[bold cyan]Step 7/7: Summary[/bold cyan]")
        fit_counts: Dict[str, int] = {}
        with capture_stage("summary") as (_stage_artifacts, _stage_warnings, stage_extra):
            fit_counts = print_scan_fit_summary(
                self.console,
                ingestion,
                sol=sol,
                target=target,
                scan=scan,
            )
            if fit_counts:
                stage_extra["fit_summary"] = fit_counts
        if fit_counts:
            metadata["fit_summary"] = fit_counts
            run_metadata.extra["fit_summary"] = fit_counts

        self.console.print("[bold green]Full pipeline completed successfully.[/bold green]")

        summary = f"Full pipeline completed successfully for {sol}/{target}/{scan}"
        if archived_path is not None:
            summary += f" (archived to {archived_path.name})"

        run_metadata.artifacts = [str(path) for path in artifacts]
        run_metadata.warnings = list(warnings)
        run_end_wall = datetime.now(timezone.utc)
        run_metadata.end_time = run_end_wall.isoformat()
        run_metadata.duration_s = time.perf_counter() - run_start_perf
        metadata["run"] = run_metadata.to_dict()

        return ServiceResult(
            summary=summary,
            artifacts=artifacts,
            warnings=warnings,
            metadata=metadata,
        )

    # Internal helpers -----------------------------------------------------------
    def _configure_logging(self, cfg: Any) -> None:
        try:
            level = str(cfg.logging.get("level", "INFO")).upper()
            logging.getLogger().setLevel(getattr(logging, level, logging.INFO))
        except Exception:
            pass

    def _resolve_layers(self, cfg: Any) -> str:
        try:
            spatial_cfg = cfg.spatial or {}
            if isinstance(spatial_cfg, dict):
                raw_layers = spatial_cfg.get("layers")
            else:
                raw_layers = getattr(spatial_cfg, "layers", None)
            if isinstance(raw_layers, (list, tuple)):
                return ",".join(str(layer) for layer in raw_layers)
            if isinstance(raw_layers, str) and raw_layers.strip():
                return raw_layers
        except Exception:
            pass
        return "minerals,organics,hydration"

    def _resolve_upscale(self, cfg: Any) -> int:
        try:
            image_cfg = cfg.image
            if isinstance(image_cfg, dict):
                return int(image_cfg.get("default_upscale_factor", 3))
            return int(getattr(image_cfg, "default_upscale_factor", 3))
        except Exception:
            return 3

    def _raise_stage_error(self, stage: str, exc: Exception, sol: str, target: str, scan: str) -> None:
        if isinstance(exc, SherlocServiceError):
            context = dict(exc.context)
            context.setdefault("stage", stage)
            pipeline_error = PipelineRunError(
                message=f"{stage} failed: {exc.message}",
                exit_code=exc.exit_code,
                context=context,
            )
            raise enrich(pipeline_error, sol=sol, target=target, scan=scan) from exc

        pipeline_error = PipelineRunError(
            message=f"{stage} failed: {exc}",
            exit_code=1,
            context={"stage": stage},
        )
        raise enrich(pipeline_error, sol=sol, target=target, scan=scan) from exc


__all__ = [
    "PipelineService",
    "archive_existing_results",
    "print_scan_fit_summary",
    "detect_silicate_hump",
    "aggregate_co_occurrences",
    "summarize_findings",
]

