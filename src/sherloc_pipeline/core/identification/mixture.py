"""Provisional sparse spectral contributions, ported from prototype/model.py.

Retrieval fixes each species' shift and width before fitting. Coefficients are
unit-L2 spectral contributions; normalized weights do not measure abundances.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from sherloc_pipeline.core.identification.prep import grid_for


def retrieve(query, reference, recipe):
    """Best broadened cosine variant; deterministic width/shift ties."""
    if query.vector is None or reference.vector is None:
        return None
    grid = grid_for(recipe)
    best = None
    source_valid = reference.mask & np.isfinite(reference.vector)
    source = np.where(source_valid, reference.vector, 0.)
    for width in recipe.extra_fwhm_cm1:
        y, broadened_valid = source, source_valid
        if width:
            sigma = width / recipe.fwhm_to_sigma / recipe.grid[2]
            y = gaussian_filter1d(source, sigma, mode=recipe.smoothing_mode)
            # The same positive kernel identifies every output touched by an
            # invalid input, including edge extension under the smoothing mode.
            contaminated = gaussian_filter1d((~source_valid).astype(float), sigma,
                                             mode=recipe.smoothing_mode) > 0
            broadened_valid = ~contaminated
        for shift in recipe.shifts:
            vector = np.interp(grid-shift, grid, y, left=np.nan, right=np.nan)
            valid = np.isfinite(vector) & (
                np.interp(grid-shift, grid, broadened_valid.astype(float), left=0, right=0) == 1
            )
            keep = query.mask & valid
            if keep.mean() < recipe.min_overlap:
                continue
            a, b = query.vector[keep], vector[keep]
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            if not np.isfinite(denominator) or denominator <= recipe.denominator_floor:
                continue
            score = float(np.clip(a @ b / denominator, -1., 1.))
            if best is None or score > best["retrieval_score"]:
                best = {"retrieval_score": score, "shift_cm_1": shift,
                        "extra_fwhm_cm1": width, "vector": np.nan_to_num(vector), "mask": valid}
    return best


def background_basis(size, keep):
    t = np.linspace(0., 1., size)[keep]
    basis = np.column_stack([1-t, 1-np.abs(2*t-1), t])
    norms = np.linalg.norm(basis, axis=0)
    return basis / np.maximum(norms, 1e-12)


def coordinate_descent(design, target, penalty, cap, iterations, tolerance, initial=None):
    """Unit-column nonnegative L1 updates with explicit finite/limit diagnostics."""
    coef = np.zeros(design.shape[1]) if initial is None else initial.copy()
    delta = None
    for iteration in range(1, iterations + 1):
        previous = coef.copy()
        for j in range(len(coef)):
            residual = target - design @ coef + design[:, j] * coef[j]
            coef[j] = np.clip(design[:, j] @ residual - penalty[j], 0., cap[j])
        if not np.isfinite(coef).all():
            return coef, {"iterations": iteration, "converged": False, "finite": False,
                          "iteration_limit": iteration == iterations, "max_change": None}
        delta = float(np.max(np.abs(coef-previous)))
        if delta < tolerance:
            break
    limit = iteration == iterations
    return coef, {"iterations": iteration, "converged": delta < tolerance and not limit,
                  "finite": True, "iteration_limit": limit, "max_change": delta}


def fit_mixture(query, candidates, recipe):
    """Fit up to five retrieved variants; retain at most two reference columns.

    Candidates contain template_id, species_name, vector, mask and retrieval_score.
    All returned curves are on the prepared grid with nulls outside joint overlap.
    """
    empty = {"support": None, "components": [], "sse": None, "background_only_sse": None,
             "background_coefficients": [], "background_only_coefficients": [],
             "diagnostics": {"converged": False, "finite_solution": False}, "evidence": None}

    def reject(reason):
        return {**empty, "reason": reason}

    if query.vector is None:
        return reject(query.diagnostics.get("reason") or "no_detectable_feature")
    if not candidates:
        return reject("no_compatible_species_references")
    candidates = candidates[:recipe.retrieval_count]
    keep = query.mask.copy()
    for item in candidates:
        keep &= item["mask"]
    if keep.mean() < recipe.min_overlap:
        return reject("insufficient_joint_overlap")
    target = query.vector[keep]
    design = np.column_stack([row["vector"][keep] for row in candidates])
    norms = np.linalg.norm(design, axis=0)
    if (not np.isfinite(target).all() or not np.isfinite(design).all()
            or np.any(norms <= recipe.denominator_floor)
            or np.linalg.norm(target) <= recipe.denominator_floor):
        return reject("degenerate_fit")
    design /= norms
    bg = background_basis(len(query.vector), keep)
    design = np.column_stack([design, bg])
    n = len(candidates)
    penalty = np.r_[np.full(n, recipe.l1_penalty), np.zeros(3)]
    cap = np.r_[np.full(n, np.inf), np.full(3, recipe.background_cap)]
    first, second, null = recipe.solver_iterations
    coef, initial_info = coordinate_descent(design, target, penalty, cap, first, recipe.solver_tolerance)
    # Stable retrieval order breaks coefficient ties, including collinear alternatives.
    active = sorted(range(n), key=lambda j: (-coef[j], j))[:recipe.max_components]
    for j in range(n):
        if j not in active:
            coef[j] = cap[j] = 0.
    coef, refit_info = coordinate_descent(design, target, penalty, cap, second, recipe.solver_tolerance, coef)
    # Prune before scoring so every fitted reference is visible in components.
    # Refitting can push another coefficient below cutoff; never reactivate it.
    cutoff_refits = []
    while True:
        retained = [j for j in active if coef[j] > recipe.coefficient_cutoff]
        if retained == active:
            break
        for j in active:
            if j not in retained:
                coef[j] = cap[j] = 0.
        active = retained
        coef, info = coordinate_descent(design, target, penalty, cap, second, recipe.solver_tolerance, coef)
        cutoff_refits.append(info)
    bgcoef, null_info = coordinate_descent(bg, target, np.zeros(3), np.full(3, recipe.background_cap),
                                          null, recipe.solver_tolerance)
    fitted = design @ coef
    residual = target - fitted
    sse = float(residual @ residual)
    null_residual = target - bg @ bgcoef
    null_sse = float(null_residual @ null_residual)
    finite = bool(np.isfinite([sse, null_sse, *coef, *bgcoef]).all())
    if not finite:
        return reject("nonfinite_solution")
    support = max(0., 1 - sse / max(null_sse, recipe.denominator_floor))
    selected = active
    total = sum(coef[j] for j in selected)
    components = [{"template_id": candidates[j]["template_id"], "species_name": candidates[j]["species_name"],
                   "coefficient": float(coef[j]), "weight": float(coef[j]/total)} for j in selected]
    converged = all(info["converged"] for info in (initial_info, refit_info, *cutoff_refits, null_info))
    reason = None
    if not converged:
        reason = "fit_not_converged"
    elif not components:
        reason = "no_component_above_cutoff"
    elif support < recipe.unexplained_support_min:
        reason = "unexplained_structure"
    elif support < recipe.support_min:
        reason = "below_support_threshold"

    def curve(values):
        result = np.full(len(keep), None, dtype=object)
        result[keep] = values
        return result.tolist()

    return {"support": support, "reason": reason, "components": components,
            "sse": sse, "background_only_sse": null_sse,
            "background_coefficients": coef[n:].tolist(), "background_only_coefficients": bgcoef.tolist(),
            "diagnostics": {"converged": converged, "finite_solution": finite,
                            "initial_fit": initial_info, "active_refit": refit_info, "cutoff_refits": cutoff_refits,
                            "background_only": null_info,
                            "coefficients": coef[:n].tolist(), "template_ids": [r["template_id"] for r in candidates],
                            "shift_at_boundary": any(abs(candidates[j]["shift_cm_1"]) == max(map(abs, recipe.shifts))
                                                     for j in selected)},
            "evidence": {"label": "mixture fit", "interpretation": "provisional_mixture_fit",
                         "converged": converged, "grid": grid_for(recipe).tolist(), "mask": keep.tolist(),
                         "prepared_trace": curve(target), "fit": curve(fitted), "residual": curve(residual),
                         "background": curve(bg @ coef[n:])}}
