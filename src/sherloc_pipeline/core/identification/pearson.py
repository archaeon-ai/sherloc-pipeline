"""Signed Pearson search with the prototype's bounded shift-only variants."""

import numpy as np

from sherloc_pipeline.core.identification.prep import Prepared, grid_for
from sherloc_pipeline.models.identification import MethodRecipe


def signed_pearson(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 1 or b.shape != a.shape or len(a) < 2:
        return None
    a, b = a-a.mean(), b-b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(denominator) or denominator <= 1e-12:
        return None
    return float(np.clip(a @ b / denominator, -1., 1.))


def compare(query: Prepared, reference: Prepared, recipe: MethodRecipe):
    if query.vector is None or reference.vector is None:
        return None
    grid = grid_for(recipe)
    best = None
    for shift in recipe.shifts:
        vector = np.interp(grid-shift, grid, reference.vector, left=np.nan, right=np.nan)
        valid = np.isfinite(vector) & (
            np.interp(grid-shift, grid, reference.mask.astype(float), left=0, right=0) == 1
        )
        keep = query.mask & valid
        if keep.mean() < recipe.min_overlap:
            continue
        support = signed_pearson(query.vector[keep], vector[keep])
        if support is not None and (best is None or support > best["support"]):
            best = {"support": support, "shift_cm_1": shift, "vector": np.nan_to_num(vector),
                    "mask": keep, "extra_fwhm_cm1": recipe.extra_fwhm_cm1[0]}
    return best
