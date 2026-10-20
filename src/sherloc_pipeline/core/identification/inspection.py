"""Scale/offset inspection on prepared vectors; does not alter support."""

import numpy as np


def inspection_fit(query, reference, mask) -> dict:
    design = np.column_stack([reference[mask], np.ones(int(mask.sum()))])
    scale, offset = np.linalg.lstsq(design, query[mask], rcond=None)[0]
    fitted = scale * reference + offset
    residual = query - fitted
    return {"space": "prepared", "interpretation": "mathematical_inspection",
            "inverted": bool(scale < 0), "scale": float(scale), "offset": float(offset),
            "fit": fitted.tolist(), "residual": residual.tolist(),
            "mask": mask.tolist(), "sse": float(residual[mask] @ residual[mask])}
