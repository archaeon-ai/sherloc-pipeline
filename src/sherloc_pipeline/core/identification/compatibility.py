"""Model eligibility is stricter than permission to display a native trace."""

import math

from sherloc_pipeline.models.identification import PointSelection
from sherloc_pipeline.models.references import ReferenceTrace


def query_exclusion(selection: PointSelection) -> str | None:
    if (selection.modality, selection.x_quantity, selection.x_unit) != ("raman", "raman_shift", "cm-1"):
        return "unsupported_query_axis"
    if not selection.excitation_evidence.strip():
        return "missing_query_excitation_evidence"
    if selection.processing:
        return "unsupported_query_processing"
    return None


def reference_exclusion(trace: ReferenceTrace, query: PointSelection) -> str | None:
    if (trace.modality, trace.x_quantity, trace.x_unit) != ("raman", "raman_shift", "cm-1"):
        return "unsupported_reference_axis"
    if not trace.x_evidence or not trace.x_evidence.strip():
        return "missing_axis_evidence"
    if (trace.excitation_value is None or not math.isfinite(trace.excitation_value)
            or trace.excitation_unit != "nm" or not trace.excitation_evidence
            or not trace.excitation_evidence.strip()):
        return "missing_excitation_evidence"
    if trace.excitation_value != query.excitation_nm:
        return "incompatible_excitation"
    if not trace.instrument or not trace.y_unit:
        return "missing_instrument_or_intensity_unit"
    if trace.column_role != "measurement" or trace.source_processing:
        return "unsupported_reference_processing"
    if trace.sample_form != "pure" or not trace.species_name:
        return "not_pure_species_reference"
    if trace.qa_flags:
        return "reference_qa_exclusion"
    return None
