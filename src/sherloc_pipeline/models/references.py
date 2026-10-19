"""Versioned local reference snapshot and serving-policy contracts."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1, max_length=1000)]


class ReferenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ReferenceTrace(ReferenceModel):
    trace_index: Annotated[int, Field(ge=0)]
    source_label: Name
    x_key: Name
    y_key: Name
    modality: Literal["raman", "fluorescence", "unknown"]
    x_quantity: Literal["raman_shift", "wavelength", "unknown"]
    x_unit: Literal["cm-1", "nm", "unknown"]
    x_evidence: str | None = None
    y_unit: str | None = None
    excitation_value: float | None = None
    excitation_unit: str | None = None
    excitation_evidence: str | None = None
    instrument: str | None = None
    species_name: str | None = None
    supported_class: str | None = None
    specimen_key: Name
    specimen_is_proxy: bool = True
    sample_form: str | None = None
    source_processing: list[str] = Field(default_factory=list)
    column_role: Literal["measurement", "uncertainty", "source_mean"] = "measurement"
    qa_flags: list[str] = Field(default_factory=list)
    qa_evidence: dict[str, Any] = Field(default_factory=dict)
    citation_text: str | None = None
    citation_doi: str | None = None
    citation_source: str | None = None
    license_identifier: str | None = None
    license_text: str | None = None
    license_source: str | None = None
    rights_status: Literal["unknown", "known"] = "unknown"

    @model_validator(mode="after")
    def compatible_units(self):
        if (self.x_quantity, self.x_unit) not in {
            ("raman_shift", "cm-1"), ("wavelength", "nm"), ("unknown", "unknown"),
        }:
            raise ValueError("inconsistent native axis quantity/unit")
        return self


class ReferenceFile(ReferenceModel):
    datatype_id: Name
    record_id: Name
    file_id: Name
    original_filename: Name
    source_url: str | None = None
    media_type: Name
    raw_path: Name
    raw_sha256: Digest
    raw_size: Annotated[int, Field(ge=0)]
    arrays_path: Name
    arrays_sha256: Digest
    arrays_size: Annotated[int, Field(ge=1)]
    original_metadata: dict[str, Any] = Field(default_factory=dict)
    parse_adjustments: list[str] = Field(default_factory=list)
    traces: list[ReferenceTrace] = Field(min_length=1, max_length=10000)


class TemplateMember(ReferenceModel):
    record_id: Name
    file_id: Name
    trace_index: Annotated[int, Field(ge=0)]
    weight: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0


class ReferenceTemplate(ReferenceModel):
    name: Name
    species_name: Name
    supported_class: str | None = None
    recipe_sha256: Digest
    grouping_evidence: Name
    members: list[TemplateMember] = Field(min_length=1, max_length=10000)
    compatibility_exclusions: list[str] = Field(default_factory=list)


class ReferenceSnapshot(ReferenceModel):
    schema_version: Literal["reference-snapshot-1"]
    source_name: Name
    source_base_url: str | None = None
    captured_at: datetime
    parser_version: Name
    qa_version: Name
    files: list[ReferenceFile] = Field(min_length=1, max_length=10000)
    templates: list[ReferenceTemplate] = Field(default_factory=list, max_length=10000)

    @model_validator(mode="after")
    def utc_capture(self):
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset().total_seconds() != 0:
            raise ValueError("captured_at must be UTC")
        return self


class ServingPolicy(ReferenceModel):
    schema_version: Literal["reference-serving-1"] = "reference-serving-1"
    approved_trace_keys: list[Name] = Field(default_factory=list)


def trace_key(record_id: str, file_id: str, trace_index: int) -> str:
    return f"{record_id}:{file_id}:{trace_index}"


def known_rights(trace: ReferenceTrace) -> bool:
    return bool(
        trace.rights_status == "known"
        and (trace.license_identifier and trace.license_identifier.strip()
         or trace.license_text and trace.license_text.strip())
        and trace.license_source and trace.license_source.strip()
    )
