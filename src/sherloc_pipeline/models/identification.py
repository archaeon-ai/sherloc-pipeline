"""Contracts for one-point, provisional Raman candidate support."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from sherloc_pipeline.models.references import Digest, Name


class IdentificationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class MethodRecipe(IdentificationModel):
    method_id: Literal["raman-pearson"]
    version: Literal["0.2.0"]
    status: Literal["provisional"]
    support_min: Literal[0.80]
    threshold_source: Literal["fixed_provisional"]
    evaluation_unit: Literal["point"]
    grid: list[float]
    shifts: list[float]
    extra_fwhm_cm1: list[Literal[0.0]] = Field(min_length=1, max_length=1)
    excluded_intervals: list[list[float]]
    min_coverage: float
    min_overlap: float
    max_gap: float
    min_samples: int
    baseline_lambda: float
    baseline_iterations: int
    baseline_asymmetry: float
    snr_min: float


class MixtureRecipe(MethodRecipe):
    method_id: Literal["raman-mixture"]
    version: Literal["0.1.0"]
    support_min: Literal[0.60]
    extra_fwhm_cm1: list[float]
    overlap_basis: Literal["eligible_after_exclusions_broadening_and_shift"]
    retrieval_metric: Literal["cosine"]
    retrieval_count: Literal[5]
    fwhm_to_sigma: float
    smoothing_mode: Literal["nearest"]
    l1_penalty: float
    background_cap: float
    background_basis: Literal["three_linear_hats_unit_l2"]
    max_components: Literal[2]
    coefficient_cutoff: float
    unexplained_support_min: float
    solver_iterations: list[int] = Field(min_length=3, max_length=3)
    solver_tolerance: float
    denominator_floor: float
    weight_basis: Literal["normalized_spectral_contributions"]


class PointSelection(IdentificationModel):
    product_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,200}$")]
    sol: Annotated[int, Field(ge=0)]
    target: Name
    point_index: Annotated[int, Field(ge=0)]
    source_sha256: Digest
    modality: Literal["raman", "fluorescence", "unknown"]
    x_quantity: Literal["raman_shift", "wavelength", "unknown"]
    x_unit: Literal["cm-1", "nm", "unknown"]
    excitation_nm: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    excitation_evidence: Name
    processing: list[str] = Field(default_factory=list)


class IdentificationRequest(IdentificationModel):
    selection: PointSelection
    reference_release_id: Digest
    reference_subset_id: Digest
    method_release_id: Digest
    template_ids: list[Digest] = Field(min_length=1, max_length=10000)


class Candidate(IdentificationModel):
    template_id: Digest
    species_name: Name
    supported_class: str | None
    trace_revision_ids: list[Digest]
    support: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)] | None
    retrieval_score: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)] | None = None
    shift_cm_1: float
    extra_fwhm_cm1: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    citations: list[dict[str, Any]] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def serialize_candidate(self, handler):
        payload = handler(self)
        if self.retrieval_score is None:
            payload.pop("retrieval_score", None)
        return payload


class MixtureComponent(IdentificationModel):
    template_id: Digest
    species_name: Name
    coefficient: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    weight: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class MixtureResult(IdentificationModel):
    support_name: Literal["background_relative_sse"] = "background_relative_sse"
    support: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None
    weight_basis: Literal["normalized_spectral_contributions"] = "normalized_spectral_contributions"
    components: list[MixtureComponent] = Field(max_length=2)
    sse: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    background_only_sse: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    background_coefficients: list[float]
    background_only_coefficients: list[float]
    diagnostics: dict[str, Any]


class IdentificationResult(IdentificationModel):
    schema_version: Literal["identification-1"] = "identification-1"
    status: Literal["candidate", "abstained"]
    reasons: list[str]
    warnings: list[str] = Field(default_factory=lambda: ["provisional_method", "experimental_candidates"])
    method: dict[str, Any]
    selection: PointSelection
    reference_release_id: Digest
    reference_subset_id: Digest
    effective_set_sha256: Digest
    ranking: list[Candidate]
    candidate: str | None
    mixture: MixtureResult | None = None
    diagnostics: dict[str, Any]
    evidence: dict[str, Any]
    result_digest: Digest


    @model_serializer(mode="wrap")
    def serialize_result(self, handler):
        payload = handler(self)
        if self.mixture is None:
            payload.pop("mixture", None)
        return payload
