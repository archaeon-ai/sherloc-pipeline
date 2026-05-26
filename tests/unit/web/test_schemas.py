"""Schema validation tests."""

import pytest
from pydantic import ValidationError

from sherloc_pipeline.web.schemas import (
    API_SCHEMA_VERSION,
    BaselineParamsSchema,
    FitParamsSchema,
    SubsetRequest,
    ScanListResponse,
)


class TestAPISchemaVersion:
    def test_version_string(self):
        assert API_SCHEMA_VERSION == "1.0.0"


class TestBaselineParamsSchema:
    def test_defaults(self):
        p = BaselineParamsSchema()
        assert p.method == "aspls"
        assert p.lam == 1_000_000.0
        assert p.max_iter == 10

    def test_custom_values(self):
        p = BaselineParamsSchema(lam=5e5, max_iter=20)
        assert p.lam == 5e5
        assert p.max_iter == 20


class TestFitParamsSchema:
    def test_defaults(self):
        p = FitParamsSchema()
        assert p.domain == "minerals"
        assert p.max_peaks == 5
        assert p.model_selection == "aicc"

    def test_invalid_domain(self):
        with pytest.raises(ValidationError):
            FitParamsSchema(domain="invalid")

    def test_invalid_model_selection(self):
        with pytest.raises(ValidationError):
            FitParamsSchema(model_selection="bic")

    def test_max_peaks_bounds(self):
        with pytest.raises(ValidationError):
            FitParamsSchema(max_peaks=0)
        with pytest.raises(ValidationError):
            FitParamsSchema(max_peaks=25)


class TestSubsetRequest:
    def test_non_empty_validation(self):
        with pytest.raises(ValidationError):
            SubsetRequest(point_indices=[])

    def test_valid_request(self):
        r = SubsetRequest(point_indices=[0, 1, 2])
        assert r.region == "R1"
        assert len(r.point_indices) == 3


class TestScanListResponse:
    def test_schema_version_default(self):
        r = ScanListResponse(scans=[], total=0, offset=0, limit=50)
        assert r.schema_version == "1.0.0"
