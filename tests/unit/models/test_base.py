"""
Unit tests for PHASE base models (bd-a43: WS2-prep).

Tests the foundational Pydantic models defined in models/base.py:
- PHASEBaseModel: Base model with strict configuration
- TimestampedModel: Model with created_at/updated_at
- IdentifiableModel: Model with UUID + timestamps
- ValidatedMixin: Common validation utilities
- ModelRegistry: Runtime model discovery
"""

import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import pytest
from pydantic import Field, ValidationError

from sherloc_pipeline.models import (
    PHASEBaseModel,
    TimestampedModel,
    IdentifiableModel,
    ValidatedMixin,
    ModelRegistry,
    utc_now,
)


class TestUtcNow:
    """Tests for the utc_now() utility function."""

    def test_returns_datetime(self):
        """utc_now() returns a datetime object."""
        result = utc_now()
        assert isinstance(result, datetime)

    def test_has_timezone_info(self):
        """utc_now() returns datetime with tzinfo set."""
        result = utc_now()
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc

    def test_is_current_time(self):
        """utc_now() returns approximately current time."""
        before = datetime.now(timezone.utc)
        result = utc_now()
        after = datetime.now(timezone.utc)
        assert before <= result <= after


class TestPHASEBaseModel:
    """Tests for PHASEBaseModel configuration."""

    def test_basic_model_creation(self):
        """Create a simple model with PHASEBaseModel."""

        class SimpleModel(PHASEBaseModel):
            name: str
            value: int

        model = SimpleModel(name="test", value=42)
        assert model.name == "test"
        assert model.value == 42

    def test_extra_fields_forbidden(self):
        """Extra fields should raise ValidationError."""

        class StrictModel(PHASEBaseModel):
            name: str

        with pytest.raises(ValidationError) as exc_info:
            StrictModel(name="test", extra_field="forbidden")

        assert "extra_field" in str(exc_info.value)

    def test_validate_assignment(self):
        """Field validation happens on assignment."""

        class ValidatedModel(PHASEBaseModel):
            value: int = Field(ge=0)

        model = ValidatedModel(value=10)
        assert model.value == 10

        with pytest.raises(ValidationError):
            model.value = -5

    def test_enum_values_serialization(self):
        """Enums serialize to their values."""
        from enum import Enum

        class Status(str, Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        class WithEnum(PHASEBaseModel):
            status: Status

        model = WithEnum(status=Status.ACTIVE)
        data = model.model_dump()
        assert data["status"] == "active"

    def test_json_serialization(self):
        """Models serialize to valid JSON."""

        class JsonModel(PHASEBaseModel):
            name: str
            values: List[float]

        model = JsonModel(name="test", values=[1.0, 2.0, 3.0])
        json_str = model.model_dump_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == "test"
        assert parsed["values"] == [1.0, 2.0, 3.0]

    def test_populate_by_name_or_alias(self):
        """Fields can be set by name or alias."""

        class AliasedModel(PHASEBaseModel):
            target_name: str = Field(alias="targetName")

        # By name
        model1 = AliasedModel(target_name="test1")
        assert model1.target_name == "test1"

        # By alias
        model2 = AliasedModel(targetName="test2")
        assert model2.target_name == "test2"

    def test_field_descriptions(self):
        """Field descriptions are preserved in schema."""

        class DescribedModel(PHASEBaseModel):
            sol: int = Field(ge=0, description="Mars sol number")

        schema = DescribedModel.model_json_schema()
        assert "Mars sol number" in str(schema)

    def test_nested_model_validation(self):
        """Nested models are validated."""

        class Inner(PHASEBaseModel):
            value: int = Field(ge=0)

        class Outer(PHASEBaseModel):
            inner: Inner

        # Valid nested
        outer = Outer(inner=Inner(value=10))
        assert outer.inner.value == 10

        # Invalid nested
        with pytest.raises(ValidationError):
            Outer(inner=Inner(value=-1))


class TestTimestampedModel:
    """Tests for TimestampedModel with automatic timestamps."""

    def test_created_at_auto_set(self):
        """created_at is automatically set on instantiation."""

        class Note(TimestampedModel):
            content: str

        before = utc_now()
        note = Note(content="test")
        after = utc_now()

        assert note.created_at is not None
        assert before <= note.created_at <= after

    def test_updated_at_initially_none(self):
        """updated_at is None when first created."""

        class Note(TimestampedModel):
            content: str

        note = Note(content="test")
        assert note.updated_at is None

    def test_updated_at_can_be_set(self):
        """updated_at can be set manually."""

        class Note(TimestampedModel):
            content: str

        note = Note(content="test")
        update_time = utc_now()
        note.updated_at = update_time

        assert note.updated_at == update_time

    def test_touch_method(self):
        """touch() sets updated_at to current time."""

        class Note(TimestampedModel):
            content: str

        note = Note(content="test")
        assert note.updated_at is None

        before = utc_now()
        note.touch()
        after = utc_now()

        assert note.updated_at is not None
        assert before <= note.updated_at <= after

    def test_datetime_serialization_iso8601(self):
        """Timestamps serialize to ISO 8601 format."""

        class Note(TimestampedModel):
            content: str

        note = Note(content="test")
        data = note.model_dump()

        # created_at should be ISO string
        assert isinstance(data["created_at"], str)
        assert "T" in data["created_at"]  # ISO format has T separator
        assert "+" in data["created_at"] or "Z" in data["created_at"]  # Has timezone

        # updated_at should be None
        assert data["updated_at"] is None

    def test_explicit_created_at(self):
        """created_at can be set explicitly."""

        class Note(TimestampedModel):
            content: str

        specific_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        note = Note(content="test", created_at=specific_time)

        assert note.created_at == specific_time

    def test_inherits_base_config(self):
        """TimestampedModel inherits PHASEBaseModel configuration."""

        class Note(TimestampedModel):
            content: str

        # Extra fields still forbidden
        with pytest.raises(ValidationError):
            Note(content="test", extra="forbidden")


class TestIdentifiableModel:
    """Tests for IdentifiableModel with UUID primary key."""

    def test_id_auto_generated(self):
        """id is automatically generated as UUID."""

        class Spectrum(IdentifiableModel):
            sol: int

        spectrum = Spectrum(sol=921)

        assert spectrum.id is not None
        assert isinstance(spectrum.id, uuid.UUID)

    def test_id_is_unique(self):
        """Each instance gets a unique id."""

        class Spectrum(IdentifiableModel):
            sol: int

        s1 = Spectrum(sol=921)
        s2 = Spectrum(sol=921)

        assert s1.id != s2.id

    def test_id_is_uuid4(self):
        """Generated id is a valid UUID v4."""

        class Spectrum(IdentifiableModel):
            sol: int

        spectrum = Spectrum(sol=921)

        # UUID v4 has version 4
        assert spectrum.id.version == 4

    def test_explicit_id(self):
        """id can be set explicitly."""

        class Spectrum(IdentifiableModel):
            sol: int

        explicit_id = uuid.uuid4()
        spectrum = Spectrum(id=explicit_id, sol=921)

        assert spectrum.id == explicit_id

    def test_id_serialization(self):
        """UUID serializes to string format."""

        class Spectrum(IdentifiableModel):
            sol: int

        spectrum = Spectrum(sol=921)
        data = spectrum.model_dump()

        assert isinstance(data["id"], str)
        # Should be parseable as UUID
        parsed = uuid.UUID(data["id"])
        assert parsed == spectrum.id

    def test_hashable(self):
        """IdentifiableModel instances are hashable."""

        class Spectrum(IdentifiableModel):
            sol: int

        spectrum = Spectrum(sol=921)

        # Should be usable in a set
        spectrum_set = {spectrum}
        assert spectrum in spectrum_set

        # Hash is based on id
        assert hash(spectrum) == hash(spectrum.id)

    def test_equality_by_id(self):
        """Equality is based on id, not field values."""

        class Spectrum(IdentifiableModel):
            sol: int
            target: str

        shared_id = uuid.uuid4()

        s1 = Spectrum(id=shared_id, sol=921, target="Target_A")
        s2 = Spectrum(id=shared_id, sol=999, target="Target_B")

        # Same id = equal
        assert s1 == s2

        # Different id = not equal
        s3 = Spectrum(sol=921, target="Target_A")
        assert s1 != s3

    def test_inequality_with_non_identifiable(self):
        """IdentifiableModel is not equal to non-IdentifiableModel objects."""

        class Spectrum(IdentifiableModel):
            sol: int

        spectrum = Spectrum(sol=921)

        assert spectrum != "not a model"
        assert spectrum != 42
        assert spectrum != {"id": str(spectrum.id)}

    def test_has_timestamps(self):
        """IdentifiableModel includes timestamps from TimestampedModel."""

        class Spectrum(IdentifiableModel):
            sol: int

        spectrum = Spectrum(sol=921)

        assert hasattr(spectrum, "created_at")
        assert hasattr(spectrum, "updated_at")
        assert spectrum.created_at is not None
        assert spectrum.updated_at is None

    def test_full_serialization(self):
        """Full model serializes correctly."""

        class Spectrum(IdentifiableModel):
            sol: int
            target: str

        spectrum = Spectrum(sol=921, target="Amherst_Point")
        data = spectrum.model_dump()

        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert "sol" in data
        assert "target" in data


class TestValidatedMixin:
    """Tests for ValidatedMixin utility methods."""

    def test_validate_positive_success(self):
        """validate_positive passes for positive values."""
        ValidatedMixin.validate_positive(1.0, "value")
        ValidatedMixin.validate_positive(0.001, "value")
        ValidatedMixin.validate_positive(1000, "value")

    def test_validate_positive_failure(self):
        """validate_positive raises for non-positive values."""
        with pytest.raises(ValueError, match="must be positive"):
            ValidatedMixin.validate_positive(0, "value")

        with pytest.raises(ValueError, match="must be positive"):
            ValidatedMixin.validate_positive(-5, "value")

    def test_validate_range_success(self):
        """validate_range passes for values in range."""
        ValidatedMixin.validate_range(5, 0, 10, "value")
        ValidatedMixin.validate_range(0, 0, 10, "value")  # inclusive min
        ValidatedMixin.validate_range(10, 0, 10, "value")  # inclusive max

    def test_validate_range_failure(self):
        """validate_range raises for values outside range."""
        with pytest.raises(ValueError, match="must be between"):
            ValidatedMixin.validate_range(-1, 0, 10, "value")

        with pytest.raises(ValueError, match="must be between"):
            ValidatedMixin.validate_range(11, 0, 10, "value")

    def test_validate_non_empty_success(self):
        """validate_non_empty passes for non-empty strings."""
        ValidatedMixin.validate_non_empty("test", "field")
        ValidatedMixin.validate_non_empty("  test  ", "field")

    def test_validate_non_empty_failure(self):
        """validate_non_empty raises for empty/whitespace strings."""
        with pytest.raises(ValueError, match="must not be empty"):
            ValidatedMixin.validate_non_empty("", "field")

        with pytest.raises(ValueError, match="must not be empty"):
            ValidatedMixin.validate_non_empty("   ", "field")

    def test_mixin_with_model(self):
        """ValidatedMixin can be combined with PHASEBaseModel."""

        class MyModel(PHASEBaseModel, ValidatedMixin):
            value: float
            name: str

            def validate_all(self) -> None:
                self.validate_positive(self.value, "value")
                self.validate_non_empty(self.name, "name")

        model = MyModel(value=10, name="test")
        model.validate_all()  # Should pass

        model2 = MyModel(value=-5, name="test")
        with pytest.raises(ValueError):
            model2.validate_all()


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        ModelRegistry.clear()

    def test_register_decorator(self):
        """@ModelRegistry.register works as decorator."""

        @ModelRegistry.register
        class MyModel(PHASEBaseModel):
            value: int

        assert ModelRegistry.get("MyModel") is MyModel

    def test_register_returns_class(self):
        """register() returns the class unchanged."""

        class MyModel(PHASEBaseModel):
            value: int

        result = ModelRegistry.register(MyModel)
        assert result is MyModel

    def test_get_unregistered(self):
        """get() returns None for unregistered models."""
        assert ModelRegistry.get("NotRegistered") is None

    def test_all_returns_copy(self):
        """all() returns a copy of the registry."""

        @ModelRegistry.register
        class Model1(PHASEBaseModel):
            pass

        @ModelRegistry.register
        class Model2(PHASEBaseModel):
            pass

        all_models = ModelRegistry.all()
        assert "Model1" in all_models
        assert "Model2" in all_models

        # Modifications don't affect registry
        all_models["Model3"] = None
        assert ModelRegistry.get("Model3") is None

    def test_clear(self):
        """clear() removes all registered models."""

        @ModelRegistry.register
        class MyModel(PHASEBaseModel):
            pass

        assert ModelRegistry.get("MyModel") is not None

        ModelRegistry.clear()
        assert ModelRegistry.get("MyModel") is None


class TestModelIntegration:
    """Integration tests combining multiple base model features."""

    def test_complex_model_hierarchy(self):
        """Test a realistic model hierarchy."""

        class Spectrum(IdentifiableModel):
            sol: int = Field(ge=0, description="Mars sol number")
            target: str = Field(min_length=1)
            x_values: List[float]
            y_values: List[float]

        spectrum = Spectrum(
            sol=921,
            target="Amherst_Point",
            x_values=[700.0, 800.0, 900.0],
            y_values=[100.0, 150.0, 120.0],
        )

        # Check all fields
        assert spectrum.sol == 921
        assert spectrum.target == "Amherst_Point"
        assert len(spectrum.x_values) == 3
        assert len(spectrum.y_values) == 3

        # Check inherited fields
        assert spectrum.id is not None
        assert spectrum.created_at is not None

        # Serialization
        data = spectrum.model_dump()
        assert all(k in data for k in ["id", "created_at", "sol", "target"])

    def test_json_round_trip(self):
        """Model survives JSON serialization round-trip."""

        class Spectrum(IdentifiableModel):
            sol: int
            target: str

        original = Spectrum(sol=921, target="Amherst_Point")

        # Serialize
        json_str = original.model_dump_json()

        # Deserialize
        restored = Spectrum.model_validate_json(json_str)

        # Compare (using id equality)
        assert restored.id == original.id
        assert restored.sol == original.sol
        assert restored.target == original.target

    def test_model_with_optional_fields(self):
        """Models with optional fields work correctly."""

        class Spectrum(IdentifiableModel):
            sol: int
            target: str
            notes: Optional[str] = None
            quality: Optional[float] = None

        # Without optional
        s1 = Spectrum(sol=921, target="Test")
        assert s1.notes is None
        assert s1.quality is None

        # With optional
        s2 = Spectrum(sol=921, target="Test", notes="Good signal", quality=0.95)
        assert s2.notes == "Good signal"
        assert s2.quality == 0.95
