"""
Base Pydantic model configuration for PHASE data structures.

This module provides the foundational base models for all PHASE data structures,
implementing the Pydantic-first design philosophy described in PHASE_SPEC.md.

All data flowing through PHASE will be represented as Pydantic models, providing:
- Runtime validation at data boundaries
- JSON/dict serialization for database and API
- Self-documenting schemas with field descriptions
- Full type hints and IDE autocomplete support
- OpenAPI generation for automatic API documentation

Classes:
    PHASEBaseModel: Base model with strict configuration for all PHASE models
    TimestampedModel: Model with automatic created_at/updated_at timestamps
    IdentifiableModel: Model with UUID primary key and timestamps

Example:
    >>> from sherloc_pipeline.models.base import PHASEBaseModel, IdentifiableModel
    >>> from pydantic import Field
    >>>
    >>> class MyModel(PHASEBaseModel):
    ...     name: str = Field(description="A name field")
    ...     value: float = Field(ge=0, description="A positive value")
    >>>
    >>> obj = MyModel(name="test", value=42.0)
    >>> obj.model_dump()
    {'name': 'test', 'value': 42.0}
"""

from datetime import datetime, timezone
from typing import Optional, Any, Dict
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_serializer


def utc_now() -> datetime:
    """Return current UTC datetime with timezone info.

    Returns:
        datetime: Current UTC datetime with tzinfo set to UTC
    """
    return datetime.now(timezone.utc)


class PHASEBaseModel(BaseModel):
    """Base model for all PHASE data structures.

    This base model provides consistent configuration across all PHASE models:

    - **Strict mode**: No extra fields allowed (extra="forbid")
    - **Validation on assignment**: Fields are validated when modified
    - **Enum values**: Enums serialize to their values, not names
    - **ISO8601 timedelta**: Timedeltas serialize in ISO format
    - **Populate by name**: Fields can be set by name or alias

    Example:
        >>> class Spectrum(PHASEBaseModel):
        ...     sol: int = Field(ge=0, description="Mars sol number")
        ...     target: str = Field(min_length=1, description="Target name")
        >>>
        >>> s = Spectrum(sol=921, target="Amherst_Point")
        >>> s.model_dump_json()
        '{"sol":921,"target":"Amherst_Point"}'

    Note:
        All subclasses inherit these configuration options. Override specific
        settings in a subclass's model_config if needed.
    """

    model_config = ConfigDict(
        # Strict mode: no extra fields allowed
        extra="forbid",
        # Enable validation on assignment
        validate_assignment=True,
        # Use enum values for serialization
        use_enum_values=True,
        # Serialize timedelta as ISO format
        ser_json_timedelta="iso8601",
        # Allow population by field name or alias
        populate_by_name=True,
        # Generate JSON schema with better descriptions
        json_schema_extra={"title": "PHASE Model"},
    )


class TimestampedModel(PHASEBaseModel):
    """Model with automatic timestamp tracking.

    Provides created_at and updated_at fields for tracking when records
    are created and modified. The created_at field is automatically set
    to the current UTC time when the model is instantiated.

    Attributes:
        created_at: UTC datetime when the record was created (auto-set)
        updated_at: UTC datetime when the record was last updated (None initially)

    Example:
        >>> class Note(TimestampedModel):
        ...     content: str
        >>>
        >>> note = Note(content="Important finding")
        >>> note.created_at  # doctest: +SKIP
        datetime.datetime(2026, 1, 24, 2, 30, 0, tzinfo=datetime.timezone.utc)
        >>> note.updated_at is None
        True
        >>> note.updated_at = utc_now()  # Mark as updated

    Note:
        The updated_at field is not automatically managed; callers must set
        it explicitly when updating records. This allows flexibility in
        deciding what constitutes an "update" (e.g., only significant changes).
    """

    created_at: datetime = Field(
        default_factory=utc_now,
        description="UTC datetime when the record was created"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="UTC datetime when the record was last updated"
    )

    @field_serializer("created_at", "updated_at")
    def serialize_datetime(self, dt: Optional[datetime]) -> Optional[str]:
        """Serialize datetime fields to ISO 8601 format with timezone.

        Args:
            dt: The datetime to serialize, or None

        Returns:
            ISO 8601 formatted string with timezone, or None
        """
        if dt is None:
            return None
        return dt.isoformat()

    def touch(self) -> None:
        """Update the updated_at timestamp to the current UTC time.

        Convenience method for marking a record as modified. This sets
        the updated_at field to the current UTC datetime.

        Example:
            >>> note = Note(content="test")
            >>> note.updated_at is None
            True
            >>> note.touch()
            >>> note.updated_at is not None  # doctest: +SKIP
            True
        """
        self.updated_at = utc_now()


class IdentifiableModel(TimestampedModel):
    """Model with UUID primary key and timestamps.

    Extends TimestampedModel with a UUID primary key field. Each instance
    gets a unique identifier automatically generated at creation time.

    Attributes:
        id: UUID primary key (auto-generated)
        created_at: UTC datetime when the record was created (from TimestampedModel)
        updated_at: UTC datetime when the record was last updated (from TimestampedModel)

    Example:
        >>> class Spectrum(IdentifiableModel):
        ...     sol: int
        ...     target: str
        >>>
        >>> s = Spectrum(sol=921, target="Amherst_Point")
        >>> s.id  # doctest: +SKIP
        UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')
        >>> s.created_at  # doctest: +SKIP
        datetime.datetime(2026, 1, 24, 2, 30, 0, tzinfo=datetime.timezone.utc)

    Note:
        The UUID is generated using uuid4() which creates a random UUID.
        For database persistence, this ID should be used as the primary key.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier (UUID v4)"
    )

    @field_serializer("id")
    def serialize_uuid(self, id_value: uuid.UUID) -> str:
        """Serialize UUID to string format.

        Args:
            id_value: The UUID to serialize

        Returns:
            String representation of the UUID
        """
        return str(id_value)

    def __hash__(self) -> int:
        """Make the model hashable using its UUID.

        This allows IdentifiableModel instances to be used in sets
        and as dictionary keys.

        Returns:
            Hash of the UUID
        """
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        """Check equality based on UUID.

        Two IdentifiableModel instances are considered equal if they
        have the same UUID, regardless of other field values.

        Args:
            other: Object to compare with

        Returns:
            True if other is an IdentifiableModel with the same UUID
        """
        if not isinstance(other, IdentifiableModel):
            return False
        return self.id == other.id


class ValidatedMixin:
    """Mixin providing common validation utilities.

    This mixin can be combined with PHASEBaseModel subclasses to add
    common validation methods and utilities.

    Methods:
        validate_positive: Ensure a value is positive
        validate_range: Ensure a value is within a range
        validate_non_empty: Ensure a string is not empty or whitespace

    Example:
        >>> class MyModel(PHASEBaseModel, ValidatedMixin):
        ...     value: float
        ...
        ...     def validate_value(self) -> None:
        ...         self.validate_positive(self.value, "value")
    """

    @staticmethod
    def validate_positive(value: float, field_name: str) -> None:
        """Validate that a value is positive (> 0).

        Args:
            value: The value to check
            field_name: Name of the field (for error message)

        Raises:
            ValueError: If value is not positive
        """
        if value <= 0:
            raise ValueError(f"{field_name} must be positive, got {value}")

    @staticmethod
    def validate_range(
        value: float,
        min_val: float,
        max_val: float,
        field_name: str
    ) -> None:
        """Validate that a value is within a range (inclusive).

        Args:
            value: The value to check
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            field_name: Name of the field (for error message)

        Raises:
            ValueError: If value is outside the range
        """
        if not (min_val <= value <= max_val):
            raise ValueError(
                f"{field_name} must be between {min_val} and {max_val}, got {value}"
            )

    @staticmethod
    def validate_non_empty(value: str, field_name: str) -> None:
        """Validate that a string is not empty or whitespace-only.

        Args:
            value: The string to check
            field_name: Name of the field (for error message)

        Raises:
            ValueError: If string is empty or whitespace-only
        """
        if not value or not value.strip():
            raise ValueError(f"{field_name} must not be empty")


class ModelRegistry:
    """Registry for tracking all PHASE model types.

    This registry allows runtime discovery of all registered model types,
    useful for schema generation, migration, and introspection.

    Class Attributes:
        _models: Dictionary mapping model names to model classes

    Example:
        >>> @ModelRegistry.register
        ... class Spectrum(PHASEBaseModel):
        ...     pass
        >>>
        >>> ModelRegistry.get("Spectrum")  # doctest: +SKIP
        <class 'Spectrum'>
    """

    _models: Dict[str, type] = {}

    @classmethod
    def register(cls, model_class: type) -> type:
        """Register a model class (can be used as a decorator).

        Args:
            model_class: The model class to register

        Returns:
            The model class (unchanged)
        """
        cls._models[model_class.__name__] = model_class
        return model_class

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get a registered model class by name.

        Args:
            name: The model class name

        Returns:
            The model class, or None if not registered
        """
        return cls._models.get(name)

    @classmethod
    def all(cls) -> Dict[str, type]:
        """Get all registered model classes.

        Returns:
            Dictionary mapping names to model classes
        """
        return cls._models.copy()

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models (mainly for testing).
        """
        cls._models.clear()
