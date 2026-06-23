"""
Base classes for PHASE ML infrastructure.

This module provides foundational abstractions for ML models and transformers,
following scikit-learn-like interfaces for consistency and familiarity.

Classes:
    MLBaseModel: Base Pydantic model for ML configurations
    Transformer: Base class for data transformations (fit_transform pattern)
    Estimator: Base class for models that learn from data
    ModelPersistence: Mixin for saving/loading models

Example:
    >>> from sherloc_pipeline.ml.base import Transformer, Estimator
    >>>
    >>> class MyNormalizer(Transformer):
    ...     def fit(self, X):
    ...         self.mean_ = X.mean(axis=0)
    ...         self.std_ = X.std(axis=0)
    ...         return self
    ...
    ...     def transform(self, X):
    ...         return (X - self.mean_) / self.std_
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import pickle

import numpy as np
from pydantic import Field

from sherloc_pipeline.models.base import PHASEBaseModel, utc_now


class MLBaseModel(PHASEBaseModel):
    """Base Pydantic model for ML configuration objects.

    Extends PHASEBaseModel with ML-specific utilities like
    parameter serialization and compatibility checking.

    Attributes:
        name: Human-readable name for the model/configuration
        description: Optional description of purpose
        version: Version string for compatibility tracking

    Example:
        >>> class ClusterConfig(MLBaseModel):
        ...     n_clusters: int = Field(ge=1, default=5)
        ...     random_state: int = 42
        >>>
        >>> config = ClusterConfig(name="my_config")
    """

    name: str = Field(
        default="unnamed",
        description="Human-readable name for this configuration"
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of purpose"
    )
    version: str = Field(
        default="1.0.0",
        description="Version string for compatibility tracking"
    )

    def get_params(self) -> Dict[str, Any]:
        """Get all configuration parameters as a dictionary.

        Returns:
            Dictionary of parameter names to values
        """
        return self.model_dump(exclude={"name", "description", "version"})


class Transformer(ABC):
    """Base class for data transformations.

    Implements the scikit-learn-style fit/transform pattern for
    data preprocessing and feature extraction.

    The fit() method learns parameters from data, and transform()
    applies the learned transformation. fit_transform() combines both.

    Subclasses must implement:
        - fit(X): Learn transformation parameters
        - transform(X): Apply the transformation

    Example:
        >>> class Normalizer(Transformer):
        ...     def fit(self, X):
        ...         self.mean_ = np.mean(X, axis=0)
        ...         self.std_ = np.std(X, axis=0)
        ...         return self
        ...
        ...     def transform(self, X):
        ...         return (X - self.mean_) / self.std_
        >>>
        >>> norm = Normalizer()
        >>> X_normalized = norm.fit_transform(data)
    """

    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Transformer":
        """Fit the transformer to the data.

        Args:
            X: Input data array of shape (n_samples, n_features)
            y: Optional target values (ignored by most transformers)

        Returns:
            self: The fitted transformer instance
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned transformation to data.

        Args:
            X: Input data array of shape (n_samples, n_features)

        Returns:
            Transformed data array

        Raises:
            RuntimeError: If transform() called before fit()
        """
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit to data and then transform it.

        Convenience method equivalent to calling fit(X) then transform(X).

        Args:
            X: Input data array of shape (n_samples, n_features)
            y: Optional target values (ignored by most transformers)

        Returns:
            Transformed data array
        """
        return self.fit(X, y).transform(X)

    def _check_is_fitted(self) -> None:
        """Check if the transformer has been fitted.

        Raises:
            RuntimeError: If not yet fitted
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call fit() before transform()."
            )


class Estimator(ABC):
    """Base class for ML estimators (models that learn from data).

    Provides a consistent interface for training and prediction,
    following scikit-learn conventions.

    Subclasses must implement:
        - fit(X, y): Train the model
        - predict(X): Make predictions

    Example:
        >>> class SimpleClassifier(Estimator):
        ...     def fit(self, X, y):
        ...         self.classes_ = np.unique(y)
        ...         return self
        ...
        ...     def predict(self, X):
        ...         return np.zeros(len(X))  # Always predict class 0
    """

    def __init__(self):
        self._is_fitted = False
        self._fit_timestamp: Optional[datetime] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Estimator":
        """Fit the estimator to training data.

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values (optional for unsupervised methods)

        Returns:
            self: The fitted estimator instance
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for input samples.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Predictions array of shape (n_samples,)

        Raises:
            RuntimeError: If predict() called before fit()
        """
        pass

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit to data and then predict.

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values (optional for unsupervised methods)

        Returns:
            Predictions array
        """
        return self.fit(X, y).predict(X)

    def _check_is_fitted(self) -> None:
        """Check if the estimator has been fitted.

        Raises:
            RuntimeError: If not yet fitted
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call fit() before predict()."
            )

    def _mark_fitted(self) -> None:
        """Mark the estimator as fitted with current timestamp."""
        self._is_fitted = True
        self._fit_timestamp = utc_now()


class ModelPersistence:
    """Mixin class providing save/load functionality for models.

    Supports both pickle (full model) and JSON (parameters only)
    serialization formats.

    Example:
        >>> class MyModel(Estimator, ModelPersistence):
        ...     pass
        >>>
        >>> model = MyModel()
        >>> model.fit(X)
        >>> model.save("model.pkl")
        >>>
        >>> loaded = MyModel.load("model.pkl")
    """

    def save(self, path: Union[str, Path], format: str = "pickle") -> None:
        """Save the model to disk.

        Args:
            path: File path to save to
            format: Serialization format ("pickle" or "json")

        Raises:
            ValueError: If format is not supported
        """
        path = Path(path)

        if format == "pickle":
            with open(path, "wb") as f:
                pickle.dump(self, f)
        elif format == "json":
            # Save parameters only (for retraining)
            params = self._get_saveable_params()
            with open(path, "w") as f:
                json.dump(params, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'.")

    @classmethod
    def load(cls, path: Union[str, Path], format: str = "pickle") -> "ModelPersistence":
        """Load a model from disk.

        Args:
            path: File path to load from
            format: Serialization format ("pickle" or "json")

        Returns:
            Loaded model instance

        Raises:
            ValueError: If format is not supported
        """
        path = Path(path)

        if format == "pickle":
            with open(path, "rb") as f:
                return pickle.load(f)
        elif format == "json":
            with open(path, "r") as f:
                params = json.load(f)
            return cls._from_params(params)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'.")

    def _get_saveable_params(self) -> Dict[str, Any]:
        """Get parameters suitable for JSON serialization.

        Override in subclasses to customize saved parameters.

        Returns:
            Dictionary of serializable parameters
        """
        return {
            "class": self.__class__.__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def _from_params(cls, params: Dict[str, Any]) -> "ModelPersistence":
        """Reconstruct model from saved parameters.

        Override in subclasses to customize loading.

        Args:
            params: Dictionary of saved parameters

        Returns:
            New model instance
        """
        return cls()
