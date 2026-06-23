"""
Unit tests for ML base classes.

Tests the foundational classes in ml/base.py:
- MLBaseModel
- Transformer
- Estimator
- ModelPersistence
"""

import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pytest

from sherloc_pipeline.ml.base import (
    MLBaseModel,
    Transformer,
    Estimator,
    ModelPersistence,
)


class TestMLBaseModel:
    """Tests for MLBaseModel."""

    def test_basic_creation(self):
        """Create basic MLBaseModel instance."""

        class TestModel(MLBaseModel):
            value: int = 42

        model = TestModel(name="test")
        assert model.name == "test"
        assert model.value == 42
        assert model.version == "1.0.0"

    def test_get_params(self):
        """get_params returns configuration dict."""

        class TestModel(MLBaseModel):
            a: int = 1
            b: str = "hello"

        model = TestModel()
        params = model.get_params()

        assert params["a"] == 1
        assert params["b"] == "hello"
        assert "name" not in params  # Excluded
        assert "version" not in params  # Excluded

    def test_description_optional(self):
        """Description is optional."""

        class TestModel(MLBaseModel):
            pass

        model = TestModel()
        assert model.description is None

        model_with_desc = TestModel(description="A test model")
        assert model_with_desc.description == "A test model"


class TestTransformer:
    """Tests for Transformer base class."""

    def test_abstract_methods(self):
        """Transformer requires fit and transform implementation."""

        class IncompleteTransformer(Transformer):
            pass

        with pytest.raises(TypeError):
            IncompleteTransformer()

    def test_concrete_transformer(self):
        """Concrete transformer works correctly."""

        class ScaleTransformer(Transformer):
            def __init__(self, scale: float = 1.0):
                super().__init__()
                self.scale = scale

            def fit(self, X, y=None):
                self._is_fitted = True
                return self

            def transform(self, X):
                self._check_is_fitted()
                return X * self.scale

        transformer = ScaleTransformer(scale=2.0)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        # fit_transform
        result = transformer.fit_transform(X)
        assert np.allclose(result, X * 2.0)

    def test_not_fitted_error(self):
        """Transform before fit raises error."""

        class SimpleTransformer(Transformer):
            def fit(self, X, y=None):
                self._is_fitted = True
                return self

            def transform(self, X):
                self._check_is_fitted()
                return X

        transformer = SimpleTransformer()

        with pytest.raises(RuntimeError, match="not been fitted"):
            transformer.transform(np.array([[1.0]]))


class TestEstimator:
    """Tests for Estimator base class."""

    def test_abstract_methods(self):
        """Estimator requires fit and predict implementation."""

        class IncompleteEstimator(Estimator):
            pass

        with pytest.raises(TypeError):
            IncompleteEstimator()

    def test_concrete_estimator(self):
        """Concrete estimator works correctly."""

        class ConstantPredictor(Estimator):
            def __init__(self, value: float = 0.0):
                super().__init__()
                self.value = value

            def fit(self, X, y=None):
                self._mark_fitted()
                return self

            def predict(self, X):
                self._check_is_fitted()
                return np.full(len(X), self.value)

        predictor = ConstantPredictor(value=42.0)
        X = np.array([[1.0], [2.0], [3.0]])

        # fit_predict
        predictions = predictor.fit_predict(X)
        assert np.allclose(predictions, [42.0, 42.0, 42.0])

    def test_not_fitted_error(self):
        """Predict before fit raises error."""

        class SimpleEstimator(Estimator):
            def fit(self, X, y=None):
                self._mark_fitted()
                return self

            def predict(self, X):
                self._check_is_fitted()
                return np.zeros(len(X))

        estimator = SimpleEstimator()

        with pytest.raises(RuntimeError, match="not been fitted"):
            estimator.predict(np.array([[1.0]]))

    def test_fit_timestamp(self):
        """Fit records timestamp."""

        class SimpleEstimator(Estimator):
            def fit(self, X, y=None):
                self._mark_fitted()
                return self

            def predict(self, X):
                return np.zeros(len(X))

        estimator = SimpleEstimator()
        assert estimator._fit_timestamp is None

        estimator.fit(np.array([[1.0]]))
        assert estimator._fit_timestamp is not None
        assert isinstance(estimator._fit_timestamp, datetime)


class TestModelPersistence:
    """Tests for ModelPersistence mixin.

    Note: We use KMeansClusterer for pickle tests because locally-defined
    classes cannot be pickled. The JSON tests use a local class since
    JSON serialization only saves parameters, not the class itself.
    """

    def test_save_pickle(self, tmp_path):
        """Save model as pickle using KMeansClusterer."""
        from sherloc_pipeline.ml.clustering import KMeansClusterer

        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        X = np.array([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]])
        clusterer.fit(X)

        save_path = tmp_path / "model.pkl"
        clusterer.save(save_path, format="pickle")

        assert save_path.exists()

    def test_load_pickle(self, tmp_path):
        """Load model from pickle using KMeansClusterer."""
        from sherloc_pipeline.ml.clustering import KMeansClusterer

        clusterer = KMeansClusterer(n_clusters=2, random_state=42)
        X = np.array([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]])
        clusterer.fit(X)

        save_path = tmp_path / "model.pkl"
        clusterer.save(save_path, format="pickle")

        loaded = KMeansClusterer.load(save_path, format="pickle")

        assert loaded.n_clusters == 2
        assert loaded.centroids_ is not None
        assert loaded._is_fitted

    def test_save_json(self, tmp_path):
        """Save parameters as JSON."""

        class LocalEstimator(Estimator, ModelPersistence):
            def __init__(self, param: int = 0):
                Estimator.__init__(self)
                self.param = param

            def fit(self, X, y=None):
                self._mark_fitted()
                return self

            def predict(self, X):
                return np.zeros(len(X))

            def _get_saveable_params(self):
                params = ModelPersistence._get_saveable_params(self)
                params["param"] = self.param
                return params

        estimator = LocalEstimator(param=7)
        estimator.fit(np.array([[1.0]]))

        save_path = tmp_path / "params.json"
        estimator.save(save_path, format="json")

        assert save_path.exists()

        with open(save_path) as f:
            params = json.load(f)

        assert params["param"] == 7

    def test_invalid_format(self, tmp_path):
        """Invalid format raises error."""
        from sherloc_pipeline.ml.clustering import KMeansClusterer

        clusterer = KMeansClusterer(n_clusters=2)

        with pytest.raises(ValueError, match="Unsupported format"):
            clusterer.save(tmp_path / "model.xyz", format="xyz")

    def test_pickle_preserves_state(self, tmp_path):
        """Pickle preserves full model state including learned parameters."""
        from sherloc_pipeline.ml.clustering import KMeansClusterer

        clusterer = KMeansClusterer(n_clusters=2, random_state=42)
        X = np.array([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]])
        clusterer.fit(X)

        save_path = tmp_path / "model.pkl"
        clusterer.save(save_path)

        loaded = KMeansClusterer.load(save_path)

        # Can predict with loaded model
        predictions = loaded.predict(np.array([[0.0, 0.0], [10.0, 10.0]]))
        assert len(predictions) == 2
