"""Adapter conversion tests."""

from datetime import datetime, timezone

import numpy as np
import pytest

from sherloc_pipeline.web.adapters import _format_dt, numpy_to_list


class TestFormatDt:
    def test_none(self):
        assert _format_dt(None) is None

    def test_utc_datetime(self):
        dt = datetime(2026, 3, 19, 14, 30, 0, tzinfo=timezone.utc)
        assert _format_dt(dt) == "2026-03-19T14:30:00Z"

    def test_naive_datetime_treated_as_utc(self):
        dt = datetime(2026, 3, 19, 14, 30, 0)
        result = _format_dt(dt)
        assert result.endswith("Z")


class TestNumpyToList:
    def test_basic(self):
        arr = np.array([1.0, 2.5, 3.7])
        result = numpy_to_list(arr)
        assert result == [1.0, 2.5, 3.7]
        assert all(isinstance(v, float) for v in result)

    def test_empty(self):
        assert numpy_to_list(np.array([])) == []

    def test_integer_array(self):
        arr = np.array([1, 2, 3])
        result = numpy_to_list(arr)
        assert all(isinstance(v, float) for v in result)
