"""Golden-vector tests for the certified 8-channel featurization.

Pins the featurization math against hand-constructed synthetic frames
whose robust statistics are known by design. Expected
values are computed with explicit scalar arithmetic in the test — never
by calling the module under test — and committed as code, not binaries.
End-to-end numeric equivalence to the certified implementation is proven
separately by the LOCAL parity harness.
"""

import math

import numpy as np
import pytest

from sherloc_pipeline.ml_despike.featurize import (
    N_CHANNELS,
    REGIONS,
    featurize,
    featurize_batch,
)
from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST


def _flat_frame(value: float) -> np.ndarray:
    return np.full(N_CHANNELS, value, dtype=np.float64)


class TestGoldenVectors:
    def test_r1_zero_mad_frame(self):
        """Constant window + one spike: med/mad/scale known exactly.

        R1 window [52, 575) holds 523 samples; with all but one equal to
        100.0, ``med = 100.0`` and ``mad = 0.0`` exactly, so
        ``scale = 1.4826 * 0 + 1.0 = 1.0`` and x0 is simply
        ``active - 100``.
        """
        active = _flat_frame(100.0)
        active[60] = 105.0  # in-window spike
        active[10] = 999.0  # out-of-window value: must not perturb stats
        dark = _flat_frame(50.0)
        dark[70] = 53.0

        x = featurize(active, dark, "R1")

        assert x.dtype == np.float32
        assert x.shape == (8, N_CHANNELS)
        # x0 = (active - 100) / 1.0
        assert x[0, 60] == np.float32(5.0)
        assert x[0, 10] == np.float32(899.0)
        assert x[0, 0] == np.float32(0.0)
        assert x[0, 600] == np.float32(0.0)
        # x1 = (dark - 50) / 1.0
        assert x[1, 70] == np.float32(3.0)
        assert x[1, 0] == np.float32(0.0)
        # one-hot for R1
        assert np.all(x[2] == np.float32(1.0))
        assert np.all(x[3] == np.float32(0.0))
        assert np.all(x[4] == np.float32(0.0))
        # x5/x6 = log10(scale)/4 with scale exactly 1.0
        assert np.all(x[5] == np.float32(0.0))
        assert np.all(x[6] == np.float32(0.0))
        # x7 = log10(1 + |median(active window)|)/4 = log10(101)/4
        expected_x7 = np.float32(math.log10(101.0) / 4.0)
        assert np.all(x[7] == expected_x7)
        assert x[7, 0] == pytest.approx(0.5010803, abs=1e-6)

    def test_r2_nonzero_mad_frame(self):
        """Two-level window: med and mad known exactly, scale nontrivial.

        R2 window [575, 1677) holds 1102 samples. Alternating 100/104
        gives ``med = 102.0`` and ``|win - med| = 2.0`` everywhere, so
        ``mad = 2.0`` and ``scale = 1.4826 * 2 + 1 = 3.9652``.
        """
        active = _flat_frame(0.0)
        lo, hi = DEFAULT_MANIFEST.region_windows["R2"]
        window = np.empty(hi - lo, dtype=np.float64)
        window[0::2] = 100.0
        window[1::2] = 104.0
        active[lo:hi] = window
        dark = _flat_frame(50.0)

        x = featurize(active, dark, "R2")

        scale_a = 1.4826 * 2.0 + 1.0  # 3.9652
        assert x[0, lo] == np.float32((100.0 - 102.0) / scale_a)
        assert x[0, lo + 1] == np.float32((104.0 - 102.0) / scale_a)
        # out-of-window channels normalized with in-window stats
        assert x[0, 0] == np.float32((0.0 - 102.0) / scale_a)
        assert x[0, lo] == pytest.approx(-0.5043882, abs=1e-6)
        # one-hot for R2
        assert np.all(x[2] == np.float32(0.0))
        assert np.all(x[3] == np.float32(1.0))
        assert np.all(x[4] == np.float32(0.0))
        # x5 = log10(3.9652)/4; x6 = log10(1.0)/4 = 0 (flat dark)
        assert np.all(x[5] == np.float32(math.log10(scale_a) / 4.0))
        assert x[5, 0] == pytest.approx(0.1495663, abs=1e-6)
        assert np.all(x[6] == np.float32(0.0))
        # x7 = log10(1 + 102)/4
        assert np.all(x[7] == np.float32(math.log10(103.0) / 4.0))

    def test_r3_one_hot_and_window(self):
        active = _flat_frame(10.0)
        dark = _flat_frame(5.0)
        x = featurize(active, dark, "R3")
        assert np.all(x[2] == np.float32(0.0))
        assert np.all(x[3] == np.float32(0.0))
        assert np.all(x[4] == np.float32(1.0))
        # x7 uses the R3 window median (10.0)
        assert np.all(x[7] == np.float32(math.log10(11.0) / 4.0))

    def test_negative_window_median_uses_absolute_value(self):
        """x7 takes |median(active window)| before the log."""
        active = _flat_frame(-100.0)
        dark = _flat_frame(0.0)
        x = featurize(active, dark, "R1")
        assert np.all(x[7] == np.float32(math.log10(101.0) / 4.0))

    def test_integer_dn_input_accepted(self):
        """Raw DN often arrives as integer arrays; math must match float."""
        active_int = np.full(N_CHANNELS, 100, dtype=np.int32)
        active_int[60] = 105
        dark_int = np.full(N_CHANNELS, 50, dtype=np.int32)
        x_int = featurize(active_int, dark_int, "R1")
        x_float = featurize(
            active_int.astype(np.float64), dark_int.astype(np.float64), "R1"
        )
        assert np.array_equal(x_int, x_float)


class TestBatch:
    def test_batch_shape_and_order(self):
        frames = []
        for i, region in enumerate(REGIONS):
            active = _flat_frame(100.0 + i)
            dark = _flat_frame(50.0)
            frames.append((active, dark, region))
        actives, darks, regions = zip(*frames)

        batch = featurize_batch(actives, darks, regions)

        assert batch.dtype == np.float32
        assert batch.shape == (3, 8, N_CHANNELS)
        for i, (active, dark, region) in enumerate(frames):
            assert np.array_equal(batch[i], featurize(active, dark, region))

    def test_empty_batch(self):
        batch = featurize_batch([], [], [])
        assert batch.shape == (0, 8, N_CHANNELS)
        assert batch.dtype == np.float32

    def test_mismatched_lengths_rejected(self):
        active = _flat_frame(1.0)
        with pytest.raises(ValueError, match="equal lengths"):
            featurize_batch([active], [active, active], ["R1"])


class TestValidation:
    def test_unknown_region_rejected(self):
        active = _flat_frame(1.0)
        with pytest.raises(ValueError, match="R1, R2, R3"):
            featurize(active, active, "R4")

    def test_wrong_length_rejected(self):
        short = np.zeros(100)
        good = _flat_frame(1.0)
        with pytest.raises(ValueError, match="2148"):
            featurize(short, good, "R1")
        with pytest.raises(ValueError, match="2148"):
            featurize(good, short, "R1")

    def test_wrong_ndim_rejected(self):
        good = _flat_frame(1.0)
        with pytest.raises(ValueError, match="1-D"):
            featurize(good.reshape(1, -1), good, "R1")
