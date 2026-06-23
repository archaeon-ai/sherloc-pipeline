"""Stub-model chain tests for the ONNX CR detector.

Deterministically asserts the certified numeric structure (MLD-DET-004
AC1/AC2, MLD-QUA-003 AC2): featurize → infer → float64 sigmoid → strict
``p > tau`` → window-restricted absolute-index masks — with the stub
artifact flowing through the real ``resolve_artifact`` chain. CI never
sees the certified artifact; bit-exact parity against it is the LOCAL
gate (MLD-QUA-003 AC1).
"""

import dataclasses
from types import MappingProxyType

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")

from sherloc_pipeline.ml_despike import artifact as artifact_mod  # noqa: E402
from sherloc_pipeline.ml_despike.artifact import ArtifactDigestError  # noqa: E402
from sherloc_pipeline.ml_despike.detector import MLCRDetector  # noqa: E402
from sherloc_pipeline.ml_despike.featurize import N_CHANNELS, featurize  # noqa: E402

from .conftest import stub_weights  # noqa: E402

# sigmoid(3.0) ~ 0.95: comfortably above both frozen taus.
STRONG_SPIKE_LOGIT = 3.0
# sigmoid(-0.93) ~ 0.283: strictly between tau_fluor (~0.2734) and
# tau_R1 (~0.2910) — flagged in R2/R3, not in R1.
BETWEEN_TAUS_LOGIT = -0.93


def make_frame(spike_channels=(), spike_logit=STRONG_SPIKE_LOGIT):
    """Frame whose stub logits are ~0 -4 baseline except engineered spikes.

    The constant background (med 100, mad 0 → scale 1.0) makes the
    normalized-active feature exactly ``active - 100``; setting
    ``active[ch] = 100 + (logit - w[ch])`` steers the stub's logit at
    ``ch`` to ``logit`` up to float32 rounding.
    """
    weights = stub_weights()
    active = np.full(N_CHANNELS, 100.0)
    for ch in spike_channels:
        active[ch] = 100.0 + (spike_logit - float(weights[ch]))
    dark = np.full(N_CHANNELS, 50.0)
    return active, dark


def expected_probability(active, dark, region, channel):
    """The probability the detector must produce at one channel.

    Replicates the certified arithmetic independently: float32 feature +
    float32 stub weight (IEEE single addition, exactly what the ORT Add
    node computes), then the float64 sigmoid.
    """
    features = featurize(active, dark, region)
    logit = np.float32(features[0, channel] + stub_weights()[channel])
    return float(1.0 / (1.0 + np.exp(-np.float64(logit))))


@pytest.fixture
def detector(stub_manifest, stub_model_path):
    return MLCRDetector(manifest=stub_manifest, artifact_path=stub_model_path)


class TestDetectionChain:
    def test_strong_spikes_flagged_at_absolute_indices(self, detector):
        active, dark = make_frame(spike_channels=(100, 300, 574))
        masks = detector.detect([active], [dark], ["R1"])
        assert len(masks) == 1
        assert masks[0].tolist() == [100, 300, 574]
        assert masks[0].dtype == np.int64

    def test_unspiked_frame_yields_empty_mask(self, detector):
        active, dark = make_frame(spike_channels=())
        masks = detector.detect([active], [dark], ["R1"])
        assert masks[0].size == 0

    def test_masks_sorted_and_inside_window(self, detector, stub_manifest):
        active, dark = make_frame(spike_channels=(560, 90, 401))
        masks = detector.detect([active], [dark], ["R1"])
        lo, hi = stub_manifest.region_windows["R1"]
        assert masks[0].tolist() == sorted(masks[0].tolist())
        assert all(lo <= ch < hi for ch in masks[0])

    def test_flags_restricted_to_certified_window(self, detector):
        """Out-of-window spikes never flag, in-window edges do (MLD-SYS-015 AC3)."""
        # R1 window is [52, 575): 10 and 600 are outside, 52/574 are edges.
        active, dark = make_frame(spike_channels=(10, 52, 574, 600))
        masks = detector.detect([active], [dark], ["R1"])
        assert masks[0].tolist() == [52, 574]

        # R3 window is [1677, 2140): channel 2140 is the first excluded one.
        active, dark = make_frame(spike_channels=(2139, 2140, 2147))
        masks = detector.detect([active], [dark], ["R3"])
        assert masks[0].tolist() == [2139]

    def test_per_region_tau_applied(self, detector):
        """A probability between the two taus flags fluor regions only."""
        r1_channel, r2_channel = 300, 800
        active_r1, dark_r1 = make_frame((r1_channel,), spike_logit=BETWEEN_TAUS_LOGIT)
        active_r2, dark_r2 = make_frame((r2_channel,), spike_logit=BETWEEN_TAUS_LOGIT)

        p_r1 = expected_probability(active_r1, dark_r1, "R1", r1_channel)
        p_r2 = expected_probability(active_r2, dark_r2, "R2", r2_channel)
        tau_r1 = detector.manifest.tau["R1"]
        tau_fluor = detector.manifest.tau["R2"]
        # Guard the construction: both probabilities strictly between taus.
        assert tau_fluor < p_r1 < tau_r1
        assert tau_fluor < p_r2 < tau_r1

        masks = detector.detect(
            [active_r1, active_r2], [dark_r1, dark_r2], ["R1", "R2"]
        )
        assert r1_channel not in masks[0]  # below R1 tau
        assert masks[1].tolist() == [r2_channel]  # above fluor tau

    def test_strict_inequality_at_exact_tau(self, stub_manifest, stub_model_path):
        """p == tau is NOT flagged; the comparison is strictly greater-than."""
        channel = 300
        active, dark = make_frame((channel,), spike_logit=BETWEEN_TAUS_LOGIT)
        p_exact = expected_probability(active, dark, "R1", channel)

        manifest_eq = dataclasses.replace(
            stub_manifest,
            tau=MappingProxyType({"R1": p_exact, "R2": p_exact, "R3": p_exact}),
        )
        detector_eq = MLCRDetector(manifest=manifest_eq, artifact_path=stub_model_path)
        masks = detector_eq.detect([active], [dark], ["R1"])
        assert channel not in masks[0]

        tau_below = float(np.nextafter(p_exact, 0.0))
        manifest_below = dataclasses.replace(
            stub_manifest,
            tau=MappingProxyType({"R1": tau_below, "R2": tau_below, "R3": tau_below}),
        )
        detector_below = MLCRDetector(
            manifest=manifest_below, artifact_path=stub_model_path
        )
        masks = detector_below.detect([active], [dark], ["R1"])
        assert channel in masks[0]

    def test_r1_alone_first_class(self, detector):
        """An all-R1 call is a fully supported path (MLD-SYS-006)."""
        frames = [make_frame((ch,)) for ch in (60, 200, 400)]
        actives = [f[0] for f in frames]
        darks = [f[1] for f in frames]
        masks = detector.detect(actives, darks, ["R1", "R1", "R1"])
        assert [m.tolist() for m in masks] == [[60], [200], [400]]

    def test_all_three_regions_detected(self, detector):
        """Each region flags through its own window and tau (MLD-SYS-006 AC1)."""
        spec = {"R1": 300, "R2": 800, "R3": 1800}
        actives, darks, regions = [], [], []
        for region, channel in spec.items():
            active, dark = make_frame((channel,))
            actives.append(active)
            darks.append(dark)
            regions.append(region)
        masks = detector.detect(actives, darks, regions)
        assert [m.tolist() for m in masks] == [[300], [800], [1800]]

    def test_internal_batching_transparent(self, detector):
        """35 frames (>2 internal batches of 16) match per-frame results."""
        rng = np.random.default_rng(7)
        regions = ["R1", "R2", "R3"] * 12
        regions = regions[:35]
        actives, darks = [], []
        for i, region in enumerate(regions):
            lo, hi = detector.manifest.region_windows[region]
            channel = int(rng.integers(lo, hi))
            active, dark = make_frame((channel,) if i % 3 else ())
            actives.append(active)
            darks.append(dark)

        batched = detector.detect(actives, darks, regions)
        single = [
            detector.detect([a], [d], [r])[0]
            for a, d, r in zip(actives, darks, regions)
        ]
        assert len(batched) == 35
        for got, want in zip(batched, single):
            assert np.array_equal(got, want)

    def test_empty_input(self, detector):
        assert detector.detect([], [], []) == []

    def test_mismatched_lengths_rejected(self, detector):
        active, dark = make_frame()
        with pytest.raises(ValueError, match="equal lengths"):
            detector.detect([active], [dark, dark], ["R1"])


class TestSessionConfiguration:
    def test_certified_session_config(self, detector):
        """CPU EP only, 2 intra-op / 1 inter-op threads (MLD-DET-004 AC2)."""
        assert detector._session.get_providers() == ["CPUExecutionProvider"]
        assert detector.intra_op_threads == 2
        assert detector.inter_op_threads == 1

    def test_ort_version_recorded_for_provenance(self, detector):
        assert detector.ort_version == ort.__version__


class TestArtifactResolutionChain:
    def test_tampered_artifact_creates_no_session(
        self, tmp_path, stub_manifest, stub_model_path
    ):
        """Digest mismatch fails construction before any session exists
        (MLD-SEC-002 AC1)."""
        corrupt = tmp_path / "corrupt.onnx"
        corrupt.write_bytes(stub_model_path.read_bytes() + b"!")
        with pytest.raises(ArtifactDigestError):
            MLCRDetector(manifest=stub_manifest, artifact_path=corrupt)

    def test_detector_resolves_via_fetch_and_cache(
        self, tmp_path, monkeypatch, stub_manifest, stub_model_path
    ):
        """Without an explicit path the detector uses the real
        fetch-and-cache resolution (network mocked)."""
        payload = stub_model_path.read_bytes()
        calls = {"n": 0}

        def fake_urlopen(url, timeout=None):
            calls["n"] += 1
            import io

            return io.BytesIO(payload)

        monkeypatch.setattr(artifact_mod, "urlopen", fake_urlopen)
        monkeypatch.setattr(
            artifact_mod, "default_cache_dir", lambda: tmp_path / "cache"
        )

        detector = MLCRDetector(manifest=stub_manifest)
        assert calls["n"] == 1
        active, dark = make_frame((300,))
        assert detector.detect([active], [dark], ["R1"])[0].tolist() == [300]

        # Second construction hits the verified cache: no further network.
        MLCRDetector(manifest=stub_manifest)
        assert calls["n"] == 1
