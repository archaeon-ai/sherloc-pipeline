"""Frozen-identity tests for the ML despike model manifest.

Pins the manifest to the model's release record at full precision
(MLD-DET-001 AC1, MLD-SYS-005 AC1) and asserts runtime immutability
(MLD-DET-001 AC2). The literal digests and taus below are the test's
authority, not the manifest module.
"""

import dataclasses

import pytest

from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST, ModelManifest

CERTIFIED_ONNX_SHA256 = "9668a0b2ca257ce333d57e3f76598dda8cb5c1839e2fde6bd955086d959be0ba"
CERTIFIED_CHECKPOINT_SHA256 = (
    "a77cd435d65631a8728c9d39c01c31dd30805ac37062b8c48937be6fb3594881"
)
CERTIFIED_TAU_R1 = 0.29882812500038747
CERTIFIED_TAU_FLUOR = 0.2656250000008831


class TestCertifiedValues:
    def test_taus_full_precision(self):
        assert DEFAULT_MANIFEST.tau["R1"] == CERTIFIED_TAU_R1
        assert DEFAULT_MANIFEST.tau["R2"] == CERTIFIED_TAU_FLUOR
        assert DEFAULT_MANIFEST.tau["R3"] == CERTIFIED_TAU_FLUOR
        assert set(DEFAULT_MANIFEST.tau) == {"R1", "R2", "R3"}

    def test_artifact_digests(self):
        assert DEFAULT_MANIFEST.sha256 == CERTIFIED_ONNX_SHA256
        assert DEFAULT_MANIFEST.checkpoint_sha256 == CERTIFIED_CHECKPOINT_SHA256

    def test_region_windows(self):
        assert DEFAULT_MANIFEST.region_windows["R1"] == (52, 575)
        assert DEFAULT_MANIFEST.region_windows["R2"] == (575, 1677)
        assert DEFAULT_MANIFEST.region_windows["R3"] == (1677, 2140)
        assert set(DEFAULT_MANIFEST.region_windows) == {"R1", "R2", "R3"}

    def test_identity_fields(self):
        assert DEFAULT_MANIFEST.name == "v1_stageB_v13c"
        assert DEFAULT_MANIFEST.artifact_filename == "v1_stageB_v13c.onnx"
        assert DEFAULT_MANIFEST.provenance_label == "ml_v1.3_tau_matched"
        assert DEFAULT_MANIFEST.n_channels == 2148
        assert DEFAULT_MANIFEST.opset == 18
        assert DEFAULT_MANIFEST.certified_runtime == (
            "onnxruntime 1.26.0, CPUExecutionProvider, fp32"
        )

    def test_download_url_https_versioned(self):
        url = DEFAULT_MANIFEST.download_url
        assert url.startswith("https://")
        # Model-artifact hosting tag (non-v*, C4-safe; spec §4.9).
        assert "model-cr-despike-1.3" in url
        assert url.endswith(DEFAULT_MANIFEST.artifact_filename)


class TestImmutability:
    def test_dataclass_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULT_MANIFEST.sha256 = "0" * 64

    def test_tau_mapping_not_writable(self):
        with pytest.raises(TypeError):
            DEFAULT_MANIFEST.tau["R1"] = 0.5

    def test_region_windows_not_writable(self):
        with pytest.raises(TypeError):
            DEFAULT_MANIFEST.region_windows["R1"] = (0, 2148)

    def test_replace_builds_variant_without_mutating_default(self):
        # Tests build stub-model variants this way; the certified
        # constant must be unaffected.
        variant = dataclasses.replace(DEFAULT_MANIFEST, sha256="0" * 64)
        assert variant.sha256 == "0" * 64
        assert DEFAULT_MANIFEST.sha256 == CERTIFIED_ONNX_SHA256

    def test_manifest_is_module_constant_instance(self):
        assert isinstance(DEFAULT_MANIFEST, ModelManifest)
