"""ML despike service path on the fixture workspace with the stub model.

Covers detection-on-raw-planes, per-region taus and
R1-alone scope, replacement via the shared helper,
provenance metadata, the
``cr_masks.json`` artifact, certified-window confinement,
and the fail-loud paths.
"""

import json
import shutil

import numpy as np
import pytest

from sherloc_pipeline.core.data_ingestion import DataIngestion
from sherloc_pipeline.services.errors import PreprocessingError

WORKSPACE_REL = (
    "loupe/sol_0921/detail_1/SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
)


@pytest.fixture(scope="module")
def workspace(fixtures_path):
    return fixtures_path / WORKSPACE_REL


@pytest.fixture
def ingestion(fixtures_path, tmp_path):
    return DataIngestion(
        base_data_dir=fixtures_path / "loupe",
        results_dir=tmp_path / "results",
        sol="0921",
        target="Test_Target",
        scan="detail_1",
    )


@pytest.fixture
def normalized_spectra(ingestion, workspace):
    return ingestion.process_normalized_spectra(workspace)


def _run(service, ingestion, workspace, normalized_spectra):
    from sherloc_pipeline.config import get_config

    return service._run_ml_despike(
        ingestion=ingestion,
        working_dir=workspace,
        normalized_spectra=normalized_spectra,
        sol="0921",
        target="Test_Target",
        scan="detail_1",
        generate_plots=False,
        despike_example_point=50,
        config=get_config(),
    )


class TestProvenanceMetadata:
    """The four run-level provenance fields, exact."""

    def test_provenance_fields(
        self, preprocessing_service_with_stub, stub_manifest,
        ingestion, workspace, normalized_spectra,
    ):
        import onnxruntime

        meta = _run(
            preprocessing_service_with_stub, ingestion, workspace,
            normalized_spectra,
        )
        assert meta["method"] == "ml_v1.3_tau_matched"
        assert meta["model_sha256"] == stub_manifest.sha256
        assert meta["tau"] == {
            "R1": 0.29882812500038747,
            "R2": 0.2656250000008831,
            "R3": 0.2656250000008831,
        }
        assert meta["ort_version"] == onnxruntime.__version__
        assert set(meta["n_flagged"]) == {"R1", "R2", "R3"}

    def test_cr_masks_json_round_trip(
        self, preprocessing_service_with_stub, ingestion, workspace,
        normalized_spectra,
    ):
        meta = _run(
            preprocessing_service_with_stub, ingestion, workspace,
            normalized_spectra,
        )
        cr_path = (
            ingestion.get_results_path(
                target="Test_Target", sol="0921", scan="detail_1"
            )
            / "cr_masks.json"
        )
        assert cr_path.exists()
        with open(cr_path) as f:
            stored = json.load(f)
        assert stored == meta

    def test_no_dn_values_in_artifact(
        self, preprocessing_service_with_stub, ingestion, workspace,
        normalized_spectra,
    ):
        """C5/raw-DN governance: channel indices + provenance only."""
        meta = _run(
            preprocessing_service_with_stub, ingestion, workspace,
            normalized_spectra,
        )
        assert set(meta) == {
            "method", "model_sha256", "tau", "ort_version", "n_flagged",
            "masks",
        }
        for region_masks in meta["masks"].values():
            for channels in region_masks.values():
                assert all(isinstance(c, int) for c in channels)


class TestDetectionInputs:
    """Detection consumes the raw planes, paired per
    point and region — not any normalized representation."""

    def test_detector_receives_raw_plane_values(
        self, preprocessing_service_with_stub, ingestion, workspace,
        normalized_spectra, monkeypatch,
    ):
        service = preprocessing_service_with_stub
        captured = {}
        real_factory = service._build_ml_detector

        def capturing_factory():
            detector = real_factory()
            real_detect = detector.detect

            def detect(actives, darks, regions):
                captured["actives"] = [np.asarray(a) for a in actives]
                captured["darks"] = [np.asarray(d) for d in darks]
                captured["regions"] = list(regions)
                return real_detect(actives, darks, regions)

            detector.detect = detect
            return detector

        monkeypatch.setattr(
            type(service), "_build_ml_detector",
            lambda self: capturing_factory(),
        )
        _run(service, ingestion, workspace, normalized_spectra)

        planes = ingestion.load_active_dark_planes(workspace)
        n_points = planes["active"]["R1"].shape[0]
        assert len(captured["regions"]) == 3 * n_points

        # Frame i of region block r must equal raw plane row i (paired).
        idx = 0
        for region in ("R1", "R2", "R3"):
            for point in range(n_points):
                assert captured["regions"][idx] == region
                np.testing.assert_array_equal(
                    captured["actives"][idx], planes["active"][region][point]
                )
                np.testing.assert_array_equal(
                    captured["darks"][idx], planes["dark"][region][point]
                )
                idx += 1

        # AC2: raw inputs differ from the normalized forms the chain holds.
        r1_norm_point0 = normalized_spectra["R1"][0].to_numpy(dtype=float)
        raw_windowed = captured["actives"][0][52:575]
        assert not np.allclose(r1_norm_point0, raw_windowed)


class TestScopeAndWindows:
    def test_r1_alone_scope_absent_r2_r3(
        self, preprocessing_service_with_stub, ingestion, workspace,
        normalized_spectra,
    ):
        """An R1-alone run featurizes and
        infers R1 only — no R2/R3 entries anywhere in metadata or
        cr_masks.json."""
        r1_only = {"R1": normalized_spectra["R1"]}
        meta = _run(
            preprocessing_service_with_stub, ingestion, workspace, r1_only
        )
        assert set(meta["tau"]) == {"R1"}
        assert set(meta["n_flagged"]) == {"R1"}
        for region_masks in meta["masks"].values():
            assert set(region_masks) == {"R1"}
        assert "R1_despiked" in r1_only

        cr_path = (
            ingestion.get_results_path(
                target="Test_Target", sol="0921", scan="detail_1"
            )
            / "cr_masks.json"
        )
        with open(cr_path) as f:
            stored = json.dumps(json.load(f))
        assert '"R2"' not in stored
        assert '"R3"' not in stored

    def test_three_region_run_flags_all_regions(
        self, preprocessing_service_with_stub, ingestion, workspace,
        normalized_spectra,
    ):
        """Three-region run yields per-region masks, each
        thresholded with its region's tau (R1 vs fluor differ)."""
        meta = _run(
            preprocessing_service_with_stub, ingestion, workspace,
            normalized_spectra,
        )
        assert set(meta["n_flagged"]) == {"R1", "R2", "R3"}
        assert meta["tau"]["R1"] != meta["tau"]["R2"]
        assert meta["tau"]["R2"] == meta["tau"]["R3"]

    def test_masks_confined_to_certified_windows(
        self, preprocessing_service_with_stub, stub_manifest,
        ingestion, workspace, normalized_spectra,
    ):
        """No flags outside [lo, hi) per region."""
        meta = _run(
            preprocessing_service_with_stub, ingestion, workspace,
            normalized_spectra,
        )
        for region_masks in meta["masks"].values():
            for region, channels in region_masks.items():
                lo, hi = stub_manifest.region_windows[region]
                assert all(lo <= c < hi for c in channels)


class TestReplacement:
    def test_r1_replacement_equals_shared_helper(
        self, preprocessing_service_with_stub, ingestion, workspace,
        normalized_spectra,
    ):
        """At the service level: the despiked R1 column
        equals apply_mask_replacement applied to the same mask."""
        from sherloc_pipeline.config import get_config
        from sherloc_pipeline.core.mask_application import (
            derive_region_channel_masks,
        )
        from sherloc_pipeline.core.preprocessing import apply_mask_replacement

        r1_before = normalized_spectra["R1"].copy()
        meta = _run(
            preprocessing_service_with_stub, ingestion, workspace,
            normalized_spectra,
        )
        r1_despiked = normalized_spectra["R1_despiked"]

        _, channel_masks = derive_region_channel_masks(get_config())
        selected = np.where(channel_masks["R1"])[0]
        row_of_channel = {int(ch): row for row, ch in enumerate(selected)}

        # First point with flags
        point = next(
            int(p) for p, rm in meta["masks"].items() if rm["R1"]
        )
        row_mask = np.zeros(len(r1_before), dtype=bool)
        for ch in meta["masks"][str(point)]["R1"]:
            row = row_of_channel.get(ch)
            if row is not None:
                row_mask[row] = True
        expected = apply_mask_replacement(r1_before[point], row_mask, "linear")
        np.testing.assert_array_equal(
            r1_despiked[point].to_numpy(), expected.to_numpy()
        )

    def test_fluor_and_r123_frames_despiked_in_place(
        self, preprocessing_service_with_stub, ingestion, workspace,
        normalized_spectra,
    ):
        """The fluorescence (and R123) frames follow
        despike_method=ml — replaced in the carried dict."""
        fluor_before = normalized_spectra["fluorescence"].copy()
        r123_before = normalized_spectra["R123"].copy()
        _run(
            preprocessing_service_with_stub, ingestion, workspace,
            normalized_spectra,
        )
        assert not normalized_spectra["fluorescence"].equals(fluor_before)
        assert not normalized_spectra["R123"].equals(r123_before)
        # R1 frame itself is untouched; despiked copy is a new key.
        assert "R1_despiked" in normalized_spectra


class TestFailLoud:
    def test_missing_raw_planes_fails_with_remedy(
        self, preprocessing_service_with_stub, workspace, tmp_path,
    ):
        """Missing planes name the
        alternative methods; no substitute input."""
        partial = tmp_path / "no_dark_workspace"
        partial.mkdir()
        shutil.copy(workspace / "loupe.csv", partial / "loupe.csv")
        shutil.copy(
            workspace / "activeSpectra.csv", partial / "activeSpectra.csv"
        )
        shutil.copy(
            workspace / "darkSubSpectraN.csv", partial / "darkSubSpectraN.csv"
        )
        ingestion = DataIngestion(
            base_data_dir=tmp_path, results_dir=tmp_path / "results",
            sol="0921", target="Test_Target", scan="detail_1",
        )
        normalized = ingestion.process_normalized_spectra(partial)

        with pytest.raises(PreprocessingError) as excinfo:
            _run(
                preprocessing_service_with_stub, ingestion, partial,
                normalized,
            )
        message = str(excinfo.value)
        assert "activeSpectra.csv and darkSpectra.csv" in message
        assert "--despike-method modz or none" in message

    def test_missing_extra_preserves_install_remedy(
        self, ingestion, workspace, normalized_spectra, monkeypatch,
    ):
        """ImportError from the lazy onnxruntime import
        surfaces as PreprocessingError with the exact install command."""
        from sherloc_pipeline.ml_despike.detector import _INSTALL_HINT
        from sherloc_pipeline.services.preprocessing import (
            PreprocessingService,
        )

        service = PreprocessingService()

        def raise_import_error(self):
            raise ImportError(_INSTALL_HINT)

        monkeypatch.setattr(
            PreprocessingService, "_build_ml_detector", raise_import_error
        )
        with pytest.raises(PreprocessingError) as excinfo:
            _run(service, ingestion, workspace, normalized_spectra)
        message = str(excinfo.value)
        assert "[ml-despike]" in message
        assert "pip install" in message
