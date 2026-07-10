"""Plot despike integration (spec §4.7, review F8).

``plot`` never runs inference: ``ml`` applies stored DB masks via the
pipeline's own mask-application helpers; ``modz`` computes the legacy
despike live; ``none`` renders raw (the pre-integration behavior).
Missing masks render non-despiked with a once-per-invocation aggregated
console note and a per-render effective-state summary.
"""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app
from sherloc_pipeline.config import get_config, reset_config
from sherloc_pipeline.core.mask_application import derive_region_channel_masks
from sherloc_pipeline.core.preprocessing import apply_mask_replacement
from sherloc_pipeline.database.connection import get_session
from sherloc_pipeline.database.models import ScanORM
from sherloc_pipeline.services.cr_masks import CRMaskService
from sherloc_pipeline.services.ingestion import IngestionService
from sherloc_pipeline.services.runtime import RuntimeContext
from sherloc_pipeline.services.spectral import SpectralPlotRequest, SpectralService

runner = CliRunner()

WORKSPACE_REL = (
    "loupe/sol_0921/detail_1/SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
)

#: Stored-mask fixture content. Channels are absolute (0-2147), inside
#: the certified windows: R1 [52, 575), R2 [575, 1677), R3 [1677, 2140).
R1_POINT5 = [100, 101, 200]
FLUOR_POINT0 = {"R2": [600, 1200], "R3": [1700]}

MASKS = {
    "5": {"R1": R1_POINT5},
    "0": FLUOR_POINT0,
    # Point 1 has an R2 record but no R3 record: the fluorescence
    # composite is all-or-none per point (spec §4.6 rule mirrored).
    "1": {"R2": [800]},
}

DESPIKE_METADATA = {
    "method": "ml_v1.3_tau_matched",
    "model_sha256": "ab" * 32,
    "tau": {
        "R1": 0.29882812500038747,
        "R2": 0.2656250000008831,
        "R3": 0.2656250000008831,
    },
    "masks": MASKS,
}


@pytest.fixture(autouse=True)
def _clean_config():
    yield
    reset_config()


@pytest.fixture(scope="module")
def mask_db(tmp_path_factory, fixtures_path):
    """Temp DB with the fixture scan ingested and known masks persisted."""
    tmp = tmp_path_factory.mktemp("plot_despike_db")
    db_path = tmp / "phase.db"
    service = IngestionService(
        database_path=db_path,
        include_spectra=True,
        ingestion_mode="all_regions",
    )
    service.ingest_workspace(fixtures_path / WORKSPACE_REL)
    with get_session(service.engine) as session:
        scan = session.query(ScanORM).first()
        scan.target = "Amherst Point"
    CRMaskService().persist_masks(
        sol="0921",
        target="Amherst_Point",
        scan="detail_1",
        despike_metadata=DESPIKE_METADATA,
        database_path=db_path,
    )
    return db_path


def _run_plot(
    fixtures_path,
    results_dir,
    monkeypatch,
    db_path,
    method,
    point,
    domain="raman",
):
    """Run a point-mode plot through the service; return (csv_df, result)."""
    monkeypatch.setenv("SHERLOC_DB_PATH", str(db_path))
    context = RuntimeContext.bootstrap(
        data_dir=fixtures_path / "loupe",
        results_dir=results_dir,
    )
    from rich.console import Console

    service = SpectralService(console=Console(quiet=True), context=context)
    request = SpectralPlotRequest(
        sol="0921",
        target="Amherst_Point",
        scan="detail_1",
        mode="point",
        point=point,
        domain=domain,
        baseline=False,
        export="csv",
        no_metadata=True,
        despike_method=method,
    )
    result = service.process(request)
    csv_path = next(a for a in result.artifacts if a.suffix == ".csv")
    return pd.read_csv(csv_path), result


class TestStoredMaskApplication:
    """ml renders mask-replaced intensities; none raw."""

    def test_ml_point_render_equals_mask_replacement(
        self, fixtures_path, tmp_path, monkeypatch, mask_db
    ):
        df_none, _ = _run_plot(
            fixtures_path, tmp_path / "none", monkeypatch, mask_db,
            method="none", point=5,
        )
        df_ml, result_ml = _run_plot(
            fixtures_path, tmp_path / "ml", monkeypatch, mask_db,
            method="ml", point=5,
        )

        # Map absolute channels to R1 frame rows the way the frame was
        # built (config wavelength bounds), never by hardcoded offsets.
        _, channel_masks = derive_region_channel_masks(get_config(), 2148)
        selected = np.where(channel_masks["R1"])[0]
        row_of = {int(ch): row for row, ch in enumerate(selected)}
        rows = [row_of[ch] for ch in R1_POINT5]

        row_mask = np.zeros(len(df_none), dtype=bool)
        row_mask[rows] = True
        expected = apply_mask_replacement(
            df_none["intensity"], row_mask, "linear"
        )
        np.testing.assert_allclose(
            df_ml["intensity"].values, expected.values,
            err_msg="ml render must equal stored-mask replacement",
        )
        # Outside the masked rows the render is untouched
        np.testing.assert_array_equal(
            df_ml["intensity"].values[~row_mask],
            df_none["intensity"].values[~row_mask],
        )

        despike_meta = result_ml.metadata["despike"]
        assert despike_meta["method"] == "ml"
        assert despike_meta["source"] == "stored_masks"
        assert despike_meta["points_total"] == 1
        assert despike_meta["points_despiked"] == 1
        assert not result_ml.warnings  # masked point -> no no-mask note

    def test_point_without_stored_mask_renders_raw_with_note(
        self, fixtures_path, tmp_path, monkeypatch, mask_db
    ):
        df_none, _ = _run_plot(
            fixtures_path, tmp_path / "none", monkeypatch, mask_db,
            method="none", point=7,
        )
        df_ml, result_ml = _run_plot(
            fixtures_path, tmp_path / "ml", monkeypatch, mask_db,
            method="ml", point=7,
        )
        np.testing.assert_array_equal(
            df_ml["intensity"].values, df_none["intensity"].values
        )
        assert result_ml.metadata["despike"]["points_despiked"] == 0
        assert any("no stored CR mask" in w for w in result_ml.warnings)

    def test_no_database_renders_raw_with_note(
        self, fixtures_path, tmp_path, monkeypatch
    ):
        df_none, _ = _run_plot(
            fixtures_path, tmp_path / "none", monkeypatch,
            tmp_path / "absent.db", method="none", point=5,
        )
        df_ml, result_ml = _run_plot(
            fixtures_path, tmp_path / "ml", monkeypatch,
            tmp_path / "absent.db", method="ml", point=5,
        )
        np.testing.assert_array_equal(
            df_ml["intensity"].values, df_none["intensity"].values
        )
        assert any("no stored CR mask" in w for w in result_ml.warnings)


class TestModzLive:
    """plot modz = the legacy method computed live (spec §4.7)."""

    def test_modz_point_render_equals_legacy_despike(
        self, fixtures_path, tmp_path, monkeypatch
    ):
        from sherloc_pipeline.core.preprocessing import (
            DespikeParams,
            despike_r1_dataframe,
        )

        db_path = tmp_path / "absent.db"  # modz never touches the DB
        df_none, _ = _run_plot(
            fixtures_path, tmp_path / "none", monkeypatch, db_path,
            method="none", point=5,
        )
        df_modz, result_modz = _run_plot(
            fixtures_path, tmp_path / "modz", monkeypatch, db_path,
            method="modz", point=5,
        )

        dsp = get_config().preprocessing.get('despike', {})
        params = DespikeParams(
            window_size=dsp.get('window_size', 7),
            zscore_threshold=dsp.get('zscore_threshold', 6.0),
            max_iterations=dsp.get('max_iterations', 1),
            interpolation_method=dsp.get('interpolation_method', 'linear'),
        )
        frame = pd.DataFrame({
            "raman_shift": df_none["raman_shift"],
            5: df_none["intensity"].astype(float),
        })
        expected, _ = despike_r1_dataframe(frame, params)
        np.testing.assert_allclose(
            df_modz["intensity"].values, expected[5].values,
            err_msg="modz render must equal the legacy despike output",
        )
        assert result_modz.metadata["despike"]["source"] == "live_modz"
        assert not result_modz.warnings  # no-mask note is ml-only


class TestFluorAllOrNone:
    """Fluorescence sum despike is all-or-none per point (§3.3/§4.6)."""

    def test_point_with_both_contributors_despiked(
        self, fixtures_path, tmp_path, monkeypatch, mask_db
    ):
        df_none, _ = _run_plot(
            fixtures_path, tmp_path / "none", monkeypatch, mask_db,
            method="none", point=0, domain="fluor",
        )
        df_ml, result_ml = _run_plot(
            fixtures_path, tmp_path / "ml", monkeypatch, mask_db,
            method="ml", point=0, domain="fluor",
        )
        # Identity row map: row i = channel i (full 2148-channel sum)
        channels = FLUOR_POINT0["R2"] + FLUOR_POINT0["R3"]
        row_mask = np.zeros(len(df_none), dtype=bool)
        row_mask[channels] = True
        expected = apply_mask_replacement(
            df_none["intensity"], row_mask, "linear"
        )
        np.testing.assert_allclose(df_ml["intensity"].values, expected.values)
        assert result_ml.metadata["despike"]["points_despiked"] == 1

    def test_missing_contributor_renders_raw(
        self, fixtures_path, tmp_path, monkeypatch, mask_db
    ):
        """Point 1 has R2 but no R3 record: never partially despiked."""
        df_none, _ = _run_plot(
            fixtures_path, tmp_path / "none", monkeypatch, mask_db,
            method="none", point=1, domain="fluor",
        )
        df_ml, result_ml = _run_plot(
            fixtures_path, tmp_path / "ml", monkeypatch, mask_db,
            method="ml", point=1, domain="fluor",
        )
        np.testing.assert_array_equal(
            df_ml["intensity"].values, df_none["intensity"].values
        )
        assert result_ml.metadata["despike"]["points_despiked"] == 0
        assert any("no stored CR mask" in w for w in result_ml.warnings)


class TestCliNoMaskNote:
    """Review F8: the no-mask note is once per invocation, aggregated
    with a count; every render prints an effective-state summary."""

    def test_averaged_no_mask_note_once_with_count(
        self, fixtures_path, tmp_path
    ):
        result = runner.invoke(
            app,
            [
                "plot",
                "--sol", "0921",
                "--target", "Amherst_Point",
                "--scan", "detail_1",
                "--despike-method", "ml",
                "--export", "csv",
                "--no-metadata",
                "--data-dir", str(fixtures_path / "loupe"),
                "--results-dir", str(tmp_path),
            ],
            env={"SHERLOC_DB_PATH": str(tmp_path / "absent.db")},
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        # Aggregated once — never per spectrum
        assert result.output.count("have no stored CR mask") == 1
        assert "100 of 100 spectra have no stored CR mask" in result.output
        # Effective-state summary line (method, masks applied/total)
        assert (
            "Despike: ml — stored masks applied to 0/100 point spectra"
            in result.output
        )

    def test_level_path_not_reprocessed(self, fixtures_path, tmp_path):
        # --level mode reads from and writes plots under results-dir;
        # copy the fixture outputs to tmp so the fixture tree stays clean.
        import json
        import shutil

        results_dir = tmp_path / "results"
        shutil.copytree(fixtures_path / "pipeline_outputs", results_dir)
        result = runner.invoke(
            app,
            [
                "plot",
                "--sol", "0921",
                "--target", "Amherst_Point",
                "--scan", "detail_1",
                "--point", "5",
                "--level", "normalized",
                "--despike-method", "ml",
                "--export", "csv",
                "--results-dir", str(results_dir),
            ],
            env={"SHERLOC_DB_PATH": str(tmp_path / "absent.db")},
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "not re-processed" in result.output
        assert "have no stored CR mask" not in result.output

        # The exported JSON sidecar communicates the stored-level despike
        # state (spec §4.7: state is communicated, data not re-processed).
        # "normalized" is pre-despike output — no method is claimed and
        # the data is reported non-despiked.
        json_paths = list(
            (results_dir / "Amherst_Point" / "plots").glob("*_p5_normalized.json")
        )
        assert json_paths, "expected a JSON metadata sidecar"
        sidecar = json.loads(json_paths[0].read_text())
        assert sidecar["despike"] == {
            "source": "stored_level",
            "level": "normalized",
            "despiked": False,
        }
        assert "is pre-despike stored output" in result.output

    def test_level_despiked_data_reported_despiked(
        self, fixtures_path, tmp_path
    ):
        """A despiked stored level reports despiked=True without claiming
        a method — the producing run's method is not recorded in level
        CSVs."""
        import json
        import shutil

        results_dir = tmp_path / "results"
        shutil.copytree(fixtures_path / "pipeline_outputs", results_dir)
        result = runner.invoke(
            app,
            [
                "plot",
                "--sol", "0921",
                "--target", "Amherst_Point",
                "--scan", "detail_1",
                "--point", "5",
                "--level", "normalized_despiked_baselined",
                "--export", "csv",
                "--results-dir", str(results_dir),
            ],
            env={"SHERLOC_DB_PATH": str(tmp_path / "absent.db")},
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "was despiked at processing time" in result.output
        json_paths = list(
            (results_dir / "Amherst_Point" / "plots").glob(
                "*_p5_normalized_despiked_baselined.json"
            )
        )
        assert json_paths, "expected a JSON metadata sidecar"
        sidecar = json.loads(json_paths[0].read_text())
        assert sidecar["despike"] == {
            "source": "stored_level",
            "level": "normalized_despiked_baselined",
            "despiked": True,
        }
