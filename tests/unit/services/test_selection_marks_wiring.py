"""Selection marks through the services: the per-point size log beside each peak
table, aicc_gain / step_p in the table, and kept_by decided at the persist read by putting the
F-test pick's rows through the same gates as the persisted rows. Records only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sherloc_pipeline.core.fitting import SIZE_LOG_VERSION
from sherloc_pipeline.database.connection import get_session
from sherloc_pipeline.database.models import FittedPeakORM
from sherloc_pipeline.services.fitting import (
    KEPT_BY_MATCH_CM1,
    _fit_point_minerals,
    _kept_by,
    size_sidecar_path,
)
from tests.unit.services.test_raman_persistence import (  # noqa: F401  (pytest fixtures)
    _call_persist_raman,
    _make_service,
    engine,
    populated_db,
    results_all,
)

CFG = yaml.safe_load((Path(__file__).resolve().parents[3] / "src/sherloc_pipeline/config.yaml").read_text())["fitting"]


def _gauss(x, c, a, f):
    return a * np.exp(-4 * np.log(2) * (x - c) ** 2 / f ** 2)


@pytest.mark.parametrize("name, sidecar", [
    ("S_T_s_R1_point3_fit_peaks.csv", "S_T_s_R1_point3_fit_sizes.json"),
    ("S_T_s_R1_point3_organics_g_peaks.csv", "S_T_s_R1_point3_organics_g_sizes.json"),
    ("S_T_s_R1_point3_organics_dg_peaks.csv", "S_T_s_R1_point3_organics_dg_sizes.json"),
    ("S_T_s_R1_point3_hydration_peaks.csv", "S_T_s_R1_point3_hydration_sizes.json"),
])
def test_sidecar_sits_beside_its_peak_table(name, sidecar):
    assert size_sidecar_path(Path("/r") / name) == Path("/r") / sidecar


def test_sidecar_name_keeps_the_point_token_and_is_never_a_peak_table():
    from sherloc_pipeline.services.fitting import RAMAN_CSV_PATTERNS
    import re

    side = size_sidecar_path(Path("S_T_s_R1_point3_fit_peaks.csv")).name
    assert re.match(r"^S_T_s_R1_point3[_.]", side)  # swept by _remove_obsolete_point_artifacts
    assert not any(re.search(p, side) for p in RAMAN_CSV_PATTERNS.values())


def _run_worker(out, cfg=CFG):
    x = np.linspace(500.0, 2200.0, 1800)
    y = np.random.default_rng(7).normal(0.0, 10.0, x.size) + _gauss(x, 1015.0, 600.0, 30.0) + _gauss(x, 880.0, 150.0, 30.0)
    out.mkdir(parents=True, exist_ok=True)
    return _fit_point_minerals(
        {"point_idx": 3, "y": y}, x=x, fit_cfg=cfg, fit_roi=(700.0, 1200.0), plot_roi=(700.0, 1300.0),
        out_dir=str(out), sol="0001", target="T", scan="detail_1", region="R1", snr_min_cfg=3.0, fwhm_min_cfg=25.0,
    )


def test_worker_writes_the_size_log_and_the_marks(tmp_path):
    out = _run_worker(tmp_path)
    csv = tmp_path / "0001_T_detail_1_R1_point3_fit_peaks.csv"
    side = size_sidecar_path(csv)
    assert side in out["artifacts"] and not side.with_name(side.name + ".tmp").exists()
    rec = json.loads(side.read_text())
    assert rec["version"] == SIZE_LOG_VERSION and rec["aicc_size"] >= 1 and rec["sizes"]
    assert {"p", "seed_ranks", "params", "rss", "aicc", "merge_skip", "failed", "step_aicc_gain", "step_p"} <= set(rec["sizes"][0])
    df = pd.read_csv(csv)
    assert df["aicc_gain"].notna().all() and df["step_p"].notna().all()


def test_a_fit_with_no_size_search_drops_an_earlier_log(tmp_path):
    _run_worker(tmp_path)
    side = size_sidecar_path(tmp_path / "0001_T_detail_1_R1_point3_fit_peaks.csv")
    assert side.exists()
    out = _run_worker(tmp_path, {**CFG, "parsimony": {**CFG["parsimony"], "model_selection": "ftest"}})
    assert not side.exists() and side not in out["artifacts"]
    assert pd.read_csv(tmp_path / "0001_T_detail_1_R1_point3_fit_peaks.csv")["aicc_gain"].isna().all()


def test_kept_by_matching():
    assert _kept_by(1000.0, None) is None
    assert _kept_by(1000.0, []) == "aicc"
    assert _kept_by(1000.0, [1000.0 + KEPT_BY_MATCH_CM1]) == "both"
    assert _kept_by(1000.0, [1000.0 + KEPT_BY_MATCH_CM1 + 0.01]) == "aicc"


def _sidecar(csv, ftest_rows, ftest_r2=0.9):
    rec = dict(version=SIZE_LOG_VERSION, n=50, rss0=1.0, aicc0=0.0, ftest_alpha=0.01, sizes=[],
               aicc_size=2, ftest_size=len(ftest_rows), ftest_r2=ftest_r2, ftest_peaks=ftest_rows)
    size_sidecar_path(csv).write_text(json.dumps(rec))


def _row(center, snr=15.0, **kw):
    return {"center_cm1": center, "fwhm_cm1": 30.0, "amplitude_a": 1000.0, "area": 50.0, "snr": snr,
            "pass_sharpness": True, **kw}


class TestPersistRead:
    def test_kept_by_and_marks_reach_fitted_peaks(self, populated_db, results_all):
        engine, spectrum_ids, _ = populated_db
        d = results_all / "minerals_fit"
        p0 = d / "0851_Lake_Haiyaha_detail_1_R1_point0_fit_peaks.csv"
        p1 = d / "0851_Lake_Haiyaha_detail_1_R1_point1_fit_peaks.csv"
        pd.DataFrame([_row(1085.5, aicc_gain=40.5, step_p=1e-9),
                      _row(1300.0, aicc_gain=3.25, step_p=0.04)]).to_csv(p0, index=False)
        # The F-test pick agrees at 1085 (within 2 cm-1); its 1301 fails the persist SNR gate.
        _sidecar(p0, [_row(1086.9), _row(1301.0, snr=0.5)])
        pd.DataFrame([_row(700.0, aicc_gain=12.0, step_p=0.001)]).to_csv(p1, index=False)  # no size log
        size_sidecar_path(p1).unlink(missing_ok=True)

        result = _call_persist_raman(_make_service(engine), results_all, "minerals")
        assert result.metadata["peaks_inserted"] == 3
        with get_session(engine) as session:
            got = sorted((p.center_cm1, p.kept_by, p.aicc_gain, p.step_p) for p in session.query(FittedPeakORM).all())
            pyd = session.query(FittedPeakORM).first().to_pydantic()
        assert got == [(700.0, None, 12.0, 0.001), (1085.5, "both", 40.5, 1e-9), (1300.0, "aicc", 3.25, 0.04)]
        assert pyd.kept_by in (None, "aicc", "both")

    def test_the_ftest_picks_own_r2_floor_applies(self, populated_db, results_all):
        engine, _, _ = populated_db
        p0 = results_all / "minerals_fit" / "0851_Lake_Haiyaha_detail_1_R1_point0_fit_peaks.csv"
        pd.DataFrame([_row(1085.5)]).to_csv(p0, index=False)
        _sidecar(p0, [_row(1085.5)], ftest_r2=-0.1)
        _call_persist_raman(_make_service(engine), results_all, "minerals")
        with get_session(engine) as session:
            rows = {p.center_cm1: p.kept_by for p in session.query(FittedPeakORM).all()}
        assert rows[1085.5] == "aicc"

    def test_an_unreadable_log_is_a_warning_not_a_failure(self, populated_db, results_all):
        engine, _, _ = populated_db
        p0 = results_all / "minerals_fit" / "0851_Lake_Haiyaha_detail_1_R1_point0_fit_peaks.csv"
        pd.DataFrame([_row(1085.5)]).to_csv(p0, index=False)
        size_sidecar_path(p0).write_text("{not json")
        result = _call_persist_raman(_make_service(engine), results_all, "minerals")
        assert any("unreadable" in w for w in result.warnings)
        with get_session(engine) as session:
            assert {p.kept_by for p in session.query(FittedPeakORM).all()} == {None}

    @pytest.mark.parametrize("text", [
        "[1, 2]",                                                                    # root not an object
        json.dumps({"version": SIZE_LOG_VERSION, "ftest_peaks": "x"}),                # rows not a list
        json.dumps({"version": SIZE_LOG_VERSION, "ftest_peaks": [{"center_cm1": 1085.5, "fwhm_cm1": 30.0}]}),  # no snr
        json.dumps({"version": SIZE_LOG_VERSION, "ftest_peaks": [{"center_cm1": "a", "fwhm_cm1": 30.0, "snr": 9.0}]}),
        '{"version": "%s", "ftest_peaks": [{"center_cm1": NaN, "fwhm_cm1": 30.0, "snr": 9.0}]}' % SIZE_LOG_VERSION,
        json.dumps({"version": SIZE_LOG_VERSION, "ftest_r2": "high", "ftest_peaks": [_row(1085.5)]}),
    ])
    def test_a_malformed_log_is_a_warning_not_a_failure(self, populated_db, results_all, text):
        engine, _, _ = populated_db
        p0 = results_all / "minerals_fit" / "0851_Lake_Haiyaha_detail_1_R1_point0_fit_peaks.csv"
        pd.DataFrame([_row(1085.5)]).to_csv(p0, index=False)
        size_sidecar_path(p0).write_text(text)
        result = _call_persist_raman(_make_service(engine), results_all, "minerals")
        assert any("unreadable" in w for w in result.warnings)
        with get_session(engine) as session:
            kept = {p.center_cm1: p.kept_by for p in session.query(FittedPeakORM).all()}
        assert kept[1085.5] is None

    def test_tables_without_the_columns_persist_null_marks(self, populated_db, results_all):
        engine, _, _ = populated_db
        _call_persist_raman(_make_service(engine), results_all, "minerals")
        with get_session(engine) as session:
            peaks = session.query(FittedPeakORM).all()
        assert peaks and {(p.kept_by, p.aicc_gain, p.step_p) for p in peaks} == {(None, None, None)}
