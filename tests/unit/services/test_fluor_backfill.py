"""backfill-fluor-inputs fails closed and backfills a copy by default."""

import hashlib
import sqlite3

import pytest
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app
from sherloc_pipeline.database.connection import get_session
from sherloc_pipeline.database.models import ScanORM
from sherloc_pipeline.services import fluor_backfill
from sherloc_pipeline.services.cr_masks import CRMaskService
from sherloc_pipeline.services.ingestion import IngestionService

WORKSPACE_REL = (
    "loupe/sol_0921/detail_1/SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
)
TARGET = "Fluor_Backfill"
MASKS = {"0": {"R1": [60], "R2": [600], "R3": [1700]}, "1": {"R1": [], "R2": [800], "R3": []}}


class _Config:
    preprocessing = {"despike": {"interpolation_method": "linear"}}


@pytest.fixture
def source(tmp_path, fixtures_path):
    db_path = tmp_path / "phase.db"
    service = IngestionService(
        database_path=db_path, include_spectra=True, ingestion_mode="all_regions"
    )
    service.ingest_workspace(fixtures_path / WORKSPACE_REL)
    with get_session(service.engine) as session:
        session.query(ScanORM).first().target = TARGET.replace("_", " ")
    CRMaskService().persist_masks(
        sol="0921", target=TARGET, scan="detail_1", database_path=db_path,
        despike_metadata={
            "method": "ml_v1.3_tau_matched", "model_sha256": "ab" * 32,
            "tau": {"R1": 0.3, "R2": 0.27, "R3": 0.27}, "ort_version": "1.26.0",
            "n_flagged": {"R1": 1, "R2": 2, "R3": 1}, "masks": MASKS,
        },
    )
    service.engine.dispose()
    return db_path


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def _despiked(path):
    with sqlite3.connect(path) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM spectra WHERE spectrum_type = 'dark_sub_despiked'"
        ).fetchone()[0]


def test_the_default_backfills_a_copy_and_leaves_the_source_alone(source, tmp_path):
    before = _md5(source)
    result = fluor_backfill.backfill(source, _Config(), out=tmp_path / "copy.db")

    assert (result.scans, result.rows_written) == (1, 6)
    assert _despiked(tmp_path / "copy.db") == 6
    assert _md5(source) == before


def test_in_place_needs_allow_live(source):
    before = _md5(source)
    with pytest.raises(fluor_backfill.BackfillRefused, match="allow_live"):
        fluor_backfill.backfill(source, _Config(), in_place=True)
    assert _md5(source) == before


def test_in_place_with_allow_live_writes_the_source(source):
    result = fluor_backfill.backfill(source, _Config(), in_place=True, allow_live=True)
    assert result.rows_written == 6 and _despiked(source) == 6


def test_an_existing_copy_path_is_refused(source, tmp_path):
    (tmp_path / "copy.db").write_bytes(b"")
    with pytest.raises(fluor_backfill.BackfillRefused, match="exists"):
        fluor_backfill.backfill(source, _Config(), out=tmp_path / "copy.db")


def test_a_second_backfill_is_refused_while_the_lock_is_held(source):
    with fluor_backfill.exclusive_lock(source):
        with pytest.raises(fluor_backfill.BackfillRefused, match="another backfill"):
            fluor_backfill.backfill(source, _Config(), in_place=True, allow_live=True)


def test_in_place_is_refused_while_a_writer_runs(source, monkeypatch):
    monkeypatch.setattr(fluor_backfill, "writers_on", lambda path: [4242])
    with pytest.raises(fluor_backfill.BackfillRefused, match="4242"):
        fluor_backfill.backfill(source, _Config(), in_place=True, allow_live=True)
    assert _despiked(source) == 0


def _fake_proc(root, pid, argv, cwd, env=""):
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    (d / "environ").write_bytes(env.encode() + b"\0")
    if cwd is not None:
        (d / "cwd").symlink_to(cwd)


def test_writer_detection_resolves_args_env_and_the_implicit_default(source, tmp_path):
    proc, here, elsewhere = tmp_path / "proc", source.parent, tmp_path / "other"
    elsewhere.mkdir()
    (elsewhere / "phase.db").write_bytes(b"")
    _fake_proc(proc, 101, ["python", "-m", "sherloc_pipeline.cli.app", "full-pipeline", "0921"],
               elsewhere, env=f"SHERLOC_DB_PATH={source}")
    _fake_proc(proc, 102, ["sherloc", "ingest", "x", "--database", source.name], here)
    _fake_proc(proc, 103, ["sherloc", "full-pipeline", "0921", "T", "detail_1"], here)  # implicit ./phase.db
    _fake_proc(proc, 104, ["sherloc", "serve", "--database", str(source)], here)  # not a writer
    _fake_proc(proc, 105, ["sherloc", "full-pipeline", "0921", "T", "detail_1"], elsewhere)  # other phase.db
    _fake_proc(proc, 106, ["sherloc", "ingest", "x"], None)  # cwd unreadable: counted

    assert sorted(fluor_backfill.writers_on(source, proc_root=proc)) == [101, 102, 103, 106]


def test_in_place_is_refused_without_proc(source, tmp_path, monkeypatch):
    real = fluor_backfill.writers_on
    monkeypatch.setattr(
        fluor_backfill, "writers_on", lambda path: real(path, proc_root=tmp_path / "no-proc")
    )
    with pytest.raises(fluor_backfill.BackfillRefused, match="cannot inspect"):
        fluor_backfill.backfill(source, _Config(), in_place=True, allow_live=True)


def test_a_busy_database_stops_the_backfill(source, monkeypatch):
    monkeypatch.setattr(fluor_backfill, "BUSY_TIMEOUT_MS", 100)
    holder = sqlite3.connect(source, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        result = fluor_backfill.backfill(source, _Config(), in_place=True, allow_live=True)
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert result.stopped and "busy" in result.stopped
    assert result.scans == 0 and _despiked(source) == 0


def test_cli_dry_run_and_copy(source, tmp_path):
    runner = CliRunner()
    dry = runner.invoke(app, ["backfill-fluor-inputs", "--source-db", str(source), "--dry-run"])
    assert dry.exit_code == 0 and "1 scan(s) with stored masks" in dry.output

    refused = runner.invoke(
        app, ["backfill-fluor-inputs", "--source-db", str(source), "--in-place"]
    )
    assert refused.exit_code == 2 and "Refused" in refused.output

    ok = runner.invoke(app, [
        "backfill-fluor-inputs", "--source-db", str(source), "--out-db", str(tmp_path / "c.db"),
    ])
    assert ok.exit_code == 0, ok.output
    assert _despiked(tmp_path / "c.db") == 6
