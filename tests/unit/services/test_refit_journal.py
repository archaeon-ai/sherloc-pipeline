"""Refit batch journal: success predicate, identity check on resume, crash-safe appends."""

import inspect
import json
from pathlib import Path

import pytest

from sherloc_pipeline.services import pipeline, refit_journal
from sherloc_pipeline.services.refit_journal import (
    IDENTITY_FIELDS,
    REQUIRED_STAGES,
    BatchJournal,
    JournalError,
    batch_identity,
    code_digest,
    pending,
    run_failures,
)


def _stage(name, **extra):
    return {"name": name, "warnings": [], "extra": extra}


def _complete():
    stages = [_stage(name) for name in REQUIRED_STAGES]
    persistence = next(s for s in stages if s["name"] == "raman_persistence")
    persistence["extra"] = {d: {"peaks_inserted": 0} for d in ("minerals", "organics", "hydration")}
    return {"run": {"stages": stages}}


def _identity(tmp_path, **changes):
    identity = batch_identity(
        database=tmp_path / "synthetic.db",
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
        selection={"science": True, "no_engineering": False},
        config_digest="c" * 64,
        code="d" * 64,
    )
    identity.update(changes)
    return identity


class TestRunFailures:
    def test_complete_run(self):
        assert run_failures(_complete()) == []

    def test_warnings_alone_do_not_fail(self):
        metadata = _complete()
        for stage in metadata["run"]["stages"]:
            stage["warnings"] = ["0 peaks persisted for 2 points"]
        assert run_failures(metadata) == []

    @pytest.mark.parametrize("name", REQUIRED_STAGES)
    def test_missing_stage(self, name):
        metadata = _complete()
        metadata["run"]["stages"] = [s for s in metadata["run"]["stages"] if s["name"] != name]
        assert f"{name}: did not run" in run_failures(metadata)

    @pytest.mark.parametrize("name", ["fluorescence_inputs", "fluorescence_fitting", "review", "cr_mask_persistence"])
    def test_caught_exception(self, name):
        metadata = _complete()
        metadata["run"]["stages"].append(_stage(name, exception="synthetic failure"))
        assert f"{name}: synthetic failure" in run_failures(metadata)

    def test_raman_domain_error(self):
        metadata = _complete()
        persistence = next(s for s in metadata["run"]["stages"] if s["name"] == "raman_persistence")
        del persistence["extra"]["organics"]
        persistence["extra"]["organics_error"] = "synthetic failure"
        assert run_failures(metadata) == ["raman_persistence: synthetic failure"]

    def test_raman_domain_missing(self):
        metadata = _complete()
        persistence = next(s for s in metadata["run"]["stages"] if s["name"] == "raman_persistence")
        del persistence["extra"]["hydration"]
        assert run_failures(metadata) == ["raman_persistence: hydration did not run"]

    def test_no_run_metadata(self):
        assert len(run_failures({})) == len(REQUIRED_STAGES)

    def test_required_stages_exist_in_the_pipeline(self):
        source = inspect.getsource(pipeline.PipelineService.run_full_pipeline)
        for name in REQUIRED_STAGES:
            assert f'capture_stage("{name}")' in source


class TestJournal:
    def test_create_resume_skips_only_recorded_scans(self, tmp_path):
        path = tmp_path / "batch.jsonl"
        with BatchJournal.create(path, _identity(tmp_path)) as batch:
            batch.record(10, "Synthetic A", "scan_1")
        with BatchJournal.resume(path, _identity(tmp_path)) as batch:
            to_run, skipped = pending(
                [(10, "Synthetic A", "scan_1"), (10, "Synthetic A", "scan_2"), (11, None, "scan_1")],
                batch.done,
            )
        assert skipped == [(10, "Synthetic A", "scan_1")]
        assert to_run == [(10, "Synthetic A", "scan_2"), (11, None, "scan_1")]

    def test_create_refuses_existing(self, tmp_path):
        path = tmp_path / "batch.jsonl"
        BatchJournal.create(path, _identity(tmp_path)).close()
        with pytest.raises(JournalError, match="already exists"):
            BatchJournal.create(path, _identity(tmp_path))

    @pytest.mark.parametrize("field", [f for f in IDENTITY_FIELDS if f != "schema"])
    def test_resume_refuses_any_identity_change(self, tmp_path, field):
        path = tmp_path / "batch.jsonl"
        BatchJournal.create(path, _identity(tmp_path)).close()
        changed = {"selection": {"science": False, "no_engineering": False}}.get(field, "other")
        with pytest.raises(JournalError, match=f"Cannot resume .*{field}"):
            BatchJournal.resume(path, _identity(tmp_path, **{field: changed}))

    def test_resume_refuses_other_files(self, tmp_path):
        path = tmp_path / "other.jsonl"
        path.write_text('{"schema": "something-else"}\n')
        with pytest.raises(JournalError, match="Not a refit journal"):
            BatchJournal.resume(path, _identity(tmp_path))
        with pytest.raises(JournalError, match="not found"):
            BatchJournal.resume(tmp_path / "missing.jsonl", _identity(tmp_path))

    def test_unterminated_last_line_is_dropped_and_appends_stay_whole(self, tmp_path):
        path = tmp_path / "batch.jsonl"
        with BatchJournal.create(path, _identity(tmp_path)) as batch:
            batch.record(10, "Synthetic A", "scan_1")
        with open(path, "a") as f:
            f.write('{"sol": 10, "target": "Synthetic A", "scan": "sc')
        with BatchJournal.resume(path, _identity(tmp_path)) as batch:
            assert batch.done == {(10, "Synthetic A", "scan_1")}
            batch.record(10, "Synthetic A", "scan_2")
        lines = path.read_text().splitlines()
        assert [json.loads(line)["scan"] for line in lines[1:]] == ["scan_1", "scan_2"]

    def test_unreadable_middle_line_refuses(self, tmp_path):
        path = tmp_path / "batch.jsonl"
        BatchJournal.create(path, _identity(tmp_path)).close()
        with open(path, "a") as f:
            f.write("not json\n")
        with pytest.raises(JournalError, match="line 2 is unreadable"):
            BatchJournal.resume(path, _identity(tmp_path))

    def test_one_process_at_a_time(self, tmp_path):
        path = tmp_path / "batch.jsonl"
        with BatchJournal.create(path, _identity(tmp_path)):
            with pytest.raises(JournalError, match="in use"):
                BatchJournal.resume(path, _identity(tmp_path))


def test_code_digest_tracks_content_not_bytecode(tmp_path):
    package = tmp_path / "pkg"
    (package / "__pycache__").mkdir(parents=True)
    (package / "a.py").write_text("x = 1\n")
    before = code_digest(package)
    (package / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
    assert code_digest(package) == before
    (package / "a.py").write_text("x = 2\n")
    assert code_digest(package) != before


def test_code_digest_defaults_to_the_installed_package():
    assert code_digest() == code_digest(Path(refit_journal.__file__).resolve().parents[1])
