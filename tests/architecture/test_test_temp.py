"""Test temp stays bounded locally and belongs to the CI job."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_pytest_keeps_only_failed_test_temp(pytestconfig):
    assert pytestconfig.getini("tmp_path_retention_policy") == "failed"


def test_pytest_keeps_only_the_last_run(pytestconfig):
    assert int(pytestconfig.getini("tmp_path_retention_count")) == 1


@pytest.mark.parametrize(
    ("workflow", "job"),
    [("ci.yml", "test"), ("contract-smoke.yml", "smoke")],
)
def test_ci_passes_job_temp_on_the_command_line(workflow, job, tmp_path):
    steps = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())[
        "jobs"
    ][job]["steps"]
    step = next(
        step
        for step in steps
        if "pytest " in step.get("run", "")
        and "--collect-only" not in step["run"]
    )
    commands = tmp_path / "bin"
    commands.mkdir()
    recorder = commands / "pytest"
    recorder.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['ARGV_LOG']).write_text(json.dumps({\n"
        "    'argv': sys.argv[1:],\n"
        "    'addopts': os.environ.get('PYTEST_ADDOPTS', ''),\n"
        "}))\n"
    )
    recorder.chmod(0o755)
    job_temp = tmp_path / "job temp"
    job_temp.mkdir()
    argv_log = tmp_path / "argv.json"
    env = {
        **os.environ,
        **step.get("env", {}),
        "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(job_temp),
        "ARGV_LOG": str(argv_log),
    }
    result = subprocess.run(
        ["bash", "-eu", "-c", step["run"]],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    captured = json.loads(argv_log.read_text())
    assert f"--basetemp={job_temp / 'pytest'}" in captured["argv"]
    assert "--basetemp" not in captured["addopts"]
    assert str(job_temp / "pytest") in result.stdout


def test_collection_guard_does_not_pass_basetemp():
    steps = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())[
        "jobs"
    ]["test"]["steps"]
    guard = next(step for step in steps if "--collect-only" in step.get("run", ""))
    assert "--basetemp" not in guard["run"]
    assert "--basetemp" not in guard.get("env", {}).get("PYTEST_ADDOPTS", "")
