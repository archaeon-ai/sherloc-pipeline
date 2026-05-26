"""Contract test: live container smoke.

Pins DEPLOYMENT_CONTRACT.md §3 (healthcheck) + §5 (validator gates) +
§6 (public-mode DB invariant) + §7 (boot order). Five cases:

  A — positive, dev-mode internal tier: container reaches health 200;
      ``alembic current`` matches EXPECTED_HEAD.
  B — negative, missing SHERLOC_DB: container exits non-zero with
      `missing required variable: SHERLOC_DB` on stderr.
  C — negative, public-mode + non-compliant DB filename: container
      exits non-zero with ValueError referencing `phase_pds.db`.
  D — positive, public-mode + compliant DB filename: container reaches
      health 200.
  E — negative, PHASE_DATABASE_PATH mismatch: container exits non-zero
      with `differ` error from config_check.

Double-marked ``docker`` + ``slow`` so the cheap CI lane filters it out.
The dedicated ``.github/workflows/contract-smoke.yml`` workflow opts it
in. Skipped automatically when ``docker`` is not on PATH or when
``SKIP_DOCKER_TESTS=1``.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

from tests.contract.test_alembic_contract import EXPECTED_HEAD

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_TAG = "sherloc-pipeline:contract-test"
BUILD_TIMEOUT = 600  # 10 min — slow CI worst case
HEALTH_TIMEOUT = 90  # poll budget for /api/health
HEALTH_POLL_INTERVAL = 2.0

pytestmark = [pytest.mark.docker, pytest.mark.slow]


def _docker_available() -> bool:
    return shutil.which("docker") is not None and not os.environ.get(
        "SKIP_DOCKER_TESTS"
    )


if not _docker_available():
    pytest.skip(
        "docker not available or SKIP_DOCKER_TESTS=1; contract smoke skipped",
        allow_module_level=True,
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _container_name() -> str:
    return f"sct-{uuid.uuid4().hex[:8]}"


def _docker_rm(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        check=False,
        capture_output=True,
    )


def _write_env_file(path: Path, env: dict[str, str]) -> None:
    path.write_text("\n".join(f"{k}={v}" for k, v in env.items()) + "\n")


@pytest.fixture(scope="module")
def built_image():
    """Build the runtime stage once per test module."""
    result = subprocess.run(
        ["docker", "build", "--target", "runtime", "-t", IMAGE_TAG, str(REPO_ROOT)],
        timeout=BUILD_TIMEOUT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            "docker build failed:\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    yield IMAGE_TAG
    if os.environ.get("CONTRACT_TEST_PRUNE") == "1":
        subprocess.run(
            ["docker", "rmi", "-f", IMAGE_TAG], check=False, capture_output=True
        )


def _poll_health(port: int, timeout: float = HEALTH_TIMEOUT) -> tuple[bool, str]:
    """Poll http://127.0.0.1:port/api/health until 200 or timeout.
    Returns (ok, last_error_text)."""
    deadline = time.monotonic() + timeout
    last_err = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/health", timeout=5
            ) as resp:
                if resp.status == 200:
                    return True, ""
                last_err = f"http {resp.status}"
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_err = repr(exc)
        time.sleep(HEALTH_POLL_INTERVAL)
    return False, last_err


def _wait_for_exit(name: str, timeout: float = 30.0) -> tuple[int, str]:
    """Wait for a container to exit. Returns (exit_code, combined logs)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}/{{.State.ExitCode}}", name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            time.sleep(0.5)
            continue
        status_field = result.stdout.strip()
        # Expected shape: "exited/<code>" or "running/0" or "created/0"
        if "/" in status_field:
            status, code = status_field.rsplit("/", 1)
            if status == "exited":
                logs = subprocess.run(
                    ["docker", "logs", name],
                    capture_output=True,
                    text=True,
                )
                return int(code), logs.stdout + logs.stderr
        time.sleep(0.5)
    # Force-stop and capture what we have.
    subprocess.run(["docker", "stop", name], check=False, capture_output=True)
    logs = subprocess.run(
        ["docker", "logs", name], capture_output=True, text=True
    )
    return -1, logs.stdout + logs.stderr


# =====================================================================
# Case A — positive, dev-mode internal tier
# =====================================================================


def test_case_a_dev_mode_internal_boots_healthy(built_image, tmp_path):
    name = _container_name()
    host_port = _free_port()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # mkdir(mode=) is filtered by the process umask; the container runs
    # as uid/gid 1000 (per DEPLOYMENT_CONTRACT.md §2 + §4) and must be
    # able to write /data. Force the mode after creation to bypass umask.
    os.chmod(data_dir, 0o777)
    db_path = "/data/phase.db"
    env_file = tmp_path / "env"
    _write_env_file(
        env_file,
        {
            "SHERLOC_AUTH_MODE": "dev",
            "SHERLOC_DB": db_path,
        },
    )
    try:
        run = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", name,
                "--env-file", str(env_file),
                "-v", f"{data_dir}:/data",
                "-p", f"127.0.0.1:{host_port}:8000",
                built_image,
            ],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, (
            f"docker run failed: {run.stdout}\n{run.stderr}"
        )
        ok, last_err = _poll_health(host_port)
        if not ok:
            logs = subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True
            )
            pytest.fail(
                f"Case A: /api/health did not return 200 within "
                f"{HEALTH_TIMEOUT}s (last={last_err}).\nlogs:\n"
                f"{logs.stdout}\n{logs.stderr}"
            )
        # Verify alembic head matches contract pin. `docker exec` runs
        # the binary directly in a fresh exec process whose environment
        # is the container's image config + `--env-file` plus explicit
        # `-e` overrides; it does NOT inherit variables exported inside
        # the entrypoint's already-running process (no shell is
        # involved). The Q1 collapse intentionally derives
        # PHASE_DATABASE_PATH from SHERLOC_DB inside the entrypoint, so
        # we must pass it explicitly here to give alembic env.py a
        # usable URL. The image itself is fine (the entrypoint already
        # ran `alembic upgrade head` before /api/health 200).
        head_result = subprocess.run(
            [
                "docker", "exec",
                "-e", f"PHASE_DATABASE_PATH={db_path}",
                name, "alembic", "current",
            ],
            capture_output=True,
            text=True,
        )
        assert EXPECTED_HEAD in head_result.stdout, (
            f"alembic current did not include EXPECTED_HEAD={EXPECTED_HEAD!r}: "
            f"stdout={head_result.stdout!r} stderr={head_result.stderr!r}"
        )
    finally:
        _docker_rm(name)


# =====================================================================
# Case B — negative, missing SHERLOC_DB
# =====================================================================


def test_case_b_missing_db_var_fails(built_image, tmp_path):
    name = _container_name()
    env_file = tmp_path / "env"
    _write_env_file(env_file, {"SHERLOC_AUTH_MODE": "dev"})
    try:
        run = subprocess.run(
            ["docker", "run", "-d", "--name", name, "--env-file", str(env_file), built_image],
            capture_output=True, text=True,
        )
        assert run.returncode == 0, f"docker run failed: {run.stderr}"
        exit_code, logs = _wait_for_exit(name)
        assert exit_code != 0, f"expected non-zero exit, got {exit_code}; logs:\n{logs}"
        assert "missing required variable: SHERLOC_DB" in logs, (
            f"missing required-var message not in logs:\n{logs}"
        )
    finally:
        _docker_rm(name)


# =====================================================================
# Case C — negative, public-mode + non-compliant DB filename
# =====================================================================


def test_case_c_public_mode_non_compliant_db_rejected(built_image, tmp_path):
    name = _container_name()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # mkdir(mode=) is filtered by the process umask; the container runs
    # as uid/gid 1000 (per DEPLOYMENT_CONTRACT.md §2 + §4) and must be
    # able to write /data. Force the mode after creation to bypass umask.
    os.chmod(data_dir, 0o777)
    env_file = tmp_path / "env"
    _write_env_file(
        env_file,
        {
            "SHERLOC_AUTH_MODE": "dev",
            "SHERLOC_ACCESS_MODE": "public",
            "SHERLOC_DB": "/data/phase.db",
        },
    )
    try:
        run = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", name,
                "--env-file", str(env_file),
                "-v", f"{data_dir}:/data",
                built_image,
            ],
            capture_output=True, text=True,
        )
        assert run.returncode == 0, f"docker run failed: {run.stderr}"
        exit_code, logs = _wait_for_exit(name)
        assert exit_code != 0, f"expected non-zero exit, got {exit_code}; logs:\n{logs}"
        # The check raises ValueError at create_app(); uvicorn surfaces
        # it. Either the literal phrase or the substring `phase_pds.db`
        # in the error message is acceptable evidence.
        assert (
            "phase_pds.db" in logs or "Public mode" in logs
        ), f"public-mode DB invariant message not in logs:\n{logs}"
    finally:
        _docker_rm(name)


# =====================================================================
# Case D — positive, public-mode + compliant DB filename
# =====================================================================


def test_case_d_public_mode_compliant_db_boots(built_image, tmp_path):
    name = _container_name()
    host_port = _free_port()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # mkdir(mode=) is filtered by the process umask; the container runs
    # as uid/gid 1000 (per DEPLOYMENT_CONTRACT.md §2 + §4) and must be
    # able to write /data. Force the mode after creation to bypass umask.
    os.chmod(data_dir, 0o777)
    env_file = tmp_path / "env"
    _write_env_file(
        env_file,
        {
            "SHERLOC_AUTH_MODE": "dev",
            "SHERLOC_ACCESS_MODE": "public",
            "SHERLOC_DB": "/data/phase_pds.db",
        },
    )
    try:
        run = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", name,
                "--env-file", str(env_file),
                "-v", f"{data_dir}:/data",
                "-p", f"127.0.0.1:{host_port}:8000",
                built_image,
            ],
            capture_output=True, text=True,
        )
        assert run.returncode == 0, f"docker run failed: {run.stderr}"
        ok, last_err = _poll_health(host_port)
        if not ok:
            logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
            pytest.fail(
                f"Case D: /api/health did not return 200 within "
                f"{HEALTH_TIMEOUT}s (last={last_err}).\nlogs:\n"
                f"{logs.stdout}\n{logs.stderr}"
            )
    finally:
        _docker_rm(name)


# =====================================================================
# Case E — negative, PHASE_DATABASE_PATH mismatch
# =====================================================================


def test_case_e_phase_database_path_mismatch_fails(built_image, tmp_path):
    name = _container_name()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # mkdir(mode=) is filtered by the process umask; the container runs
    # as uid/gid 1000 (per DEPLOYMENT_CONTRACT.md §2 + §4) and must be
    # able to write /data. Force the mode after creation to bypass umask.
    os.chmod(data_dir, 0o777)
    env_file = tmp_path / "env"
    _write_env_file(
        env_file,
        {
            "SHERLOC_AUTH_MODE": "dev",
            "SHERLOC_DB": "/data/phase.db",
            "PHASE_DATABASE_PATH": "/data/different.db",
        },
    )
    try:
        run = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", name,
                "--env-file", str(env_file),
                "-v", f"{data_dir}:/data",
                built_image,
            ],
            capture_output=True, text=True,
        )
        assert run.returncode == 0, f"docker run failed: {run.stderr}"
        exit_code, logs = _wait_for_exit(name)
        assert exit_code != 0, f"expected non-zero exit, got {exit_code}; logs:\n{logs}"
        assert "differ" in logs.lower(), (
            f"mismatch `differ` message not in logs:\n{logs}"
        )
        assert "PHASE_DATABASE_PATH" in logs, (
            f"PHASE_DATABASE_PATH name not in logs:\n{logs}"
        )
    finally:
        _docker_rm(name)
