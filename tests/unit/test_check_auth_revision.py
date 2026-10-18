"""scripts/check-auth-revision.py against invented build inputs."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check-auth-revision.py"
SHA = "0123456789abcdef0123456789abcdef01234567"
URL = f"git+https://github.com/archaeon-ai/phase-platform-auth@{SHA}"


def _load(root):
    spec = importlib.util.spec_from_file_location("check_auth_revision", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    return module


def _write(root, lock_line):
    (root / "pyproject.toml").write_text(f'dependencies = ["phase-platform-auth @ {URL}"]\n')
    (root / "Dockerfile").write_text(f"RUN pip install \\\n      'phase-platform-auth @ {URL}'\n")
    (root / "requirements-lock.txt").write_text(f"anyio==4.0.0\n{lock_line}\n    # via sherloc\n")


def test_exact_lock_line_passes(tmp_path):
    _write(tmp_path, f"phase-platform-auth @ {URL}")
    assert _load(tmp_path).main() == 0


@pytest.mark.parametrize("lock_line", [
    f"  phase-platform-auth @ {URL}",
    f"phase-platform-auth  @ {URL}",
    f"phase-platform-auth @  {URL}",
    f"phase-platform-auth\t@ {URL}",
    f"phase-platform-auth @ {URL} ; python_version >= '3.12'",
])
def test_lock_line_the_dockerfile_sed_would_miss_fails(tmp_path, lock_line, capsys):
    _write(tmp_path, lock_line)
    assert _load(tmp_path).main() == 1
    assert "must match the Dockerfile sed" in capsys.readouterr().err
