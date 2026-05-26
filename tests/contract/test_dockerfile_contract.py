"""Contract test: Dockerfile bakes in the documented image-shape surface.

Pins DEPLOYMENT_CONTRACT.md §2 + §3 against the static Dockerfile text.
Tolerates patch-version bumps of the python base (regex on the minor)
but a Python 3.11 / 3.13 jump trips the test deliberately — that is a
contract change.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"


def _dockerfile_text() -> str:
    return DOCKERFILE.read_text()


def test_dockerfile_exists():
    assert DOCKERFILE.is_file(), f"missing {DOCKERFILE}"


def test_runtime_stage_base_is_python_3_12():
    text = _dockerfile_text()
    # `FROM python:3.12.<patch>-slim-bookworm AS runtime` shape; we accept
    # patch + flavor variations but require the 3.12 minor.
    assert re.search(
        r"^FROM\s+python:3\.12\.[0-9a-zA-Z._-]+\s+AS\s+runtime\s*$",
        text,
        re.MULTILINE,
    ), "runtime stage must use python:3.12.* base"


def test_expose_8000():
    text = _dockerfile_text()
    assert re.search(r"^EXPOSE\s+8000\s*$", text, re.MULTILINE)


def test_healthcheck_targets_localhost_8000_health():
    text = _dockerfile_text()
    assert "http://localhost:8000/api/health" in text


def test_user_sherloc_before_expose():
    text = _dockerfile_text()
    user_match = re.search(r"^USER\s+sherloc\s*$", text, re.MULTILINE)
    expose_match = re.search(r"^EXPOSE\s+8000\s*$", text, re.MULTILINE)
    assert user_match, "missing `USER sherloc` line"
    assert expose_match, "missing `EXPOSE 8000` line"
    assert user_match.start() < expose_match.start(), (
        "USER sherloc must appear before EXPOSE 8000"
    )


def test_entrypoint_uses_tini_and_entrypoint_script():
    text = _dockerfile_text()
    # ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker-entrypoint.sh"]
    assert re.search(
        r'^ENTRYPOINT\s+\[\s*"/usr/bin/tini"\s*,\s*"--"\s*,\s*"/app/docker-entrypoint\.sh"\s*\]\s*$',
        text,
        re.MULTILINE,
    )


def test_default_cmd_is_web():
    text = _dockerfile_text()
    assert re.search(r'^CMD\s+\[\s*"web"\s*\]\s*$', text, re.MULTILINE)


def test_runtime_user_uid_gid_1000():
    text = _dockerfile_text()
    assert re.search(r"groupadd\s+-g\s+1000\s+sherloc", text)
    assert re.search(r"useradd\s+-u\s+1000\s+-g\s+sherloc", text)


def test_build_time_smoke_imports_present():
    """The build-time `RUN python -c ...` block guarantees the [web]/[pds]
    import surface lands in the runtime image. Without these, a future
    extras-spec regression would silently break first-request startup."""
    text = _dockerfile_text()
    required = [
        "numpy",
        "fastapi",
        "uvicorn",
        "jwt",
        "httpx",
        "websockets",
        "boto3",
        "sherloc_pipeline.web.app",
    ]
    for symbol in required:
        assert symbol in text, (
            f"build-time smoke must import {symbol!r}; missing from Dockerfile"
        )
