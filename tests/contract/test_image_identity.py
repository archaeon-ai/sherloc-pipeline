"""Contract test: image identity claims match the package version + workflow.

Pins DEPLOYMENT_CONTRACT.md §2 + §10:

- ``pyproject.toml::project.version`` coheres with the latest ``v*`` git
  tag using a three-state rule (ahead / equal / behind).
- On tag push (GITHUB_REF_TYPE=tag), strict equality is required —
  ``GITHUB_REF_NAME == f"v{project.version}"``. That is the gate which
  prevents a tag publish from racing ahead of a pyproject bump.
- ``.github/workflows/publish.yml`` IMAGE_NAME equals
  ``ghcr.io/archaeon-ai/sherloc-pipeline``.
- ``publish.yml`` ``platforms:`` line equals ``linux/amd64,linux/arm64``
  (multi-arch added at v4.1.16 per FOUNDATION §3.5 step 6).

The PR-mode three-state rule (per Chunk B design §5.10 Round 2 fix):
- ``project.version`` AHEAD of latest tag → ``pytest.skip`` (the
  operator-coordinated post-Chunk-D window before the new tag is
  pushed; see Activity B in chunk_b_design.md §10.5).
- ``project.version`` EQUAL to latest tag → PASS.
- ``project.version`` BEHIND latest tag → FAIL (the regression Codex
  caught: pyproject lagging the live image tag).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PUBLISH_YML = REPO_ROOT / ".github" / "workflows" / "publish.yml"

EXPECTED_IMAGE_NAME = "ghcr.io/archaeon-ai/sherloc-pipeline"
EXPECTED_PLATFORMS = "linux/amd64,linux/arm64"


def _project_version() -> str:
    data = tomllib.loads(PYPROJECT.read_text())
    return data["project"]["version"]


def _semver_tuple(version: str) -> tuple[int, ...]:
    """Loose semver split — accepts X.Y.Z and X.Y.Z.PRE forms by taking
    the leading integer components only."""
    parts = re.split(r"[^0-9]+", version.lstrip("v"))
    return tuple(int(p) for p in parts if p.isdigit())


def _latest_v_tag() -> str | None:
    """Return the latest `v*` tag by creator date, or None if no such tag
    exists in the current checkout. Returns None on git failure too."""
    try:
        result = subprocess.run(
            ["git", "tag", "--list", "v*", "--sort=-creatordate"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    tags = [t.strip() for t in result.stdout.splitlines() if t.strip()]
    return tags[0] if tags else None


def test_project_version_parseable():
    version = _project_version()
    assert version, "pyproject.toml::project.version is empty"
    assert _semver_tuple(version), f"unparseable version {version!r}"


def test_tag_push_strict_equality():
    """When a tag is pushed (GITHUB_REF_TYPE=tag with a v* ref), the tag
    name MUST equal f"v{project.version}". This is the publish-gate."""
    ref_type = os.environ.get("GITHUB_REF_TYPE", "")
    ref_name = os.environ.get("GITHUB_REF_NAME", "")
    if ref_type != "tag" or not ref_name.startswith("v"):
        pytest.skip("not a tag-push CI invocation (GITHUB_REF_TYPE!=tag)")
    expected = f"v{_project_version()}"
    assert ref_name == expected, (
        f"tag push {ref_name!r} disagrees with pyproject "
        f"project.version → expected tag {expected!r}. Bump pyproject "
        "before tagging."
    )


def test_pr_mode_version_not_behind_latest_tag():
    """PR mode (no tag in env). Three-state rule:

    - project.version EQUAL to latest tag → PASS.
    - project.version AHEAD of latest tag → SKIP (operator-coordinated
      tag-push window; see chunk_b_design.md §10.5 Activity B).
    - project.version BEHIND latest tag → FAIL.
    """
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        pytest.skip("tag-push CI is covered by test_tag_push_strict_equality")
    latest = _latest_v_tag()
    if latest is None:
        pytest.skip("no v* git tags in checkout (shallow clone or fresh repo)")
    version = _project_version()
    v_version = f"v{version}"
    if v_version == latest:
        return
    proj_tuple = _semver_tuple(version)
    tag_tuple = _semver_tuple(latest)
    if proj_tuple > tag_tuple:
        pytest.skip(
            f"pyproject.toml version {version!r} is AHEAD of latest tag "
            f"{latest!r} — awaiting operator tag publish per "
            "chunk_b_design.md §10.5 Activity B"
        )
    pytest.fail(
        f"pyproject.toml version {version!r} is BEHIND latest tag "
        f"{latest!r}. Bump pyproject.toml::project.version to at least "
        f"{latest!r} (strip leading 'v')."
    )


def test_publish_yml_image_name():
    text = PUBLISH_YML.read_text()
    # `IMAGE_NAME: ghcr.io/archaeon-ai/sherloc-pipeline` in the env block.
    assert re.search(
        rf"^\s*IMAGE_NAME:\s*{re.escape(EXPECTED_IMAGE_NAME)}\s*$",
        text,
        re.MULTILINE,
    ), f"publish.yml IMAGE_NAME must equal {EXPECTED_IMAGE_NAME!r}"


def test_publish_yml_platforms_multi_arch():
    """v4.1.16 adds linux/arm64 alongside linux/amd64 per FOUNDATION §3.5
    step 6 (sunset deliverable for the public CLI install audience —
    Apple Silicon / AWS Graviton). DEPLOYMENT_CONTRACT.md §2.

    The build-push action's ``platforms:`` line is the contract surface
    (the manifest the registry sees). The QEMU setup step also carries
    a ``platforms:`` field, but that one names the emulator targets and
    uses the short form (``arm64``), so we scope this assertion to the
    build-push step explicitly.
    """
    text = PUBLISH_YML.read_text()
    # The build-push step is the contract surface. Match the `platforms:`
    # line that lives under the `Build and push runtime image` step, NOT
    # the QEMU setup step's emulator-targets list.
    build_push_block = re.search(
        r"name:\s*Build and push runtime image[\s\S]*?(?=^\s+- name:|\Z)",
        text,
        re.MULTILINE,
    )
    assert build_push_block, "publish.yml has no `Build and push runtime image` step"
    matches = re.findall(
        r"^\s*platforms:\s*(.+?)\s*$",
        build_push_block.group(0),
        re.MULTILINE,
    )
    assert matches, (
        "build-push step has no `platforms:` declaration"
    )
    for value in matches:
        assert value.strip() == EXPECTED_PLATFORMS, (
            f"publish.yml build-push platforms must equal "
            f"{EXPECTED_PLATFORMS!r}, got {value!r}"
        )
