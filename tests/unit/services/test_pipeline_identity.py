"""pipeline_identity: the code identity written on each fitted scan."""

import subprocess

import pytest

from sherloc_pipeline import __version__
from sherloc_pipeline.services.scan_processing import pipeline_identity


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    package = repo / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init")
    return repo, package


def test_clean_checkout_is_the_full_commit(checkout):
    repo, package = checkout
    assert pipeline_identity(package) == _git(repo, "rev-parse", "HEAD")
    assert len(pipeline_identity(package)) == 40


def test_modified_tracked_file_marks_the_commit_dirty(checkout):
    repo, package = checkout
    (package / "__init__.py").write_text("x = 2\n")
    assert pipeline_identity(package) == _git(repo, "rev-parse", "HEAD") + "-dirty"


def test_untracked_files_do_not_mark_it_dirty(checkout):
    repo, package = checkout
    (repo / "notes.txt").write_text("local\n")
    assert pipeline_identity(package) == _git(repo, "rev-parse", "HEAD")


def test_without_a_work_tree_it_is_the_package_version(tmp_path):
    package = tmp_path / "site-packages" / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    assert pipeline_identity(package) == __version__


def test_package_not_tracked_by_the_enclosing_tree_is_the_package_version(checkout):
    repo, _ = checkout
    vendored = repo / "vendor" / "pkg"
    vendored.mkdir(parents=True)
    (vendored / "__init__.py").write_text("")
    assert pipeline_identity(vendored) == __version__
