"""
Enforcement tests for the fixture-provenance ledger and its checker.

Two layers:
1. Unit tests of scripts/check-fixture-provenance.py internals (sol
   extraction, prefix matching, envelope enforcement, drift detection,
   fail-closed coverage, stale-entry detection).
2. A live gate: the tracked tree must pass the checker — the same
   assertion CI runs via ``scripts/check-fixture-provenance.py --tree``,
   kept here too so a plain ``pytest`` run catches provenance violations.

See CONTRIBUTING.md "Fixture provenance".
"""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check-fixture-provenance.py"

_spec = importlib.util.spec_from_file_location("check_fixture_provenance", SCRIPT)
cfp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfp)


def _ledger(entries, max_sol=1739):
    return {
        "pds_envelope": {"sherloc": {"max_sol": max_sol}},
        "entries": entries,
    }


# ---------------------------------------------------------------- extraction

@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/fixtures/loupe/sol_0852/detail_1/loupe.csv", {852}),
        ("tests/golden/sol_921_detail_1/fitted_peaks.json", {921}),
        (
            "tests/fixtures/loupe/sol_0852/detail_1/"
            "SC3_0852_0742603923_718ECM_N0410188SRLC11376_0000LMJ01.IMG",
            {852},
        ),
        (
            "tests/fixtures/loupe/sol_0852/x/"
            "SS__0852_0742604032_130RLS__0410188SRLC11376_104___J01.csv",
            {852},
        ),
        ("a/plots/0921_Amherst_Point_detail_1_R1_p0_normalized.csv", {921}),
        # Ambiguous trailing numerics are deliberately NOT extracted: the
        # ledger entry is authoritative for these.
        (
            "tests/fixtures/background/"
            "Arm_Stowed_post-anomaly_900ppp_trimmed_mean_1266.csv",
            set(),
        ),
        # 'sol' must be its own token — no false positive on e.g. 'solver_2000'.
        ("tests/fixtures/solver_2000/out.csv", set()),
        ("src/sherloc_pipeline/web/frontend/package.json", set()),
    ],
)
def test_extract_sols(path, expected):
    assert cfp.extract_sols(path) == expected


def test_extract_sols_conflicting_tokens_all_reported():
    # Directory says 852, product id says 921 — both surface, so the
    # cross-check against the ledger entry fails loudly.
    path = "tests/fixtures/loupe/sol_0852/SC3_0921_0748731308_359ECM_X.IMG"
    assert cfp.extract_sols(path) == {852, 921}


# ------------------------------------------------------------ prefix matching

def test_find_entry_longest_prefix_wins():
    outer = {"path": "tests/fixtures/", "category": "metadata"}
    inner = {"path": "tests/fixtures/loupe/sol_0852/", "category": "pds-archived"}
    entries = [outer, inner]
    hit = cfp.find_entry("tests/fixtures/loupe/sol_0852/detail_1/loupe.csv", entries)
    assert hit is inner
    assert cfp.find_entry("tests/fixtures/manifest.json", entries) is outer


def test_find_entry_exact_file_and_miss():
    entry = {"path": "tests/fixtures/manifest.json", "category": "metadata"}
    assert cfp.find_entry("tests/fixtures/manifest.json", [entry]) is entry
    # Directory semantics require the trailing slash — a non-slash entry
    # must not prefix-match other files.
    assert cfp.find_entry("tests/fixtures/manifest.json.bak", [entry]) is None


# ----------------------------------------------------------------- check_files

def test_unledgered_data_file_fails_closed():
    violations = cfp.check_files(["tests/fixtures/new_thing.csv"], _ledger([]))
    assert len(violations) == 1
    assert "no provenance entry" in violations[0]


def test_envelope_violation_detected():
    entries = [
        {
            "path": "tests/fixtures/loupe/sol_1800/",
            "category": "pds-archived",
            "instrument": "sherloc",
            "sol": 1800,
        }
    ]
    violations = cfp.check_files(
        ["tests/fixtures/loupe/sol_1800/loupe.csv"], _ledger(entries)
    )
    assert any("exceeds the sherloc archived envelope" in v for v in violations)


def test_within_envelope_passes():
    entries = [
        {
            "path": "tests/fixtures/loupe/sol_0852/",
            "category": "pds-archived",
            "instrument": "sherloc",
            "sol": 852,
        }
    ]
    assert cfp.check_files(
        ["tests/fixtures/loupe/sol_0852/loupe.csv"], _ledger(entries)
    ) == []


def test_path_sol_vs_ledger_sol_drift_fails():
    entries = [
        {
            "path": "tests/fixtures/loupe/sol_0852/",
            "category": "pds-archived",
            "instrument": "sherloc",
            "sol": 921,
        }
    ]
    violations = cfp.check_files(
        ["tests/fixtures/loupe/sol_0852/loupe.csv"], _ledger(entries)
    )
    assert any("declares sol 921" in v for v in violations)


def test_embedded_sol_requires_declared_sol():
    entries = [{"path": "tests/fixtures/pipeline_outputs/", "category": "synthetic"}]
    violations = cfp.check_files(
        ["tests/fixtures/pipeline_outputs/0921_target_R1_p0.csv"], _ledger(entries)
    )
    assert any("declares no sol" in v for v in violations)


def test_synthetic_with_matching_nominal_sol_passes():
    entries = [
        {"path": "tests/fixtures/pipeline_outputs/", "category": "synthetic", "sol": 921}
    ]
    assert cfp.check_files(
        ["tests/fixtures/pipeline_outputs/0921_target_R1_p0.csv"], _ledger(entries)
    ) == []


def test_operator_derived_with_sol_is_envelope_checked():
    entries = [
        {
            "path": "tests/golden/sol_1900_x/",
            "category": "operator-derived-archived",
            "instrument": "sherloc",
            "sol": 1900,
        }
    ]
    violations = cfp.check_files(
        ["tests/golden/sol_1900_x/out.json"], _ledger(entries)
    )
    assert any("exceeds the sherloc archived envelope" in v for v in violations)


def test_metadata_passes_without_sol():
    entries = [{"path": "pkg/package.json", "category": "metadata"}]
    assert cfp.check_files(["pkg/package.json"], _ledger(entries)) == []


# -------------------------------------------------------------- stale entries

def test_stale_entries_detected_for_dir_and_file():
    ledger = _ledger(
        [
            {"path": "tests/fixtures/gone/", "category": "synthetic"},
            {"path": "tests/fixtures/gone.csv", "category": "synthetic"},
            {"path": "tests/fixtures/here/", "category": "synthetic"},
        ]
    )
    tracked = ["tests/fixtures/here/a.csv", "README.md"]
    violations = cfp.stale_entries(tracked, ledger)
    assert len(violations) == 2
    assert all("stale entry" in v for v in violations)


# ------------------------------------------------- staged-mode integration


@pytest.fixture()
def git_sandbox(tmp_path):
    """Minimal repo with one ledgered fixture, for staged-mode tests."""

    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        )

    run("init", "-q")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "test")
    fixdir = tmp_path / "tests" / "fixtures" / "loupe" / "sol_0852"
    fixdir.mkdir(parents=True)
    (fixdir / "loupe.csv").write_text("a,b\n1,2\n")
    ledger = {
        "pds_envelope": {"sherloc": {"max_sol": 1739}},
        "entries": [
            {"path": "tests/fixtures/PROVENANCE.json", "category": "metadata"},
            {
                "path": "tests/fixtures/loupe/sol_0852/",
                "category": "pds-archived",
                "instrument": "sherloc",
                "sol": 852,
            },
        ],
    }
    (tmp_path / "tests" / "fixtures" / "PROVENANCE.json").write_text(
        json.dumps(ledger)
    )
    run("add", "-A")
    run("commit", "-qm", "seed")
    return tmp_path, run


def _checker(cwd, mode):
    return subprocess.run([str(SCRIPT), mode], cwd=cwd, capture_output=True, text=True)


def test_staged_clean_sandbox_passes(git_sandbox):
    root, run = git_sandbox
    assert _checker(root, "--staged").returncode == 0
    assert _checker(root, "--tree").returncode == 0


def test_staged_rename_to_unledgered_path_fails(git_sandbox):
    # Renames are listed under their NEW path (--diff-filter includes R),
    # so moving a fixture out of its ledgered prefix fails at commit time.
    root, run = git_sandbox
    run(
        "mv",
        "tests/fixtures/loupe/sol_0852/loupe.csv",
        "tests/fixtures/loupe/sol_0852_moved.csv",
    )
    result = _checker(root, "--staged")
    assert result.returncode == 1
    assert "no provenance entry" in result.stderr


def test_staged_deletion_orphaning_ledger_entry_fails(git_sandbox):
    # Stale-entry detection runs against the git index in BOTH modes, so a
    # staged deletion that orphans a ledger entry fails at commit time.
    root, run = git_sandbox
    run("rm", "-q", "tests/fixtures/loupe/sol_0852/loupe.csv")
    result = _checker(root, "--staged")
    assert result.returncode == 1
    assert "stale entry" in result.stderr


def test_ledger_self_entry_keeps_coverage_exact(git_sandbox):
    # The ledger is itself a tracked .json data file; it is covered by its
    # own metadata entry rather than a checker special-case. Removing the
    # self-entry must fail tree mode.
    root, run = git_sandbox
    lpath = root / "tests" / "fixtures" / "PROVENANCE.json"
    ledger = json.loads(lpath.read_text())
    ledger["entries"] = [e for e in ledger["entries"] if "PROVENANCE" not in e["path"]]
    lpath.write_text(json.dumps(ledger))
    run("add", "-A")
    result = _checker(root, "--tree")
    assert result.returncode == 1
    assert "PROVENANCE.json: no provenance entry" in result.stderr


# ------------------------------------------------------------- ledger schema

def test_live_ledger_loads_and_validates():
    ledger = cfp.load_ledger(REPO)
    assert ledger["pds_envelope"]["sherloc"]["max_sol"] >= 1739
    categories = {e["category"] for e in ledger["entries"]}
    assert categories <= cfp.VALID_CATEGORIES


# ------------------------------------------------------------------ live gate

def test_repo_tree_passes_provenance_check():
    """The tracked tree must satisfy the ledger — same gate CI runs."""
    result = subprocess.run(
        [str(SCRIPT), "--tree"], cwd=REPO, capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"fixture-provenance --tree failed:\n{result.stdout}{result.stderr}"
    )
