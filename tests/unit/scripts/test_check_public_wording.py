"""scripts/check-public-wording.py: what it flags, what it lets through, and its modes."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check-public-wording.py"
spec = importlib.util.spec_from_file_location("check_public_wording", _SCRIPT)
cpw = importlib.util.module_from_spec(spec)
sys.modules["check_public_wording"] = cpw
spec.loader.exec_module(cpw)

CEILING = 1859


def rules(text: str, path: str | None = None) -> set[str]:
    return {f.rule for f in cpw.scan_line(path or "x.py", 1, text, CEILING, path)}


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("data lives in /home/alice/projects/run", "home-path"),
        (r"C:\Users\alice\data", "home-path"),
        ('DB = "/mnt/share/sherloc/phase.db"', "store-path"),
        ('ROOT = "/data/group/sherloc/data"', "store-path"),
        ("see https://viewer.build-box.corp/scan", "host"),
        ("contact ops@internal-lab.net", "host"),
        ("ssh to 100.101.102.103 first", "ip-address"),
        ("server at 52.14.200.7", "ip-address"),
        ("fixed in review round 3", "review-label"),
        ("Review r1 F2: keys never resolve", "review-label"),
        ("r2 NO-GO on the head", "review-label"),
        ("# (connector P2.1)", "review-label"),
        ('(Dana 2026-01-05: "keep it")', "ruling"),
        ("threshold 25 (Dana 1/5: fwhm 25)", "ruling"),
        ("the domain owner's ruling", "ruling"),
        ("reserved for the operator's go", "ruling"),
        ("awaiting ratification", "ruling"),
        ("Dana ruled 'B' on the call", "ruling"),
        ("see WS-1 section 4", "process"),
        ("(fix; surface: config.yaml; no deploy)", "process"),
        ("PM-approved footer", "process"),
        ("parked this PR at the boundary", "process"),
        ("the change must not merge", "process"),
        ("use for the canary rollout", "process"),
        ("class TestCodexFindings:", "tool-attribution"),
        ("reviewed by gpt-5.5", "tool-attribution"),
        ("Generated with a coding assistant", "tool-attribution"),
        ("Session: https://claude.ai/code/session_x", "tool-attribution"),
        ("process sol 1900 next", "sol-ceiling"),
        ("data/loupe/sol_2100/survey", "sol-ceiling"),
        ("SC3_1900_0000000000_000ECM_N0000000SRLC00000_0000LMJ01", "sol-ceiling"),
    ],
)
def test_flags(text, rule):
    assert rule in rules(text)


@pytest.mark.parametrize(
    "text",
    [
        "a second peak was ruled out by the F-test",
        "this does not rule out a carbonate",
        "from sherloc_pipeline.services import handoff",
        "the cheap CI lane filters it out",
        "the coroutine stays parked on its waiter",
        "Exported from OPUS 8.5",
        "--color-surface: #ffffff;",
        "a contract surface: a misconfigured container",
        "Step 1/7: Preprocessing",
        "Round 1: veto off, the CR is accepted",
        "file generated without error",
        "use $HOME/data or ~/data, or /home/<user>/data",
        "see https://pds-geosciences.wustl.edu/missions/mars2020/sherloc.htm",
        "point a browser at http://127.0.0.1:8000",
        "base_url='http://testserver'",
        "trusted_hosts='10.99.0.1'",
        "X-Forwarded-For: 192.0.2.1",
        "version 1.11.1.0 of the model",
        "spec section 3.9.8.3",
        "e.g. your-team.cloudflareaccess.com",
        "request.app.state.config",
        "sol 1853 line_1 has 25 points",
        'sol_dir = tmp_path / "sol_9999"',
        "ss__9999_0000000000_000rrs__synthetic",
    ],
)
def test_passes(text):
    assert rules(text) == set()


def test_deploy_files_may_name_container_paths():
    assert rules("VOLUME /data/sherloc/store/x", "Dockerfile") == set()
    assert rules("VOLUME /data/sherloc/store/x", "src/x.py") == {"store-path"}


def test_lockfiles_skip_the_host_rule():
    assert rules("resolved https://registry.yarnpkg.example-mirror.net/x", "web/package-lock.json") == set()


def _repo(tmp_path: Path, max_sol: int = CEILING) -> Path:
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "dev@example.com")
    git("config", "user.name", "Dev")
    ledger = tmp_path / "tests" / "fixtures" / "PROVENANCE.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"pds_envelope": {"sherloc": {"max_sol": max_sol}}}))
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "chore: start")
    git("checkout", "-q", "-b", "topic")
    return tmp_path


def _run(cwd: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_SCRIPT), *args], cwd=cwd, capture_output=True,
                          text=True, env=env)


def test_branch_mode_reads_added_lines_and_messages(tmp_path):
    repo = _repo(tmp_path)
    (repo / "a.py").write_text("x = 1\npath = '/home/alice/x'\n")
    subprocess.run(["git", "commit", "-qam", "fix: thing (review round 2)"], cwd=repo, check=True)
    r = _run(repo, "--base", "main")
    assert r.returncode == 1
    assert "a.py:2: [home-path]" in r.stdout
    assert "[review-label]" in r.stdout and "commit " in r.stdout


def test_branch_mode_ignores_lines_it_does_not_add(tmp_path):
    repo = _repo(tmp_path)
    (repo / "old.py").write_text("p = '/home/alice/x'\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "chore: old"], cwd=repo, check=True)
    subprocess.run(["git", "branch", "-qf", "main"], cwd=repo, check=True)
    (repo / "a.py").write_text("x = 2\n")
    subprocess.run(["git", "commit", "-qam", "fix: change x"], cwd=repo, check=True)
    assert _run(repo, "--base", "main").returncode == 0


def test_pr_text_from_environment(tmp_path):
    repo = _repo(tmp_path)
    env = {"PATH": "/usr/bin:/bin", "PR_TITLE": "fix: x", "PR_BODY": "Codex r2 GO on the head"}
    r = _run(repo, "--base", "main", env=env)
    assert r.returncode == 1 and "PR body:1:" in r.stdout


def test_work_state_file_is_flagged(tmp_path):
    repo = _repo(tmp_path)
    (repo / "WORK_STATE.yaml").write_text("next_action: x\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "chore: add"], cwd=repo, check=True)
    assert "[work-state]" in _run(repo, "--base", "main").stdout


def test_commit_msg_mode_skips_comment_lines(tmp_path):
    repo = _repo(tmp_path)
    msg = tmp_path / "MSG"
    msg.write_text("fix: x\n# Please enter the commit message (surface: ignored)\n")
    assert _run(repo, "--commit-msg", str(msg)).returncode == 0
    msg.write_text("fix: x\n\nClaude-Session: link\n")
    assert _run(repo, "--commit-msg", str(msg)).returncode == 1


def test_ceiling_comes_from_the_ledger(tmp_path):
    repo = _repo(tmp_path, max_sol=2000)
    msg = tmp_path / "MSG"
    msg.write_text("docs: sol 1950 example\n")
    assert _run(repo, "--commit-msg", str(msg)).returncode == 0
    msg.write_text("docs: sol 2050 example\n")
    assert _run(repo, "--commit-msg", str(msg)).returncode == 1


def test_allow_file_exempts_one_line_and_needs_a_reason(tmp_path):
    repo = _repo(tmp_path)
    (repo / "a.py").write_text("x = 1\nroot = '/mnt/share/x'\n")
    subprocess.run(["git", "commit", "-qam", "feat: mount"], cwd=repo, check=True)
    allow = repo / ".github" / "public-wording-allow.txt"
    allow.parent.mkdir()
    allow.write_text("a.py | store-path | /mnt/share/x | documented example mount\n")
    assert _run(repo, "--base", "main").returncode == 0
    allow.write_text("a.py | store-path | /mnt/share/x |\n")
    assert _run(repo, "--base", "main").returncode == 2


def test_missing_ledger_is_a_usage_error(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tests" / "fixtures" / "PROVENANCE.json").write_text("{}")
    assert _run(repo, "--tree").returncode == 2
