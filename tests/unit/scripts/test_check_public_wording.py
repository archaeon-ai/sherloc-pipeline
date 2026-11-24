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
        ('(Dana 2026-01-05: "keep it")', "decision-note"),
        ("threshold 25 (Dana 1/5: fwhm 25)", "decision-note"),
        ("the domain owner's ruling", "decision-note"),
        ("reserved for the operator's go", "decision-note"),
        ("awaiting ratification", "decision-note"),
        ("Dana ruled 'B' on the call", "decision-note"),
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
        ("ss__1900_0000000000_000rrs__x", "sol-ceiling"),
        ('tmp_path / "1934_Target_line_R1_avg-4p_fit.png"', "sol-ceiling"),
        ("new observation at sol 2860", "sol-ceiling"),
        ("build.internal", "host"),
        ("build.secret.corp/docs", "host"),
        ("https://acme-lab.cloudflareaccess.com/admin", "host"),
        ("https://mycompany.example-bad.net/x", "host"),
        ("2001:4860:4860::8888", "ipv6-address"),
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
        "issuer https://tenant.us.auth0.com/",
        "bucket at https://account.r2.cloudflarestorage.com",
        "request.app.state.config",
        "sol 1853 line_1 has 25 points",
        'sol_dir = tmp_path / "sol_9999"',
        "ss__9999_0000000000_000rrs__synthetic",
        "run_2026_11_23.log",
        "1234_",
        "ports 8000:8000 and 3000:3000",
        "the documentation address 2001:db8::1",
        "see CLAUDE.local.md",
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


def test_an_allowed_match_does_not_hide_another_on_the_same_line():
    allow = cpw.Allow("*", "host", "ok-site.org", "test")
    found = cpw.scan_line("x.md", 1, "mail a@ok-site.org or https://build.secret.corp/", CEILING, "x.md")
    kept = [f for f in found if not allow.covers(f)]
    assert [f.match for f in kept] == ["https://build.secret.corp"]


def test_an_added_line_that_looks_like_a_diff_header_is_still_read(tmp_path):
    repo = _repo(tmp_path)
    (repo / "a.py").write_text("x = 1\n++ /home/alice/one\npath = '/home/alice/two'\n")
    subprocess.run(["git", "commit", "-qam", "fix: lines"], cwd=repo, check=True)
    out = _run(repo, "--base", "main").stdout
    assert "a.py:2: [home-path]" in out and "a.py:3: [home-path]" in out


def test_messages_mode_reads_one_commit_even_the_root(tmp_path):
    repo = _repo(tmp_path)
    root = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()
    assert _run(repo, "--messages", root).returncode == 0
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "fix: x (review round 2)"],
                   cwd=repo, check=True)
    assert _run(repo, "--messages", "HEAD").returncode == 1
    assert _run(repo, "--messages", "main..HEAD").returncode == 1


def test_commits_file_mode_reads_each_listed_commit(tmp_path):
    repo = _repo(tmp_path)
    for msg in ("fix: clean one", "fix: x (review round 2)", "fix: clean two"):
        subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", msg], cwd=repo, check=True)
    shas = subprocess.run(["git", "rev-list", "main..HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout
    listed = tmp_path / "list"
    listed.write_text(shas)
    r = _run(repo, "--commits-file", str(listed))
    assert r.returncode == 1 and r.stdout.count("[review-label]") == 1
    listed.write_text(shas.splitlines()[0] + "\n")
    assert _run(repo, "--commits-file", str(listed)).returncode == 0


@pytest.mark.parametrize("text", ["see build.local now", "cache.home.arpa"])
def test_more_internal_suffixes(text):
    assert "host" in rules(text)


@pytest.mark.parametrize("text", ["https://demo.secret.corp/admin", "https://sample.build.internal/logs",
                                  "see test.lab.corp", "mail ops@demo.acme.tech"])
def test_placeholder_labels_do_not_pass_private_hosts(text):
    assert "host" in rules(text)


@pytest.mark.parametrize("text", ["https://your-team.workers.dev/x", "https://tenant.cloudflareaccess.com",
                                  "https://demo.example.com/x"])
def test_placeholder_labels_pass_hosting_service_examples(text):
    assert "host" not in rules(text)


@pytest.mark.parametrize("text", ["cache.private", "build.acme.tech", "see files.home now"])
def test_bare_hosts_with_private_and_public_endings(text):
    assert "host" in rules(text)


@pytest.mark.parametrize("text", ["return self.app", "info = user.info", "load model.xyz"])
def test_bare_names_with_other_endings_are_not_hosts(text):
    assert "host" not in rules(text)


def test_an_allow_entry_for_a_file_covers_only_the_text_it_names(tmp_path):
    repo = _repo(tmp_path)
    (repo / "a.py").write_text("x = 1\n"
                               "p = '/home/alice/one'\n"
                               "q = '/home/alice/two'\n")
    subprocess.run(["git", "commit", "-qam", "test: inputs"], cwd=repo, check=True)
    allow = repo / ".github" / "public-wording-allow.txt"
    allow.parent.mkdir()
    allow.write_text("a.py | home-path | /home/alice/one | test input\n")
    out = _run(repo, "--base", "main").stdout
    assert "a.py:3: [home-path]" in out and "a.py:2:" not in out


def test_the_checker_files_hold_only_the_matches_the_allow_file_names():
    root = _SCRIPT.parents[1]
    r = subprocess.run([sys.executable, str(_SCRIPT), "--tree"], cwd=root, capture_output=True, text=True)
    own = [l for l in r.stdout.splitlines()
           if l.startswith(("scripts/check-public-wording.py:", "tests/unit/scripts/test_check_public_wording.py:"))]
    assert own == []


_LIST = _SCRIPT.parent / "list-pushed-commits.sh"
_ZERO = "0" * 40


def _pushed(tmp_path: Path):
    """An origin with main, and a clone to run the push listing in."""
    origin = tmp_path / "origin.git"
    (tmp_path / "work").mkdir()
    work = _repo(tmp_path / "work")
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=work, check=True, capture_output=True,  # noqa: E731
                                    text=True).stdout.strip()
    git("remote", "add", "origin", str(origin))
    git("push", "-q", "origin", "main")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    return work, clone, git


def _list(clone: Path, before: str, after: str, out: Path) -> subprocess.CompletedProcess:
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=clone, check=True)
    return subprocess.run(["bash", str(_LIST), before, after, str(out)], cwd=clone,
                          capture_output=True, text=True)


def test_push_listing_normal_push(tmp_path):
    work, clone, git = _pushed(tmp_path)
    before = git("rev-parse", "main")
    git("checkout", "-q", "main")
    for m in ("fix: one", "fix: two"):
        git("commit", "-q", "--allow-empty", "-m", m)
    git("push", "-q", "origin", "main")
    out = tmp_path / "list"
    assert _list(clone, before, git("rev-parse", "main"), out).returncode == 0
    assert out.read_text().split() == git("rev-list", f"{before}..main").split()


def test_push_listing_new_branch_lists_only_commits_on_no_other_branch(tmp_path):
    work, clone, git = _pushed(tmp_path)
    git("checkout", "-q", "-b", "other", "main")
    git("commit", "-q", "--allow-empty", "-m", "fix: on other")
    git("push", "-q", "origin", "other")
    git("checkout", "-q", "-b", "fresh", "main")
    git("commit", "-q", "--allow-empty", "-m", "fix: new")
    subprocess.run(["git", "symbolic-ref", "HEAD", "refs/heads/other"], cwd=tmp_path / "origin.git",
                   check=True)
    git("push", "-q", "origin", "--delete", "main")
    git("push", "-q", "origin", "fresh:main")  # main created anew: BEFORE is all zeros
    out = tmp_path / "list"
    assert _list(clone, _ZERO, git("rev-parse", "fresh"), out).returncode == 0
    assert out.read_text().split() == [git("rev-parse", "fresh")]


def test_push_listing_forced_push_fetches_the_old_tip(tmp_path):
    work, clone, git = _pushed(tmp_path)
    git("checkout", "-q", "main")
    git("commit", "-q", "--allow-empty", "-m", "fix: old tip")
    old = git("rev-parse", "main")
    git("push", "-q", "origin", "main")
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=clone, check=True)
    base = git("rev-parse", "main~1")
    git("reset", "-q", "--hard", base)
    git("commit", "-q", "--allow-empty", "-m", "fix: rewritten")
    git("push", "-q", "-f", "origin", "main")
    git("push", "-q", "origin", f"{old}:refs/heads/keep")  # the old tip stays fetchable by id
    fresh = tmp_path / "fresh"
    subprocess.run(["git", "clone", "-q", "--single-branch", str(tmp_path / "origin.git"), str(fresh)],
                   check=True)
    out = tmp_path / "list"
    r = _list(fresh, old, git("rev-parse", "main"), out)
    assert r.returncode == 0, r.stderr
    assert out.read_text().split() == [git("rev-parse", "main")]


def test_push_listing_fails_when_the_old_tip_cannot_be_fetched(tmp_path):
    work, clone, git = _pushed(tmp_path)
    out = tmp_path / "list"
    r = _list(clone, "1" * 40, git("rev-parse", "main"), out)
    assert r.returncode == 1 and "could not be fetched" in r.stderr


@pytest.mark.parametrize("text", ["the rulings on fits", "Ruling: keep it",
                                  "def test_config_carries_the_rulings():", "see the rulings."])
def test_ruling_words_are_flagged(text):
    assert "decision-note" in rules(text)


def test_a_file_name_with_ruling_in_it_is_not_a_decision_note():
    assert "decision-note" not in rules("`tests/unit/core/test_fitting_rulings.py` is renamed")


@pytest.mark.parametrize("text", ['sol_dir = tmp_path / "sol_3000"', "sol 8000", "sol 42424", "sol 9999"])
def test_invented_far_future_sols_pass(text):
    assert "sol-ceiling" not in rules(text)


@pytest.mark.parametrize("text", ["sol 1860", "sol 2999"])
def test_sols_between_the_archive_and_the_invented_range_are_flagged(text):
    assert "sol-ceiling" in rules(text)


def test_a_dated_measurement_note_is_not_a_decision_note():
    assert rules("measured on the 190 release-16 P-flag rrs/rcs labels, 2026-11-22,") == set()


@pytest.mark.parametrize("max_sol", [2500, 2999, 3000, 3500])
def test_a_ceiling_near_the_invented_range_is_a_usage_error(tmp_path, max_sol):
    repo = _repo(tmp_path, max_sol=max_sol)
    r = _run(repo, "--tree")
    assert r.returncode == 2 and "SOL_PLACEHOLDER_MIN" in r.stderr


def test_a_ceiling_below_the_margin_still_runs(tmp_path):
    assert _run(_repo(tmp_path, max_sol=2499), "--tree").returncode == 0
