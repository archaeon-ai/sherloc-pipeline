#!/usr/bin/env python3
"""Check that text entering this public repository is written for outside readers.

This repository is public. Code, docs, commit messages and pull request text
should not carry local machine details (home or data-store paths, private host
names), notes about how a change was reviewed or decided, tool attribution, or
references to observations newer than the archived PDS release. This script
finds those patterns and reports each one with its file, line and reason.

Usage:
  scripts/check-public-wording.py                  # this branch vs origin/main
  scripts/check-public-wording.py --base <ref>     # this branch vs <ref>
  scripts/check-public-wording.py --tree           # every tracked file
  scripts/check-public-wording.py --title T --body-file F   # also PR text
  scripts/check-public-wording.py --commit-msg FILE     # one commit message
  scripts/check-public-wording.py --messages A..B       # commit messages only
  scripts/check-public-wording.py --commits-file F      # messages of the commits listed in F

In a pull request workflow the title and body come from the PR_TITLE and
PR_BODY environment variables. Branch mode reads the lines the branch adds
(not the lines it keeps or deletes), the branch's commit messages and any PR
text given.

A justified exception is one line in .github/public-wording-allow.txt, so it
shows in the diff that adds it. Exits 0 when clean, 1 on findings, 2 on usage
or allow-file errors. Stdlib only, so it runs without a virtualenv. See
CONTRIBUTING.md "Public-repo discipline".
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LEDGER_PATH = "tests/fixtures/PROVENANCE.json"
ALLOW_PATH = ".github/public-wording-allow.txt"
# Sols at or above this value are invented examples (sol 9999 is the usual one;
# tests also use round sols such as 3000 to 8000 and 42424). The rover is far
# below sol 3000, so every sol past the archived envelope and below it is
# flagged. The check stops with an error once the archived envelope comes
# within SOL_PLACEHOLDER_MARGIN of this value, so it cannot go quiet on new sols.
SOL_PLACEHOLDER_MIN = 3000
SOL_PLACEHOLDER_MARGIN = 500

# Public sites a document may link to. A host is allowed when it equals one of
# these or ends with "." plus one of them. Anything else, including a service
# subdomain set up for one deployment, should be an example domain or come
# from configuration.
PUBLIC_HOSTS = (
    "example.com", "example.org", "example.net", "example", "invalid", "test",
    "localhost",
    "github.com", "githubusercontent.com", "github.io", "ghcr.io",
    "nasa.gov", "wustl.edu", "doi.org", "zenodo.org", "arxiv.org",
    "python.org", "pypi.org", "pythonhosted.org", "readthedocs.io",
    "readthedocs.org", "npmjs.com", "npmjs.org", "nodejs.org",
    "docker.com", "docker.io", "astral.sh", "shields.io", "w3.org",
    "mozilla.org", "svelte.dev", "vitejs.dev", "fastapi.tiangolo.com",
    "sqlalchemy.org", "scipy.org", "numpy.org", "matplotlib.org",
    "pydantic.dev", "pytest.org", "auth0.com", "keepachangelog.com",
    "semver.org", "conventionalcommits.org", "spdx.org", "opensource.org",
    "creativecommons.org", "schema.org", "oclc.org", "apache.org",
    "gstatic.com", "googleapis.com", "cloudflare.com", "jsdelivr.net", "unpkg.com", "mit.edu", "rapids.ai",
)

# Hosting services whose subdomains belong to one deployment: any subdomain is
# reported (placeholders such as your-team.<service> pass).
TENANT_SUFFIXES = (
    "cloudflareaccess.com", "cloudflarestorage.com", "r2.dev", "workers.dev",
    "pages.dev", "auth0.com", "amazonaws.com", "azurewebsites.net",
    "herokuapp.com", "vercel.app", "netlify.app", "fly.dev", "ngrok.io",
    "ngrok-free.app", "ts.net",
)

# A host name with no scheme is recognized only when it ends in one of these:
# public endings that rarely end a code identifier, and endings used inside
# private networks. Any other ending (".app", ".info", ".site") is common in
# code such as self.app, so a bare name there is not checked; a URL or an
# email address is checked whatever its ending.
BARE_HOST_SUFFIXES = (
    "com", "net", "org", "io", "edu", "gov", "dev", "ai", "tech", "cloud",
    "online", "biz",
    "internal", "corp", "lan", "intranet", "localdomain", "local", "arpa",
    "private", "home", "lab",
)

# A first label that marks a hosting-service subdomain (TENANT_SUFFIXES) as an
# invented example, such as your-team.workers.dev. It applies to those
# services only: demo.<private host> is reported like any other private host.
PLACEHOLDER_LABEL = re.compile(
    r"your[\w-]*|tenant|team|account|acct|test|test-[\w-]+|example|invented|change-me"
    r"|placeholder|sample|demo|attacker[\w-]*")

# Files where container and deployment paths such as /data are the subject.
DEPLOY_GLOBS = ("Dockerfile*", "*/Dockerfile*", "docker-compose*", "compose*.y*ml",
                "deploy/*", "docker-entrypoint.sh", ".dockerignore")
# Generated dependency locks list registry hosts by the hundred.
LOCK_GLOBS = ("*.lock", "*-lock.json", "*-lock.yaml", "*lock.txt", "uv.lock")


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    reason: str


def _r(pattern: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(pattern, flags)


_DATE = r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?)"
_MONTHS_DAYS = (r"(?!(?:January|February|March|April|May|June|July|August|September|"
                r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|"
                r"Nov|Dec|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
                r"Mon|Tue|Wed|Thu|Fri|Sat|Sun|Released?|Release|Updated|Sol|Version|Step|Stage|Phase|"
                r"Page|Part|Pass|Run|Line|Item)\b)")

RULES: tuple[Rule, ...] = (
    Rule("home-path",
         _r(r"(?<![\w$.])(?:/home|/Users)/(?!<|\$|\{|user/|username/|runner/|you/)[A-Za-z0-9._-]+[\w./-]*"
            r"|\b[A-Z]:\\\\?Users\\\\?(?!<|%)[A-Za-z0-9._-]+"),
         "home-directory path: use $HOME, ~ or a repository-relative path"),
    Rule("store-path",
         _r(r"(?<![\w.~$/:-])/(?:nas|mnt|work|volume\d*)/[\w.$-][\w.$/-]*"
            r"|(?<![\w.~$/:-])/(?:data|srv)/[\w.$-]+/[\w.$-]+/[\w.$/-]*"),
         "absolute data-store path: use a configured root (environment variable) or a repository-relative path"),
    Rule("host",
         _r(r"(?i)\b[a-z][a-z0-9+.-]*://(?:[^@/\s]+@)?([a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)"
            r"|\b[\w.+-]+@((?:[a-z0-9-]+\.)+[a-z]{2,})\b"
            r"|(?<![\w@./-])((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+"
            r"(?:" + "|".join(BARE_HOST_SUFFIXES) + r"))\b(?![\w(-])(?!\.\w)"),
         "host name that is not a public site: use an example domain (example.com) or configuration"),
    Rule("ip-address",
         _r(r"(?<![\w.\u00a7])(?!127\.|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|192\.0\.2\.|198\.51\.100\.|203\.0\.113\.)"
            r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]\d)\.(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){2}"
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w.])"),
         "public or shared IP address: use localhost, a documentation address (192.0.2.x) or configuration"),
    Rule("ipv6-address",
         _r(r"(?i)(?<![\w:.])(?!2001:db8:)[23][0-9a-f]{3}:(?:[0-9a-f]{0,4}:){2,6}[0-9a-f]{0,4}(?![\w:])"),
         "public IPv6 address: use a documentation address (2001:db8::) or configuration"),
    Rule("review-label",
         _r(r"\b(?:[Rr]eview|[Rr]eviewer|Codex)\s+round\s*\d+|\b[Rr]ound[ -]?\d+\s+(?:F\d+|findings?|review|verdict)\b"
            r"|\b[Rr]\d+\s+(?:NO-GO|GO)\b|\bNO-GO\b|\bCONDITIONAL[_ ]GO\b"
            r"|\b(?:[Rr]eview|[Rr]eviewer|[Cc]onnector)\s+(?:[Rr]\d+\s+)?[FP]\d+(?:\.\d+)?\b"
            r"|\b[Rr]\d+\s+F\d+\b|\bPRIOR_RESOLVED\b|\b[Pp]rior round\b|\bexact head\b"),
         "review-round or finding label: describe the behavior, not how it was reviewed"),
    Rule("decision-note",
         _r(r"\b[Pp]er\s+(?:the\s+)?(?:operator|PM|domain owner|science lead)\b"
            r"|\b[Dd]omain[ -]owner\b|\b[Ss]cience lead\b"
            r"|\b[Oo]perator'?s\s+(?:go|word|ruling|ratification|approval|decision|sign-off)\b"
            r"|\b[Rr]atif(?:y|ied|ication)\b|\b[Rr]uled\b(?!\s+out)|(?<![A-Za-z])[Rr]ulings?(?![A-Za-z])"
            r"(?!\.(?:py|md|txt|ya?ml|json|toml)\b)"
            r"|\(\s*" + _MONTHS_DAYS + r"[A-Z][a-z]+,?\s+" + _DATE + r"\b"
            r"|\b" + _MONTHS_DAYS + r"[A-Z][a-z]+,?\s+(?:on\s+)?" + _DATE
            + r"(?:\s+at\s+\d{1,2}:\d{2})?\s*:?\s*[\"“]"),
         "a person's decision or its date: state the rule and its reason instead"),
    Rule("process",
         _r(r"\bW[PS]-\d+[a-z]?\b|(?:^|[;(,]\s*)[Ss]urface:\s|\b[Nn]o deploy\b|\bPM-approved\b"
            r"|\b[Pp]arked (?:this|the|at)\b|\b[Bb]urndown\b|\bmust not merge\b|\bcanary rollout\b"),
         "internal project-management wording: write for a reader outside the project"),
    Rule("tool-attribution",
         _r(r"Claude|Codex|\bcodex\b|Anthropic|OpenAI|\bGPT\b|\bGPT-?\d|\bgpt-\d|\bOpus\b|\bSonnet\b"
            r"|\bGemini\b|claude\.ai|\bGenerated with\b|\U0001F916"),
         "AI tool name or attribution: leave tooling out of code, docs and history"),
    Rule("sol-ceiling",
         _r(r"(?i)(?<![A-Za-z])sols?[ _-]?0*(\d{1,5})(?!\d)"
            r"|(?:^|(?<=[\s/\\\"'`(=,]))(\d{4})_(?=[A-Za-z])"
            r"|\b[A-Z]{2,4}\d?_{1,2}(\d{4})_\d{9,10}"
            r"|\b[a-z]{2,4}\d?_{1,2}(\d{4})_\d{9,10}"),
         "sol past the newest archived PDS release (pds_envelope in tests/fixtures/PROVENANCE.json; "
         "invented examples use sol 9999)"),
)

WORK_STATE_RULE = Rule(
    "work-state", _r(r"^(?:.*/)?WORK_STATE[^/]*$|^\.claude/|^_scratch/|^\.codex/"),
    "work-tracking file or folder: keep it outside the repository")


@dataclass(frozen=True)
class Finding:
    where: str      # file path, "PR title", "PR body" or "commit <sha>"
    line: int
    rule: str
    reason: str
    text: str
    match: str = ""  # the text the rule matched; allow entries are compared with it

    def render(self) -> str:
        snippet = self.text.strip()
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        return f"{self.where}:{self.line}: [{self.rule}] {self.reason}\n    {snippet}"


@dataclass(frozen=True)
class Allow:
    glob: str
    rule: str
    text: str
    reason: str

    def covers(self, f: Finding) -> bool:
        return (fnmatch.fnmatchcase(f.where, self.glob)
                and self.rule in ("*", f.rule)
                and (self.text == "*" or self.text in (f.match or f.text)))


def max_archived_sol(root: Path) -> int | None:
    try:
        ledger = json.loads((root / LEDGER_PATH).read_text(encoding="utf-8"))
        sols = [int(v["max_sol"]) for v in ledger.get("pds_envelope", {}).values()]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return max(sols) if sols else None


def load_allow(root: Path) -> list[Allow]:
    path = root / ALLOW_PATH
    if not path.exists():
        return []
    out: list[Allow] = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4 or not all(parts):
            raise ValueError(f"{ALLOW_PATH}:{n}: expected 'path glob | rule | text | reason'")
        known = {r.name for r in RULES} | {WORK_STATE_RULE.name, "*"}
        if parts[1] not in known:
            raise ValueError(f"{ALLOW_PATH}:{n}: unknown rule {parts[1]!r}")
        out.append(Allow(*parts))
    return out


def _host_ok(host: str) -> bool:
    host = host.lower().rstrip(".")
    if re.match(r"(?:<|\{|\$)", host):
        return True     # a template such as <host> or ${HOST}
    if re.fullmatch(r"[\d.]+", host):
        return True     # an address: the ip-address rule decides
    if "." not in host:
        return True     # a single-label test host such as http://testserver
    if any(host.endswith("." + h) for h in TENANT_SUFFIXES):
        # A deployment's own subdomain of a hosting service, unless its first
        # label is a placeholder such as your-team.<service> or tenant.<service>.
        return bool(PLACEHOLDER_LABEL.fullmatch(host.split(".")[0]))
    return any(host == h or host.endswith("." + h) for h in PUBLIC_HOSTS)


def _matches(glob_set: tuple[str, ...], path: str) -> bool:
    return any(fnmatch.fnmatchcase(path, g) for g in glob_set)


def scan_line(where: str, line_no: int, text: str, ceiling: int | None,
              path: str | None = None) -> list[Finding]:
    found: list[Finding] = []
    for rule in RULES:
        if rule.name == "store-path" and path and _matches(DEPLOY_GLOBS, path):
            continue
        if rule.name == "host" and path and _matches(LOCK_GLOBS, path):
            continue
        for m in rule.pattern.finditer(text):
            if rule.name == "host" and _host_ok(next(g for g in m.groups() if g)):
                continue
            if rule.name == "sol-ceiling":
                value = int(next(g for g in m.groups() if g))
                if ceiling is None or not ceiling < value < SOL_PLACEHOLDER_MIN:
                    continue
            found.append(Finding(where, line_no, rule.name, rule.reason, text, m.group(0)))
    return found


def scan_text(where: str, text: str, ceiling: int | None) -> list[Finding]:
    out: list[Finding] = []
    for n, line in enumerate(text.splitlines(), 1):
        out.extend(scan_line(where, n, line, ceiling))
    return out


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


def _decode_text(data: bytes) -> str | None:
    """Return the file as text, or None for binary or non-UTF-8 content."""
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_tree(root: Path, ceiling: int | None) -> list[Finding]:
    out: list[Finding] = []
    for path in _git(root, "ls-files", "-z").split("\0"):
        if not path:
            continue
        if WORK_STATE_RULE.pattern.search(path):
            out.append(Finding(path, 0, WORK_STATE_RULE.name, WORK_STATE_RULE.reason, path))
        try:
            data = (root / path).read_bytes()
        except OSError:
            continue
        text = _decode_text(data)
        if text is None:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            out.extend(scan_line(path, n, line, ceiling, path))
    return out


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def scan_branch(root: Path, base: str, ceiling: int | None) -> list[Finding]:
    out: list[Finding] = []
    for path in _git(root, "diff", "--name-only", "--diff-filter=AR", "-z",
                     f"{base}...HEAD").split("\0"):
        if path and WORK_STATE_RULE.pattern.search(path):
            out.append(Finding(path, 0, WORK_STATE_RULE.name, WORK_STATE_RULE.reason, path))
    diff = _git(root, "diff", "--no-color", "--unified=0", "--no-ext-diff",
                f"{base}...HEAD")
    path: str | None = None
    line_no = 0
    in_header = False
    for raw in diff.splitlines():
        if raw.startswith("diff --git "):
            in_header, path = True, None
            continue
        if in_header:
            # Header lines ("index", "---", "+++", mode lines) end at the first hunk.
            if raw.startswith("+++ "):
                target = raw[4:]
                path = (None if target == "/dev/null"
                        else target[2:] if target.startswith("b/") else target)
            m = _HUNK.match(raw)
            if m:
                in_header = False
                line_no = int(m.group(1))
            continue
        m = _HUNK.match(raw)
        if m:
            line_no = int(m.group(1))
            continue
        if path is None:
            continue
        if raw.startswith("+"):
            out.extend(scan_line(path, line_no, raw[1:], ceiling, path))
            line_no += 1
    out.extend(scan_messages(root, f"{base}..HEAD", ceiling))
    return out


def scan_messages(root: Path, rev_range: str, ceiling: int | None) -> list[Finding]:
    """Check commit messages in A..B, or the one commit named when there is no '..'."""
    out: list[Finding] = []
    sel = [rev_range] if ".." in rev_range else ["-1", rev_range]
    log = _git(root, "log", "--format=%H%x1f%B%x1e", *sel)
    for entry in log.split("\x1e"):
        entry = entry.strip("\n")
        if not entry:
            continue
        sha, _, body = entry.partition("\x1f")
        out.extend(scan_text(f"commit {sha[:12]}", body, ceiling))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--tree", action="store_true", help="scan every tracked file")
    mode.add_argument("--base", default="origin/main",
                      help="compare this branch with BASE (default origin/main)")
    mode.add_argument("--commit-msg", metavar="FILE",
                      help="check one commit message file (the pre-commit commit-msg hook)")
    mode.add_argument("--messages", metavar="RANGE",
                      help="check only the commit messages in RANGE (A..B), or one commit")
    mode.add_argument("--commits-file", metavar="FILE",
                      help="check only the messages of the commits listed in FILE, one id per line")
    ap.add_argument("--title", help="pull request title (default: $PR_TITLE)")
    ap.add_argument("--body-file", help="file holding the pull request body (default: $PR_BODY)")
    args = ap.parse_args(argv)

    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    try:
        allows = load_allow(root)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    ceiling = max_archived_sol(root)
    if ceiling is None:
        print(f"error: no pds_envelope max_sol in {LEDGER_PATH}", file=sys.stderr)
        return 2
    if ceiling >= SOL_PLACEHOLDER_MIN - SOL_PLACEHOLDER_MARGIN:
        print(f"error: archived max_sol {ceiling} is within {SOL_PLACEHOLDER_MARGIN} of "
              f"SOL_PLACEHOLDER_MIN ({SOL_PLACEHOLDER_MIN}); raise SOL_PLACEHOLDER_MIN and "
              f"move invented test sols above it", file=sys.stderr)
        return 2

    try:
        if args.tree:
            findings = scan_tree(root, ceiling)
        elif args.commit_msg:
            text = Path(args.commit_msg).read_text(encoding="utf-8")
            text = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
            findings = scan_text("commit message", text, ceiling)
        elif args.messages:
            findings = scan_messages(root, args.messages, ceiling)
        elif args.commits_file:
            findings = []
            for sha in Path(args.commits_file).read_text(encoding="utf-8").split():
                findings += scan_messages(root, sha, ceiling)
        else:
            findings = scan_branch(root, args.base, ceiling)
    except subprocess.CalledProcessError as exc:
        print(f"error: git failed: {exc.stderr.strip()}", file=sys.stderr)
        return 2

    title = args.title if args.title is not None else os.environ.get("PR_TITLE", "")
    body = (Path(args.body_file).read_text(encoding="utf-8") if args.body_file
            else os.environ.get("PR_BODY", ""))
    findings += scan_text("PR title", title, ceiling)
    findings += scan_text("PR body", body, ceiling)

    kept: list[Finding] = []
    seen: set[tuple[str, int, str]] = set()
    for f in findings:  # one report per line and rule, from matches no allow entry covers
        key = (f.where, f.line, f.rule)
        if key not in seen and not any(a.covers(f) for a in allows):
            seen.add(key)
            kept.append(f)
    for f in kept:
        print(f.render())
    if kept:
        print(f"\n{len(kept)} finding(s). This repository is public; see CONTRIBUTING.md "
              f"\"Public-repo discipline\". A justified exception goes in {ALLOW_PATH}.")
        return 1
    print("public wording: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
