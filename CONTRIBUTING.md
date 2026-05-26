# Contributing to SHERLOC Pipeline

Thanks for your interest in contributing. The sections below cover the ground
rules for working in this repository.

## Quick start

```bash
git clone https://github.com/archaeon-ai/sherloc-pipeline.git
cd sherloc-pipeline
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pre-commit install
pytest -m "not slow"
```

`pre-commit install` is required — it activates the local hooks described under
[Public-repo discipline](#public-repo-discipline) below.

## Issues and pull requests

- Bug reports and feature requests: use the issue templates under
  `.github/ISSUE_TEMPLATE/`.
- Security-sensitive reports: see [`SECURITY.md`](SECURITY.md).
- Pull requests: include a concise summary, a test plan, and link any related
  issue. CI must pass before review.

## Conventional commits

Commit messages follow `<type>(<scope>): <subject>`, where `<type>` is one of
`feat`, `fix`, `docs`, `refactor`, `test`, `chore`, or `spec`. Keep each commit
to one logical change.

## Public-repo discipline

This is a public repository. Tracked content must not embed operator-local
infrastructure details. The same code base also runs in private operator
environments with idiosyncratic paths and hostnames; the convention is to
reach those via environment variables rather than hardcoding them.

### Forbidden patterns and substitutions

| Forbidden pattern | Use instead |
|-------------------|-------------|
| Absolute data paths (e.g. local `data` mounts) | `SHERLOC_DATA_ROOT` env var, or repo-relative paths |
| Absolute NAS paths (e.g. shared-storage mounts) | `SHERLOC_NAS_ROOT` env var, or repo-relative paths |
| Absolute home paths (e.g. `/home/<user>`) | `$HOME` or `~` |
| Operator hostnames | A generic placeholder such as `devhost`, or env-driven |
| Operator-owned domains | `example.com` or env-driven hostnames |
| Internal agent codenames | Do not appear in tracked content |

The exact patterns enforced live in `scripts/check-forbidden-strings.sh` so
that script and policy stay in lockstep.

### Enforcement

Two checks back this rule:

1. **Pre-commit hook** — `.pre-commit-config.yaml` registers a local hook that
   runs the check against staged files. Activate once per clone with
   `pre-commit install`.
2. **CI workflow** — `.github/workflows/ci.yml` runs the check across every
   tracked file on each push and pull request.

You can run either mode manually:

```bash
scripts/check-forbidden-strings.sh --staged   # files staged for commit
scripts/check-forbidden-strings.sh --tree     # the entire tracked tree
```

If a check fails, the script prints the offending pattern and file. Replace
the value with the substitution from the table above (or with an env-driven
configuration entry) before re-staging.

### Rationale

Conflating an operator's local environment with the public package makes the
repository harder to clone, fork, and reproduce. Env-driven configuration
keeps the same code path working both in the operator's private workspace
and in clean public deployments without per-environment patches.

## Coding standards

- Python 3.12+. The project targets `ruff` defaults; run `ruff check .` before
  pushing.
- Add tests under `tests/` for new behavior. The `not slow` selector should
  pass in under ~10 minutes; the full suite (including the regression golden)
  takes ~25 minutes and runs in CI on demand.
- Keep public-API docstrings short and informative; long-form explanations
  belong in `docs/`.

### System invariants

[`docs/INVARIANTS.md`](docs/INVARIANTS.md) records system-wide constraints
that hold across changes — service return patterns, error hierarchy, CLI
surface stability, spectral calibration rules, and so on. Read it before
modifying core paths, and update it in the same commit when an invariant is
intentionally extended or refined.

## Where to look

- Architecture overview: [`docs/architecture.md`](docs/architecture.md)
- Spectral calibration and region definitions: [`docs/schema/SPECTRAL_REGIONS.md`](docs/schema/SPECTRAL_REGIONS.md)
- Active specs: `docs/specs/`
- Archived spec history: `docs/archive/`
