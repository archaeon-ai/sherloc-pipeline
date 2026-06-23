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

### Enforcement

Review changed content against the table above before committing. If you
introduce an absolute path, internal hostname, or other operator-local string,
replace it with the substitution from the table (or with an env-driven
configuration entry) before staging.

### Rationale

Conflating an operator's local environment with the public package makes the
repository harder to clone, fork, and reproduce. Env-driven configuration
keeps the same code path working both in the operator's private workspace
and in clean public deployments without per-environment patches.

### Pre-publication and internal content

Keep pre-publication science and internal engineering process out of the
public tree:

- **Pre-publication findings** — conclusion language (feasibility verdicts,
  organics/carbonate determinations) and the mission-target names tied to
  unpublished work belong in the gitignored local working tree until they are
  published.
- **Internal engineering artifacts** — verification reports, review/revision
  spec drafts, and deploy-pipeline material belong in the gitignored
  internal-docs area, not the public tree. Keep one canonical,
  public-appropriate document per topic under `docs`.
- **Internal-only directory references and development-tool attribution** —
  references to the gitignored internal directories should not appear in
  tracked content other than the ignore-config files that exist to name them.

### Fixture provenance

This is a public repository, so every tracked **data file** (spectra, images,
fixtures, golden outputs — anything matching the data extensions listed in
`scripts/check-fixture-provenance.py`) must contain only publicly releasable
values. Concretely, a data file may enter git only if it is one of:

| Category | Meaning |
|----------|---------|
| `pds-archived` | Mars 2020 instrument data whose sol is within the instrument's archived PDS release envelope |
| `raw-images-feed` | A product published on the public Mars 2020 raw-images site (public on receipt) |
| `synthetic` | Generated/simulated values; any sol or target in the file name is nominal |
| `reference-standard` | Earth-laboratory reference or calibration data (not Mars data) |
| `operator-derived-archived` | An operator-produced derivation whose inputs are all archived/public |
| `metadata` | A value-free structural/config file (manifests, lockfiles) |

The ledger at `tests/fixtures/PROVENANCE.json` records the category for every
tracked data file (directory entries end with `/` and cover everything under
them). The check is **fail-closed**: a tracked data file with no ledger entry
is a violation. High-confidence sol tokens embedded in paths (`sol_NNNN`
directories, PDS product IDs, `NNNN_` file prefixes) are cross-checked
against the ledger entry, so a mislabeled fixture fails loudly.

**Adding a fixture:** add the file(s) plus a ledger entry in the same commit.
For Mars 2020 data, confirm the sol is within the instrument's archived PDS
release before adding it; if a new PDS release extends the envelope, bump
`pds_envelope.<instrument>` in the ledger (release id, date, `max_sol`) in
the same change-set.

Enforcement is a pre-commit hook plus a CI step, with a third gate in
`tests/architecture/test_fixture_provenance.py` so a plain `pytest` run also
catches violations:

```bash
scripts/check-fixture-provenance.py --staged   # files staged for commit
scripts/check-fixture-provenance.py --tree     # the entire tracked tree
```

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
- Scientific methods: [`docs/METHODS.md`](docs/METHODS.md)
- System invariants: [`docs/INVARIANTS.md`](docs/INVARIANTS.md)
