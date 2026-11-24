# Contributing to SHERLOC Pipeline

From the checkout, install the test dependencies in a Python 3.12 virtual environment:

```bash
pip install -e '.[dev,web,pds,ml-despike]'
pre-commit install
```

Format locally with the Makefile target, then run the CI lint and test selection:

```bash
make format
ruff check .
pytest -m "not slow and not docker"
```

Push your branch and open a pull request against `main`. Include a summary,
the tests you ran, and a link to any related issue.

## Public-repo discipline

Tracked files, commit messages and pull requests must not contain local
absolute paths, private host names or domains. Use `SHERLOC_DATA_DIR`, a
command-line path option, `$HOME`, `example.com` or a repo-relative path
instead. `scripts/check-public-wording.py` checks added lines and commit
messages; CI runs it on every pull request. A justified exception is one line
in `.github/public-wording-allow.txt` naming the file, the rule and the exact
text.

## Fixture provenance

Every tracked data file must have an entry in `tests/fixtures/PROVENANCE.json`
with one of the categories `pds-archived`, `raw-images-feed`, `synthetic`,
`reference-standard`, `operator-derived-archived` or `metadata`. Mars 2020 data
must come from a sol within the archived PDS release. Add the file and its
entry in the same commit. `scripts/check-fixture-provenance.py --tree` checks
this in CI.
