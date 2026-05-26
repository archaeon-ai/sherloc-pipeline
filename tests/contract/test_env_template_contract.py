"""Contract test: deploy/env-templates/sherloc.env.example is sanitized + complete.

Pins DEPLOYMENT_CONTRACT.md §5 against the example template:

- Every always-required + auth0-mode required + R2 required var appears
  as a non-commented assignment.
- The retired-var literal does NOT appear anywhere in the file (not even
  commented or in "do not add" blocks).
- Secret-shaped values use `change-me*` placeholders or are empty.

The retired-var literal is constructed at runtime so this test file is
also free of it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_TEMPLATE = REPO_ROOT / "deploy" / "env-templates" / "sherloc.env.example"

# REQ set per DEPLOYMENT_CONTRACT.md §5.1 + §5.2 (auth0 mode) + §5.3.
# SHERLOC_AUTH_MODE itself is included because the template sets it
# explicitly (mode=auth0 makes the subsequent reqs apply).
REQUIRED_VARS_NON_COMMENTED = (
    "SHERLOC_AUTH_MODE",
    "SHERLOC_AUTH0_DOMAIN",
    "SHERLOC_AUTH0_AUDIENCE",
    "SHERLOC_AUTH0_SPA_CLIENT_ID",
    "SHERLOC_AUTH0_IDENTITY_CLAIM_URI",
    "SHERLOC_DB",
    "PHASE_TIER",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ENDPOINT_URL",
)

# Names that look like secrets — every match must have a placeholder
# value or be empty.
_SECRET_NAME_PATTERN = re.compile(
    r"^([A-Z][A-Z0-9_]*_(SECRET|KEY|PASSWORD|TOKEN))\s*=\s*(.*)$", re.MULTILINE
)
_PLACEHOLDER_PATTERN = re.compile(
    r"^(change-me.*|your-.*|\$\{.*\}|)$", re.IGNORECASE
)


def _lines() -> list[str]:
    return ENV_TEMPLATE.read_text().splitlines()


def _non_commented_assignments() -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in _lines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        # Strip trailing inline comments (`KEY=value   # comment`).
        value = value.split("#", 1)[0].strip()
        out[key.strip()] = value
    return out


def test_env_template_exists():
    assert ENV_TEMPLATE.is_file(), f"missing {ENV_TEMPLATE}"


def test_required_vars_are_non_commented_assignments():
    assignments = _non_commented_assignments()
    missing = [v for v in REQUIRED_VARS_NON_COMMENTED if v not in assignments]
    assert not missing, (
        f"required vars missing as non-commented assignments: {missing}. "
        "See DEPLOYMENT_CONTRACT.md §5.1 + §5.2."
    )


def test_retired_role_claim_uri_var_absent_everywhere():
    """The legacy split-claim env var was retired in v4.1.0 and MUST NOT
    appear ANYWHERE in the template — not commented, not in a "do not
    add" block, not in any heading. Re-introducing it is a contract
    violation."""
    text = ENV_TEMPLATE.read_text()
    legacy_literal = "SHERLOC_AUTH0_" + "ROLE_CLAIM_URI"
    assert legacy_literal not in text, (
        f"Retired env var {legacy_literal!r} reappeared in the env "
        "template — see DEPLOYMENT_CONTRACT.md §5.5"
    )


def test_secret_shaped_values_use_placeholders():
    text = ENV_TEMPLATE.read_text()
    for match in _SECRET_NAME_PATTERN.finditer(text):
        name, _suffix, value = match.group(1), match.group(2), match.group(3)
        value = value.split("#", 1)[0].strip()
        assert _PLACEHOLDER_PATTERN.fullmatch(value), (
            f"secret-shaped var {name!r} has non-placeholder value "
            f"{value!r}; use change-me*/your-* or leave empty"
        )
