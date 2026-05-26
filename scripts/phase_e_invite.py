#!/usr/bin/env python
"""Phase E.1 helper — invite team members to the Auth0 tenant.

Per spec §19.1 Phase E.1: "Auth0 Management API supports invite-via-email;
small helper script lives in scripts/phase_e_invite.py".

Pattern:
  1. POST /api/v2/users  — create the user with a random password and
     email_verified=false.
  2. POST /api/v2/tickets/password-change — generate a password-change
     ticket; the user follows the URL to set their own password.
  3. POST /api/v2/users/{id}/roles — assign the requested role.
  4. Emit one CSV row per invitee with the ticket URL.

The operator emails each invitee the ticket URL (the `phase-e-team-letter.md`
runbook is the comms template). On first visit, Auth0 sends them through
email-verification + password-set, then back to the SPA.

This script is operator-tooling only. It is not invoked by the application
at runtime. It uses standard library only so it runs without the project
virtualenv if needed.

Usage:
    # From a CSV with one email per line (header optional)
    ./scripts/phase_e_invite.py --emails-file invitees.csv --role sherloc:internal

    # From stdin
    cat invitees.csv | ./scripts/phase_e_invite.py --role sherloc:internal

    # Cross-deployment access (both internal + public)
    ./scripts/phase_e_invite.py --emails-file team.csv \
        --role sherloc:internal --extra-role sherloc:public

Required env vars (operator pulls from Infisical):
    SHERLOC_AUTH0_MGMT_DOMAIN     e.g. sherloc.us.auth0.com
    SHERLOC_AUTH0_MGMT_CLIENT_ID  Management API M2M client id
    SHERLOC_AUTH0_MGMT_CLIENT_SECRET
    SHERLOC_AUTH0_CONNECTION      (optional; default: Username-Password-Authentication)
    SHERLOC_AUTH0_RESULT_URL      REQUIRED — URL Auth0 redirects to after
                                   password-set. No built-in default; the
                                   script exits non-zero at start if unset.
                                   Set to the operator-canonical post-login
                                   URL for the target deployment.

Output: CSV to stdout with columns
    email,user_id,role,role_assignment_status,ticket_url,ticket_expires_at,error
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import string
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

DEFAULT_CONNECTION = "Username-Password-Authentication"
# DEFAULT_RESULT_URL removed (was an operator-local hostname literal,
# superseded by the m2020-phase deployment hostnames). Operator MUST
# set SHERLOC_AUTH0_RESULT_URL in the runtime env; the script enforces
# this at start via `_require_env("SHERLOC_AUTH0_RESULT_URL")`. Falling
# back to a hardcoded operator-local URL would leak deployment topology
# into invitation emails.
TICKET_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.stderr.write(f"ERROR: required env var not set: {name}\n")
        sys.exit(2)
    return val


def _api_call(
    method: str,
    url: str,
    token: str,
    body: Optional[dict] = None,
) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body_bytes = resp.read()
    return json.loads(body_bytes) if body_bytes else {}


def _management_token(domain: str, client_id: str, client_secret: str) -> str:
    body = {
        "client_id": client_id,
        "client_secret": client_secret,
        "audience": f"https://{domain}/api/v2/",
        "grant_type": "client_credentials",
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"https://{domain}/oauth/token",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read())
    return payload["access_token"]


def _random_password() -> str:
    # Auth0 free-tier default policy requires upper, lower, digit, length>=8.
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
    return "".join(secrets.choice(alphabet) for _ in range(32))


def _resolve_role_id(domain: str, token: str, role_name: str) -> str:
    page = 0
    while True:
        url = (
            f"https://{domain}/api/v2/roles?per_page=100&page={page}&include_totals=true"
        )
        data = _api_call("GET", url, token)
        for role in data.get("roles", []):
            if role.get("name") == role_name:
                return role["id"]
        if (page + 1) * 100 >= data.get("total", 0):
            break
        page += 1
    raise RuntimeError(
        f"Auth0 role not found: {role_name!r}. "
        f"Verify it exists in the tenant per spec §13.0.4."
    )


def _create_user(domain: str, token: str, email: str, connection: str) -> dict:
    return _api_call(
        "POST",
        f"https://{domain}/api/v2/users",
        token,
        {
            "email": email,
            "password": _random_password(),
            "connection": connection,
            "email_verified": False,
            "verify_email": True,
        },
    )


def _create_password_change_ticket(
    domain: str, token: str, user_id: str, result_url: str
) -> dict:
    return _api_call(
        "POST",
        f"https://{domain}/api/v2/tickets/password-change",
        token,
        {
            "user_id": user_id,
            "result_url": result_url,
            "ttl_sec": TICKET_TTL_SECONDS,
            "mark_email_as_verified": False,
        },
    )


def _assign_role(domain: str, token: str, user_id: str, role_id: str) -> None:
    _api_call(
        "POST",
        f"https://{domain}/api/v2/users/{urllib.parse.quote(user_id)}/roles",
        token,
        {"roles": [role_id]},
    )


def _read_emails(path: Optional[str]) -> list[str]:
    if path:
        with open(path, encoding="utf-8") as f:
            lines = list(f)
    else:
        lines = list(sys.stdin)
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Allow CSV with email in first column
        first = line.split(",", 1)[0].strip().strip('"')
        if first.lower() == "email":  # skip header
            continue
        if "@" not in first:
            sys.stderr.write(f"WARN: skipping non-email line: {first!r}\n")
            continue
        out.append(first)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emails-file",
        help="Path to CSV/text file with one email per line (header optional). "
        "If omitted, reads stdin.",
    )
    parser.add_argument(
        "--role",
        required=True,
        choices=["sherloc:internal", "sherloc:public", "sherloc:admin"],
        help="Primary role to assign (per spec §13.0.4).",
    )
    parser.add_argument(
        "--extra-role",
        choices=["sherloc:internal", "sherloc:public", "sherloc:admin"],
        help="Optional second role for cross-deployment access.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve env + role IDs but do not create users.",
    )
    args = parser.parse_args()

    domain = _require_env("SHERLOC_AUTH0_MGMT_DOMAIN")
    client_id = _require_env("SHERLOC_AUTH0_MGMT_CLIENT_ID")
    client_secret = _require_env("SHERLOC_AUTH0_MGMT_CLIENT_SECRET")
    connection = os.environ.get("SHERLOC_AUTH0_CONNECTION", DEFAULT_CONNECTION)
    result_url = _require_env("SHERLOC_AUTH0_RESULT_URL")

    sys.stderr.write(f"Resolving Management API token for {domain} ...\n")
    token = _management_token(domain, client_id, client_secret)

    sys.stderr.write(f"Resolving role id for {args.role} ...\n")
    primary_role_id = _resolve_role_id(domain, token, args.role)
    extra_role_id = (
        _resolve_role_id(domain, token, args.extra_role) if args.extra_role else None
    )

    emails = _read_emails(args.emails_file)
    if not emails:
        sys.stderr.write("ERROR: no emails to process\n")
        return 64
    sys.stderr.write(f"Will invite {len(emails)} user(s)\n")

    if args.dry_run:
        sys.stderr.write("--dry-run: not creating users; exiting.\n")
        return 0

    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "email",
            "user_id",
            "role",
            "role_assignment_status",
            "ticket_url",
            "ticket_expires_at",
            "error",
        ]
    )

    exit_code = 0
    for email in emails:
        try:
            user = _create_user(domain, token, email, connection)
            user_id = user["user_id"]
            try:
                _assign_role(domain, token, user_id, primary_role_id)
                if extra_role_id:
                    _assign_role(domain, token, user_id, extra_role_id)
                role_status = "ok"
            except Exception as role_exc:  # noqa: BLE001
                role_status = f"failed: {type(role_exc).__name__}: {role_exc}"
                exit_code = 1
            ticket = _create_password_change_ticket(
                domain, token, user_id, result_url
            )
            writer.writerow(
                [
                    email,
                    user_id,
                    args.role + (f"+{args.extra_role}" if args.extra_role else ""),
                    role_status,
                    ticket.get("ticket", ""),
                    ticket.get("ticket_expires_at", ""),
                    "",
                ]
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            writer.writerow([email, "", args.role, "error", "", "", f"HTTP {e.code}: {body}"])
            exit_code = 1
        except Exception as exc:  # noqa: BLE001
            writer.writerow(
                [email, "", args.role, "error", "", "", f"{type(exc).__name__}: {exc}"]
            )
            exit_code = 1
        sys.stdout.flush()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
