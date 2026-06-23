# Security

## Reporting a vulnerability

Please **do not file a public issue** for security problems. Email
`ken@bmsis.org` with details (affected version, reproduction steps,
expected vs. observed behavior). Best-effort response within a few
working days.

## Deployment requirements (web UI)

The SHERLOC web UI is **not** intended to be exposed to the open
internet without an authenticating proxy. Deployments are expected
to:

- Bind `uvicorn` to `127.0.0.1` (or a private interface) — never
  `0.0.0.0` on a public interface.
- Front the service with a Cloudflare Access (or equivalent)
  authenticating proxy.
- Set `SHERLOC_CORS_ALLOWED_ORIGINS` to an explicit allowlist of the
  origin(s) that need cross-origin access (default is empty — no
  cross-origin requests).

## Authentication model

When fronted by Cloudflare Access, the application validates the
`Cf-Access-Jwt-Assertion` JWT on every authenticated request:
signature (against the live JWKS), issuer (must equal
`https://<SHERLOC_CF_TEAM_DOMAIN>`), audience (must equal
`SHERLOC_CF_AUDIENCE`), and expiry. The convenience header
`Cf-Access-Authenticated-User-Email` is **not** trusted; identity is
read from validated JWT claims only.

Required environment variables in production:

- `SHERLOC_CF_TEAM_DOMAIN` — e.g. `your-team.cloudflareaccess.com`.
  The expected issuer is `https://<this value>`.
- `SHERLOC_CF_AUDIENCE` — the Cloudflare Access application AUD tag
  (Zero Trust dashboard → Access → Applications → Overview).

If either is unset, the auth path fails closed with HTTP 500. There
is no header-trust fallback.

JWKS handling: keys are cached in-process for one hour. On a transient
fetch failure, the cache is reused for up to 24 hours; outside that
window — or with no cache — authenticated requests return HTTP 503,
**not** 401, so a JWKS outage is not misread as a credentials problem.

## Development escape hatch

Setting `SHERLOC_AUTH_MODE=dev` bypasses JWT validation and resolves
all authenticated requests to a hardcoded `dev@local` identity. The
server logs a prominent warning at startup when this mode is on. **Do
not enable in production.**
