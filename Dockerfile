# syntax=docker/dockerfile:1.7
# ============================================================
# Stage 1: frontend build — Svelte/Vite SPA
# ============================================================
FROM node:20.18.0-bookworm-slim AS frontend-build
WORKDIR /frontend
COPY src/sherloc_pipeline/web/frontend/package.json \
     src/sherloc_pipeline/web/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY src/sherloc_pipeline/web/frontend/ ./
RUN npm run build
# Output: /frontend/dist/

# ============================================================
# Stage 2: Python build — produce wheels for runtime install
# ============================================================
FROM python:3.12.12-slim-bookworm AS python-build
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# `git` is required because phase-platform-auth is declared as a git URL
# extra dep in pyproject.toml (`phase-platform-auth @ git+https://...`)
# until its PyPI publish; pip-wheel needs to clone and build it locally
# in stage 2 so stage 3 can install offline from /wheels/. Without git
# here, stage 3's `pip install --no-index 'sherloc-pipeline[web,pds]'`
# fails with "Cannot find command 'git'" when resolving the extra.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements-lock.txt pyproject.toml ./
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./

# Build a wheel for sherloc-pipeline from source
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /wheels/

# Pre-build all dependency wheels (locked) into /wheels/
RUN pip wheel --no-cache-dir --wheel-dir=/wheels/ -r requirements-lock.txt

# Pre-build the [web] extra's git-URL dep into /wheels/ so stage 3 can
# resolve it offline. Pinned to the exact ref pyproject.toml declares.
# Synced manually until phase-platform-auth lands on PyPI; if the version
# pin in pyproject.toml moves, update this URL too.
RUN pip wheel --no-cache-dir --wheel-dir=/wheels/ \
      'phase-platform-auth @ git+https://github.com/archaeon-ai/phase-platform-auth@v0.1.0'

# ============================================================
# Stage 3: runtime — install ONLY from wheels, no compilers
# ============================================================
FROM python:3.12.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# `git` here (stage 3) is required because pip resolves the
# `phase-platform-auth @ git+https://...` direct-URL spec from the
# sherloc-pipeline wheel's metadata by CLONING the URL — even with
# `--no-index --find-links=/wheels` and even when /wheels/ contains a
# pre-built wheel of the same package. Direct-URL specifiers in pip are
# authoritative; `--no-index` does not disable them. So git must exist
# in the runtime image. Stage 2's pre-built wheel of phase-platform-auth
# is kept as defense in depth (in case pip's resolver behavior changes
# in a future minor version), but stage 3's git is the load-bearing
# fix. Adds ~30MB to the runtime image.
#
# Sunset path: when phase-platform-auth lands on PyPI and pyproject.toml
# can replace the git URL with a versioned spec (`phase-platform-auth>=0.1.0,<1`),
# both stage 2's git apt + RUN pip wheel step AND this stage 3 git
# install can be removed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 curl tini libgomp1 git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 sherloc \
    && useradd -u 1000 -g sherloc -m sherloc

COPY --from=python-build /wheels /wheels

# Install with [web,pds] extras so fastapi / uvicorn / pyjwt[crypto] /
# httpx / websockets all land in the runtime image. Without the extras
# spec the entrypoint's `uvicorn sherloc_pipeline.web.app:app` and
# `web/auth.py`'s `import jwt`/`import httpx` would fail at first
# request.
#
# We DO NOT pass --no-index here because the [web] extra includes a
# direct-URL specifier (`phase-platform-auth @ git+https://...`) and
# pip ALWAYS clones direct-URL deps + builds them via PEP 517 (here
# hatchling), even when a matching wheel is in /wheels/. The PEP 517
# build needs to fetch hatchling from PyPI; --no-index blocks that.
# So /wheels/ + PyPI is the working combination: pip prefers /wheels/
# (via --find-links) for the locked deps, falls back to PyPI only for
# build-system deps of direct-URL packages. The locked-deps wheels
# in /wheels/ are still authoritative for everything except the
# direct-URL extra.
#
# Sunset path: when phase-platform-auth lands on PyPI and pyproject.toml
# can replace the git URL with a versioned spec, this can revert to
# `--no-index --find-links=/wheels` for fully offline installs (drop
# git from both stages, drop the stage-2 pip-wheel-phase-platform-auth
# step).
RUN pip install --no-cache-dir --find-links=/wheels \
      'sherloc-pipeline[web,pds]' \
    && rm -rf /wheels

WORKDIR /app
COPY --chown=sherloc:sherloc alembic/ alembic/
COPY --chown=sherloc:sherloc alembic.ini ./
COPY --chown=sherloc:sherloc docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod 0755 /app/docker-entrypoint.sh

# Frontend dist installed into the package's resource directory
COPY --from=frontend-build --chown=sherloc:sherloc \
     /frontend/dist/ /usr/local/lib/python3.12/site-packages/sherloc_pipeline/web/frontend/dist/

# Build-time smoke check — fail the build early if imports or
# entrypoints are broken (as root, before USER switch). Includes the
# web/auth surface explicitly so a future regression in the [web]
# extras spec (B.13 F6) fails at build time rather than at first
# request — without these imports here, the base build would pass
# while production startup blew up on `import uvicorn`.
RUN python -c "import numpy, scipy, skimage, sherloc_pipeline; print('base imports ok')" \
 && python -c "import fastapi, uvicorn, jwt, httpx, websockets, boto3; print('web imports ok')" \
 && python -c "from sherloc_pipeline.web.app import create_app; print('web app importable')" \
 && [ -f /usr/local/lib/python3.12/site-packages/sherloc_pipeline/web/frontend/dist/index.html ] \
 && sherloc --help >/dev/null \
 && alembic --version >/dev/null

USER sherloc

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker-entrypoint.sh"]
CMD ["web"]

# ============================================================
# Stage 4: test — runtime + tests/ + dev deps for §18.3 G18.2
# ============================================================
# Build with `docker build --target test -t sherloc-pipeline:v4.0.0-test .`
# Run with `docker run --rm sherloc-pipeline:v4.0.0-test pytest -m "not slow"`.
#
# Same wheel-installed sherloc-pipeline as the runtime stage; only
# pytest + test deps + the tests/ tree are layered on top. This keeps
# the production image lean while making the §18.3 container test gate
# executable from CI on the same Dockerfile.
FROM runtime AS test
USER root

# Install dev test deps. Versions mirror pyproject.toml [dev] extra plus
# cryptography for the auth tests (key generation in test_auth_routes.py
# and test_auth0_validator.py). Listed explicitly here rather than via
# pip install '.[dev]' because the runtime layer has no source tree to
# install from.
RUN pip install --no-cache-dir \
        "pytest>=7.0.0" \
        "pytest-asyncio>=0.23.0" \
        "pytest-httpx>=0.30.0" \
        "cryptography>=42.0.0"

# Copy the test tree. /app is the WORKDIR set by the runtime stage.
COPY --chown=sherloc:sherloc tests/ /app/tests/
COPY --chown=sherloc:sherloc pyproject.toml /app/pyproject.toml

USER sherloc

ENTRYPOINT []
CMD ["pytest", "-m", "not slow", "-q"]
