# syntax=docker/dockerfile:1.7
# ============================================================
# Stage 1: frontend build — Svelte/Vite SPA
# ============================================================
FROM node:20.18.0-bookworm-slim@sha256:28fbbb764069c698ead61d6a739a7615e8f0e07a4b8fe1473ceca70c1c3d6aaa AS frontend-build
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
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS python-build
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Git is needed only here to build auth at its reviewed commit. All
# other locked dependencies must have binary wheels for the target.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements-lock.txt requirements-build.txt pyproject.toml Dockerfile ./
COPY scripts/check-auth-revision.py scripts/check-auth-revision.py
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./

# Install exact build tools once; disabled isolation prevents either
# source build from fetching a different PEP 517 backend.
RUN python scripts/check-auth-revision.py \
    && pip install --no-cache-dir --only-binary=:all: -r requirements-build.txt \
    && python -m build --no-isolation --wheel --outdir /wheels/

# Download only the reviewed dependency wheels, without resolving their
# open dependency ranges. Auth is built separately from its Git commit.
RUN sed '/^phase-platform-auth @ /d' requirements-lock.txt > requirements-index.txt \
    && pip wheel --no-cache-dir --no-deps --only-binary=:all: \
         --wheel-dir=/wheels/ -r requirements-index.txt

# The auth revisions in pyproject.toml, requirements-lock.txt, and this
# Dockerfile must move together; scripts/check-auth-revision.py enforces it.
RUN pip wheel --no-cache-dir --no-deps --no-build-isolation --wheel-dir=/wheels/ \
      'phase-platform-auth @ git+https://github.com/archaeon-ai/phase-platform-auth@890be041d7b4e6144a3a132df38bf02e5b599d53'

# A direct Git requirement would clone even with --no-index. Replace only
# that requirement in the runtime lock copy with the exact wheel version.
# PEP 427 wheel filenames encode the normalized distribution and version.
RUN set -eu; set -- /wheels/phase_platform_auth-*.whl; \
    [ "$#" -eq 1 ]; [ -f "$1" ]; \
    auth_version=${1##*/}; auth_version=${auth_version#phase_platform_auth-}; \
    auth_version=${auth_version%%-*}; \
    sed "s/^phase-platform-auth @ .*/phase-platform-auth==$auth_version/" \
      requirements-lock.txt > /wheels/requirements-lock.txt

# ============================================================
# Stage 3: runtime — install ONLY from wheels, no compilers
# ============================================================
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Declare the canonical source repository for this image. GHCR uses this
# OCI label to link the published package to its home repo (and derive
# inherited access), so the package stays self-contained on this repo.
LABEL org.opencontainers.image.source="https://github.com/archaeon-ai/sherloc-pipeline"

# Runtime system libraries and process tools; no Git or build toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 curl tini libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 sherloc \
    && useradd -u 1000 -g sherloc -m sherloc

COPY --from=python-build /wheels /wheels

# The lock includes the base package's entire [web,pds] dependency set.
# Install that exact graph offline, then the project wheel without
# resolving its direct-URL extra. Missing wheels fail with no fallback.
RUN --network=none pip install --no-cache-dir --no-index --find-links=/wheels \
      -r /wheels/requirements-lock.txt \
    && pip install --no-cache-dir --no-index --no-deps /wheels/sherloc_pipeline-*.whl \
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
