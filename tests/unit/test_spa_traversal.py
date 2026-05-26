"""Regression tests for the SPA catch-all path-traversal vulnerability.

Pre-patch the `@app.get("/{full_path:path}")` handler in
`sherloc_pipeline.web.app.create_app` joined `full_path` onto the dist
directory without normalizing dot-segments. `%2e%2e`-encoded `..` segments
survived ASGI URL decoding and reached the handler as literal `..`,
allowing unauthenticated reads of arbitrary container files (e.g.
`/etc/passwd`). Live-exploited 2026-05-23 against
`sherloc-public.m2020-phase.net` (882-byte `/etc/passwd` leak). Patched in
v4.1.15 by resolving the candidate path and asserting it stays inside the
dist directory before serving.

Each parametrized traversal URL targets a sentinel that actually exists at
the pre-patch resolved path, so a vulnerable implementation would 200 the
sentinel body instead of falling through to `index.html`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sherloc_pipeline.database.connection import create_all_tables, get_engine
from sherloc_pipeline.web.app import create_app


SENT_BODY_1 = "LEAK_ME_IF_VULNERABLE_LEVEL_1"
SENT_BODY_2 = "LEAK_ME_IF_VULNERABLE_LEVEL_2"
ALL_SENT_BODIES = (SENT_BODY_1, SENT_BODY_2)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Build an app whose SPA catch-all is wired to a temp dist with
    depth-matched sentinel files. Layout::

        tmp_path/
        ├── SENT2.txt           ← targeted by two-level escapes (../..)
        └── frontend/
            ├── SENT1.txt       ← targeted by one-level escapes (..)
            └── dist/           ← catch-all root
                ├── assets/app.js
                └── index.html

    A vulnerable pre-patch implementation would resolve each parametrized
    URL to the matching sentinel and serve its body. The fix must instead
    fall through to ``index.html`` for every escape attempt.
    """
    dist = tmp_path / "frontend" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>SPA</title>")
    (dist / "assets" / "app.js").write_text("// app")
    (tmp_path / "frontend" / "SENT1.txt").write_text(SENT_BODY_1)
    (tmp_path / "SENT2.txt").write_text(SENT_BODY_2)

    db_path = tmp_path / "spa_traversal.db"
    engine = get_engine(str(db_path))
    create_all_tables(engine)
    app = create_app(engine=engine, frontend_dist=dist)
    return TestClient(app)


@pytest.mark.parametrize(
    "url",
    [
        # One-level escape: `dist/.. == frontend/`, where SENT1.txt lives.
        "/%2e%2e/SENT1.txt",
        # Mixed-encoding variant of the one-level escape.
        "/%2e%2e%2fSENT1.txt",
        # Two-level escape: `dist/../.. == tmp_path/`, where SENT2.txt lives.
        "/%2e%2e/%2e%2e/SENT2.txt",
    ],
)
def test_spa_catchall_rejects_encoded_path_traversal(
    client: TestClient, url: str
) -> None:
    """Encoded `..` segments must not escape the dist dir. Each URL above
    resolves under the pre-patch join to an existing sentinel; the patched
    catch-all must fall through to the SPA index instead."""
    response = client.get(url)
    assert response.status_code == 200
    for body in ALL_SENT_BODIES:
        assert body not in response.text, (
            f"Traversal succeeded for {url!r}: sentinel {body!r} leaked"
        )
    assert "<!doctype html>" in response.text.lower()


@pytest.mark.skipif(
    not Path("/etc/passwd").exists(),
    reason="POSIX-only: mirrors the 2026-05-23 live exploit against /etc/passwd",
)
def test_spa_catchall_rejects_live_exploit_etc_passwd(client: TestClient) -> None:
    """Verbatim replay of the live exploit URL pattern from seed §4.3:
    ten encoded ``..`` segments targeting ``/etc/passwd``. The pre-patch
    handler would `os.stat`-resolve this and serve the host's passwd file
    (HTTP 200, text/plain). The patched handler must fall through to the
    SPA index without ever leaking the ``root:x:0:0`` marker."""
    response = client.get(
        "/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/etc/passwd"
    )
    assert response.status_code == 200
    assert "root:x:0:0" not in response.text, (
        "Live-exploit pattern leaked /etc/passwd"
    )
    assert "<!doctype html>" in response.text.lower()


def test_spa_catchall_serves_real_files_inside_dist(client: TestClient) -> None:
    """Legitimate paths inside the dist tree still resolve normally."""
    response = client.get("/index.html")
    assert response.status_code == 200
    assert "<!doctype html>" in response.text.lower()


def test_spa_catchall_serves_index_for_unknown_path(client: TestClient) -> None:
    """Unknown in-tree paths fall through to the SPA index (existing
    SPA-routing behavior; preserved by the patch)."""
    response = client.get("/some/route/the/spa/handles")
    assert response.status_code == 200
    assert "<!doctype html>" in response.text.lower()


def test_spa_catchall_serves_index_for_root(client: TestClient) -> None:
    """Empty `full_path` (root request) returns the SPA index."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<!doctype html>" in response.text.lower()
