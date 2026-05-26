"""Contract test: Alembic migration graph has exactly one head, matching pin.

Pins DEPLOYMENT_CONTRACT.md §7. Every new migration MUST bump
``EXPECTED_HEAD`` below — that is the deliberate review-time signal.

Implementation uses AST + ``ast.literal_eval`` so tuple / list / str /
None ``down_revision`` values are all handled uniformly. The migration
graph contains at least one merge migration (``1ed5cea6c32d``) with a
tuple ``down_revision = ('23b44fd37fd9', '73d700403ed9')`` — regex
parsing would either miscompute heads or silently drop merge parents.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional, Sequence, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSIONS_DIR = REPO_ROOT / "alembic" / "versions"

EXPECTED_HEAD = "412fc1e3ee92"  # add_user_sub_column_for_auth0_identity

DownRevision = Union[None, str, Sequence[str]]


def _extract_revision_pair(path: Path) -> tuple[Optional[str], DownRevision]:
    """Return (revision, down_revision) from a migration file's module
    assignments. Both are extracted via ast.literal_eval to handle
    string / tuple / list / None uniformly."""
    tree = ast.parse(path.read_text(), filename=str(path))
    revision: Optional[str] = None
    down_revision: DownRevision = "<unparsed>"
    for node in ast.iter_child_nodes(tree):
        # AnnAssign handles annotated forms like
        #   revision: str = '...'
        #   down_revision: Union[str, Sequence[str], None] = ('a', 'b')
        # Assign handles plain `revision = '...'`.
        targets: list[ast.AST] = []
        value: Optional[ast.AST] = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        else:
            continue
        if value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                if target.id == "revision":
                    revision = literal
                elif target.id == "down_revision":
                    down_revision = literal
    return revision, down_revision


def _all_migrations() -> list[tuple[Path, str, DownRevision]]:
    """Return [(path, revision, down_revision)] for every migration."""
    out: list[tuple[Path, str, DownRevision]] = []
    for path in sorted(VERSIONS_DIR.glob("*.py")):
        if path.name.startswith("_") or path.name == "__init__.py":
            continue
        rev, down = _extract_revision_pair(path)
        assert rev is not None, f"no revision found in {path.name}"
        assert down != "<unparsed>", f"no down_revision found in {path.name}"
        out.append((path, rev, down))
    return out


def _down_parents(down_revision: DownRevision) -> list[str]:
    if down_revision is None:
        return []
    if isinstance(down_revision, str):
        return [down_revision]
    if isinstance(down_revision, (list, tuple)):
        return [r for r in down_revision if isinstance(r, str)]
    raise TypeError(
        f"unexpected down_revision type {type(down_revision).__name__}: "
        f"{down_revision!r}"
    )


def test_migrations_directory_exists():
    assert VERSIONS_DIR.is_dir(), f"missing {VERSIONS_DIR}"


def test_at_least_one_merge_migration_exists():
    """Sanity check: the AST handling above is meaningful only if the
    graph actually contains a merge (tuple/list down_revision). If this
    fails the merge migration was removed; the AST approach still works
    but the comment about merges should be updated."""
    migrations = _all_migrations()
    merges = [
        path.name
        for path, _, down in migrations
        if isinstance(down, (list, tuple))
    ]
    assert merges, "expected at least one merge migration (tuple down_revision)"


def test_single_head_matches_pin():
    migrations = _all_migrations()
    all_revisions = {rev for _, rev, _ in migrations}
    referenced_as_parent: set[str] = set()
    for _, _, down in migrations:
        for parent in _down_parents(down):
            referenced_as_parent.add(parent)
    heads = sorted(all_revisions - referenced_as_parent)
    assert len(heads) == 1, (
        f"expected exactly one alembic head, found {len(heads)}: {heads}"
    )
    head = heads[0]
    assert head == EXPECTED_HEAD, (
        f"alembic head drifted: expected {EXPECTED_HEAD!r}, "
        f"found {head!r}. Update EXPECTED_HEAD in this test and "
        "DEPLOYMENT_CONTRACT.md §7 when adding a migration."
    )
