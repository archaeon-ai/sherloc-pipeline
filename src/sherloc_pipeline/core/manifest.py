from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class ManifestResolutionError(RuntimeError):
    """Raised when Loupe manifest discovery cannot uniquely resolve a working directory."""


@dataclass(frozen=True)
class ManifestCandidate:
    workspace: str
    working_dir: Path
    source: str
    metadata: Dict[str, str]


def _iter_working_directories(sol_dir: Path) -> Iterable[Path]:
    if not sol_dir.exists():
        return

    for scan_dir in sol_dir.iterdir():
        if not scan_dir.is_dir():
            continue
        if scan_dir.name.startswith(".") or "archive" in scan_dir.name.lower():
            continue
        for candidate in scan_dir.iterdir():
            if candidate.is_dir() and candidate.name.endswith("_Loupe_working"):
                yield candidate


def _read_loupe_manifest(loupe_path: Path) -> Optional[Dict[str, str]]:
    try:
        with loupe_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            manifest: Dict[str, str] = {}
            for row in reader:
                if not row:
                    continue
                key = str(row[0]).strip()
                if not key:
                    continue
                value = ""
                if len(row) > 1:
                    value = str(row[1]).strip()
                manifest[key] = value
    except Exception as exc:
        raise ManifestResolutionError(f"Failed to parse Loupe manifest at {loupe_path}: {exc}") from exc

    return manifest or None


def _build_manifest_candidates(sol_dir: Path) -> List[ManifestCandidate]:
    candidates: List[ManifestCandidate] = []
    for working_dir in _iter_working_directories(sol_dir):
        loupe_path = working_dir / "loupe.csv"
        if not loupe_path.exists():
            continue
        manifest = _read_loupe_manifest(loupe_path)
        if not manifest:
            continue
        workspace = (
            manifest.get("human_readable_workspace")
            or manifest.get("workspace")
            or manifest.get("scan")
        )
        if not workspace:
            continue
        candidates.append(
            ManifestCandidate(
                workspace=str(workspace).strip(),
                working_dir=working_dir,
                source=str(loupe_path),
                metadata=manifest,
            )
        )
    return candidates


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    previous_row = list(range(len_b + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            substitute_cost = previous_row[j - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, substitute_cost))
        previous_row = current_row
    return previous_row[-1]


# Composite reduction vocabulary (mirrors the recognized multishot reduction
# suffixes, e.g. ``_sum_active_median_dark``, ``_sum_active_sum_dark``,
# ``_median_all``). These tokens are drawn from the pipeline's own controlled
# naming, never freehand-transcribed by Loupe, so a mismatch here means two
# genuinely different composite reductions -- not a typo -- and must not be
# corrected by edit distance.
_STRUCTURAL_TOKENS = frozenset({"sum", "active", "median", "dark", "all", "mean"})

# Per-token cap on how far a single editable (root) token may drift under
# edit distance before it's rejected as unrelated rather than a typo. Real
# single-word Loupe typos observed so far (e.g. sol 1521's ``meteroite`` for
# ``meteorite``) are 1-2 edits; this is deliberately tight -- it exists to
# stop the aggregate, length-scaled threshold below from being satisfied by
# a root token that's simply a different word.
_MAX_ROOT_TOKEN_EDIT_DISTANCE = 2

# Sub-scan suffix letters, mirroring the sub-scan rule in
# ``models.spectra.classify_scan_class``/``derive_parent_name`` (last char in
# a/b/c preceded by a digit or the token boundary).
_SUB_SCAN_SUFFIXES = ("a", "b", "c")

# A sub-scan suffix (one or more digits then a/b/c) trailing a token,
# whether or not it's split into its own underscore token. Mirrors
# ``classify_scan_class``'s rule, which checks the raw scan name's last
# char rather than an underscore-delimited token -- so ``detail1a`` (a real
# Loupe naming variant with no separator before the index) is a sub-scan
# suffix glued onto free text just as much as the ``1a`` in ``detail_1a``,
# and ``detail1a``/``detail1b`` are distinct sub-scans with distinct
# spectral data, never edit-distance neighbors of each other.
_SUB_SCAN_SUFFIX_PATTERN = re.compile(r"\d+[abc]$")


def _is_index_token(token: str) -> bool:
    """True when a token is, or ends with, a repeat- or sub-scan index.

    Covers a bare repeat index (``1``, ``2``), an alphanumeric sub-scan
    index (``1a``, ``500b``), a bare sub-scan suffix (``a``, ``b``, ``c``),
    and a sub-scan suffix glued directly onto free text (``detail1a``) --
    all of which distinguish genuinely different sibling scans with
    distinct spectral data, never a misspelling of one another.
    """
    if token.isdigit():
        return True
    if token in _SUB_SCAN_SUFFIXES:
        return True
    return bool(_SUB_SCAN_SUFFIX_PATTERN.search(token))


def _typo_distance(normalized_scan: str, workspace: str) -> Optional[int]:
    """Edit distance restricted to the free-text root of underscore-delimited names.

    Loupe composite/repeat-scan names encode structure in ``_``-separated
    tokens -- a repeat index (``detail_1`` vs ``detail_2``), an alphanumeric
    or bare-letter sub-scan index (``detail_1a`` vs ``detail_1b``, ``HDR_a``
    vs ``HDR_b``, or the glued ``detail1a`` vs ``detail1b``), or a composite
    reduction method (``..._sum_active_median_dark``). Those tokens
    distinguish genuinely different sibling scans, not a misspelling of the
    same one, so they must match exactly; only a non-index, non-structural
    token (the scan/target root, e.g. sol 1521's ``meteroite`` for
    ``meteorite``) may be corrected by edit distance, and only up to
    ``_MAX_ROOT_TOKEN_EDIT_DISTANCE`` per token. Returns ``None`` when the
    two names aren't structurally comparable (different token count, an
    index/structural token that doesn't match verbatim, or a root token
    that drifts past the per-token cap) rather than a candidate for typo
    correction.
    """
    scan_tokens = normalized_scan.split("_")
    workspace_tokens = workspace.split("_")
    if len(scan_tokens) != len(workspace_tokens):
        return None

    total = 0
    for scan_token, workspace_token in zip(scan_tokens, workspace_tokens):
        if scan_token == workspace_token:
            continue
        if (
            _is_index_token(scan_token)
            or _is_index_token(workspace_token)
            or scan_token in _STRUCTURAL_TOKENS
            or workspace_token in _STRUCTURAL_TOKENS
        ):
            return None
        token_distance = _edit_distance(scan_token, workspace_token)
        if token_distance > _MAX_ROOT_TOKEN_EDIT_DISTANCE:
            return None
        total += token_distance
    return total


def _editable_root_length(normalized_scan: str) -> int:
    """Length of the free-text root tokens only, excluding structural/index ones.

    Used to scale the fuzzy-match tolerance -- the invariant composite
    suffix (``_sum_active_median_dark`` and friends) and repeat/sub-scan
    index tokens (see ``_is_index_token``) must match verbatim (see
    ``_typo_distance``) and contribute nothing to the actual typo budget, so
    they must not inflate the tolerance applied to the editable root either.
    """
    tokens = normalized_scan.split("_")
    return sum(
        len(token)
        for token in tokens
        if not _is_index_token(token) and token not in _STRUCTURAL_TOKENS
    )


def _fuzzy_match_candidates(
    candidates: List[ManifestCandidate], normalized_scan: str
) -> List[ManifestCandidate]:
    """Typo-tolerant fallback when no candidate's workspace matches exactly.

    Resolve by nearest typo distance (see ``_typo_distance``), but only when
    exactly one candidate is unambiguously closest -- never guess between
    two similarly-misspelled scans, since silently picking the wrong one
    would point the pipeline at the wrong spectral data.
    """
    threshold = max(1, round(_editable_root_length(normalized_scan) * 0.2))
    scored = []
    for candidate in candidates:
        distance = _typo_distance(normalized_scan, candidate.workspace.strip().casefold())
        if distance is None:
            continue
        scored.append((distance, candidate))
    scored.sort(key=lambda pair: pair[0])

    if not scored or scored[0][0] > threshold:
        return []
    if len(scored) > 1 and scored[1][0] == scored[0][0]:
        return []  # Ambiguous -- refuse to guess.
    return [scored[0][1]]


def resolve_manifest_working_directory(
    base_data_dir: Path,
    sol: str,
    scan: str,
) -> Optional[Path]:
    sol_dir = base_data_dir / f"sol_{sol}"
    manifest_candidates = _build_manifest_candidates(sol_dir)
    if not manifest_candidates:
        return None

    normalized_scan = scan.strip().casefold()
    matches: List[ManifestCandidate] = [
        candidate
        for candidate in manifest_candidates
        if candidate.workspace.strip().casefold() == normalized_scan
    ]

    if not matches:
        matches = _fuzzy_match_candidates(manifest_candidates, normalized_scan)

    if not matches:
        return None

    if len(matches) > 1:
        paths = ", ".join(str(candidate.working_dir) for candidate in matches)
        raise ManifestResolutionError(
            f"Multiple Loupe manifests match sol {sol} scan {scan}: {paths}. "
            "Prune archives or verify loupe.csv entries."
        )

    working_dir = matches[0].working_dir
    required_files = [
        "loupe.csv",
        "spatial.csv",
        "darkSubSpectra.csv",
        "photodiodeRaw.csv",
    ]
    missing = [name for name in required_files if not (working_dir / name).exists()]
    if missing:
        raise ManifestResolutionError(
            f"Manifest match {working_dir} missing required files: {', '.join(missing)}"
        )

    return working_dir.resolve()


__all__ = ["ManifestResolutionError", "resolve_manifest_working_directory"]
