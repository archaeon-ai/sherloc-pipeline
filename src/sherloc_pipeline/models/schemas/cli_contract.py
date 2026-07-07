"""Self-contained contract for the ``--json`` CLI output envelope (§3.2.9).

The versioned CLI output models (:class:`CLIResult`, :class:`CLIError`,
:class:`PipelineManifest`) are the wire surface that downstream consumers vendor
to read this tool's structured output. This module freezes their structural
*shape* as a committed golden schema so that:

* any change to the envelope shape is caught by ``tests/contract/`` at merge
  time — a breaking change can never land silently; and
* regenerating the frozen golden after a shape change **requires bumping**
  ``schema_version`` first, mechanically coupling a breaking (or additive)
  interface change to a semver decision.

This is a purely self-contained guarantee: the contract references only this
package's own models and its own committed golden. Consumers that vendor this
envelope detect breaking changes via ``schema_version``; how any consumer
re-binds afterward is its own concern and is deliberately not modeled here.

Regenerate after an intentional change::

    python -m sherloc_pipeline.models.schemas.cli_contract        # regen (guarded)
    python -m sherloc_pipeline.models.schemas.cli_contract --check # CI/verify mode
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from sherloc_pipeline.models.schemas.cli import CLIError, CLIResult, PipelineManifest

# The models whose JSON shape is frozen, in a stable order for a deterministic
# golden. All three carry the same ``schema_version`` default; they version as
# one envelope.
_CONTRACT_MODELS = {
    "CLIResult": CLIResult,
    "CLIError": CLIError,
    "PipelineManifest": PipelineManifest,
}

#: The committed golden lives beside this module so it ships with the package
#: (a consumer could read it directly) and moves with the models it freezes.
GOLDEN_PATH = Path(__file__).with_name("cli_contract.golden.json")

#: Single source of truth for the envelope version (all models share it).
SCHEMA_VERSION: str = CLIResult.model_fields["schema_version"].default

# Placeholder substituted for the ``schema_version`` default when computing the
# shape fingerprint, so the fingerprint tracks *shape* only and is independent
# of the version string. The field's presence and type still count.
_VERSION_SENTINEL = "<version>"


def _model_schemas() -> Dict[str, Any]:
    return {name: model.model_json_schema() for name, model in _CONTRACT_MODELS.items()}


def _shape_fingerprint(model_schemas: Dict[str, Any]) -> str:
    """SHA-256 over the canonical structural shape, with the ``schema_version``
    default value normalized out. A pure version bump therefore does NOT change
    the fingerprint, while any field add/remove/retype does."""
    normalized = json.loads(json.dumps(model_schemas))  # deep copy
    for schema in normalized.values():
        prop = schema.get("properties", {}).get("schema_version")
        if isinstance(prop, dict) and "default" in prop:
            prop["default"] = _VERSION_SENTINEL
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_live_contract() -> Dict[str, Any]:
    """The contract implied by the current, in-tree models."""
    schemas = _model_schemas()
    return {
        "schema_version": SCHEMA_VERSION,
        "shape_fingerprint": _shape_fingerprint(schemas),
        "models": schemas,
    }


def load_golden() -> Dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text())


def diff() -> List[str]:
    """Human-readable mismatches between the live models and the committed
    golden. An empty list means the envelope is in sync with its contract."""
    live = build_live_contract()
    try:
        golden = load_golden()
    except FileNotFoundError:
        return [f"golden contract missing: {GOLDEN_PATH.name}"]

    problems: List[str] = []
    if live["schema_version"] != golden.get("schema_version"):
        problems.append(
            f"schema_version: live {live['schema_version']!r} != "
            f"golden {golden.get('schema_version')!r}"
        )
    if live["shape_fingerprint"] != golden.get("shape_fingerprint"):
        problems.append("envelope shape changed (shape_fingerprint mismatch)")
    if live["models"] != golden.get("models"):
        problems.append("model JSON schema(s) differ from the golden")
    return problems


class VersionBumpRequired(RuntimeError):
    """Raised when the shape changed but ``schema_version`` was not bumped."""


def regen(enforce_bump: bool = True) -> Dict[str, Any]:
    """Rewrite the golden from the live models.

    When ``enforce_bump`` is set and the shape changed relative to the existing
    golden while ``schema_version`` did NOT, refuse: a shape change must be
    accompanied by a conscious semver bump so vendoring consumers can detect it.
    """
    live = build_live_contract()
    if enforce_bump and GOLDEN_PATH.exists():
        golden = load_golden()
        shape_changed = live["shape_fingerprint"] != golden.get("shape_fingerprint")
        version_changed = live["schema_version"] != golden.get("schema_version")
        if shape_changed and not version_changed:
            raise VersionBumpRequired(
                "The CLI JSON envelope shape changed but schema_version is still "
                f"{live['schema_version']!r}. Bump schema_version in "
                "models/schemas/cli.py (semver: breaking -> major, additive -> "
                "minor) before regenerating this contract, so consumers that "
                "vendor the envelope can detect the change."
            )
    GOLDEN_PATH.write_text(json.dumps(live, indent=2, sort_keys=True) + "\n")
    return live


def _main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Freeze/verify the --json CLI envelope contract."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed golden is stale (does not write).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate without the schema_version-bump guard.",
    )
    args = parser.parse_args(argv)

    if args.check:
        problems = diff()
        if problems:
            print("CLI envelope contract DRIFT:")
            for problem in problems:
                print(f"  - {problem}")
            print(
                "\nRegenerate (after deciding the semver impact):\n"
                "  python -m sherloc_pipeline.models.schemas.cli_contract"
            )
            return 1
        print("CLI envelope contract: in sync")
        return 0

    try:
        regen(enforce_bump=not args.force)
    except VersionBumpRequired as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(f"Wrote {GOLDEN_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
