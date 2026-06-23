"""
Architecture enforcement tests for unidirectional dependency flow.

Verifies the layering rule: cli/ -> api/ -> services/ -> core/ -> models/

Uses AST inspection to catch all import forms (module-level and lazy/inline),
with an explicit allowlist for intentional backward-compat shims that use
lazy imports inside function bodies.

Phase 2.5, Gate 1 of the Public Release spec (§3.2.5).
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "sherloc_pipeline"

# Intentional backward-compat shims: lazy imports inside function bodies.
# These are allowlisted because they delegate to visualization/ at call time
# (not at import time) and were explicitly moved there to keep core/ otherwise
# matplotlib-free. The key is (relative_path_from_SRC, forbidden_prefix).
#
# To add a new exception, document the rationale here and add the tuple.
ALLOWED_EXCEPTIONS: set[tuple[str, str]] = {
    # core/spatial.py shims: _compose_with_skimage_rings, overlay_points_on_aci,
    # _compose_rings_on_image, render_pointloc_full, render_pointloc_zoomed,
    # render_pointloc_with_colorbar, build_combined_grid — all delegate to
    # visualization/spatial.py for backward compatibility.
    ("core/spatial.py", "sherloc_pipeline.visualization"),
    # core/laser_normalization.py: optional plot generation inside
    # normalize_laser_power(); gracefully swallowed on ImportError.
    ("core/laser_normalization.py", "sherloc_pipeline.visualization"),
    # core/data_ingestion.py: plot_average_spectrum() and
    # plot_both_average_spectra() delegate to visualization/ingestion_plots.py.
    ("core/data_ingestion.py", "sherloc_pipeline.visualization"),
}


def _collect_imports(filepath: Path) -> list[str]:
    """Parse a Python file and return all imported module names (all scopes)."""
    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _get_python_files(package_dir: Path) -> list[Path]:
    """Return all .py files recursively under package_dir."""
    return list(package_dir.rglob("*.py"))


def _check_no_imports_from(
    package_name: str,
    forbidden_packages: list[str],
) -> list[str]:
    """Return a list of violation strings for files in package_name that import
    from any of the forbidden_packages.

    Entries in ALLOWED_EXCEPTIONS are skipped silently.
    """
    package_dir = SRC / package_name
    if not package_dir.exists():
        return []

    violations = []
    for pyfile in _get_python_files(package_dir):
        rel = pyfile.relative_to(SRC)
        rel_str = rel.as_posix()
        imports = _collect_imports(pyfile)
        for imp in imports:
            for forbidden in forbidden_packages:
                forbidden_prefix = f"sherloc_pipeline.{forbidden}"
                if imp.startswith(forbidden_prefix):
                    # Check allowlist
                    if (rel_str, forbidden_prefix) in ALLOWED_EXCEPTIONS:
                        continue
                    # Also check prefix matches (e.g. visualization.spatial
                    # matched by sherloc_pipeline.visualization)
                    allowed = any(
                        rel_str == exc_path and imp.startswith(exc_prefix)
                        for exc_path, exc_prefix in ALLOWED_EXCEPTIONS
                    )
                    if not allowed:
                        violations.append(f"{rel_str}: imports {imp}")
    return violations


class TestLayeringRules:
    """Enforce the unidirectional dependency flow:
    cli/ -> api/ -> services/ -> core/ -> models/
    """

    def test_core_does_not_import_services_cli_api_web_visualization(self):
        """core/ must not import from services/, cli/, api/, web/, or visualization/.

        Exceptions: intentional backward-compat shims listed in ALLOWED_EXCEPTIONS.
        """
        violations = _check_no_imports_from(
            "core", ["services", "cli", "api", "web", "visualization"]
        )
        assert violations == [], (
            "core/ has forbidden imports (layering violation):\n"
            + "\n".join(violations)
        )

    def test_services_does_not_import_cli_api_web(self):
        """services/ must not import from cli/, api/, or web/."""
        violations = _check_no_imports_from("services", ["cli", "api", "web"])
        assert violations == [], (
            "services/ has forbidden imports (layering violation):\n"
            + "\n".join(violations)
        )

    def test_models_does_not_import_logic_layers(self):
        """models/ must not import from core/, services/, cli/, api/, or web/.

        models/ is a pure data layer; it must not depend on business logic.
        """
        violations = _check_no_imports_from(
            "models", ["core", "services", "cli", "api", "web"]
        )
        assert violations == [], (
            "models/ has forbidden imports (layering violation):\n"
            + "\n".join(violations)
        )

    def test_api_does_not_import_core(self):
        """api/ must route through services/, not import core/ directly.

        Phase 1 invariant — enforce permanently.
        """
        violations = _check_no_imports_from("api", ["core"])
        assert violations == [], (
            "api/ has forbidden imports from core/ (must route through services/):\n"
            + "\n".join(violations)
        )

    def test_cli_does_not_import_core(self):
        """cli/ must route through services/ or api/, not import core/ directly.

        Phase 1 invariant — enforce permanently.
        """
        violations = _check_no_imports_from("cli", ["core"])
        assert violations == [], (
            "cli/ has forbidden imports from core/ (must route through services/):\n"
            + "\n".join(violations)
        )

    def test_no_matplotlib_in_core(self):
        """core/ must have no matplotlib imports.

        Phase 1 invariant: all plotting must live in visualization/.
        Enforce permanently so that future changes cannot accidentally
        re-introduce matplotlib dependencies into the algorithm layer.
        """
        core_dir = SRC / "core"
        violations = []
        for pyfile in _get_python_files(core_dir):
            rel = pyfile.relative_to(SRC).as_posix()
            imports = _collect_imports(pyfile)
            for imp in imports:
                if "matplotlib" in imp:
                    violations.append(f"{rel}: imports {imp}")
        assert violations == [], (
            "core/ has matplotlib imports (Phase 1 invariant violated):\n"
            + "\n".join(violations)
        )
