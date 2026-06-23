"""Cold-path isolation (MLD-QUA-002) and pipeline wiring shape.

The ML runtime must never load unless the ml branch executes: importing
the preprocessing/pipeline/cr_masks services — and resolving the
modz/none methods — must not import onnxruntime or ml_despike. Run in a
subprocess so other tests' imports cannot contaminate sys.modules.
"""

import inspect
import subprocess
import sys
import textwrap


def _run_isolated(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"isolation subprocess failed:\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


class TestModuleIsolation:
    def test_service_imports_do_not_load_ml_runtime(self):
        _run_isolated(
            """
            import sys
            import sherloc_pipeline.services.preprocessing
            import sherloc_pipeline.services.pipeline
            import sherloc_pipeline.services.cr_masks
            import sherloc_pipeline.core.mask_application
            forbidden = [m for m in sys.modules
                         if m.startswith(("onnxruntime",
                                          "sherloc_pipeline.ml_despike"))]
            assert not forbidden, f"ML runtime leaked at import: {forbidden}"
            """
        )

    def test_web_app_import_does_not_load_ml_runtime(self):
        """MLD-QUA-002 AC2: building the web app (and importing the spectra
        route that adds the stored-mask despike toggle) must not pull
        onnxruntime — the serving host never runs inference."""
        _run_isolated(
            """
            import sys
            from sherloc_pipeline.web.app import create_app  # noqa: F401
            import sherloc_pipeline.web.routes.spectra  # noqa: F401
            leaked = [m for m in sys.modules if m.startswith("onnxruntime")]
            assert not leaked, f"onnxruntime leaked into the web app: {leaked}"
            """
        )

    def test_modz_and_none_resolution_do_not_load_ml_runtime(self):
        _run_isolated(
            """
            import sys
            from types import SimpleNamespace
            from sherloc_pipeline.services.preprocessing import (
                resolve_despike_method,
            )
            for method in ("modz", "none"):
                cfg = SimpleNamespace(
                    preprocessing={"despike": {"method": method}}
                )
                assert resolve_despike_method(None, cfg) == method
            forbidden = [m for m in sys.modules
                         if m.startswith(("onnxruntime",
                                          "sherloc_pipeline.ml_despike"))]
            assert not forbidden, f"ML runtime leaked: {forbidden}"
            """
        )


class TestPipelineWiring:
    """Source-shape assertions, following the existing
    test_service_pipeline_integration inspect-based precedent."""

    def test_run_full_pipeline_has_cr_mask_persistence_stage(self):
        from sherloc_pipeline.services.pipeline import PipelineService

        source = inspect.getsource(PipelineService.run_full_pipeline)
        assert 'capture_stage("cr_mask_persistence")' in source
        assert "CRMaskService" in source
        # Non-fatal: persists inside the warn-and-continue pattern
        assert "CR mask persistence skipped" in source

    def test_run_full_pipeline_defers_method_to_config(self):
        """The sole run_scan call site passes despike_method=None (defer
        to config resolution); the legacy despike_r1 kwarg is gone."""
        from sherloc_pipeline.services.pipeline import PipelineService

        source = inspect.getsource(PipelineService.run_full_pipeline)
        assert "despike_method=None" in source
        assert "despike_r1" not in source

    def test_persistence_gated_on_masks_and_db(self):
        from sherloc_pipeline.services.pipeline import PipelineService

        source = inspect.getsource(PipelineService.run_full_pipeline)
        assert 'despike_metadata.get("masks")' in source

    def test_run_scan_has_no_despike_r1_parameter(self):
        from sherloc_pipeline.services.preprocessing import (
            PreprocessingService,
        )

        params = inspect.signature(
            PreprocessingService.run_scan
        ).parameters
        assert "despike_r1" not in params
        assert "despike_method" in params
        assert params["despike_method"].default is None
