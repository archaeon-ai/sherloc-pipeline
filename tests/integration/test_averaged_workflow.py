"""Integration tests for averaged spectrum workflow.

These tests exercise the complete _process_averaged() workflow from Loupe data
loading through export, using real fixture data.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from sherloc_pipeline.services.spectral import (
    SpectralService,
    SpectralPlotRequest,
)
from sherloc_pipeline.services.base import ServiceResult


class TestAveragedWorkflowIntegration:
    """Integration tests for the averaged spectrum workflow."""

    def test_minimal_workflow_amherst_point(self, test_context, tmp_results):
        """Test minimal workflow: load → average → export (no processing)."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="mean",
            background=None,
            baseline=False,
            fit=False,
            export="csv",
        )
        
        result = service.process(request)
        
        # Verify result structure
        assert isinstance(result, ServiceResult)
        assert result.summary is not None
        assert "0921" in result.summary
        assert "Amherst_Point" in result.summary
        
        # Verify artifacts created (CSV + JSON metadata)
        assert len(result.artifacts) == 2
        csv_path = [p for p in result.artifacts if p.suffix == ".csv"][0]
        json_path = [p for p in result.artifacts if p.suffix == ".json"][0]
        assert csv_path.exists()
        assert json_path.exists()
        
        # Verify CSV structure
        df = pd.read_csv(csv_path)
        assert "raman_shift" in df.columns
        assert "intensity" in df.columns
        assert len(df) > 0
        
        # Verify reasonable value ranges
        assert df["raman_shift"].min() > 200
        assert df["raman_shift"].max() < 5000
        
        # Verify metadata
        assert result.metadata["n_points"] == 100
        assert result.metadata["ppp"] == 500.0
        assert result.metadata["avg_method"] == "mean"

    def test_full_workflow_with_background_baseline(self, test_context, tmp_results):
        """Test workflow with FS background subtraction and baseline correction."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="trim-mean",
            trim_pct=2.0,
            background="fs",
            bgscale="auto",
            baseline=True,
            fit=False,
            export="both",
        )
        
        result = service.process(request)
        
        # Verify result structure
        assert isinstance(result, ServiceResult)
        assert len(result.artifacts) == 3  # CSV, PNG, and JSON metadata
        
        # Find artifacts by extension
        csv_path = [p for p in result.artifacts if p.suffix == ".csv"][0]
        png_path = [p for p in result.artifacts if p.suffix == ".png"][0]
        json_path = [p for p in result.artifacts if p.suffix == ".json"][0]
        
        assert csv_path.exists()
        assert png_path.exists()
        assert json_path.exists()
        
        # Verify CSV structure
        df = pd.read_csv(csv_path)
        assert "raman_shift" in df.columns
        assert "intensity" in df.columns
        
        # Verify metadata
        assert result.metadata["background"] == "fs"
        assert result.metadata["baseline"] is True
        assert "bgscale" in result.metadata
        # Auto-scale for 500 PPP should be 500/900
        assert abs(result.metadata["bgscale"] - 500/900) < 0.01

    def test_full_workflow_with_fitting(self, test_context, tmp_results):
        """Test complete workflow including Gaussian fitting."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="trim-mean",
            trim_pct=2.0,
            background="fs",
            bgscale="auto",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
            export="both",
        )
        
        result = service.process(request)
        
        # Verify result structure
        assert isinstance(result, ServiceResult)
        assert len(result.artifacts) == 3  # CSV, PNG, and JSON metadata
        
        # Verify fit metadata
        assert result.metadata["fit"] is True
        assert "n_peaks" in result.metadata
        assert "r2" in result.metadata
        
        # R² should be between 0 and 1
        assert 0 <= result.metadata["r2"] <= 1.0

    def test_workflow_lake_haiyaha(self, test_context, tmp_results):
        """Test workflow on Lake Haiyaha fixture (pure olivine)."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0852",
            target="Lake_Haiyaha",
            scan="detail_1",
            mode="averaged",
            avg_method="median",
            background="fs",
            baseline=True,
            fit=True,
            export="csv",
        )
        
        result = service.process(request)
        
        # Verify result
        assert isinstance(result, ServiceResult)
        assert len(result.artifacts) == 2  # CSV + JSON
        
        # Verify metadata
        assert result.metadata["n_points"] == 100
        assert result.metadata["ppp"] == 500.0
        assert result.metadata["avg_method"] == "median"

    def test_workflow_stigbreen_line_scan(self, test_context, tmp_results):
        """Test workflow on Stigbreen line scan (900 PPP)."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="1634",
            target="Stigbreen",
            scan="line_1",
            mode="averaged",
            avg_method="trim-mean",
            trim_pct=2.0,
            background="fs",
            bgscale="auto",
            baseline=True,
            fit=False,
            export="csv",
        )
        
        result = service.process(request)
        
        # Verify result
        assert isinstance(result, ServiceResult)
        
        # Verify 900 PPP metadata and scaling
        assert result.metadata["ppp"] == 900.0
        # Auto-scale for 900 PPP should be 900/900 = 1.0
        assert abs(result.metadata["bgscale"] - 1.0) < 0.01

    def test_output_path_structure(self, test_context, tmp_results):
        """Verify outputs go to correct directory structure: results/<target>/plots/"""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
            baseline=True,
            export="both",
        )
        
        result = service.process(request)
        
        # Verify output paths include /plots/ directory
        for artifact in result.artifacts:
            assert "plots" in str(artifact)
            assert str(artifact).endswith(".csv") or str(artifact).endswith(".png") or str(artifact).endswith(".json")
            # Path should be: <results_root>/Amherst_Point/plots/<filename>
            assert artifact.parent.name == "plots"
            assert artifact.parent.parent.name == "Amherst_Point"

    def test_filename_convention_averaged(self, test_context, tmp_results):
        """Verify output filenames follow convention: <sol>_<target>_<scan>_R1_avg-<method>..."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="trim-mean",
            trim_pct=2.0,
            background="fs",
            baseline=True,
            export="csv",
        )
        
        result = service.process(request)
        
        csv_path = result.artifacts[0]
        filename = csv_path.stem  # Filename without extension
        
        # Verify filename components
        assert "0921" in filename
        assert "Amherst_Point" in filename
        assert "detail_1" in filename
        assert "R1" in filename
        assert "avg-2p_trim_mean" in filename
        assert "fs" in filename
        assert "baselined" in filename

    def test_explicit_bgscale_override(self, test_context, tmp_results):
        """Test explicit background scale override."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
            bgscale=0.5,  # Explicit override
            baseline=True,
            export="csv",
        )
        
        result = service.process(request)
        
        # Verify explicit scale was used
        assert result.metadata["bgscale"] == 0.5

    def test_arm_stowed_background(self, test_context, tmp_results):
        """Test workflow with arm stowed (AS) background."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="as",
            baseline=True,
            export="csv",
        )
        
        result = service.process(request)
        
        # Verify AS background was used
        assert result.metadata["background"] == "as"
        assert len(result.artifacts) == 2  # CSV + JSON

    def test_warnings_propagated(self, test_context, tmp_results):
        """Test that fitting warnings are propagated to result."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
            baseline=True,
            fit=True,
            export="csv",
        )
        
        result = service.process(request)
        
        # warnings should be a list (may be empty)
        assert isinstance(result.warnings, list)


class TestAveragedWorkflowEdgeCases:
    """Edge case tests for averaged workflow."""

    def test_no_background_no_baseline(self, test_context, tmp_results):
        """Test workflow with no processing (just averaging)."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="mean",
            background=None,
            baseline=False,
            fit=False,
            export="csv",
        )
        
        result = service.process(request)
        
        # Verify no processing metadata
        assert "background" not in result.metadata
        assert "baseline" not in result.metadata or result.metadata.get("baseline") is None

    def test_png_only_export(self, test_context, tmp_results):
        """Test PNG-only export."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
            baseline=True,
            export="png",
        )
        
        result = service.process(request)
        
        # Should have PNG + JSON
        assert len(result.artifacts) == 2
        png_path = [p for p in result.artifacts if p.suffix == ".png"][0]
        json_path = [p for p in result.artifacts if p.suffix == ".json"][0]
        assert png_path.exists()
        assert json_path.exists()

    def test_different_trim_percentages(self, test_context, tmp_results):
        """Test different trim percentages produce different filenames."""
        service = SpectralService(context=test_context)
        
        # 2% trim
        request_2pct = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="trim-mean",
            trim_pct=2.0,
            export="csv",
        )
        
        # 5% trim
        request_5pct = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="trim-mean",
            trim_pct=5.0,
            export="csv",
        )
        
        result_2pct = service.process(request_2pct)
        result_5pct = service.process(request_5pct)
        
        # Filenames should differ
        assert result_2pct.artifacts[0].stem != result_5pct.artifacts[0].stem
        assert "2p_trim_mean" in result_2pct.artifacts[0].stem
        assert "5p_trim_mean" in result_5pct.artifacts[0].stem

    def test_axis_limits_applied(self, test_context, tmp_results):
        """Test that xlim and ylim are applied to plot (file generated without error)."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
            baseline=True,
            xlim=(700, 1300),
            ylim=(-100, 5000),
            export="png",
        )
        
        result = service.process(request)
        
        # Should generate PNG + JSON without error
        assert len(result.artifacts) == 2
        png_path = [p for p in result.artifacts if p.suffix == ".png"][0]
        assert png_path.exists()

