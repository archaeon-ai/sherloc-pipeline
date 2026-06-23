"""
Enhanced Spectrogram Visualization Pipeline for Detail Scans

Implements bd-17f: Advanced spatial visualization capabilities including:
- Detail scan 10x10 grid layouts
- Heatmaps at specific wavenumbers
- Band ratio maps
- PCA component spatial maps
- PNG and HTML export with interactive features

This module extends the existing spectrogram infrastructure to support
spatial visualization of detailed scans with enhanced analytical views.
"""

import json
import sqlite3
import zlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Union, Any
import uuid
from datetime import datetime, timezone
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LinearSegmentedColormap
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

from sherloc_pipeline.models import SpectralRegion, ProcessingLevel
from sherloc_pipeline.visualization.plotting import configure_matplotlib, apply_plot_config
from sherloc_pipeline.services.base import ServiceResult


@dataclass
class SpectrogramVisualizationConfig:
    """Configuration for enhanced spectrogram visualizations."""

    # Grid layout settings
    grid_width: int = 10
    grid_height: int = 10

    # Heatmap settings
    target_wavenumbers: List[float] = None
    wavenumber_tolerance: float = 10.0  # cm-1 tolerance for band selection

    # Band ratio settings
    band_ratios: List[Dict[str, Any]] = None  # List of {name, num_bands, den_bands}

    # PCA settings
    n_components: int = 5
    standardize_before_pca: bool = True

    # Export settings
    export_png: bool = True
    export_html: bool = True
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300

    def __post_init__(self):
        """Set default values after initialization."""
        if self.target_wavenumbers is None:
            # Common Raman bands for Mars minerals
            self.target_wavenumbers = [
                285,   # Fe-O stretch (hematite)
                412,   # Fe-O stretch (hematite)
                500,   # Si-O bend (olivine)
                610,   # Fe-O stretch (magnetite)
                816,   # Si-O stretch (olivine)
                970,   # SO4 stretch (sulfates)
                1086,  # CO3 stretch (carbonates)
                1320,  # C-H/C-C (organics)
                1600   # C=C stretch (organics)
            ]

        if self.band_ratios is None:
            # Common band ratio indicators
            self.band_ratios = [
                {
                    "name": "Olivine_Index",
                    "numerator": [800, 830],    # Olivine Si-O stretch
                    "denominator": [980, 1020], # Background
                    "description": "Olivine detection ratio"
                },
                {
                    "name": "Carbonate_Index",
                    "numerator": [1080, 1090],  # CO3 stretch
                    "denominator": [1050, 1060], # Background
                    "description": "Carbonate detection ratio"
                },
                {
                    "name": "Organic_Index",
                    "numerator": [1590, 1610],  # C=C stretch
                    "denominator": [1400, 1420], # Background
                    "description": "Organic signature ratio"
                }
            ]


@dataclass
class SpatialSpectralData:
    """Container for spatial spectral data with coordinates."""

    spectra: np.ndarray  # Shape: (n_points, n_wavenumbers)
    wavenumbers: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    scan_name: str
    target: str
    metadata: Dict[str, Any]


class SpectrogramVisualizationPipeline:
    """Enhanced spectrogram visualization pipeline for detail scans."""

    def __init__(self, config: SpectrogramVisualizationConfig = None):
        self.config = config or SpectrogramVisualizationConfig()
        self.plot_config = configure_matplotlib()

    def load_detail_scan_data(self, scan_name: str = None, target: str = None) -> List[SpatialSpectralData]:
        """Load detail scan data with spatial coordinates.

        Args:
            scan_name: Specific scan name to load
            target: Target filter (e.g., 'detail', 'Mars')

        Returns:
            List of SpatialSpectralData objects for detail scans
        """
        conn = sqlite3.connect('./phase.db')

        # Query for detail scans (100 points in 10x10 grid)
        where_conditions = ["s.n_points = 100"]
        params = []

        if scan_name:
            where_conditions.append("s.scan_name = ?")
            params.append(scan_name)

        if target:
            where_conditions.append("s.target LIKE ?")
            params.append(f"%{target}%")

        query = f"""
        SELECT
            s.scan_name, s.target, s.sol_number,
            sp.point_index, sp.x_pixel, sp.y_pixel,
            spec.intensities, spec.region
        FROM scans s
        JOIN scan_points sp ON s.id = sp.scan_id
        JOIN spectra spec ON sp.id = spec.scan_point_id
        WHERE {" AND ".join(where_conditions)}
          AND spec.spectrum_type = 'active'
          AND spec.processing_level = 'raw'
          AND spec.intensities IS NOT NULL
          AND spec.region = 'R1'  -- R1 Raman region (see docs/schema/SPECTRAL_REGIONS.md)
        ORDER BY s.scan_name, sp.point_index
        """

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        print(f"Loaded {len(df)} spectral measurements")

        # Group by scan and process each
        scan_data_list = []
        for scan_name, group in df.groupby('scan_name'):
            try:
                scan_data = self._process_scan_group(group)
                scan_data_list.append(scan_data)
                print(f"Processed scan: {scan_name} ({scan_data.target}) - {len(scan_data.spectra)} points")
            except Exception as e:
                print(f"Warning: Failed to process scan {scan_name}: {e}")

        return scan_data_list

    def _process_scan_group(self, group: pd.DataFrame) -> SpatialSpectralData:
        """Process a single scan's data into SpatialSpectralData."""

        # Extract metadata
        scan_name = group['scan_name'].iloc[0]
        target = group['target'].iloc[0]
        sol_number = group['sol_number'].iloc[0]

        # Process spectra
        spectra_list = []
        x_coords = []
        y_coords = []
        wavenumbers = None

        for _, row in group.iterrows():
            try:
                # Decompress spectrum
                intensities = np.frombuffer(zlib.decompress(row['intensities']), dtype=np.float32)

                # Use point index as coordinates if pixel coords are missing
                x_coord = row['x_pixel'] if pd.notna(row['x_pixel']) else (row['point_index'] % 10)
                y_coord = row['y_pixel'] if pd.notna(row['y_pixel']) else (row['point_index'] // 10)

                spectra_list.append(intensities)
                x_coords.append(x_coord)
                y_coords.append(y_coord)

                # Generate wavenumbers from intensity length (common SHERLOC range)
                if wavenumbers is None:
                    n_channels = len(intensities)
                    wavenumbers = np.linspace(200, 4000, n_channels)  # Typical SHERLOC range

            except Exception as e:
                print(f"Warning: Failed to process point {row['point_index']}: {e}")
                continue

        if not spectra_list:
            raise ValueError(f"No valid spectra found for scan {scan_name}")

        return SpatialSpectralData(
            spectra=np.array(spectra_list),
            wavenumbers=wavenumbers,
            x_coords=np.array(x_coords),
            y_coords=np.array(y_coords),
            scan_name=scan_name,
            target=target,
            metadata={'sol_number': sol_number, 'n_points': len(spectra_list)}
        )

    def create_wavenumber_heatmaps(self, scan_data: SpatialSpectralData,
                                 output_dir: Path) -> List[Path]:
        """Create heatmaps for specific wavenumber ranges.

        Args:
            scan_data: Spatial spectral data
            output_dir: Directory to save heatmaps

        Returns:
            List of paths to generated heatmap files
        """
        output_paths = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for wavenumber in self.config.target_wavenumbers:
            # Find closest wavenumber index
            wn_idx = np.argmin(np.abs(scan_data.wavenumbers - wavenumber))
            actual_wn = scan_data.wavenumbers[wn_idx]

            # Extract intensities at this wavenumber
            intensities = scan_data.spectra[:, wn_idx]

            # Create spatial grid
            fig, ax = plt.subplots(1, 1, figsize=self.config.figure_size)

            # Create 2D grid for heatmap
            grid = np.full((self.config.grid_height, self.config.grid_width), np.nan)
            for i, (x, y, intensity) in enumerate(zip(scan_data.x_coords,
                                                    scan_data.y_coords,
                                                    intensities)):
                if 0 <= x < self.config.grid_width and 0 <= y < self.config.grid_height:
                    grid[int(y), int(x)] = intensity

            # Plot heatmap
            im = ax.imshow(grid, cmap='viridis', interpolation='nearest')
            ax.set_title(f'{scan_data.scan_name} - {wavenumber:.0f} cm⁻¹ (actual: {actual_wn:.1f})')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Intensity (counts)')

            # Apply plot configuration
            config, bbox = apply_plot_config(fig)

            # Save PNG
            if self.config.export_png:
                png_path = output_dir / f"{scan_data.scan_name}_heatmap_{wavenumber:.0f}cm.png"
                fig.savefig(png_path, dpi=config.savefig_dpi, bbox_inches=bbox)
                output_paths.append(png_path)

            plt.close(fig)

        return output_paths

    def create_band_ratio_maps(self, scan_data: SpatialSpectralData,
                             output_dir: Path) -> List[Path]:
        """Create band ratio maps for mineral identification.

        Args:
            scan_data: Spatial spectral data
            output_dir: Directory to save maps

        Returns:
            List of paths to generated ratio map files
        """
        output_paths = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for ratio_config in self.config.band_ratios:
            name = ratio_config['name']
            num_bands = ratio_config['numerator']
            den_bands = ratio_config['denominator']

            # Find wavenumber indices for numerator and denominator bands
            num_mask = (scan_data.wavenumbers >= num_bands[0]) & (scan_data.wavenumbers <= num_bands[1])
            den_mask = (scan_data.wavenumbers >= den_bands[0]) & (scan_data.wavenumbers <= den_bands[1])

            if not np.any(num_mask) or not np.any(den_mask):
                print(f"Warning: Band range not found for {name}")
                continue

            # Calculate band ratios
            numerator = np.mean(scan_data.spectra[:, num_mask], axis=1)
            denominator = np.mean(scan_data.spectra[:, den_mask], axis=1)

            # Avoid division by zero
            denominator = np.maximum(denominator, 1e-6)
            ratios = numerator / denominator

            # Create spatial grid
            fig, ax = plt.subplots(1, 1, figsize=self.config.figure_size)

            grid = np.full((self.config.grid_height, self.config.grid_width), np.nan)
            for x, y, ratio in zip(scan_data.x_coords, scan_data.y_coords, ratios):
                if 0 <= x < self.config.grid_width and 0 <= y < self.config.grid_height:
                    grid[int(y), int(x)] = ratio

            # Plot with custom colormap for ratios
            im = ax.imshow(grid, cmap='RdYlBu_r', interpolation='nearest')
            ax.set_title(f'{scan_data.scan_name} - {name}')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f'Ratio ({num_bands[0]}-{num_bands[1]} / {den_bands[0]}-{den_bands[1]})')

            # Apply plot configuration
            config, bbox = apply_plot_config(fig)

            # Save PNG
            if self.config.export_png:
                png_path = output_dir / f"{scan_data.scan_name}_ratio_{name}.png"
                fig.savefig(png_path, dpi=config.savefig_dpi, bbox_inches=bbox)
                output_paths.append(png_path)

            plt.close(fig)

        return output_paths

    def create_pca_component_maps(self, scan_data: SpatialSpectralData,
                                output_dir: Path) -> Tuple[List[Path], Dict[str, Any]]:
        """Create PCA component spatial maps.

        Args:
            scan_data: Spatial spectral data
            output_dir: Directory to save maps

        Returns:
            Tuple of (output_paths, pca_results)
        """
        output_paths = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data for PCA
        X = scan_data.spectra.copy()

        # Optional standardization
        if self.config.standardize_before_pca:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Fit PCA
        pca = PCA(n_components=self.config.n_components, random_state=42)
        components = pca.fit_transform(X)

        # Create maps for each component
        for i in range(self.config.n_components):
            component_values = components[:, i]

            fig, ax = plt.subplots(1, 1, figsize=self.config.figure_size)

            # Create spatial grid
            grid = np.full((self.config.grid_height, self.config.grid_width), np.nan)
            for x, y, value in zip(scan_data.x_coords, scan_data.y_coords, component_values):
                if 0 <= x < self.config.grid_width and 0 <= y < self.config.grid_height:
                    grid[int(y), int(x)] = value

            # Plot with diverging colormap (since PCA components can be positive/negative)
            im = ax.imshow(grid, cmap='RdBu_r', interpolation='nearest')
            variance_pct = pca.explained_variance_ratio_[i] * 100
            ax.set_title(f'{scan_data.scan_name} - PC{i+1} ({variance_pct:.1f}% variance)')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f'PC{i+1} Score')

            # Apply plot configuration
            config, bbox = apply_plot_config(fig)

            # Save PNG
            if self.config.export_png:
                png_path = output_dir / f"{scan_data.scan_name}_PCA_PC{i+1}.png"
                fig.savefig(png_path, dpi=config.savefig_dpi, bbox_inches=bbox)
                output_paths.append(png_path)

            plt.close(fig)

        # Prepare PCA results
        pca_results = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': float(np.sum(pca.explained_variance_ratio_)),
            'component_loadings': pca.components_.tolist(),
            'wavenumbers': scan_data.wavenumbers.tolist(),
            'n_components': self.config.n_components
        }

        return output_paths, pca_results

    def create_interactive_html(self, scan_data: SpatialSpectralData,
                              output_path: Path,
                              pca_results: Dict[str, Any] = None) -> Path:
        """Create interactive HTML visualization with Plotly.

        Args:
            scan_data: Spatial spectral data
            output_path: Path for HTML output
            pca_results: Optional PCA results for component plots

        Returns:
            Path to generated HTML file
        """
        # Create subplots
        n_plots = 2 + len(self.config.target_wavenumbers[:3])  # Overview + 3 wavenumber heatmaps
        if pca_results:
            n_plots += min(3, self.config.n_components)  # Up to 3 PCA components

        rows = (n_plots + 1) // 2
        fig = make_subplots(
            rows=rows, cols=2,
            subplot_titles=['Scan Overview'] + [f'{wn:.0f} cm⁻¹' for wn in self.config.target_wavenumbers[:3]] +
                          [f'PC{i+1}' for i in range(min(3, self.config.n_components) if pca_results else 0)],
            vertical_spacing=0.08
        )

        plot_idx = 0

        # 1. Scan overview (mean intensity map)
        mean_intensities = np.mean(scan_data.spectra, axis=1)
        grid = np.full((self.config.grid_height, self.config.grid_width), np.nan)
        for x, y, intensity in zip(scan_data.x_coords, scan_data.y_coords, mean_intensities):
            if 0 <= x < self.config.grid_width and 0 <= y < self.config.grid_height:
                grid[int(y), int(x)] = intensity

        row, col = divmod(plot_idx, 2)
        fig.add_trace(
            go.Heatmap(z=grid, colorscale='Viridis', showscale=True,
                      colorbar=dict(x=0.45 if col == 0 else 1.02, len=0.4)),
            row=row+1, col=col+1
        )
        plot_idx += 1

        # 2. Wavenumber heatmaps (first 3)
        for wavenumber in self.config.target_wavenumbers[:3]:
            wn_idx = np.argmin(np.abs(scan_data.wavenumbers - wavenumber))
            intensities = scan_data.spectra[:, wn_idx]

            grid = np.full((self.config.grid_height, self.config.grid_width), np.nan)
            for x, y, intensity in zip(scan_data.x_coords, scan_data.y_coords, intensities):
                if 0 <= x < self.config.grid_width and 0 <= y < self.config.grid_height:
                    grid[int(y), int(x)] = intensity

            row, col = divmod(plot_idx, 2)
            fig.add_trace(
                go.Heatmap(z=grid, colorscale='Viridis', showscale=True,
                          colorbar=dict(x=0.45 if col == 0 else 1.02, len=0.4)),
                row=row+1, col=col+1
            )
            plot_idx += 1

        # 3. PCA component maps (first 3)
        if pca_results:
            X = scan_data.spectra.copy()
            if self.config.standardize_before_pca:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            pca = PCA(n_components=self.config.n_components, random_state=42)
            components = pca.fit_transform(X)

            for i in range(min(3, self.config.n_components)):
                component_values = components[:, i]

                grid = np.full((self.config.grid_height, self.config.grid_width), np.nan)
                for x, y, value in zip(scan_data.x_coords, scan_data.y_coords, component_values):
                    if 0 <= x < self.config.grid_width and 0 <= y < self.config.grid_height:
                        grid[int(y), int(x)] = value

                row, col = divmod(plot_idx, 2)
                fig.add_trace(
                    go.Heatmap(z=grid, colorscale='RdBu', showscale=True,
                              colorbar=dict(x=0.45 if col == 0 else 1.02, len=0.4)),
                    row=row+1, col=col+1
                )
                plot_idx += 1

        # Update layout
        fig.update_layout(
            title=f"Interactive Spectrogram Analysis: {scan_data.scan_name} ({scan_data.target})",
            height=300 * rows,
        )

        # Save HTML
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pyo.plot(fig, filename=str(output_path), auto_open=False)

        return output_path

    def get_git_sha(self) -> str:
        """Get current git commit SHA, or 'unknown' if git is unavailable."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def generate_visualization_suite(self, scan_name: str = None,
                                   target: str = None,
                                   output_dir: Path = None) -> Dict[str, Any]:
        """Generate complete visualization suite for detail scans.

        Args:
            scan_name: Optional specific scan name
            target: Optional target filter
            output_dir: Output directory (default: outputs/spectrograms/)

        Returns:
            Dictionary with generation results and metadata
        """
        if output_dir is None:
            output_dir = Path("outputs/spectrograms")

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Spectrogram Visualization Pipeline ===")
        print(f"Bead: bd-17f")
        print(f"Output directory: {output_dir}")

        # Load data
        scan_data_list = self.load_detail_scan_data(scan_name, target)

        if not scan_data_list:
            raise ValueError("No detail scan data found matching criteria")

        results = {
            "metadata": {
                "schema_version": "1.0.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "code_sha": self.get_git_sha(),
                "config": self.config.__dict__,
                "description": "Enhanced spectrogram visualizations for detail scans"
            },
            "scans_processed": [],
            "output_files": {
                "png_heatmaps": [],
                "png_ratios": [],
                "png_pca": [],
                "html_interactive": []
            }
        }

        # Process each scan
        for scan_data in scan_data_list:
            print(f"\nProcessing: {scan_data.scan_name}")

            scan_dir = output_dir / scan_data.scan_name
            scan_dir.mkdir(exist_ok=True)

            scan_result = {
                "scan_name": scan_data.scan_name,
                "target": scan_data.target,
                "n_points": len(scan_data.spectra),
                "n_wavenumbers": len(scan_data.wavenumbers),
                "wavenumber_range": [float(scan_data.wavenumbers.min()), float(scan_data.wavenumbers.max())]
            }

            # 1. Wavenumber heatmaps
            heatmap_paths = self.create_wavenumber_heatmaps(scan_data, scan_dir / "heatmaps")
            results["output_files"]["png_heatmaps"].extend([str(p) for p in heatmap_paths])
            scan_result["heatmap_files"] = [str(p) for p in heatmap_paths]

            # 2. Band ratio maps
            ratio_paths = self.create_band_ratio_maps(scan_data, scan_dir / "ratios")
            results["output_files"]["png_ratios"].extend([str(p) for p in ratio_paths])
            scan_result["ratio_files"] = [str(p) for p in ratio_paths]

            # 3. PCA component maps
            pca_paths, pca_results = self.create_pca_component_maps(scan_data, scan_dir / "pca")
            results["output_files"]["png_pca"].extend([str(p) for p in pca_paths])
            scan_result["pca_files"] = [str(p) for p in pca_paths]
            scan_result["pca_results"] = pca_results

            # 4. Interactive HTML
            if self.config.export_html:
                html_path = self.create_interactive_html(
                    scan_data,
                    scan_dir / f"{scan_data.scan_name}_interactive.html",
                    pca_results
                )
                results["output_files"]["html_interactive"].append(str(html_path))
                scan_result["html_file"] = str(html_path)

            results["scans_processed"].append(scan_result)
            print(f"  Generated {len(heatmap_paths)} heatmaps, {len(ratio_paths)} ratio maps, {len(pca_paths)} PCA maps")

        # Save results summary
        results_path = output_dir / "visualization_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n=== Visualization Complete ===")
        print(f"Processed {len(scan_data_list)} scans")
        print(f"Generated {len(results['output_files']['png_heatmaps'])} heatmap files")
        print(f"Generated {len(results['output_files']['png_ratios'])} ratio map files")
        print(f"Generated {len(results['output_files']['png_pca'])} PCA map files")
        print(f"Generated {len(results['output_files']['html_interactive'])} interactive HTML files")
        print(f"Results saved to: {results_path}")

        return results


def main():
    """CLI entry point for spectrogram visualization pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Spectrogram Visualization Pipeline")
    parser.add_argument("--scan", help="Specific scan name to process")
    parser.add_argument("--target", help="Target filter (e.g., 'detail', 'Mars')")
    parser.add_argument("--output", help="Output directory", default="outputs/spectrograms")

    args = parser.parse_args()

    config = SpectrogramVisualizationConfig()
    pipeline = SpectrogramVisualizationPipeline(config)

    results = pipeline.generate_visualization_suite(
        scan_name=args.scan,
        target=args.target,
        output_dir=Path(args.output)
    )


if __name__ == "__main__":
    main()