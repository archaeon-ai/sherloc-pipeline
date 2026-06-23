"""
Grain Morphometry Analysis Module.

This module provides comprehensive grain morphometry analysis including:
- Size distribution statistics
- Shape analysis (circularity, aspect ratio, solidity)
- Grain-spectrum linkage (which spectra were measured on which grain)
- Area conversion to physical units (um^2)

Key Features:
- Statistical summaries (mean, median, std, min, max, percentiles)
- Size class distribution
- Grain-spectrum spatial linkage
- Export to JSON and Markdown reports

ACI Image Specifications:
- Resolution: 10.1 um/pixel (default)
- Typical grain count: 30-100 per image

Usage:
    from sherloc_pipeline.vision.morphometry import (
        GrainMorphometryAnalyzer,
        MorphometryStats,
        compute_grain_spectrum_linkage,
    )

    analyzer = GrainMorphometryAnalyzer()
    stats = analyzer.compute_statistics()
    report = analyzer.generate_report()
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from sqlalchemy import text

from sherloc_pipeline.database import get_engine, get_session


logger = logging.getLogger(__name__)


# Default ACI pixel scale
DEFAULT_PIXEL_SCALE_UM = 10.1


@dataclass
class MorphometryStats:
    """Statistics for a set of grain measurements.

    Attributes:
        count: Number of grains
        mean: Mean value
        median: Median value
        std: Standard deviation
        min_val: Minimum value
        max_val: Maximum value
        p25: 25th percentile
        p75: 75th percentile
        p95: 95th percentile
    """
    count: int
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0

    @classmethod
    def from_array(cls, values: np.ndarray) -> "MorphometryStats":
        """Compute statistics from numpy array."""
        if len(values) == 0:
            return cls(0, 0, 0, 0, 0, 0, 0, 0, 0)

        return cls(
            count=len(values),
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            std=float(np.std(values)),
            min_val=float(np.min(values)),
            max_val=float(np.max(values)),
            p25=float(np.percentile(values, 25)),
            p75=float(np.percentile(values, 75)),
            p95=float(np.percentile(values, 95)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "mean": round(self.mean, 2),
            "median": round(self.median, 2),
            "std": round(self.std, 2),
            "min": round(self.min_val, 2),
            "max": round(self.max_val, 2),
            "p25": round(self.p25, 2),
            "p75": round(self.p75, 2),
            "p95": round(self.p95, 2),
        }


@dataclass
class GrainSpectralLink:
    """Link between a grain and overlapping scan points.

    Attributes:
        grain_id: ID of the grain segment
        image_id: ID of the context image
        scan_id: ID of the associated scan
        point_indices: List of scan point indices within this grain
        n_points: Number of overlapping points
        grain_area_px: Grain area in pixels
        grain_centroid: Grain centroid (x, y)
    """
    grain_id: str
    image_id: str
    scan_id: str
    point_indices: List[int]
    n_points: int
    grain_area_px: int
    grain_centroid: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "grain_id": self.grain_id,
            "image_id": self.image_id,
            "scan_id": self.scan_id,
            "point_indices": self.point_indices,
            "n_points": self.n_points,
            "grain_area_px": self.grain_area_px,
            "grain_centroid": list(self.grain_centroid),
        }


@dataclass
class SizeClass:
    """Definition of a grain size class.

    Attributes:
        name: Human-readable name (e.g., "Fine sand")
        min_diameter_um: Minimum equivalent diameter in micrometers
        max_diameter_um: Maximum equivalent diameter in micrometers
        count: Number of grains in this class
        percentage: Percentage of total grains
    """
    name: str
    min_diameter_um: float
    max_diameter_um: float
    count: int = 0
    percentage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "min_diameter_um": self.min_diameter_um,
            "max_diameter_um": self.max_diameter_um,
            "count": self.count,
            "percentage": round(self.percentage, 2),
        }


# Wentworth grain size classes (for geological context)
WENTWORTH_SIZE_CLASSES = [
    SizeClass("Very coarse sand", 1000, 2000),
    SizeClass("Coarse sand", 500, 1000),
    SizeClass("Medium sand", 250, 500),
    SizeClass("Fine sand", 125, 250),
    SizeClass("Very fine sand", 62.5, 125),
    SizeClass("Coarse silt", 31.25, 62.5),
    SizeClass("Medium silt", 15.6, 31.25),
    SizeClass("Fine silt", 7.8, 15.6),
]


class GrainMorphometryAnalyzer:
    """Analyzer for grain morphometry from segmentation results.

    Provides:
    - Size distribution analysis
    - Shape statistics
    - Grain-spectrum linkage
    - Report generation

    Example:
        >>> analyzer = GrainMorphometryAnalyzer()
        >>> stats = analyzer.compute_statistics()
        >>> print(f"Analyzed {stats['total_grains']} grains")
    """

    def __init__(
        self,
        database_path: Optional[Path] = None,
        pixel_scale_um: float = DEFAULT_PIXEL_SCALE_UM,
    ):
        """Initialize the analyzer.

        Args:
            database_path: Path to SQLite database
            pixel_scale_um: Pixel scale in micrometers (default 10.1)
        """
        if database_path is None:
            database_path = Path("./phase.db")

        self.database_path = database_path
        self.pixel_scale_um = pixel_scale_um
        self.engine = get_engine(database_path)

    def _pixels_to_um2(self, area_px: float) -> float:
        """Convert area from pixels to square micrometers."""
        return area_px * (self.pixel_scale_um ** 2)

    def _equivalent_diameter_um(self, area_px: float) -> float:
        """Compute equivalent circular diameter in micrometers."""
        area_um2 = self._pixels_to_um2(area_px)
        return 2 * np.sqrt(area_um2 / np.pi)

    def get_all_grains(self) -> List[Dict[str, Any]]:
        """Retrieve all grain segments from the database.

        Returns:
            List of grain dictionaries with morphometry data
        """
        with get_session(self.engine) as session:
            result = session.execute(text("""
                SELECT
                    gs.id, gs.image_id, gs.segment_index,
                    gs.area_px, gs.perimeter_px, gs.aspect_ratio, gs.circularity,
                    gs.centroid_x, gs.centroid_y, gs.confidence, gs.stability_score,
                    ci.scan_id, ci.pixel_scale_um
                FROM grain_segments gs
                JOIN context_images ci ON gs.image_id = ci.id
                ORDER BY gs.image_id, gs.segment_index
            """)).fetchall()

            grains = []
            for row in result:
                pixel_scale = row[12] or self.pixel_scale_um
                area_px = row[3] or 0

                grains.append({
                    "id": row[0],
                    "image_id": row[1],
                    "segment_index": row[2],
                    "area_px": area_px,
                    "area_um2": self._pixels_to_um2(area_px),
                    "equivalent_diameter_um": self._equivalent_diameter_um(area_px),
                    "perimeter_px": row[4] or 0,
                    "perimeter_um": (row[4] or 0) * pixel_scale,
                    "aspect_ratio": row[5] or 1.0,
                    "circularity": row[6] or 0,
                    "centroid_x": row[7] or 0,
                    "centroid_y": row[8] or 0,
                    "confidence": row[9] or 0,
                    "stability_score": row[10] or 0,
                    "scan_id": row[11],
                })

            return grains

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive morphometry statistics.

        Returns:
            Dictionary with statistics for all morphometric properties
        """
        grains = self.get_all_grains()

        if not grains:
            return {
                "total_grains": 0,
                "images_analyzed": 0,
                "error": "No grain segments found. Run segmentation first.",
            }

        # Extract arrays
        areas_px = np.array([g["area_px"] for g in grains])
        areas_um2 = np.array([g["area_um2"] for g in grains])
        diameters_um = np.array([g["equivalent_diameter_um"] for g in grains])
        perimeters = np.array([g["perimeter_um"] for g in grains])
        aspect_ratios = np.array([g["aspect_ratio"] for g in grains])
        circularities = np.array([g["circularity"] for g in grains])

        # Unique images
        image_ids = set(g["image_id"] for g in grains)

        # Compute statistics
        stats = {
            "total_grains": len(grains),
            "images_analyzed": len(image_ids),
            "grains_per_image": len(grains) / len(image_ids) if image_ids else 0,
            "pixel_scale_um": self.pixel_scale_um,
            "area_px": MorphometryStats.from_array(areas_px).to_dict(),
            "area_um2": MorphometryStats.from_array(areas_um2).to_dict(),
            "equivalent_diameter_um": MorphometryStats.from_array(diameters_um).to_dict(),
            "perimeter_um": MorphometryStats.from_array(perimeters).to_dict(),
            "aspect_ratio": MorphometryStats.from_array(aspect_ratios).to_dict(),
            "circularity": MorphometryStats.from_array(circularities).to_dict(),
        }

        # Size class distribution
        size_classes = self._compute_size_classes(diameters_um)
        stats["size_classes"] = [sc.to_dict() for sc in size_classes]

        return stats

    def _compute_size_classes(self, diameters_um: np.ndarray) -> List[SizeClass]:
        """Classify grains by Wentworth size classes.

        Args:
            diameters_um: Array of equivalent diameters in micrometers

        Returns:
            List of SizeClass objects with counts
        """
        classes = []
        total = len(diameters_um)

        for sc in WENTWORTH_SIZE_CLASSES:
            count = np.sum(
                (diameters_um >= sc.min_diameter_um) &
                (diameters_um < sc.max_diameter_um)
            )
            classes.append(SizeClass(
                name=sc.name,
                min_diameter_um=sc.min_diameter_um,
                max_diameter_um=sc.max_diameter_um,
                count=int(count),
                percentage=(count / total * 100) if total > 0 else 0,
            ))

        return classes

    def compute_grain_spectrum_linkage(
        self,
        image_id: Optional[str] = None,
    ) -> List[GrainSpectralLink]:
        """Link grains to overlapping scan points.

        For each grain, find which scan points fall within its boundaries.
        This enables queries like "what spectra were measured on this grain?"

        Args:
            image_id: Optional image ID to filter by

        Returns:
            List of GrainSpectralLink objects
        """
        links = []

        with get_session(self.engine) as session:
            # Get grains with their mask info
            grain_query = """
                SELECT
                    gs.id, gs.image_id, gs.bbox_x, gs.bbox_y, gs.bbox_width, gs.bbox_height,
                    gs.centroid_x, gs.centroid_y, gs.area_px, gs.mask_rle,
                    ci.scan_id
                FROM grain_segments gs
                JOIN context_images ci ON gs.image_id = ci.id
            """
            if image_id:
                grain_query += f" WHERE gs.image_id = '{image_id}'"

            grains = session.execute(text(grain_query)).fetchall()

            for grain in grains:
                grain_id = grain[0]
                img_id = grain[1]
                bbox_x, bbox_y, bbox_w, bbox_h = grain[2], grain[3], grain[4], grain[5]
                centroid = (grain[6] or 0, grain[7] or 0)
                area_px = grain[8] or 0
                mask_rle = grain[9]
                scan_id = grain[10]

                if not scan_id:
                    continue

                # Get scan points within bounding box
                points_query = text("""
                    SELECT point_index, x_pixel, y_pixel
                    FROM scan_points
                    WHERE scan_id = :scan_id
                    AND x_pixel IS NOT NULL AND y_pixel IS NOT NULL
                    AND x_pixel >= :min_x AND x_pixel <= :max_x
                    AND y_pixel >= :min_y AND y_pixel <= :max_y
                """)

                points = session.execute(points_query, {
                    "scan_id": scan_id,
                    "min_x": bbox_x,
                    "max_x": bbox_x + bbox_w,
                    "min_y": bbox_y,
                    "max_y": bbox_y + bbox_h,
                }).fetchall()

                # For now, use bbox containment
                # Future: decode mask_rle and check precise containment
                point_indices = [p[0] for p in points]

                if point_indices:
                    links.append(GrainSpectralLink(
                        grain_id=grain_id,
                        image_id=img_id,
                        scan_id=scan_id,
                        point_indices=point_indices,
                        n_points=len(point_indices),
                        grain_area_px=area_px,
                        grain_centroid=centroid,
                    ))

        return links

    def update_grain_linkage_in_db(self) -> int:
        """Update grain segments with linked point indices.

        Returns:
            Number of grains updated
        """
        links = self.compute_grain_spectrum_linkage()
        updated = 0

        with get_session(self.engine) as session:
            for link in links:
                session.execute(text("""
                    UPDATE grain_segments
                    SET linked_point_indices = :indices
                    WHERE id = :grain_id
                """), {
                    "grain_id": link.grain_id,
                    "indices": json.dumps(link.point_indices),
                })
                updated += 1

            session.commit()

        logger.info(f"Updated {updated} grains with spectral linkage")
        return updated

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate a comprehensive morphometry report.

        Args:
            output_path: Optional path to save the report

        Returns:
            Markdown report string
        """
        stats = self.compute_statistics()
        links = self.compute_grain_spectrum_linkage()

        # Build report
        lines = [
            "# Grain Morphometry Analysis Report",
            "",
            f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Bead:** bd-1ct - WS4-C: Grain Morphometry Analysis",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
        ]

        if stats.get("error"):
            lines.append(f"**Error:** {stats['error']}")
            lines.append("")
            lines.append("No grain segments found in database. Please run the segmentation service first:")
            lines.append("```python")
            lines.append("from sherloc_pipeline.services.segmentation import SegmentationService")
            lines.append("service = SegmentationService()")
            lines.append("result = service.process_all_images()")
            lines.append("```")
        else:
            lines.extend([
                f"- **Total Grains Analyzed:** {stats['total_grains']:,}",
                f"- **Images Analyzed:** {stats['images_analyzed']}",
                f"- **Average Grains per Image:** {stats['grains_per_image']:.1f}",
                f"- **Pixel Scale:** {stats['pixel_scale_um']} um/pixel",
                "",
                "---",
                "",
                "## 1. Size Distribution",
                "",
                "### 1.1 Area Statistics",
                "",
                "| Metric | Pixels | um^2 |",
                "|--------|--------|------|",
                f"| Mean | {stats['area_px']['mean']:,.0f} | {stats['area_um2']['mean']:,.0f} |",
                f"| Median | {stats['area_px']['median']:,.0f} | {stats['area_um2']['median']:,.0f} |",
                f"| Std Dev | {stats['area_px']['std']:,.0f} | {stats['area_um2']['std']:,.0f} |",
                f"| Min | {stats['area_px']['min']:,.0f} | {stats['area_um2']['min']:,.0f} |",
                f"| Max | {stats['area_px']['max']:,.0f} | {stats['area_um2']['max']:,.0f} |",
                f"| P25 | {stats['area_px']['p25']:,.0f} | {stats['area_um2']['p25']:,.0f} |",
                f"| P75 | {stats['area_px']['p75']:,.0f} | {stats['area_um2']['p75']:,.0f} |",
                f"| P95 | {stats['area_px']['p95']:,.0f} | {stats['area_um2']['p95']:,.0f} |",
                "",
                "### 1.2 Equivalent Diameter (um)",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Mean | {stats['equivalent_diameter_um']['mean']:.1f} |",
                f"| Median | {stats['equivalent_diameter_um']['median']:.1f} |",
                f"| Std Dev | {stats['equivalent_diameter_um']['std']:.1f} |",
                f"| Min | {stats['equivalent_diameter_um']['min']:.1f} |",
                f"| Max | {stats['equivalent_diameter_um']['max']:.1f} |",
                "",
                "### 1.3 Wentworth Size Classes",
                "",
                "| Size Class | Range (um) | Count | Percentage |",
                "|------------|------------|-------|------------|",
            ])

            for sc in stats['size_classes']:
                lines.append(
                    f"| {sc['name']} | {sc['min_diameter_um']:.1f} - {sc['max_diameter_um']:.1f} | "
                    f"{sc['count']:,} | {sc['percentage']:.1f}% |"
                )

            lines.extend([
                "",
                "---",
                "",
                "## 2. Shape Analysis",
                "",
                "### 2.1 Circularity",
                "",
                "Circularity = 4*pi*area / perimeter^2 (1.0 = perfect circle)",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Mean | {stats['circularity']['mean']:.3f} |",
                f"| Median | {stats['circularity']['median']:.3f} |",
                f"| Std Dev | {stats['circularity']['std']:.3f} |",
                f"| Min | {stats['circularity']['min']:.3f} |",
                f"| Max | {stats['circularity']['max']:.3f} |",
                "",
                "### 2.2 Aspect Ratio",
                "",
                "Aspect Ratio = major axis / minor axis (1.0 = equant)",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Mean | {stats['aspect_ratio']['mean']:.2f} |",
                f"| Median | {stats['aspect_ratio']['median']:.2f} |",
                f"| Std Dev | {stats['aspect_ratio']['std']:.2f} |",
                f"| Min | {stats['aspect_ratio']['min']:.2f} |",
                f"| Max | {stats['aspect_ratio']['max']:.2f} |",
                "",
                "---",
                "",
                "## 3. Grain-Spectrum Linkage",
                "",
                f"**Total Links Found:** {len(links)}",
                "",
            ])

            if links:
                # Summary by number of points
                points_per_grain = [link.n_points for link in links]
                lines.extend([
                    f"- Grains with spectra: {len(links)}",
                    f"- Average points per grain: {np.mean(points_per_grain):.1f}",
                    f"- Max points in a grain: {max(points_per_grain)}",
                    "",
                    "### Sample Linkages",
                    "",
                    "| Grain ID | Image ID | Points | Area (px) |",
                    "|----------|----------|--------|-----------|",
                ])

                for link in links[:10]:  # Show first 10
                    lines.append(
                        f"| {link.grain_id[:8]}... | {link.image_id[:8]}... | "
                        f"{link.n_points} | {link.grain_area_px:,} |"
                    )
            else:
                lines.append("No grain-spectrum links found. Run segmentation first.")

            # Get model info from database
            with get_session(self.engine) as session:
                model_counts = session.execute(text("""
                    SELECT model_name, COUNT(*) FROM grain_segments GROUP BY model_name
                """)).fetchall()
                models_used = {row[0]: row[1] for row in model_counts}

            lines.extend([
                "",
                "---",
                "",
                "## 4. Methodology",
                "",
                "### 4.1 Segmentation",
                "",
            ])

            if "watershed" in models_used:
                lines.extend([
                    "Grain segmentation performed using **traditional watershed algorithm**:",
                    "- Gaussian smoothing (sigma=1.5)",
                    "- Otsu thresholding (factor=0.8)",
                    "- Distance transform with local maxima markers",
                    "- Minimum mask area: 100 pixels",
                    "",
                    "**Note:** Watershed tends to over-segment compared to SAM. For production use,",
                    "SAM ViT-B is recommended (see `docs/research/SAM_EVALUATION.md`).",
                ])
            else:
                lines.extend([
                    "Grain segmentation performed using SAM (Segment Anything Model) ViT-B:",
                    "- Points per side: 32",
                    "- IoU threshold: 0.86",
                    "- Stability threshold: 0.92",
                    "- Minimum mask area: 100 pixels",
                ])

            lines.extend([
                "",
                f"**Models used:** {', '.join(f'{k} ({v} grains)' for k, v in models_used.items())}",
                "",
                "### 4.2 Morphometry Computation",
                "",
                "Morphometric properties computed using scikit-image:",
                "- Area: pixel count within mask",
                "- Perimeter: contour length",
                "- Circularity: 4*pi*area / perimeter^2",
                "- Aspect ratio: major axis / minor axis from fitted ellipse",
                "- Equivalent diameter: diameter of circle with same area",
                "",
                "### 4.3 Grain-Spectrum Linkage",
                "",
                "Scan points linked to grains by spatial containment:",
                "- Points within grain bounding box included",
                "- Future: precise mask containment check",
                "",
                "---",
                "",
                "## 5. Key Findings",
                "",
            ])

            # Generate key findings
            if stats['total_grains'] > 0:
                dominant_class = max(stats['size_classes'], key=lambda x: x['count'])
                lines.extend([
                    f"1. **Dominant grain size:** {dominant_class['name']} ({dominant_class['percentage']:.1f}% of grains)",
                    f"2. **Mean equivalent diameter:** {stats['equivalent_diameter_um']['mean']:.1f} um",
                    f"3. **Mean circularity:** {stats['circularity']['mean']:.3f} (higher = more circular)",
                    f"4. **Mean aspect ratio:** {stats['aspect_ratio']['mean']:.2f} (higher = more elongated)",
                    "",
                ])

            lines.extend([
                "---",
                "",
                "## References",
                "",
                "- Wentworth, C.K. (1922). A scale of grade and class terms for clastic sediments. Journal of Geology.",
                "- Kirillov, A. et al. (2023). Segment Anything. ICCV 2023.",
                "",
            ])

        report = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Report saved to {output_path}")

        return report

    def export_statistics_json(self, output_path: Path) -> None:
        """Export statistics to JSON file.

        Args:
            output_path: Path for JSON output
        """
        stats = self.compute_statistics()
        links = self.compute_grain_spectrum_linkage()

        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "statistics": stats,
            "grain_spectrum_links": [link.to_dict() for link in links[:100]],  # First 100
            "total_links": len(links),
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))

        logger.info(f"Statistics exported to {output_path}")


def compute_grain_spectrum_linkage(
    database_path: Optional[Path] = None,
    image_id: Optional[str] = None,
) -> List[GrainSpectralLink]:
    """Convenience function to compute grain-spectrum linkage.

    Args:
        database_path: Path to database
        image_id: Optional image ID to filter by

    Returns:
        List of GrainSpectralLink objects
    """
    analyzer = GrainMorphometryAnalyzer(database_path)
    return analyzer.compute_grain_spectrum_linkage(image_id)


def analyze_morphometry(
    database_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convenience function to run full morphometry analysis.

    Args:
        database_path: Path to database
        output_dir: Optional directory for output files

    Returns:
        Dictionary with analysis results
    """
    analyzer = GrainMorphometryAnalyzer(database_path)
    stats = analyzer.compute_statistics()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate report
        report_path = output_dir / "GRAIN_MORPHOMETRY.md"
        analyzer.generate_report(report_path)

        # Export JSON
        json_path = output_dir / "morphometry_stats.json"
        analyzer.export_statistics_json(json_path)

        stats["report_path"] = str(report_path)
        stats["json_path"] = str(json_path)

    return stats
