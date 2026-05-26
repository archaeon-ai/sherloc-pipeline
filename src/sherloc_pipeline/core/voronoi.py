"""Voronoi tessellation with alpha-shape boundary for Map Mode overlays.

Computes Voronoi geometry from scan point coordinates and clips infinite
regions to an alpha-shape (concave hull) boundary. Provides edge_mask
flagging for cells that extend beyond the measured area.

Used by the Map Mode spatial overlay system (MAP_MODE_SPEC.md SS2.4).
Geometry modes:
  - Voronoi (continuous): tessellated polygons provide gap-free coverage
  - Ring (point): circles at measurement locations (fallback for degenerate cases)

The alpha-shape is computed via scipy.spatial.Delaunay + circumradius filtering
(no external alphashape package). Alpha parameter defaults to 2x median
inter-point distance per spec.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import Voronoi, Delaunay

logger = logging.getLogger(__name__)


@dataclass
class VoronoiResult:
    """Voronoi tessellation with alpha-shape boundary.

    All list outputs use plain Python types (not numpy) for JSON serialization.

    Attributes:
        vertices: All Voronoi vertices as [[x, y], ...].
        regions: Per-point vertex index lists (parallel to input points).
        boundary: Alpha-shape boundary polygon as [[x, y], ...].
        edge_mask: Per-point flag: True if cell extends beyond alpha-shape.
        pixel_scale_um: Physical scale for scale bar rendering (e.g., 10.1 um/px for ACI).
    """

    vertices: list[list[float]]
    regions: list[list[int]]
    boundary: list[list[float]]
    edge_mask: list[bool]
    pixel_scale_um: float | None = None


def compute_voronoi_geometry(
    points: np.ndarray,
    alpha_factor: float = 2.0,
    pixel_scale_um: float | None = None,
) -> VoronoiResult | None:
    """Compute Voronoi tessellation with alpha-shape boundary clipping.

    Args:
        points: (N, 2) array of ACI pixel coordinates.
        alpha_factor: Multiplier on median inter-point distance for alpha parameter.
            Default 2.0 per MAP_MODE_SPEC.md SS2.4.
        pixel_scale_um: Physical scale for scale bar (e.g., 10.1 um/px for ACI).

    Returns:
        VoronoiResult with clipped regions and boundary, or None if degenerate
        (fewer than 3 non-collinear points -- use ring mode instead).
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        logger.warning("Points must be (N, 2) array, got shape %s", points.shape)
        return None

    # --- Deduplicate ---
    unique_points, inverse = _deduplicate(points)
    n_unique = len(unique_points)

    if n_unique < 3:
        logger.info(
            "Fewer than 3 unique points (%d) -- degenerate, ring mode only",
            n_unique,
        )
        return None

    # --- Collinearity check ---
    if _is_collinear(unique_points):
        logger.info("All points are collinear -- degenerate, ring mode only")
        return None

    # --- Voronoi tessellation ---
    vor = Voronoi(unique_points)

    # --- Alpha-shape boundary ---
    median_dist = _median_interpoint_distance(unique_points)
    if median_dist <= 0:
        logger.warning("Median inter-point distance is zero -- degenerate")
        return None

    alpha = 1.0 / (alpha_factor * median_dist)
    boundary = _compute_alpha_shape(unique_points, alpha)

    if boundary is None or len(boundary) < 3:
        logger.warning(
            "Alpha-shape computation failed (alpha=%.6f) -- trying with relaxed alpha",
            alpha,
        )
        # Relax alpha by 4x the median distance
        alpha_relaxed = 1.0 / (4.0 * alpha_factor * median_dist)
        boundary = _compute_alpha_shape(unique_points, alpha_relaxed)
        if boundary is None or len(boundary) < 3:
            logger.warning("Alpha-shape failed even with relaxed alpha -- degenerate")
            return None

    # --- Clip infinite regions ---
    bbox = _alpha_shape_bbox(boundary, margin_factor=0.5, median_dist=median_dist)
    finite_regions = _clip_infinite_regions(vor, bbox)

    # --- Build per-input-point region mapping ---
    # vor.point_region maps unique_point index -> region index in vor.regions
    # inverse maps original point index -> unique_point index
    regions_per_point: list[list[int]] = []
    for orig_idx in range(len(points)):
        unique_idx = inverse[orig_idx]
        region_idx = vor.point_region[unique_idx]
        regions_per_point.append(finite_regions[region_idx])

    # --- Edge mask ---
    all_vertices = vor.vertices.tolist()
    edge_mask = _compute_edge_mask(all_vertices, regions_per_point, boundary)

    # --- Convert to plain Python types ---
    vertices_out = [[float(v[0]), float(v[1])] for v in all_vertices]
    regions_out = [[int(i) for i in r] for r in regions_per_point]
    boundary_out = [[float(b[0]), float(b[1])] for b in boundary]
    edge_mask_out = [bool(e) for e in edge_mask]

    return VoronoiResult(
        vertices=vertices_out,
        regions=regions_out,
        boundary=boundary_out,
        edge_mask=edge_mask_out,
        pixel_scale_um=pixel_scale_um,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _deduplicate(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove duplicate points, returning unique points and inverse mapping.

    Returns:
        (unique_points, inverse) where inverse[i] is the index into unique_points
        for the i-th original point.
    """
    _, idx, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    unique_points = points[np.sort(idx)]
    # np.unique sorts lexicographically; rebuild inverse for the sorted unique array
    sorted_idx = np.sort(idx)
    # Map from original unique ordering to sorted ordering
    reorder = np.argsort(idx)
    inverse_reordered = reorder[inverse]
    return unique_points, inverse_reordered


def _is_collinear(points: np.ndarray) -> bool:
    """Check if all points are collinear via matrix rank of centered points."""
    centered = points - points.mean(axis=0)
    rank = np.linalg.matrix_rank(centered, tol=1e-10)
    return rank < 2


def _median_interpoint_distance(points: np.ndarray) -> float:
    """Compute median nearest-neighbor distance among points."""
    from scipy.spatial import KDTree

    tree = KDTree(points)
    # Query 2 nearest neighbors (first is self)
    dists, _ = tree.query(points, k=2)
    nn_dists = dists[:, 1]  # distance to nearest neighbor (not self)
    return float(np.median(nn_dists))


def _circumradius(tri_points: np.ndarray) -> float:
    """Compute circumradius of a triangle given its 3 vertices.

    Uses R = (a * b * c) / (4 * area) where a, b, c are edge lengths.

    Args:
        tri_points: (3, 2) array of triangle vertex coordinates.

    Returns:
        Circumradius (float). Returns inf for degenerate (zero-area) triangles.
    """
    a = np.linalg.norm(tri_points[1] - tri_points[0])
    b = np.linalg.norm(tri_points[2] - tri_points[1])
    c = np.linalg.norm(tri_points[0] - tri_points[2])

    # Signed area via cross product
    area = 0.5 * abs(
        (tri_points[1][0] - tri_points[0][0]) * (tri_points[2][1] - tri_points[0][1])
        - (tri_points[2][0] - tri_points[0][0]) * (tri_points[1][1] - tri_points[0][1])
    )

    if area < 1e-15:
        return float("inf")

    return (a * b * c) / (4.0 * area)


def _compute_alpha_shape(
    points: np.ndarray, alpha: float
) -> Optional[np.ndarray]:
    """Compute alpha-shape (concave hull) boundary via Delaunay + circumradius filter.

    Steps:
        1. Compute Delaunay triangulation
        2. Keep triangles where circumradius < 1/alpha
        3. Extract boundary edges (edges in exactly one remaining triangle)
        4. Order boundary edges into a polygon

    Args:
        points: (N, 2) array of coordinates.
        alpha: Alpha parameter (1 / (alpha_factor * median_distance)).

    Returns:
        (M, 2) array of ordered boundary polygon vertices, or None if degenerate.
    """
    tri = Delaunay(points)
    threshold = 1.0 / alpha  # circumradius threshold

    # Filter simplices by circumradius
    kept_edges: dict[tuple[int, int], int] = {}  # edge -> count

    for simplex in tri.simplices:
        tri_pts = points[simplex]
        cr = _circumradius(tri_pts)
        if cr < threshold:
            # Add edges of this triangle
            for i in range(3):
                edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
                kept_edges[edge] = kept_edges.get(edge, 0) + 1

    # Boundary edges appear in exactly one kept triangle
    boundary_edges = [e for e, count in kept_edges.items() if count == 1]

    if len(boundary_edges) < 3:
        return None

    # Order boundary edges into a polygon
    ordered = _order_boundary_edges(boundary_edges, points)
    if ordered is None:
        return None

    return points[ordered]


def _order_boundary_edges(
    edges: list[tuple[int, int]],
    points: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Order boundary edges into a closed polygon by walking the edge chain.

    Args:
        edges: List of (vertex_a, vertex_b) pairs forming the boundary.
        points: (N, 2) coordinate array for angular ordering at branch points.

    Returns:
        Array of vertex indices in polygon order, or None if chain is broken.
    """
    if not edges:
        return None

    # Build adjacency: vertex -> list of connected vertices
    adjacency: dict[int, list[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    # Verify all vertices have degree 2 for a simple closed polygon
    # Some alpha shapes can produce branching; handle the largest ring
    for v, neighbors in adjacency.items():
        if len(neighbors) != 2:
            # Non-manifold boundary -- try to extract largest ring
            return _extract_largest_ring(edges, adjacency, points)

    # Walk the chain
    start = edges[0][0]
    ordered = [start]
    prev = start
    current = adjacency[start][0]

    while current != start:
        ordered.append(current)
        neighbors = adjacency[current]
        # Pick the neighbor that isn't the one we came from
        next_v = neighbors[0] if neighbors[1] == prev else neighbors[1]
        prev = current
        current = next_v

        if len(ordered) > len(adjacency) + 1:
            # Safety: avoid infinite loop
            logger.warning("Edge chain walk exceeded expected length")
            return None

    return np.array(ordered, dtype=int)


def _extract_largest_ring(
    edges: list[tuple[int, int]],
    adjacency: dict[int, list[int]],
    points: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Extract the outer boundary ring from a non-manifold boundary graph.

    Uses planar face traversal: at each vertex, edges are sorted by angle.
    Walking always takes the "next clockwise" edge from the incoming direction,
    which traces one face of the planar subdivision. We enumerate all faces
    and return the largest (the outer boundary).

    Args:
        edges: Boundary edges as (vertex_a, vertex_b) pairs.
        adjacency: Vertex -> neighbor list mapping.
        points: (N, 2) coordinate array for angular ordering.
            Required for correct face traversal.

    Returns:
        Array of vertex indices in polygon order, or None if no ring found.
    """
    if points is None:
        return None

    # Build angular-sorted adjacency: for each vertex, sort neighbors by angle
    sorted_adj: dict[int, list[int]] = {}
    for v, nbrs in adjacency.items():
        angles = []
        for nbr in nbrs:
            dx = points[nbr][0] - points[v][0]
            dy = points[nbr][1] - points[v][1]
            angles.append(np.arctan2(dy, dx))
        order = np.argsort(angles)
        sorted_adj[v] = [nbrs[i] for i in order]

    # Build reverse index: for each vertex v and neighbor u,
    # what is the position of u in sorted_adj[v]?
    pos_in_adj: dict[tuple[int, int], int] = {}
    for v, nbrs in sorted_adj.items():
        for i, u in enumerate(nbrs):
            pos_in_adj[(v, u)] = i

    # Enumerate all faces via half-edge traversal.
    # A directed edge (u -> v) belongs to exactly one face.
    # The next edge in the face: at v, find the position of u in sorted_adj[v],
    # then take the PREVIOUS entry (clockwise next) as the next outgoing neighbor w.
    # This traces the face to the LEFT of directed edge (u -> v).
    used_half_edges: set[tuple[int, int]] = set()
    faces: list[list[int]] = []

    for a, b in edges:
        for u, v in [(a, b), (b, a)]:
            if (u, v) in used_half_edges:
                continue

            face = []
            cu, cv = u, v
            for _ in range(len(edges) * 2 + 1):
                if (cu, cv) in used_half_edges:
                    break
                used_half_edges.add((cu, cv))
                face.append(cu)

                # At cv, find cu in the sorted neighbor list
                idx = pos_in_adj.get((cv, cu))
                if idx is None:
                    break
                nbrs = sorted_adj[cv]
                # Next clockwise neighbor: one position BEFORE cu in the sorted list
                next_idx = (idx - 1) % len(nbrs)
                w = nbrs[next_idx]

                cu, cv = cv, w

                if cu == u and cv == v:
                    # Closed the face
                    break

            if len(face) >= 3 and face[0] == cu and v == cv:
                # Verify it closed properly
                faces.append(face)

    if not faces:
        return None

    # The outer boundary is the longest face
    longest_face = max(faces, key=len)

    if len(longest_face) < 3:
        return None

    return np.array(longest_face, dtype=int)


def _alpha_shape_bbox(
    boundary: np.ndarray,
    margin_factor: float = 0.5,
    median_dist: float = 1.0,
) -> tuple[float, float, float, float]:
    """Compute bounding box around alpha-shape with margin for clipping infinite cells.

    Returns:
        (x_min, y_min, x_max, y_max) with margin.
    """
    bmin = boundary.min(axis=0)
    bmax = boundary.max(axis=0)
    extent = bmax - bmin
    # Margin is the larger of margin_factor * extent or 2 * median_dist
    margin = np.maximum(margin_factor * extent, 2.0 * median_dist)
    return (
        float(bmin[0] - margin[0]),
        float(bmin[1] - margin[1]),
        float(bmax[0] + margin[0]),
        float(bmax[1] + margin[1]),
    )


def _clip_infinite_regions(
    vor: Voronoi,
    bbox: tuple[float, float, float, float],
) -> list[list[int]]:
    """Convert Voronoi regions to finite polygons, clipping infinite regions to bbox.

    For regions with -1 (infinite) vertex indices, project the infinite ridge
    directions to the bounding box to create finite polygon vertices.

    Args:
        vor: scipy Voronoi result.
        bbox: (x_min, y_min, x_max, y_max) clipping boundary.

    Returns:
        List of regions, each a list of vertex indices (all >= 0).
        New vertices from clipping are appended to the implicit vertex list.
    """
    x_min, y_min, x_max, y_max = bbox
    center = vor.points.mean(axis=0)

    # We may need to add new vertices for clipped infinite ridges.
    # Track them as an extension of vor.vertices.
    new_vertices: list[np.ndarray] = []
    next_idx = len(vor.vertices)

    # Build a map: (point_idx, point_idx) -> list of ridge vertex pairs
    # This lets us find the infinite ridge directions for each region.
    ridge_map: dict[tuple[int, int], list[int]] = {}
    for ridge_points, ridge_verts in zip(vor.ridge_points, vor.ridge_vertices):
        key = tuple(sorted(ridge_points))
        ridge_map[key] = ridge_verts

    finite_regions: list[list[int]] = []

    for region_indices in vor.regions:
        if not region_indices:
            # Empty region (can happen for points at infinity)
            finite_regions.append([])
            continue

        if -1 not in region_indices:
            # Already finite
            finite_regions.append(list(region_indices))
            continue

        # Region has infinite vertex(es) -- need to clip
        # Find the point that owns this region
        point_idx = None
        for pi, ri in enumerate(vor.point_region):
            if ri == len(finite_regions):
                point_idx = pi
                break

        # Strategy: build the finite polygon from ridge data
        # Collect all finite vertices from this region
        finite_verts = [v for v in region_indices if v >= 0]

        # Find ridges that border this region and have infinite vertices
        for ridge_points, ridge_verts in zip(vor.ridge_points, vor.ridge_vertices):
            if point_idx is not None and point_idx not in ridge_points:
                continue

            if -1 not in ridge_verts:
                continue

            # This is an infinite ridge for our region
            finite_v = ridge_verts[0] if ridge_verts[1] == -1 else ridge_verts[1]
            finite_pt = vor.vertices[finite_v]

            # Direction: perpendicular to the ridge_points midpoint -> center direction
            p0, p1 = vor.points[ridge_points[0]], vor.points[ridge_points[1]]
            midpoint = 0.5 * (p0 + p1)
            tangent = p1 - p0
            # Normal to the line between the two generating points
            normal = np.array([-tangent[1], tangent[0]])
            # Orient away from center of all points
            if np.dot(normal, midpoint - center) < 0:
                normal = -normal
            normal = normal / (np.linalg.norm(normal) + 1e-15)

            # Project to bbox
            far_dist = max(x_max - x_min, y_max - y_min) * 2
            far_point = finite_pt + normal * far_dist

            # Clip to bbox
            far_point[0] = np.clip(far_point[0], x_min, x_max)
            far_point[1] = np.clip(far_point[1], y_min, y_max)

            new_vertices.append(far_point)
            finite_verts.append(next_idx)
            next_idx += 1

        # Order the vertices by angle around the region centroid
        if finite_verts:
            all_v = np.array(
                [
                    vor.vertices[v] if v < len(vor.vertices) else new_vertices[v - len(vor.vertices)]
                    for v in finite_verts
                ]
            )
            centroid = all_v.mean(axis=0)
            angles = np.arctan2(all_v[:, 1] - centroid[1], all_v[:, 0] - centroid[0])
            order = np.argsort(angles)
            finite_verts = [finite_verts[i] for i in order]

        finite_regions.append(finite_verts)

    # Append new vertices to vor.vertices for downstream use
    if new_vertices:
        vor.vertices = np.vstack([vor.vertices, np.array(new_vertices)])

    return finite_regions


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Test if a point is inside a polygon using ray casting algorithm.

    Args:
        point: (2,) array [x, y].
        polygon: (M, 2) array of polygon vertices (ordered, closed implicitly).

    Returns:
        True if point is inside the polygon.
    """
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def _compute_edge_mask(
    vertices: list[list[float]],
    regions: list[list[int]],
    boundary: np.ndarray,
) -> list[bool]:
    """Compute per-point edge mask: True if any vertex of the cell is outside the alpha-shape.

    Args:
        vertices: All Voronoi vertices as [[x, y], ...].
        regions: Per-point vertex index lists.
        boundary: (M, 2) alpha-shape boundary polygon.

    Returns:
        List of booleans, one per point.
    """
    vertices_arr = np.array(vertices)
    mask = []

    for region in regions:
        if not region:
            mask.append(True)
            continue

        is_edge = False
        for vi in region:
            if vi < 0 or vi >= len(vertices_arr):
                is_edge = True
                break
            if not _point_in_polygon(vertices_arr[vi], boundary):
                is_edge = True
                break

        mask.append(is_edge)

    return mask
