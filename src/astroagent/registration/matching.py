from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from astroagent.errors import PipelineError
from astroagent.execution import checkpoint
from astroagent.registration.stars import StarCatalog
from astroagent.registration.transform import apply_transform, fit_transform


@dataclass
class StarMatches:
    """Unique candidate correspondences after geometric initialization."""

    target: NDArray[np.float64]
    reference: NDArray[np.float64]
    triangle_candidates: int


def _triangles(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    tree = cKDTree(points)
    triangles: set[tuple[int, ...]] = set()
    for i, point in enumerate(points):
        checkpoint()
        indices = tree.query(point, k=min(7, len(points)))[1]
        for j, k in combinations(indices[1:], 2):
            triangles.add(tuple(sorted((i, int(j), int(k)))))
    descriptors, vertices = [], []
    for triangle in sorted(triangles):
        tri = points[list(triangle)]
        sides = np.linalg.norm(tri[[1, 2, 0]] - tri[[2, 0, 1]], axis=1)
        order = np.argsort(sides)
        small, middle, large = sides[order]
        if large < 5 or small / large < 0.15 or (small + middle) / large < 1.05:
            continue
        # Vertex opposite each sorted side fixes correspondence independent of pose.
        descriptors.append([small / large, middle / large])
        vertices.append(np.asarray(triangle)[order])
    return np.asarray(descriptors, dtype=float).reshape(-1, 2), np.asarray(vertices, dtype=np.int64)


def _unique_pairs(
    target: NDArray[np.float64],
    reference: NDArray[np.float64],
    matrix: NDArray[np.float64],
    radius: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], float]:
    transformed = apply_transform(target, matrix)
    distances, neighbors = cKDTree(reference).query(transformed, distance_upper_bound=radius)
    valid = np.flatnonzero(np.isfinite(distances))
    # Greedy shortest matches enforce a one-to-one correspondence.
    valid = valid[np.argsort(distances[valid], kind="stable")]
    used: set[int] = set()
    src, dst = [], []
    for i in valid:
        j = int(neighbors[i])
        if j not in used:
            used.add(j)
            src.append(i)
            dst.append(j)
    return (
        np.asarray(src, dtype=np.int64),
        np.asarray(dst, dtype=np.int64),
        float(np.mean(distances[src] ** 2)) if src else np.inf,
    )


def match_stars(
    reference: StarCatalog,
    target: StarCatalog,
    *,
    max_stars: int = 100,
    match_radius: float = 2,
    triangle_tolerance: float = 0.015,
    trials: int = 500,
    random_seed: int = 0,
    min_scale: float = 0.8,
    max_scale: float = 1.2,
) -> StarMatches:
    """Match local triangle side ratios before any nearest-neighbor pixel matching.

    Invariants tolerate translation, rotation and uniform scale. Similarity
    hypotheses are ranked by unique catalog-wide correspondences. No flux order
    correspondence is assumed; the bright-star cap bounds the search cost.
    """
    ref = np.array([[s.x, s.y] for s in reference.stars[:max_stars]], dtype=float)
    src = np.array([[s.x, s.y] for s in target.stars[:max_stars]], dtype=float)
    if min(len(ref), len(src)) < 3:
        raise PipelineError(f"Only {min(len(ref), len(src))} stars available for matching.")
    rd, rv = _triangles(ref)
    sd, sv = _triangles(src)
    if not len(rd) or not len(sd):
        raise PipelineError("No non-degenerate star triangles available for matching.")
    distances, neighbors = cKDTree(rd).query(sd, k=min(3, len(rd)))
    distances = np.asarray(distances).reshape(len(sd), -1)
    neighbors = np.asarray(neighbors).reshape(len(sd), -1)
    pairs = [
        (i, int(neighbors[i, j]), float(distances[i, j]))
        for i, j in zip(*np.nonzero(distances < triangle_tolerance), strict=True)
    ]
    if not pairs:
        raise PipelineError("No geometrically consistent star triangles were found.")
    candidate_count = len(pairs)
    # Preserve excellent invariant matches and reproducibly sample the remainder.
    pairs.sort(key=lambda p: (p[2], p[0], p[1]))
    if len(pairs) > trials:
        first = min(50, trials)
        rest = np.random.default_rng(random_seed).choice(
            np.arange(first, len(pairs)),
            trials - first,
            replace=False,
        )
        pairs = pairs[:first] + [pairs[int(i)] for i in rest]
    best_src = np.array([], dtype=np.int64)
    best_dst = np.array([], dtype=np.int64)
    best_error = np.inf
    for i, j, _ in pairs:
        try:
            matrix = fit_transform(src[sv[i]], ref[rv[j]])
        except (ValueError, PipelineError, np.linalg.LinAlgError):
            continue
        scale = np.sqrt(np.linalg.det(matrix[:2, :2]))
        if not min_scale <= scale <= max_scale:
            continue
        si, ri, error = _unique_pairs(src, ref, matrix, match_radius)
        if len(si) > len(best_src) or (len(si) == len(best_src) and error < best_error):
            best_src, best_dst, best_error = si, ri, error
    if len(best_src) < 3:
        raise PipelineError(f"Only {len(best_src)} matching stars were found.")
    return StarMatches(src[best_src], ref[best_dst], candidate_count)
