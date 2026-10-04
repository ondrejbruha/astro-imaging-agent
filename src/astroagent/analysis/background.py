from typing import Any

import numpy as np
from astropy.stats import sigma_clip, sigma_clipped_stats
from numpy.typing import NDArray

from astroagent.analysis.statistics import BackgroundMetrics
from astroagent.errors import AnalysisError
from astroagent.models.image import AstroImage


def luminance(image: AstroImage) -> NDArray[np.float64]:
    """Return mono data or an equal-weight RGB average for detection."""
    data = np.asarray(image.data, dtype=np.float64)
    return data if image.channels == 1 else np.mean(data, axis=2)


def polynomial_terms(degree: int) -> list[tuple[int, int]]:
    """Return all x/y exponents of total degree at most degree."""
    return [(i, j) for total in range(degree + 1) for i in range(total + 1) for j in [total - i]]


def estimate_surface(
    data: NDArray[Any], *, grid_size: int = 8, polynomial_degree: int = 2, sigma: float = 3.0
) -> NDArray[np.float64]:
    """Fit a low-order surface to sigma-clipped tile medians.

    Coordinates are scaled to [-1, 1]. Residual outlier rejection reduces the
    influence of contaminated tiles. A rank-deficient fit raises AnalysisError.
    """
    if data.ndim != 2:
        raise AnalysisError("Background surface requires a two-dimensional plane.")
    if (
        not 2 <= grid_size <= 64
        or not 0 <= polynomial_degree <= 3
        or not np.isfinite(sigma)
        or sigma <= 0
    ):
        raise AnalysisError("Invalid background grid, polynomial degree, or clipping threshold.")
    height, width = data.shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing="ij")
    y_edges = np.unique(np.linspace(0, height, min(grid_size, height) + 1, dtype=int))
    x_edges = np.unique(np.linspace(0, width, min(grid_size, width) + 1, dtype=int))
    positions: list[tuple[float, float]] = []
    levels: list[float] = []
    for y0, y1 in zip(y_edges[:-1], y_edges[1:], strict=True):
        for x0, x1 in zip(x_edges[:-1], x_edges[1:], strict=True):
            tile = data[y0:y1, x0:x1]
            values = tile[np.isfinite(tile)]
            if values.size:
                _, median, _ = sigma_clipped_stats(values, sigma=sigma, maxiters=5)
                positions.append((float(xx[y0:y1, x0:x1].mean()), float(yy[y0:y1, x0:x1].mean())))
                levels.append(float(median))
    terms = polynomial_terms(polynomial_degree)
    if len(levels) < len(terms):
        raise AnalysisError("Not enough valid background tiles for the polynomial degree.")
    coordinates = np.asarray(positions)
    matrix = np.column_stack([coordinates[:, 0] ** i * coordinates[:, 1] ** j for i, j in terms])
    values_array = np.asarray(levels)
    keep = np.ones(len(levels), dtype=bool)
    for _ in range(4):
        coefficients, _, rank, _ = np.linalg.lstsq(matrix[keep], values_array[keep], rcond=None)
        if rank < len(terms):
            raise AnalysisError("Background polynomial fit is rank deficient; use a lower degree.")
        residual = values_array - matrix @ coefficients
        clipped = sigma_clip(residual, sigma=sigma, maxiters=3)
        candidate = ~np.ma.getmaskarray(clipped)
        if np.array_equal(candidate, keep) or candidate.sum() < len(terms):
            break
        if np.linalg.matrix_rank(matrix[candidate]) < len(terms):
            break
        keep = candidate
    # Refit after the final mask update.
    coefficients = np.linalg.lstsq(matrix[keep], values_array[keep], rcond=None)[0]
    surface = np.zeros(data.shape, dtype=np.float64)
    for coefficient, (i, j) in zip(coefficients, terms, strict=True):
        surface += coefficient * xx**i * yy**j
    return surface


def analyze_background(image: AstroImage) -> BackgroundMetrics:
    """Estimate sky and noise, measuring a plane gradient separately from noise."""
    data = luminance(image)
    finite = data[np.isfinite(data)]
    if not finite.size:
        raise AnalysisError("No finite samples available for background analysis.")
    _, median, _ = sigma_clipped_stats(finite, sigma=3.0, maxiters=5)
    warnings: list[str] = []
    try:
        surface = estimate_surface(data, polynomial_degree=1)
        residual = (data - surface)[np.isfinite(data)]
        gradient = float(np.ptp(np.percentile(surface, [5, 95])))
    except AnalysisError as exc:
        residual = finite - median
        gradient = 0.0
        warnings.append(f"Gradient unavailable: {exc}")
    _, _, noise = sigma_clipped_stats(residual, sigma=3.0, maxiters=5)
    return BackgroundMetrics(
        median=float(median), sigma=float(noise), gradient_estimate=gradient, warnings=warnings
    )
