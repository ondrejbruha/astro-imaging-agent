import logging
import warnings

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from astroagent.analysis.background import estimate_surface, luminance
from astroagent.analysis.statistics import StarMetrics
from astroagent.errors import AnalysisError
from astroagent.models.image import AstroImage

logger = logging.getLogger(__name__)


def analyze_stars(
    image: AstroImage, *, threshold_sigma: float = 5.0, fwhm: float = 3.0
) -> StarMetrics:
    """Detect stars with photutils and estimate shapes from positive-weight moments.

    FWHM assumes a Gaussian PSF. Ellipticity is 1-b/a and eccentricity is
    sqrt(1-(b/a)^2). Border detections and degenerate moments are excluded.
    Failed or unmeasurable detection produces null metrics and explicit warnings.
    """
    if (
        not np.isfinite(threshold_sigma)
        or threshold_sigma <= 0
        or not np.isfinite(fwhm)
        or fwhm <= 0
    ):
        raise AnalysisError("Star threshold and FWHM must be positive finite numbers.")
    try:
        data = luminance(image)
        mask = ~np.isfinite(data)
        if mask.all():
            return StarMetrics(warnings=["Star detection requires finite pixel samples."])
        _, median, _ = sigma_clipped_stats(data[~mask])
        try:
            sky = estimate_surface(data, polynomial_degree=1)
        except AnalysisError:
            sky = np.full(data.shape, median)
        residual = data - sky
        _, _, noise = sigma_clipped_stats(residual[~mask])
        if not np.isfinite(noise) or noise <= np.finfo(float).eps * max(1.0, abs(float(median))):
            return StarMetrics(
                star_count=0, warnings=["Noise is too small for reliable star detection."]
            )
        finder = DAOStarFinder(threshold=threshold_sigma * noise, fwhm=fwhm, exclude_border=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sources = finder(np.where(mask, 0, residual), mask=mask)
        messages = list(dict.fromkeys(str(item.message) for item in caught))
        if sources is None or len(sources) == 0:
            return StarMetrics(star_count=0, warnings=messages or ["No stars detected."])
        shapes: list[tuple[float, float, float]] = []
        radius = max(3, int(np.ceil(2 * fwhm)))
        x_column = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
        y_column = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"
        for row in sources:
            x, y = int(round(float(row[x_column]))), int(round(float(row[y_column])))
            if (
                y - radius < 0
                or x - radius < 0
                or y + radius >= data.shape[0]
                or x + radius >= data.shape[1]
            ):
                continue
            cutout = residual[y - radius : y + radius + 1, x - radius : x + radius + 1]
            if not np.isfinite(cutout).all():
                continue
            weights = np.maximum(
                cutout
                - np.median(np.concatenate([cutout[0], cutout[-1], cutout[:, 0], cutout[:, -1]])),
                0,
            )
            yy, xx = np.indices(weights.shape)
            total = weights.sum()
            if total <= 0:
                continue
            dx, dy = xx - (xx * weights).sum() / total, yy - (yy * weights).sum() / total
            covariance = (
                np.array(
                    [
                        [(weights * dx * dx).sum(), (weights * dx * dy).sum()],
                        [(weights * dx * dy).sum(), (weights * dy * dy).sum()],
                    ]
                )
                / total
            )
            minor, major = np.linalg.eigvalsh(covariance)
            if minor <= 0 or major <= 0:
                continue
            ratio = np.sqrt(minor / major)
            shapes.append(
                (
                    float(2.35482 * np.sqrt((major + minor) / 2)),
                    float(1 - ratio),
                    float(np.sqrt(1 - ratio**2)),
                )
            )
        if not shapes:
            return StarMetrics(
                star_count=len(sources),
                warnings=messages + ["Star shapes could not be measured reliably."],
            )
        medians = np.median(shapes, axis=0)
        return StarMetrics(
            star_count=len(sources),
            median_fwhm=float(medians[0]),
            median_ellipticity=float(medians[1]),
            median_eccentricity=float(medians[2]),
            warnings=messages,
        )
    except Exception as exc:
        logger.debug("Star detection failed", exc_info=True)
        return StarMetrics(warnings=[f"Star detection unavailable: {exc}"])
