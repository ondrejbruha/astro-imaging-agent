import logging

import numpy as np
from pydantic import ValidationError

from astroagent.analysis.statistics import StarMetrics
from astroagent.errors import AnalysisError
from astroagent.models.image import AstroImage
from astroagent.registration.stars import DetectionParams, detect_stars

logger = logging.getLogger(__name__)


def analyze_stars(
    image: AstroImage, *, threshold_sigma: float = 5.0, fwhm: float = 3.0
) -> StarMetrics:
    """Summarize the shared photutils catalog using approximate Gaussian PSF moments.

    Ellipticity is 1-b/a, eccentricity sqrt(1-(b/a)^2). Degenerate, saturated,
    border and masked sources are excluded. Detection failure returns null
    measurements and a warning; invalid caller parameters raise AnalysisError.
    """
    try:
        params = DetectionParams(threshold_sigma=threshold_sigma, fwhm=fwhm, max_stars=100000)
    except ValidationError as exc:
        raise AnalysisError("Star threshold and FWHM must be positive finite numbers.") from exc
    if not np.isfinite(image.data).any():
        return StarMetrics(warnings=["Star detection requires finite pixel samples."])
    try:
        catalog = detect_stars(image, params)
        widths = [s.fwhm for s in catalog.stars if s.fwhm is not None]
        eccentricities = [s.eccentricity for s in catalog.stars if s.eccentricity is not None]
        ellipticities = [1 - np.sqrt(1 - value**2) for value in eccentricities]
        return StarMetrics(
            star_count=len(catalog.stars),
            median_fwhm=float(np.median(widths)) if widths else None,
            median_eccentricity=float(np.median(eccentricities)) if eccentricities else None,
            median_ellipticity=float(np.median(ellipticities)) if ellipticities else None,
            warnings=catalog.warnings,
        )
    except Exception as exc:
        logger.debug("Star detection failed", exc_info=True)
        return StarMetrics(warnings=[f"Star detection unavailable: {exc}"])
