import warnings

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from pydantic import Field

from astroagent.analysis.background import estimate_surface, luminance
from astroagent.errors import AnalysisError
from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage


class DetectedStar(SchemaModel):
    """Centroid in zero-based x/y pixels with background-subtracted PSF measurements."""

    x: float
    y: float
    flux: float = Field(gt=0)
    peak: float | None = None
    fwhm: float | None = Field(default=None, gt=0)
    eccentricity: float | None = Field(default=None, ge=0, le=1)


class StarCatalog(SchemaModel):
    """Brightest-first sources; shapes assume isolated Gaussian point sources."""

    stars: list[DetectedStar] = Field(default_factory=list)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    warnings: list[str] = Field(default_factory=list)


class DetectionParams(SchemaModel):
    """Noise-relative detection threshold and expected Gaussian FWHM."""

    threshold_sigma: float = Field(default=5, gt=0)
    fwhm: float = Field(default=3, gt=0)
    max_stars: int = Field(default=2000, ge=3, le=100000)


def detect_stars(image: AstroImage, params: DetectionParams | None = None) -> StarCatalog:
    """Use DAOStarFinder above a fitted sky and measure local second moments.

    NaNs are masked. Saturated stars and border cutouts are excluded. Moments
    are approximate, biased by blends and low SNR; they are not a PSF fit.
    """
    params = DetectionParams() if params is None else params
    if image.layout.value == "cfa":
        raise AnalysisError("Debayer CFA data before star detection and registration.")
    data = luminance(image)
    mask = ~np.isfinite(data)
    catalog = StarCatalog(image_width=data.shape[1], image_height=data.shape[0])
    if mask.all():
        catalog.warnings.append("No finite pixels for star detection.")
        return catalog
    _, median, _ = sigma_clipped_stats(data[~mask])
    try:
        sky = estimate_surface(data, polynomial_degree=1)
    except AnalysisError:
        sky = np.full_like(data, median)
    residual = data - sky
    _, _, noise = sigma_clipped_stats(residual[~mask])
    if not np.isfinite(noise) or noise <= np.finfo(float).eps * max(1, abs(median)):
        catalog.warnings.append("Noise is too small for reliable star detection.")
        return catalog
    finder = DAOStarFinder(
        threshold=params.threshold_sigma * noise,
        fwhm=params.fwhm,
        exclude_border=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        sources = finder(np.where(mask, 0, residual), mask=mask)
    catalog.warnings.extend(dict.fromkeys(str(item.message) for item in caught))
    if sources is None:
        return catalog
    xcol = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"
    radius = max(3, int(np.ceil(1.5 * params.fwhm)))
    for row in sources:
        cx, cy = float(row[xcol]), float(row[ycol])
        x, y = int(round(cx)), int(round(cy))
        if min(x, y) < radius or x + radius >= data.shape[1] or y + radius >= data.shape[0]:
            continue
        cut = residual[y - radius : y + radius + 1, x - radius : x + radius + 1]
        raw = data[y - radius : y + radius + 1, x - radius : x + radius + 1]
        if not np.isfinite(cut).all():
            continue
        if image.saturation_level is not None and raw.max() >= image.saturation_level:
            continue
        background = np.median(np.concatenate([cut[0], cut[-1], cut[:, 0], cut[:, -1]]))
        weights = np.maximum(cut - background - noise, 0)
        total = weights.sum()
        if total <= 0:
            continue
        yy, xx = np.indices(cut.shape)
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
        catalog.stars.append(
            DetectedStar(
                x=cx,
                y=cy,
                flux=float(total),
                peak=float(cut.max()),
                fwhm=float(2.35482 * np.sqrt((minor + major) / 2)),
                eccentricity=float(np.sqrt(max(0, 1 - minor / major))),
            )
        )
    catalog.stars.sort(key=lambda s: (-s.flux, s.x, s.y))
    catalog.stars = catalog.stars[: params.max_stars]
    return catalog
