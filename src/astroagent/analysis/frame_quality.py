from pathlib import Path

import numpy as np
from pydantic import Field
from scipy.stats import rankdata

from astroagent.analysis.background import analyze_background
from astroagent.analysis.statistics import inspect_image
from astroagent.execution import ExecutionContext, execution_scope
from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage
from astroagent.registration.stars import DetectionParams, StarCatalog, detect_stars


class FrameQualityMetrics(SchemaModel):
    """Frame measurements; quality is a dataset-relative percentile score in [0, 1]."""

    path: str
    star_count: int = Field(ge=0)
    median_fwhm: float | None = Field(default=None, gt=0)
    median_hfr: float | None = Field(default=None, gt=0)
    median_eccentricity: float | None = Field(default=None, ge=0, le=1)
    background_median: float
    background_sigma: float = Field(ge=0)
    saturation_fraction: float | None = Field(default=None, ge=0, le=1)
    snr_estimate: float | None = Field(default=None, ge=0)
    quality_score: float | None = Field(default=None, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


@execution_scope
def measure_frame(
    image: AstroImage,
    *,
    path: Path | str | None = None,
    detection: DetectionParams | None = None,
    context: ExecutionContext | None = None,
) -> tuple[FrameQualityMetrics, StarCatalog]:
    """Measure sky, approximate PSFs and peak/noise SNR without modifying pixels."""
    catalog = detect_stars(image, detection)
    background = analyze_background(image)
    stats = inspect_image(image)
    widths = [s.fwhm for s in catalog.stars if s.fwhm is not None]
    radii = [s.hfr for s in catalog.stars if s.hfr is not None]
    shapes = [s.eccentricity for s in catalog.stars if s.eccentricity is not None]
    peaks = [s.peak for s in catalog.stars if s.peak is not None]
    return FrameQualityMetrics(
        path=str(path or image.path or "<memory>"),
        star_count=len(catalog.stars),
        median_fwhm=float(np.median(widths)) if widths else None,
        median_hfr=float(np.median(radii)) if radii else None,
        median_eccentricity=float(np.median(shapes)) if shapes else None,
        background_median=background.median,
        background_sigma=background.sigma,
        saturation_fraction=stats.fraction_of_saturated_pixels,
        snr_estimate=(
            float(np.median(peaks) / background.sigma) if peaks and background.sigma > 0 else None
        ),
        warnings=[*catalog.warnings, *background.warnings, *stats.warnings],
    ), catalog


QUALITY_WEIGHTS = {
    "median_fwhm": 0.3,
    "median_eccentricity": 0.2,
    "background_sigma": 0.2,
    "star_count": 0.2,
    "saturation_fraction": 0.1,
}


def score_frames(metrics: list[FrameQualityMetrics]) -> list[FrameQualityMetrics]:
    """Average percentile ranks with explicit weights; ties get midrank, missing gets zero.

    More stars are better, all other components lower are better. A constant
    component scores 0.5 (including singletons). Scores cannot be compared across
    sessions and assume comparable exposure, sky brightness, and image dimensions.
    """
    scores = np.zeros(len(metrics))
    for name, weight in QUALITY_WEIGHTS.items():
        values = [getattr(m, name) for m in metrics]
        valid = np.array([v is not None for v in values])
        numbers = np.array([v for v in values if v is not None], dtype=float)
        if len(numbers) < 2:
            component = np.full(len(numbers), 0.5)
        else:
            component = (rankdata(numbers, method="average") - 1) / (len(numbers) - 1)
            if name != "star_count":
                component = 1 - component
        scores[valid] += weight * component
    return [
        m.model_copy(update={"quality_score": float(s)})
        for m, s in zip(metrics, scores, strict=True)
    ]
