from typing import Any

import numpy as np
from pydantic import Field

from astroagent.errors import AnalysisError
from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage


class BackgroundMetrics(SchemaModel):
    """Sigma-clipped sky level, residual noise, and absolute fitted gradient."""

    median: float
    sigma: float = Field(ge=0)
    gradient_estimate: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class StarMetrics(SchemaModel):
    """Approximate star shape measurements in pixels; unavailable values are null."""

    star_count: int | None = Field(default=None, ge=0)
    median_fwhm: float | None = Field(default=None, ge=0)
    median_hfr: float | None = Field(default=None, gt=0)
    median_ellipticity: float | None = Field(default=None, ge=0, le=1)
    median_eccentricity: float | None = Field(default=None, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class ImageMetrics(SchemaModel):
    """JSON-safe statistics over all finite pixel samples and channels."""

    dimensions: tuple[int, int]
    number_of_channels: int
    datatype: str
    min: float
    max: float
    mean: float
    median: float
    standard_deviation: float
    percentile_1: float
    percentile_5: float
    percentile_50: float
    percentile_95: float
    percentile_99: float
    fraction_of_saturated_pixels: float | None = Field(default=None, ge=0, le=1)
    saturation_level: float | None = None
    finite_samples: int
    nonfinite_samples: int
    fits_metadata: dict[str, Any] = Field(default_factory=dict)
    background: BackgroundMetrics | None = None
    stars: StarMetrics | None = None
    appears_linear: bool | None = None
    warnings: list[str] = Field(default_factory=list)


def inspect_image(image: AstroImage) -> ImageMetrics:
    """Measure finite samples, excluding NaN/Inf and explaining saturation assumptions.

    Saturation uses an explicit level, otherwise the integer dtype ceiling, or
    one for floating images entirely within [0, 1]. Unknown float ceilings are null.
    Dimensions are (height, width); RGB statistics aggregate channel samples.
    """
    finite = np.asarray(image.data[np.isfinite(image.data)], dtype=np.float64)
    if finite.size == 0:
        raise AnalysisError("Image does not contain any finite pixel values.")
    warnings: list[str] = []
    missing = int(image.data.size - finite.size)
    if missing:
        warnings.append(f"Excluded {missing} nonfinite pixel samples from statistics.")
    level = image.saturation_level
    if level is None and image.data.dtype.kind in "ui":
        level = float(np.iinfo(image.data.dtype).max)
    elif level is None and finite.min() >= 0 and finite.max() <= 1:
        level = 1.0
    if level is not None and not np.isfinite(level):
        level = None
    if level is None:
        warnings.append("Saturation level is unknown; provide a SATURATE FITS card for raw floats.")
    percentiles = np.percentile(finite, [1, 5, 50, 95, 99])
    return ImageMetrics(
        dimensions=(image.data.shape[0], image.data.shape[1]),
        number_of_channels=image.channels,
        datatype=str(image.data.dtype),
        min=float(finite.min()),
        max=float(finite.max()),
        mean=float(finite.mean()),
        median=float(np.median(finite)),
        standard_deviation=float(finite.std()),
        percentile_1=float(percentiles[0]),
        percentile_5=float(percentiles[1]),
        percentile_50=float(percentiles[2]),
        percentile_95=float(percentiles[3]),
        percentile_99=float(percentiles[4]),
        fraction_of_saturated_pixels=(None if level is None else float(np.mean(finite >= level))),
        saturation_level=level,
        finite_samples=int(finite.size),
        nonfinite_samples=missing,
        fits_metadata=image.metadata,
        warnings=warnings,
    )
