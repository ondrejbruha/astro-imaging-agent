import math

from pydantic import Field

from astroagent.analysis.frame_quality import FrameQualityMetrics
from astroagent.models.base import SchemaModel


class RejectionParams(SchemaModel):
    """Optional dataset-level quality cuts; missing required measurements fail the cut."""

    reject_worst_fraction: float = Field(default=0, ge=0, lt=1)
    min_quality: float | None = Field(default=None, ge=0, le=1)
    max_fwhm: float | None = Field(default=None, gt=0)
    max_eccentricity: float | None = Field(default=None, ge=0, le=1)
    max_background_sigma: float | None = Field(default=None, gt=0)
    max_saturation: float | None = Field(default=None, ge=0, le=1)
    max_registration_residual: float | None = Field(default=None, gt=0)


def reject_frames(
    metrics: list[FrameQualityMetrics],
    params: RejectionParams,
    residuals: dict[str, float | None] | None = None,
) -> dict[str, list[str]]:
    """Return explicit reasons; fractional rejection removes floor(N*fraction) survivors."""
    rejected: dict[str, list[str]] = {}
    for frame in metrics:
        tests = [
            (frame.quality_score, params.min_quality, True, "quality"),
            (frame.median_fwhm, params.max_fwhm, False, "FWHM"),
            (frame.median_eccentricity, params.max_eccentricity, False, "eccentricity"),
            (frame.background_sigma, params.max_background_sigma, False, "background sigma"),
            (frame.saturation_fraction, params.max_saturation, False, "saturation"),
            (
                (residuals or {}).get(frame.path),
                params.max_registration_residual,
                False,
                "registration residual",
            ),
        ]
        reasons = [
            f"{name} outside requested limit"
            for value, limit, lower, name in tests
            if limit is not None and (value is None or (value < limit if lower else value > limit))
        ]
        if reasons:
            rejected[frame.path] = reasons
    survivors = sorted(
        (m for m in metrics if m.path not in rejected),
        key=lambda m: (m.quality_score or 0, m.path),
    )
    for frame in survivors[: math.floor(len(survivors) * params.reject_worst_fraction)]:
        rejected[frame.path] = ["worst quality fraction"]
    return rejected
