from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from astroagent.analysis.frame_quality import FrameQualityMetrics
from astroagent.errors import PipelineError


class FrameWeightCalculator(Protocol):
    """Calculate a nonnegative scalar reliability weight without access to pixels."""

    def calculate(self, metrics: FrameQualityMetrics) -> float:
        """Return the unnormalized weight for one frame."""
        ...


class UniformWeight:
    """Give all usable frames equal statistical weight."""

    def calculate(self, metrics: FrameQualityMetrics) -> float:
        """Return unity independently of the measured quality."""
        return 1.0


class QualityWeight:
    """Use dataset percentile quality with a small floor to avoid zero coverage."""

    def calculate(self, metrics: FrameQualityMetrics) -> float:
        """Return quality floored at 0.01; require quality to have been measured."""
        if metrics.quality_score is None:
            raise PipelineError("Quality weighting requires scored frame metrics.")
        return max(0.01, metrics.quality_score)


def normalized_weights(
    metrics: list[FrameQualityMetrics],
    calculator: FrameWeightCalculator,
) -> NDArray[np.float64]:
    """Normalize finite nonnegative frame weights; reject a zero total."""
    weights = np.array([calculator.calculate(m) for m in metrics], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise PipelineError("Frame weights must be finite, nonnegative, and sum to more than zero.")
    return np.asarray(weights / weights.sum(), dtype=np.float64)
