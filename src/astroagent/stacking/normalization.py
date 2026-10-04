from typing import Any

import numpy as np
from numpy.typing import NDArray

from astroagent.errors import PipelineError


def normalization_statistics(data: NDArray[Any]) -> tuple[list[float], list[float]]:
    """Return robust per-channel median and 5..95 percentile span for affine normalization."""
    planes = [data] if data.ndim == 2 else [data[..., c] for c in range(3)]
    levels, spans = [], []
    for plane in planes:
        finite = plane[np.isfinite(plane)]
        if not finite.size:
            raise PipelineError("Cannot normalize a frame without finite samples.")
        lo, median, hi = np.percentile(finite, [5, 50, 95])
        levels.append(float(median))
        spans.append(float(hi - lo))
    return levels, spans


def normalization_coefficients(
    statistics: tuple[list[float], list[float]],
    reference: tuple[list[float], list[float]],
) -> tuple[list[float], list[float]]:
    """Match medians and percentile spans; constant frames receive offset-only correction.

    This is a global intensity heuristic, not photometric or gradient normalization.
    Scale can track noise instead of transparency on star-sparse fields.
    """
    levels, spans = np.asarray(statistics)
    ref_levels, ref_spans = np.asarray(reference)
    scale = np.divide(
        ref_spans, spans, out=np.ones_like(spans), where=(spans > 0) & (ref_spans > 0)
    )
    return scale.tolist(), (ref_levels - scale * levels).tolist()
