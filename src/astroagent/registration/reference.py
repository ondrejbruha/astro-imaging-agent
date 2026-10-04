from astroagent.analysis.frame_quality import FrameQualityMetrics
from astroagent.errors import PipelineError


def select_reference(
    metrics: list[FrameQualityMetrics], *, min_stars: int = 6
) -> FrameQualityMetrics:
    """Choose the highest-quality eligible frame, breaking ties by stable path order."""
    candidates = [m for m in metrics if m.star_count >= min_stars and m.quality_score is not None]
    if not candidates:
        raise PipelineError(f"No reference candidate has at least {min_stars} detected stars.")
    return sorted(candidates, key=lambda m: (-float(m.quality_score or 0), m.path))[0]
