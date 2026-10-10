from astroagent.analysis.background import analyze_background
from astroagent.analysis.stars import analyze_stars
from astroagent.analysis.statistics import ImageMetrics, inspect_image
from astroagent.execution import ExecutionContext, emit_progress, execution_scope
from astroagent.models.image import AstroImage


@execution_scope
def analyze_image(image: AstroImage, *, context: ExecutionContext | None = None) -> ImageMetrics:
    """Combine read-only measurements and a conservative linear-image heuristic."""
    emit_progress("analyzing-statistics")
    metrics = inspect_image(image)
    emit_progress("analyzing-background")
    metrics.background = analyze_background(image)
    emit_progress("analyzing-stars")
    metrics.stars = analyze_stars(image)
    dynamic_range = metrics.percentile_99 - metrics.percentile_1
    metrics.appears_linear = (
        not bool(image.header.get("ASTRSTR", False))
        and dynamic_range > 0
        and (metrics.median - metrics.percentile_1) / dynamic_range < 0.18
    )
    metrics.warnings.extend(metrics.background.warnings)
    metrics.warnings.extend(metrics.stars.warnings)
    return metrics
