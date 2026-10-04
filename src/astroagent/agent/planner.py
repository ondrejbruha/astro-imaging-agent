from typing import Protocol

from pydantic import JsonValue

from astroagent.agent.models import PlanResult
from astroagent.analysis.statistics import ImageMetrics
from astroagent.errors import PipelineError
from astroagent.pipeline.models import PipelineDefinition, PipelineStep
from astroagent.tools.base import ToolDescription


class Planner(Protocol):
    """Provider-independent planning interface; pixels are never passed to planners."""

    def create_plan(
        self, request: str, image_metrics: ImageMetrics, available_tools: list[ToolDescription]
    ) -> PlanResult:
        """Return a declarative plan with concise explicit reasons."""
        ...


class RuleBasedPlanner:
    """Conservative numerical heuristics demonstrating orchestration without an API.

    Free-text requests are recorded for provenance; this planner does not interpret
    their meaning. Thresholds compare sky gradient/noise with the robust image range.
    """

    def create_plan(
        self, request: str, image_metrics: ImageMetrics, available_tools: list[ToolDescription]
    ) -> PlanResult:
        """Select background correction, mild denoising, and bounded stretching."""
        if not request.strip():
            raise PipelineError("Agent request must not be empty.")
        available = {tool.name for tool in available_tools}
        steps: list[PipelineStep] = []
        reasoning: list[str] = []
        dynamic_range = image_metrics.percentile_99 - image_metrics.percentile_1
        if dynamic_range <= 0:
            return PlanResult(
                pipeline=PipelineDefinition(),
                reasoning=["Constant image: no useful processing selected."],
            )

        def add(name: str, params: dict[str, JsonValue], reason: str) -> None:
            if name in available:
                steps.append(PipelineStep(tool=name, params=params))
                reasoning.append(reason)
            else:
                reasoning.append(f"Skipped {name}: tool is unavailable.")

        background = image_metrics.background
        if background is not None and background.gradient_estimate / dynamic_range > 0.15:
            if min(image_metrics.dimensions) >= 4:
                add(
                    "background_extract",
                    {"grid_size": 8, "polynomial_degree": 1, "sigma_clipping_threshold": 3.0},
                    "Estimated background gradient exceeds 15% of the robust range; "
                    "subtract a fitted sky plane.",
                )
            else:
                reasoning.append(
                    "Background correction skipped: image is too small for a reliable grid."
                )
        if background is not None and background.sigma / dynamic_range > 0.08:
            add(
                "denoise",
                {"method": "gaussian", "sigma": 0.6},
                "Estimated noise exceeds 8% of the robust range; apply mild Gaussian smoothing.",
            )
        if image_metrics.appears_linear:
            add(
                "stretch",
                {"method": "asinh", "strength": 0.45},
                "Image appears linear; use a bounded asinh stretch with the maximum mapped to one.",
            )
        elif (
            image_metrics.min < 0
            or image_metrics.max > 1
            or any(step.tool == "background_extract" for step in steps)
        ):
            add(
                "normalize",
                {"lower": 0.0, "upper": 1.0},
                "Normalize the global range to 0..1 after sky subtraction or out-of-range input.",
            )
        if not reasoning:
            reasoning.append("No heuristic threshold exceeded; preserve the current image.")
        return PlanResult(pipeline=PipelineDefinition(steps=steps), reasoning=reasoning)
