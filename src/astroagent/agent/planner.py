import unicodedata
from typing import Any, Protocol

from pydantic import JsonValue

from astroagent.agent.models import PlanResult
from astroagent.analysis.statistics import ImageMetrics
from astroagent.errors import PipelineError
from astroagent.models.dataset import DatasetMetrics
from astroagent.pipeline.models import PipelineDefinition, PipelineStep
from astroagent.tools.base import ToolDescription


class Planner(Protocol):
    """Provider-independent planning interface; pixels are never passed to planners."""

    def create_plan(
        self,
        request: str,
        image_metrics: ImageMetrics | DatasetMetrics,
        available_tools: list[ToolDescription],
    ) -> PlanResult:
        """Return a declarative plan with concise explicit reasons."""
        ...


class RuleBasedPlanner:
    """Conservative numerical heuristics demonstrating orchestration without an API.

    Thresholds compare sky gradient/noise with the robust image range. Limited
    English/Czech keywords enable contrast, color and sharpening edits; arbitrary
    free-form intent requires an existing optional LLM provider.
    """

    def create_plan(
        self,
        request: str,
        image_metrics: ImageMetrics | DatasetMetrics,
        available_tools: list[ToolDescription],
    ) -> PlanResult:
        """Select background correction, mild denoising, and bounded stretching."""
        if not request.strip():
            raise PipelineError("Agent request must not be empty.")
        if isinstance(image_metrics, DatasetMetrics):
            from astroagent.agent.dataset_planner import DatasetRulePlanner

            plan = DatasetRulePlanner().create_plan(request, image_metrics)
            if any(
                step.tool not in {t.name for t in available_tools} for step in plan.pipeline.steps
            ):
                raise PipelineError("Required dataset processing tools are unavailable.")
            return plan
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
        hints = unicodedata.normalize("NFKD", request).encode("ascii", "ignore").decode().lower()
        display_ready = 0 <= image_metrics.min <= image_metrics.max <= 1 or any(
            s.tool in {"stretch", "normalize"} for s in steps
        )
        if display_ready:
            if "contrast" in hints or "kontrast" in hints:
                add(
                    "local_contrast",
                    {"amount": 0.2, "radius": 12},
                    "Requested local contrast; use a restrained luminance boost.",
                )
            if (
                any(word in hints for word in ("color", "barv", "satur"))
                and image_metrics.number_of_channels == 3
            ):
                add(
                    "color_adjust",
                    {"saturation": 1.1},
                    "Requested colors; mildly increase RGB saturation.",
                )
            if "sharpen" in hints or "doostr" in hints:
                add(
                    "sharpen",
                    {"amount": 0.3, "radius": 1, "threshold": 0.01},
                    "Requested sharpening; threshold small noise fluctuations.",
                )
        return PlanResult(pipeline=PipelineDefinition(steps=steps), reasoning=reasoning)

    def revise_plan(
        self,
        request: str,
        metrics: ImageMetrics | DatasetMetrics,
        available_tools: list[ToolDescription],
        feedback: dict[str, Any],
    ) -> PlanResult:
        """Reevaluate measurements, avoid repeated edits and preserve scientific masters."""
        done = set(feedback.get("executed_tools", []))
        hints = unicodedata.normalize("NFKD", request).encode("ascii", "ignore").decode().lower()
        if "stack_frames" in done and not any(
            h in hints
            for h in (
                "contrast",
                "kontrast",
                "barv",
                "color",
                "sharpen",
                "doostr",
                "stretch",
                "display",
            )
        ):
            return PlanResult(
                pipeline=PipelineDefinition(),
                reasoning=["Scientific master completed; preserve linear samples."],
            )
        plan = self.create_plan(request, metrics, available_tools)
        plan.pipeline = PipelineDefinition(
            steps=[step for step in plan.pipeline.steps if step.tool not in done]
        )
        return plan
