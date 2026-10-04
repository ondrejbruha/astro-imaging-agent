from dataclasses import dataclass
from pathlib import Path

from astroagent.agent.models import PlanResult
from astroagent.agent.planner import Planner, RuleBasedPlanner
from astroagent.analysis.quality import analyze_image
from astroagent.analysis.statistics import ImageMetrics
from astroagent.io.images import load_image
from astroagent.models.image import AstroImage
from astroagent.pipeline.executor import (
    PipelineExecutor,
    PipelineResult,
    check_output_paths,
    save_result,
)


@dataclass
class PreparedPlan:
    """Read-only analysis and proposed plan, ready for display or execution."""

    image: AstroImage
    metrics: ImageMetrics
    plan: PlanResult
    request: str


class AgentExecutor:
    """Orchestrate analysis and planning while delegating all pixel work to pipelines."""

    def __init__(
        self, planner: Planner | None = None, executor: PipelineExecutor | None = None
    ) -> None:
        """Allow planner and tool injection without modifying the engine."""
        self.planner = RuleBasedPlanner() if planner is None else planner
        self.executor = PipelineExecutor() if executor is None else executor

    def prepare(self, path: Path | str, request: str) -> PreparedPlan:
        """Load and analyze an image, then validate the proposed pipeline."""
        image = load_image(path)
        metrics = analyze_image(image)
        plan = self.planner.create_plan(
            request,
            metrics,
            [tool for tool in self.executor.registry.describe() if tool.input_kind == "image"],
        )
        self.executor.validate(plan.pipeline)
        return PreparedPlan(image, metrics, plan, request)

    def execute(
        self, prepared: PreparedPlan, output: Path | str, *, overwrite: bool = False
    ) -> PipelineResult:
        """Execute, reanalyze, and save an image, replay plan, and processing report."""
        check_output_paths(prepared.image.path, Path(output), overwrite=overwrite)
        result = self.executor.execute(prepared.image, prepared.plan.pipeline)
        result.report.metrics_before = prepared.metrics
        result.report.metrics_after = analyze_image(result.image)
        result.report.warnings = list(
            dict.fromkeys(
                [
                    *result.report.warnings,
                    *prepared.metrics.warnings,
                    *result.report.metrics_after.warnings,
                ]
            )
        )
        result.report.request = prepared.request
        result.report.reasoning = prepared.plan.reasoning
        result.report.planner = {"provider": str(getattr(self.planner, "provider", "rules"))}
        if hasattr(self.planner, "model"):
            result.report.planner["model"] = str(self.planner.model)
        save_result(result, output, overwrite=overwrite)
        return result
