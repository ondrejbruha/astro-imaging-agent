from dataclasses import dataclass
from pathlib import Path

import numpy as np

from astroagent.agent.dataset_planner import inspect_selected_dataset
from astroagent.agent.models import PlanResult
from astroagent.agent.planner import Planner, RuleBasedPlanner
from astroagent.analysis.quality import analyze_image
from astroagent.analysis.statistics import ImageMetrics
from astroagent.errors import PipelineError, PipelineValidationError
from astroagent.execution import ExecutionContext, emit_progress, execution_scope
from astroagent.io.images import load_image
from astroagent.models.dataset import AstroDataset, DatasetMetrics
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


@dataclass
class PreparedDatasetPlan:
    """Pixel-free measurements and validated candidates for explicitly selected frames."""

    metrics: DatasetMetrics
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

    @execution_scope
    def prepare(
        self,
        path: Path | str,
        request: str,
        *,
        hdu: int | None = None,
        context: ExecutionContext | None = None,
    ) -> PreparedPlan:
        """Load and analyze an image, then validate the proposed pipeline."""
        emit_progress("loading")
        image = load_image(path, hdu=hdu)
        if image.cfa is not None:
            raise PipelineError("Calibrate and debayer raw CFA before single-image planning.")
        emit_progress("analyzing")
        metrics = analyze_image(image)
        emit_progress("planning")
        masked = np.isnan(image.data).any()
        tools = [
            tool
            for tool in self.executor.registry.describe()
            if tool.input_kind == "image"
            and image.layout.value in tool.compatible_layouts
            and (not masked or tool.supports_nan)
        ]
        plan = self.planner.create_plan(request, metrics, tools)
        emit_progress("validating")
        available = {tool.name for tool in tools}
        for candidate in [plan.pipeline, *plan.alternatives]:
            self.executor.validate(candidate, input_kind="image")
            for index, step in enumerate(candidate.steps, 1):
                if step.tool not in available:
                    raise PipelineValidationError(
                        "Plan contains a tool ineligible for the selected image.",
                        [{"step_index": index, "field": ["tool"], "type": "input_eligibility"}],
                    )
        return PreparedPlan(image, metrics, plan, request)

    @execution_scope
    def prepare_dataset(
        self, dataset: AstroDataset, request: str, *, context: ExecutionContext | None = None
    ) -> PreparedDatasetPlan:
        """Inspect and plan a selected dataset without executing processing tools or saving."""
        metrics = inspect_selected_dataset(dataset)
        emit_progress("planning")
        plan = self.planner.create_plan(request, metrics, self.executor.registry.describe())
        emit_progress("validating")
        for candidate in [plan.pipeline, *plan.alternatives]:
            self.executor.validate(candidate, input_kind="dataset")
        return PreparedDatasetPlan(metrics, plan, request)

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
