import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pydantic import Field

from astroagent import __version__
from astroagent.agent.dataset_planner import inspect_dataset
from astroagent.agent.models import PlanResult
from astroagent.agent.planner import Planner, RuleBasedPlanner
from astroagent.analysis.quality import analyze_image
from astroagent.analysis.statistics import ImageMetrics
from astroagent.calibration.session import inspect_session_frames
from astroagent.errors import AstroError, PipelineError
from astroagent.io.datasets import prepare_directory, write_json
from astroagent.io.export import export_image
from astroagent.io.fits import save_fits
from astroagent.io.images import image_format, load_image, save_image
from astroagent.models.base import SchemaModel
from astroagent.models.dataset import AstroDataset, DatasetMetrics
from astroagent.models.image import AstroImage
from astroagent.pipeline.dataset_executor import execute_dataset_pipeline, load_dataset
from astroagent.pipeline.executor import PipelineExecutor, artifact_paths, image_digest
from astroagent.pipeline.models import PipelineDefinition, PipelineStep
from astroagent.pipeline.serialization import save_pipeline
from astroagent.tools.base import ToolDescription

logger = logging.getLogger(__name__)


class AgentOptions(SchemaModel):
    """Explicit limits on planner rounds and alternative numerical experiments."""

    max_iterations: int = Field(default=3, ge=1, le=20)
    max_candidates: int = Field(default=3, ge=1, le=3)
    quality_drop_tolerance: float = Field(default=0.03, ge=0, le=1)


@dataclass
class PreparedAgentRun:
    """Read-only initial input, measurements and validated candidate plans."""

    source: Path
    value: AstroImage | AstroDataset
    metrics: ImageMetrics | DatasetMetrics
    plan: PlanResult
    request: str


@dataclass
class AutonomousResult:
    """Accepted image and full plan/experiment history with an offline replay pipeline."""

    image: AstroImage
    report: dict[str, Any]


def evaluation_score(metrics: ImageMetrics | DatasetMetrics) -> float:
    """Rank alternatives using bounded contrast/noise, gradient and saturation heuristics.

    Image score = .4*range/(range+10*noise) + .35/(1+gradient/range)
    + .25*(1-saturation). This is a display heuristic, not scientific accuracy
    or aesthetic truth. Dataset scores use mean available frame quality.
    """
    if isinstance(metrics, DatasetMetrics):
        scores = [f["quality_score"] for f in metrics.frames if f.get("quality_score") is not None]
        return float(np.mean(scores)) if scores else 0
    span = max(metrics.percentile_99 - metrics.percentile_1, 1e-12)
    noise = metrics.background.sigma if metrics.background else metrics.standard_deviation
    gradient = metrics.background.gradient_estimate if metrics.background else 0
    return float(
        0.4 * span / (span + 10 * noise)
        + 0.35 / (1 + gradient / span)
        + 0.25 * (1 - (metrics.fraction_of_saturated_pixels or 0))
    )


class AutonomousAgent:
    """Bounded plan/execute/measure/replan loop; planners never receive pixels or run commands."""

    def __init__(
        self, planner: Planner | None = None, executor: PipelineExecutor | None = None
    ) -> None:
        """Reuse the existing provider-independent planner and deterministic tool registry."""
        self.planner = planner or RuleBasedPlanner()
        self.executor = executor or PipelineExecutor()

    def available_tools(self, value: AstroImage | AstroDataset) -> list[ToolDescription]:
        """Advertise tools that can consume current sampling and validity masks."""
        descriptions = self.executor.registry.describe()
        if isinstance(value, AstroDataset):
            return descriptions
        masked = np.isnan(value.data).any()
        return [
            t for t in descriptions if t.input_kind == "image" and (not masked or t.supports_nan)
        ]

    def prepare(self, source: Path | str, request: str) -> PreparedAgentRun:
        """Inspect without outputs and validate all proposed pipelines before execution."""
        source = Path(source)
        if source.is_dir():
            value: AstroImage | AstroDataset = load_dataset(source, recursive=True)
            metrics: ImageMetrics | DatasetMetrics = inspect_dataset(source)
        else:
            value = load_image(source)
            if value.cfa is not None:
                raise PipelineError(
                    "Raw CFA input must be calibrated and debayered first. "
                    "Use aia agent on the session directory or aia debayer for prepared CFA."
                )
            metrics = analyze_image(value)
        plan = self.planner.create_plan(request, metrics, self.available_tools(value))
        for candidate in [plan.pipeline, *plan.alternatives]:
            self.executor.validate(candidate)
        return PreparedAgentRun(source, value, metrics, plan, request)

    def _metrics(self, value: AstroImage | AstroDataset) -> ImageMetrics | DatasetMetrics:
        if isinstance(value, AstroImage):
            return analyze_image(value)
        session = inspect_session_frames(value.frames)
        frames = (
            [q.model_dump(mode="json") for q in value.qualities]
            if value.qualities
            else [f.model_dump(mode="json") for f in session.frames]
        )
        return DatasetMetrics(
            number_of_frames=len(value.frames),
            session_counts=session.counts(),
            frames=frames,
            selected_reference=None if value.reference is None else str(value.reference),
            warnings=session.warnings,
            registration_statistics={
                "frames": [r.model_dump(mode="json") for r in value.registrations]
            },
        )

    def run(
        self,
        prepared: PreparedAgentRun,
        output: Path | str,
        *,
        options: AgentOptions | None = None,
        overwrite: bool = False,
    ) -> AutonomousResult:
        """Evaluate candidate pipelines from a common baseline, accept one, then request a new plan.

        Accepted operations are concatenated into one deterministic replay pipeline.
        Every attempted plan, candidate FITS, measurements and failure reason remain
        in <output-stem>.agent/. Final artifacts are written only after the loop.
        """
        options = options or AgentOptions()
        output = Path(output)
        paths = artifact_paths(output)
        image_format(output)
        inputs = (
            prepared.value.frames if isinstance(prepared.value, AstroDataset) else [prepared.source]
        )
        if any(p.resolve() in {i.resolve() for i in inputs} for p in paths):
            raise PipelineError("Agent artifacts must not replace input frames.")
        if not overwrite and any(p.exists() for p in paths):
            raise PipelineError("Agent output already exists; use --overwrite.")
        work = output.parent / f"{output.stem}.agent"
        prepare_directory(work, inputs, overwrite=overwrite)
        current, metrics, plan = prepared.value, prepared.metrics, prepared.plan
        accepted: list[PipelineStep] = []
        history: list[dict[str, Any]] = []
        executed_reports: list[dict[str, Any]] = []
        reasons: list[str] = []
        seen: set[str] = set()
        stop = "iteration limit reached"
        stack_report: dict[str, Any] = {}
        for iteration in range(1, options.max_iterations + 1):
            if not plan.pipeline.steps and not plan.alternatives:
                stop = "planner finished"
                reasons.extend(plan.reasoning)
                break
            logger.info("Agent round %d/%d", iteration, options.max_iterations)
            candidates = []
            for index, proposed in enumerate(
                [plan.pipeline, *plan.alternatives][: options.max_candidates], 1
            ):
                if not proposed.steps:
                    continue
                record: dict[str, Any] = {
                    "iteration": iteration,
                    "candidate": index,
                    "accepted": False,
                    "pipeline": proposed.model_dump(mode="json"),
                    "reasoning": plan.reasoning,
                }
                tick = perf_counter()
                try:
                    resolved = self.executor.validate(proposed)
                    definition = PipelineDefinition(
                        steps=[
                            PipelineStep(tool=t.name, params=p.model_dump(mode="json"))
                            for t, p in resolved
                        ]
                    )
                    signature = json.dumps(definition.model_dump(mode="json"), sort_keys=True)
                    if signature in seen:
                        raise PipelineError(
                            "Identical pipeline already attempted; stopping repetition."
                        )
                    if len(accepted) + len(definition.steps) > 100:
                        raise PipelineError("Combined replay pipeline exceeds 100 steps.")
                    seen.add(signature)
                    prefix = work / f"round-{iteration:02d}-candidate-{index:02d}"
                    save_pipeline(
                        definition, prefix.with_suffix(".proposal.yaml"), overwrite=overwrite
                    )
                    if isinstance(current, AstroDataset):
                        # Dataset tools may update their context, so isolate candidate state.
                        baseline = AstroDataset(
                            list(current.frames),
                            source=current.source,
                            reference=current.reference,
                            catalogs=dict(current.catalogs),
                            qualities=list(current.qualities),
                            registrations=list(current.registrations),
                            masters=list(current.masters),
                            reports=dict(current.reports),
                        )
                        result = execute_dataset_pipeline(
                            baseline,
                            definition,
                            resolved,
                            prefix.with_suffix(".fit"),
                            overwrite=overwrite,
                        )
                        value, diagnostics = result.value, result.report
                    else:
                        image_result = self.executor.execute(current, definition)
                        value, diagnostics = (
                            image_result.image,
                            image_result.report.model_dump(mode="json"),
                        )
                        if value.data.shape != current.data.shape:
                            raise PipelineError(
                                "Image editing changed dimensions or channel layout."
                            )
                        save_fits(value, prefix.with_suffix(".fit"), overwrite=overwrite)
                    measured = self._metrics(value)
                    record.update(
                        {
                            "pipeline": definition.model_dump(mode="json"),
                            "metrics_before": metrics.model_dump(mode="json"),
                            "metrics_after": measured.model_dump(mode="json"),
                            "score": evaluation_score(measured),
                            "warnings": diagnostics.get("warnings", []),
                            "output": str(prefix.with_suffix(".fit")),
                        }
                    )
                    candidates.append(
                        (record["score"], -index, value, measured, definition, diagnostics, record)
                    )
                    # Other candidate pixels remain on disk; retain only the best in memory.
                    candidates = [max(candidates, key=lambda item: (item[0], item[1]))]
                except (AstroError, ValueError, RuntimeError) as exc:
                    record["error"] = str(exc)
                    logger.warning("Agent candidate failed: %s", exc)
                record["duration_ms"] = (perf_counter() - tick) * 1000
                history.append(record)
                write_json(work / "iterations.json", {"attempts": history}, overwrite=True)
            if not candidates:
                if not accepted and plan.pipeline.steps:
                    raise PipelineError(
                        "All agent candidates failed: "
                        + "; ".join(r.get("error", "") for r in history[-options.max_candidates :])
                    )
                stop = "no new successful candidate"
                break
            best = max(candidates, key=lambda item: (item[0], item[1]))
            score, _, value, measured, definition, diagnostics, record = best
            if (
                isinstance(current, AstroImage)
                and accepted
                and score < evaluation_score(metrics) - options.quality_drop_tolerance
            ):
                record["rejection_reason"] = (
                    "Measured quality declined beyond tolerance; preserved previous result."
                )
                stop = "quality decline"
                break
            if (
                isinstance(current, AstroImage)
                and isinstance(value, AstroImage)
                and image_digest(current) == image_digest(value)
            ):
                record["rejection_reason"] = "Pipeline did not change pixels."
                stop = "no pixel change"
                break
            record["accepted"] = True
            accepted.extend(definition.steps)
            executed_reports.extend(diagnostics.get("steps", []))
            if "stack" in diagnostics:
                stack_report = diagnostics
            current, metrics = value, measured
            reasons.extend(plan.reasoning)
            if iteration == options.max_iterations:
                break
            feedback = {
                "iteration": iteration,
                "executed_tools": [s.tool for s in accepted],
                "initial_input_kind": "dataset"
                if isinstance(prepared.value, AstroDataset)
                else "image",
                "attempts": deepcopy(history),
            }
            revise = getattr(self.planner, "revise_plan", None)
            try:
                plan = (
                    revise(prepared.request, metrics, self.available_tools(current), feedback)
                    if revise
                    else self.planner.create_plan(
                        prepared.request, metrics, self.available_tools(current)
                    )
                )
            except (AstroError, ValueError, RuntimeError) as exc:
                stop = f"replanning failed: {exc}"
                logger.warning("%s; preserving the accepted result", stop)
                break
        if not isinstance(current, AstroImage):
            raise PipelineError(
                "Agent stopped before producing a master image; inspect iterations.json "
                "or increase --max-iterations."
            )
        combined = PipelineDefinition(steps=accepted)
        report = {
            **stack_report,
            "input": str(prepared.source),
            "output": str(output),
            "package_version": __version__,
            "dependency_versions": {
                n: version(n) for n in ("numpy", "scipy", "astropy", "photutils", "pydantic")
            },
            "planner": {"provider": str(getattr(self.planner, "provider", "rules"))},
            "request": prepared.request,
            "reasoning": reasons,
            "initial_metrics": prepared.metrics.model_dump(mode="json"),
            "metrics_after": metrics.model_dump(mode="json"),
            "output_sha256": image_digest(current),
            "pipeline": combined.model_dump(mode="json"),
            "steps": executed_reports,
            "agent": {"options": options.model_dump(), "stop_reason": stop, "iterations": history},
            "warnings": list(
                dict.fromkeys(w for h in history if h["accepted"] for w in h.get("warnings", []))
            ),
        }
        if hasattr(self.planner, "model"):
            report["planner"]["model"] = self.planner.model
        finite = current.data[np.isfinite(current.data)]
        if image_format(output) not in {"fits", "tiff"} and finite.min() >= 0 and finite.max() <= 1:
            exported = current.with_data(np.nan_to_num(current.data, nan=0))
            save_image(exported, output, overwrite=overwrite)
            report["export"] = {
                "format": image_format(output),
                "display_transform": None,
                "invalid_fill": 0,
            }
        else:
            report["export"] = export_image(current, output, overwrite=overwrite)
        save_pipeline(combined, paths[2], overwrite=overwrite)
        write_json(paths[1], report, overwrite=overwrite)
        write_json(
            work / "iterations.json", {"attempts": history, "stop_reason": stop}, overwrite=True
        )
        return AutonomousResult(current, report)
