import hashlib
import logging
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from pydantic import ValidationError

from astroagent import __version__
from astroagent.analysis.statistics import inspect_image
from astroagent.errors import AstroError, PipelineError, PipelineValidationError
from astroagent.execution import ExecutionContext, checkpoint, emit_progress, execution_scope
from astroagent.io.images import image_format, load_image, save_image
from astroagent.models.image import AstroImage
from astroagent.pipeline.dataset_executor import (
    DatasetPipelineResult,
    execute_dataset_pipeline,
    load_dataset,
)
from astroagent.pipeline.models import (
    PipelineDefinition,
    PipelineStep,
    ProcessingReport,
    StepReport,
)
from astroagent.pipeline.serialization import save_pipeline, save_report
from astroagent.tools.base import ImageTool
from astroagent.tools.dataset import DatasetTool
from astroagent.tools.registry import ToolRegistry, default_registry

logger = logging.getLogger(__name__)


def image_digest(image: AstroImage) -> str:
    """Hash canonical little-endian float64 pixels and shape for replay comparisons."""
    digest = hashlib.sha256(str(image.data.shape).encode("ascii"))
    digest.update(np.asarray(image.data, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


@dataclass
class PipelineResult:
    """Processed image and a complete in-memory execution report."""

    image: AstroImage
    report: ProcessingReport


class PipelineExecutor:
    """Validate all tools up front, then execute without importing any agent code."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        """Accept custom tools or construct a fresh default registry."""
        self.registry = default_registry() if registry is None else registry

    def validate(
        self, pipeline: PipelineDefinition, *, input_kind: Literal["image", "dataset"] | None = None
    ) -> list[tuple[ImageTool[Any] | DatasetTool[Any], Any]]:
        """Resolve every step before modifying data; identify failures by step number."""
        resolved = []
        for index, step in enumerate(pipeline.steps, start=1):
            try:
                tool = self.registry.get(step.tool)
                params = tool.params_model.model_validate(step.params)
                resolved.append((tool, params))
            except ValidationError as exc:
                raise PipelineValidationError(
                    f"Step {index} ({step.tool}): {exc}",
                    [
                        {"step_index": index, "field": list(error["loc"]), "type": error["type"]}
                        for error in exc.errors(include_input=False, include_context=False)
                    ],
                ) from exc
            except PipelineError as exc:
                raise PipelineValidationError(
                    f"Step {index} ({step.tool}): {exc}",
                    [{"step_index": index, "field": ["tool"], "type": "unknown_tool"}],
                ) from exc
        if input_kind is not None:
            state = input_kind
            for index, (tool, _) in enumerate(resolved, 1):
                required = "dataset" if isinstance(tool, DatasetTool) else "image"
                if state != required:
                    raise PipelineValidationError(
                        f"Step {index} ({tool.name}) requires {required} input; received {state}.",
                        [{"step_index": index, "field": [], "type": "input_transition"}],
                    )
                if isinstance(tool, DatasetTool):
                    state = tool.output_kind
        return resolved

    @execution_scope
    def execute(
        self,
        image: AstroImage,
        pipeline: PipelineDefinition,
        *,
        context: ExecutionContext | None = None,
    ) -> PipelineResult:
        """Execute deterministic tools and record resolved parameters and diagnostics."""
        emit_progress("validating")
        resolved = self.validate(pipeline)
        if any(isinstance(tool, DatasetTool) for tool, _ in resolved):
            raise PipelineError("Dataset tools require directory input, not a single AstroImage.")
        report = ProcessingReport(
            input_hdu=image.input_hdu,
            input=None if image.path is None else str(image.path),
            package_version=__version__,
            dependency_versions={
                name: version(name)
                for name in ("numpy", "scipy", "astropy", "photutils", "pydantic")
            },
            input_sha256=image_digest(image),
            pipeline=PipelineDefinition(
                steps=[
                    PipelineStep(tool=tool.name, params=params.model_dump(mode="json"))
                    for tool, params in resolved
                ]
            ),
            metrics_before=inspect_image(image),
            metrics_after=inspect_image(image),
        )
        current = image
        emit_progress("pipeline", 0, len(resolved), "step")
        for index, (tool, params) in enumerate(resolved, start=1):
            checkpoint()
            assert isinstance(tool, ImageTool)
            logger.info("Step %d: %s", index, tool.name)
            started = perf_counter()
            try:
                result = tool.execute(current, params)
            except (AstroError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                raise PipelineError(f"Step {index} ({tool.name}) failed: {exc}") from exc
            report.steps.append(
                StepReport(
                    tool=tool.name,
                    params=params.model_dump(mode="json"),
                    metrics_before=result.metrics_before,
                    metrics_after=result.metrics_after,
                    duration_ms=(perf_counter() - started) * 1000,
                    warnings=result.warnings,
                )
            )
            report.warnings.extend(result.warnings)
            current = result.image
            emit_progress("pipeline", index, len(resolved), "step", step_index=index)
        report.metrics_after = inspect_image(current)
        report.output_sha256 = image_digest(current)
        return PipelineResult(current, report)

    @execution_scope
    def run(
        self,
        pipeline: PipelineDefinition,
        input_path: Path | str,
        output_path: Path | str,
        *,
        overwrite: bool = False,
        hdu: int | None = None,
        context: ExecutionContext | None = None,
    ) -> PipelineResult | DatasetPipelineResult:
        """Load an image, execute, and save it with resolved YAML and a JSON sidecar."""
        resolved = self.validate(pipeline)
        if Path(input_path).is_dir():
            if hdu is not None:
                raise PipelineError(
                    "HDU selection applies to one FITS image, not a directory dataset."
                )
            return execute_dataset_pipeline(
                load_dataset(Path(input_path), recursive=True),
                pipeline,
                resolved,
                Path(output_path),
                overwrite=overwrite,
            )
        check_output_paths(Path(input_path), Path(output_path), overwrite=overwrite)
        emit_progress("loading")
        result = self.execute(load_image(input_path, hdu=hdu), pipeline)
        emit_progress("saving")
        save_result(result, output_path, overwrite=overwrite)
        return result


def artifact_paths(output_path: Path) -> tuple[Path, Path, Path]:
    """Return image, processing report, and replay pipeline paths for an output."""
    base = output_path.with_suffix("") if output_path.suffix.lower() == ".gz" else output_path
    return output_path, base.with_suffix(".processing.json"), base.with_suffix(".pipeline.yaml")


def check_output_paths(input_path: Path | None, output_path: Path, *, overwrite: bool) -> None:
    """Reject source replacement and existing artifacts before running tools."""
    paths = artifact_paths(output_path)
    image_format(output_path)
    if len({path.resolve() for path in paths}) != 3:
        raise PipelineError("Image output must have a supported filename, distinct from sidecars.")
    if input_path is not None and any(
        input_path.resolve() == path.resolve()
        or (input_path.exists() and path.exists() and input_path.samefile(path))
        for path in paths
    ):
        raise PipelineError("Output artifacts must not replace the input image.")
    if not output_path.parent.is_dir():
        raise PipelineError(f"Output directory does not exist: {output_path.parent}")
    if not overwrite:
        for path in paths:
            if path.exists():
                raise PipelineError(
                    f"Output already exists: {path}. Use --overwrite to replace it."
                )


def save_result(
    result: PipelineResult, output_path: Path | str, *, overwrite: bool = False
) -> None:
    """Write related artifacts after checking all destinations for collisions.

    Each file is atomic. The three-file bundle is not a filesystem transaction;
    an I/O failure may leave a partial bundle and is reported to the caller.
    """
    output = Path(output_path)
    check_output_paths(result.image.path, output, overwrite=overwrite)
    image_path, report_path, pipeline_path = artifact_paths(output)
    result.report.output = str(output)
    save_image(result.image, image_path, overwrite=overwrite)
    exported = load_image(image_path)
    result.report.output_sha256 = image_digest(exported)
    result.report.output_file_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    result.report.exported_metrics = inspect_image(exported)
    if image_format(output) not in {"fits", "tiff"}:
        result.report.warnings.append(
            "Display raster export quantizes pixels; JPEG additionally uses lossy compression."
        )
    save_pipeline(result.report.pipeline, pipeline_path, overwrite=overwrite)
    save_report(result.report, report_path, overwrite=overwrite)
