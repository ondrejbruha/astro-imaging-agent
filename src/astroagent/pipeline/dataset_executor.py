import logging
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

from astroagent import __version__
from astroagent.errors import PipelineError
from astroagent.execution import ExecutionContext, checkpoint, emit_progress, execution_scope
from astroagent.io.datasets import discover_fits, write_json
from astroagent.io.export import export_image
from astroagent.io.images import image_format
from astroagent.models.dataset import AstroDataset
from astroagent.models.image import AstroImage
from astroagent.pipeline.models import PipelineDefinition, PipelineStep
from astroagent.pipeline.serialization import save_pipeline
from astroagent.tools.base import ImageTool
from astroagent.tools.dataset import DatasetContext, DatasetTool

logger = logging.getLogger(__name__)


@dataclass
class DatasetPipelineResult:
    """Final image or dataset plus resolved, provider-independent execution provenance."""

    value: AstroDataset | AstroImage
    report: dict[str, Any]


@execution_scope
def execute_dataset_pipeline(
    dataset: AstroDataset,
    pipeline: PipelineDefinition,
    resolved: list[tuple[ImageTool[Any] | DatasetTool[Any], Any]],
    output: Path,
    *,
    overwrite: bool = False,
    context: ExecutionContext | None = None,
) -> DatasetPipelineResult:
    """Execute typed dataset/image transitions using a disk-backed artifact workspace."""
    image_output = any(
        isinstance(tool, DatasetTool) and tool.output_kind == "image" for tool, _ in resolved
    )
    work = output.parent / f"{output.stem}.work" if image_output else output
    base = output.with_suffix("") if output.suffix.lower() == ".gz" else output
    report_path, pipeline_path = (
        base.with_suffix(".processing.json"),
        base.with_suffix(".pipeline.yaml"),
    )
    if image_output:
        image_format(output)
    sources = {p.resolve() for p in dataset.frames}
    if output.resolve() in sources or any(p.is_relative_to(work.resolve()) for p in sources):
        raise PipelineError("Output artifacts must not replace input frames.")
    if not overwrite:
        destinations = [
            work,
            output,
            report_path,
            pipeline_path,
        ]
        if any(p.exists() for p in destinations):
            raise PipelineError("Pipeline output or workspace already exists; use --overwrite.")
    # Reject incompatible transitions before any processing or file writes.
    state = "dataset"
    for tool, _ in resolved:
        if isinstance(tool, DatasetTool):
            if state != "dataset":
                raise PipelineError(
                    f"Tool {tool.name} requires a dataset, but the previous step produced an image."
                )
            state = tool.output_kind
        elif state != "image":
            raise PipelineError(
                f"Tool {tool.name} requires an image; stack_frames must precede it."
            )
    if state == "image" and not image_output:
        raise PipelineError("Dataset pipeline must explicitly combine frames.")
    resolved_pipeline = PipelineDefinition(
        steps=[
            PipelineStep(tool=tool.name, params=params.model_dump(mode="json"))
            for tool, params in resolved
        ]
    )
    report: dict[str, Any] = {
        "package_version": __version__,
        "dependency_versions": {
            name: version(name) for name in ("numpy", "scipy", "astropy", "photutils", "pydantic")
        },
        "input_paths": [str(p) for p in dataset.frames],
        "output": str(output),
        "pipeline": resolved_pipeline.model_dump(mode="json"),
        "steps": [],
        "warnings": [],
    }
    current: AstroDataset | AstroImage = dataset
    emit_progress("pipeline", 0, len(resolved), "step")
    for index, (tool, params) in enumerate(resolved, 1):
        checkpoint()
        logger.info("Step %d: %s", index, tool.name)
        tick = perf_counter()
        destination = work / f"{index:02d}-{tool.name}"
        if isinstance(tool, DatasetTool):
            assert isinstance(current, AstroDataset)
            result = tool.execute(current, params, DatasetContext(destination, overwrite))
            current, diagnostics = result.value, result.report
            if tool.output_kind == "image":
                report.update(diagnostics)
        else:
            assert isinstance(current, AstroImage)
            image_result = tool.execute(current, params)
            current = image_result.image
            diagnostics = {"warnings": image_result.warnings}
        report["steps"].append(
            {
                "tool": tool.name,
                "params": params.model_dump(mode="json"),
                "duration_ms": (perf_counter() - tick) * 1000,
                "report": diagnostics,
            }
        )
        emit_progress("pipeline", index, len(resolved), "step", step_index=index)
    emit_progress("saving")
    if isinstance(current, AstroImage):
        output.parent.mkdir(parents=True, exist_ok=True)
        report["export"] = export_image(current, output, overwrite=overwrite)
    else:
        output.mkdir(parents=True, exist_ok=True)
        report["output_frames"] = [str(p) for p in current.frames]
        report_path, pipeline_path = output / "processing.json", output / "pipeline.yaml"
    write_json(report_path, report, overwrite=overwrite)
    save_pipeline(resolved_pipeline, pipeline_path, overwrite=overwrite)
    return DatasetPipelineResult(current, report)


def load_dataset(source: Path, *, recursive: bool = False) -> AstroDataset:
    """Create an explicit path dataset for CLI and saved pipeline execution."""
    return AstroDataset(discover_fits(source, recursive=recursive), source=source)
