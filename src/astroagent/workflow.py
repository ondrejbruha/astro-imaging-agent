from pathlib import Path

from astroagent.pipeline.dataset_executor import DatasetPipelineResult
from astroagent.pipeline.executor import PipelineExecutor
from astroagent.pipeline.models import PipelineDefinition, PipelineStep


def process_session(
    source: Path,
    output: Path,
    *,
    cfa_pattern: str | None = None,
    method: str = "weighted-sigma-clipped",
    overwrite: bool = False,
) -> DatasetPipelineResult:
    """Compose deterministic calibration, RGB conversion, registration and stacking tools."""
    pipeline = PipelineDefinition(
        steps=[
            PipelineStep(tool="build_masters", params={"cfa_pattern": cfa_pattern}),
            PipelineStep(
                tool="calibrate_frames", params={"debayer": True, "cfa_pattern": cfa_pattern}
            ),
            PipelineStep(tool="register_frames"),
            PipelineStep(tool="stack_frames", params={"method": method}),
        ]
    )
    result = PipelineExecutor().run(pipeline, source, output / "master.fit", overwrite=overwrite)
    assert isinstance(result, DatasetPipelineResult)
    return result
