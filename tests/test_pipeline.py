import json

import numpy as np
import pytest
from pydantic import ValidationError

from astroagent.errors import PipelineError
from astroagent.io import load_image
from astroagent.pipeline.executor import PipelineExecutor, artifact_paths, image_digest
from astroagent.pipeline.models import PipelineDefinition, PipelineStep
from astroagent.pipeline.serialization import dump_pipeline, load_pipeline, save_pipeline
from astroagent.tools.base import ImageTool
from astroagent.tools.normalization import NormalizeParams
from astroagent.tools.registry import ToolRegistry


def sample_pipeline():
    return PipelineDefinition(
        steps=[
            PipelineStep(tool="normalize"),
            PipelineStep(tool="denoise", params={"sigma": 0.8}),
            PipelineStep(tool="stretch", params={"strength": 0.5}),
        ]
    )


def test_serialization_and_defaults(tmp_path):
    definition = sample_pipeline()
    path = save_pipeline(definition, tmp_path / "pipeline.yaml")
    assert load_pipeline(path) == definition
    assert dump_pipeline(definition).startswith("version: 1\nsteps:")
    with pytest.raises(PipelineError, match="exists"):
        save_pipeline(definition, path)
    save_pipeline(definition, path, overwrite=True)


@pytest.mark.parametrize(
    "text",
    [
        "version: 2\nsteps: []",
        "version: 1\nsteps: nope",
        "unknown: true",
        "!!python/object/apply:os.system ['bad']",
        "steps: [",
        "",
    ],
)
def test_invalid_pipeline_yaml(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(PipelineError, match="Cannot load"):
        load_pipeline(path)


def test_missing_yaml(tmp_path):
    with pytest.raises(PipelineError, match="Cannot load"):
        load_pipeline(tmp_path / "missing.yaml")


def test_pipeline_execution_reports_and_replay(tmp_path, fits_path):
    output = tmp_path / "processed.fit"
    result = PipelineExecutor().run(sample_pipeline(), fits_path, output)
    assert len(result.report.steps) == 3
    assert result.report.output == str(output)
    assert result.report.steps[0].params == {"lower": 0, "upper": 1}
    assert all(
        step.duration_ms >= 0 and step.metrics_before is not None for step in result.report.steps
    )
    assert result.image.header["OBJECT"] == "Synthetic nebula"
    assert result.report.package_version
    assert result.report.dependency_versions["numpy"]
    report = json.loads(output.with_suffix(".processing.json").read_text())
    assert report["output_sha256"] == image_digest(load_image(output))
    assert report["output_file_sha256"]
    replay = PipelineExecutor().run(
        load_pipeline(output.with_suffix(".pipeline.yaml")), fits_path, tmp_path / "replay.fit"
    )
    np.testing.assert_array_equal(replay.image.data, result.image.data)
    assert replay.report.output_sha256 == result.report.output_sha256


def test_empty_pipeline_preserves_pixels(image):
    result = PipelineExecutor().execute(image, PipelineDefinition())
    np.testing.assert_array_equal(result.image.data, image.data)
    assert not result.report.steps
    assert result.report.input_sha256 == result.report.output_sha256


def test_invalid_tools_and_params_are_preflighted(image):
    for steps, pattern in [
        ([PipelineStep(tool="missing")], "Unknown tool"),
        (
            [PipelineStep(tool="normalize"), PipelineStep(tool="stretch", params={"strength": 10})],
            "Step 2",
        ),
        ([PipelineStep(tool="denoise", params={"typo": 1})], "typo"),
    ]:
        before = image.data.copy()
        with pytest.raises(PipelineError, match=pattern):
            PipelineExecutor().execute(image, PipelineDefinition(steps=steps))
        np.testing.assert_array_equal(image.data, before)


def test_failure_does_not_save_outputs(tmp_path, fits_path):
    pipeline = PipelineDefinition(
        steps=[PipelineStep(tool="stretch", params={"black_point": 5000})]
    )
    with pytest.raises(PipelineError, match="Step 1.*failed"):
        PipelineExecutor().run(pipeline, fits_path, tmp_path / "output.fit")
    assert not any(path.exists() for path in artifact_paths(tmp_path / "output.fit"))


def test_outputs_and_source_cannot_be_overwritten_accidentally(tmp_path, fits_path):
    executor = PipelineExecutor()
    with pytest.raises(PipelineError, match="input image"):
        executor.run(sample_pipeline(), fits_path, fits_path, overwrite=True)
    output = tmp_path / "output.fit"
    output.with_suffix(".processing.json").write_text("existing")
    with pytest.raises(PipelineError, match="exists"):
        executor.run(sample_pipeline(), fits_path, output)
    assert not output.exists()
    executor.run(sample_pipeline(), fits_path, output, overwrite=True)
    with pytest.raises(PipelineError, match="directory"):
        executor.run(sample_pipeline(), fits_path, tmp_path / "absent" / "out.fit")


def test_compressed_paths_and_raster_export_report(tmp_path, fits_path):
    assert artifact_paths(tmp_path / "result.fits.gz")[1].name == "result.processing.json"
    output = tmp_path / "output.png"
    result = PipelineExecutor().run(sample_pipeline(), fits_path, output)
    assert result.report.exported_metrics.datatype == "uint16"
    assert result.report.output_sha256 == image_digest(load_image(output))
    assert any("quantizes" in warning for warning in result.report.warnings)


def test_custom_tool_can_execute_without_engine_changes(image):
    class OffsetTool(ImageTool[NormalizeParams]):
        name = "custom"
        description = "Test custom extension"
        params_model = NormalizeParams

        def process(self, image, params):
            return image.with_data(image.data.astype(float) + params.upper), []

    executor = PipelineExecutor(ToolRegistry([OffsetTool()]))
    result = executor.execute(image, PipelineDefinition(steps=[PipelineStep(tool="custom")]))
    np.testing.assert_array_equal(result.image.data, image.data + 1)


def test_nonfinite_tool_result_rejected(image):
    class BadTool(ImageTool[NormalizeParams]):
        name = "bad"
        description = "Broken custom tool"
        params_model = NormalizeParams

        def process(self, image, params):
            return image.with_data(np.full(image.data.shape, np.nan)), []

    with pytest.raises(PipelineError, match="nonfinite"):
        PipelineExecutor(ToolRegistry([BadTool()])).execute(
            image, PipelineDefinition(steps=[PipelineStep(tool="bad")])
        )


def test_schema_rejects_extra_fields_and_non_json_params():
    with pytest.raises(ValidationError):
        PipelineDefinition.model_validate({"version": 1, "steps": [], "unknown": True})
    with pytest.raises(ValidationError):
        PipelineStep(tool="stretch", params={"data": object()})
    for value in (True, "1", 1.0):
        with pytest.raises(ValidationError):
            PipelineDefinition.model_validate({"version": value, "steps": []})
