import json
import subprocess
import sys

import numpy as np
import pytest
from synthetic import make_session, star_image, star_positions
from typer.testing import CliRunner

from astroagent.cli.main import app
from astroagent.errors import PipelineError
from astroagent.io.fits import load_fits, save_fits
from astroagent.io.images import load_image
from astroagent.pipeline.dataset_executor import DatasetPipelineResult
from astroagent.pipeline.executor import PipelineExecutor
from astroagent.pipeline.models import PipelineDefinition, PipelineStep
from astroagent.pipeline.serialization import load_pipeline, save_pipeline

runner = CliRunner()


@pytest.fixture
def lights(tmp_path):
    root = tmp_path / "lights"
    root.mkdir()
    for i, offset in enumerate([(0, 0), (3, -2), (-1, 2)]):
        save_fits(star_image(star_positions() + offset, seed=i), root / f"light_{i}.fit")
    return root


def invoke(args):
    result = runner.invoke(app, list(map(str, args)))
    assert result.exit_code == 0, result.output
    return result


def test_analyze_select_register_stack_cli_and_report(lights, tmp_path):
    analysis = json.loads(invoke(["analyze-frames", lights, "--json"]).stdout)
    assert len(analysis["frames"]) == 3
    reference = json.loads(invoke(["select-reference", lights, "--json"]).stdout)
    assert reference["selected"].endswith(".fit")
    registered = tmp_path / "registered"
    invoke(["register", lights, "--reference", "light_0.fit", "--output", registered])
    output = tmp_path / "master.fit"
    invoke(["stack", registered, "--method", "weighted-sigma-clipped", "--output", output])
    report = json.loads(output.with_suffix(".processing.json").read_text())
    assert report["used_frames"] == 3 and report["registration"]["median_residual_rms"] < 0.1
    assert sum(f["weight"] for f in report["frames"]) == pytest.approx(1)
    master = load_fits(output)
    assert master.header["NCOMBINE"] == 3 and master.header["OBJECT"] == "Synthetic stars"
    assert "Stack method" in str(master.header["HISTORY"])


@pytest.mark.parametrize("extension", ["fit", "tiff", "jpg", "png"])
def test_combined_registration_stack_export(lights, tmp_path, extension):
    output = tmp_path / f"master.{extension}"
    invoke(["stack", lights, "--register", "--method", "mean", "--output", output])
    exported = load_image(output)
    assert exported.data.shape == (112, 128)
    report = json.loads(output.with_suffix(".processing.json").read_text())
    if extension in ("jpg", "png"):
        assert report["export"]["display_transform"]["method"] == "asinh"
        assert exported.data.dtype.kind == "u"
    else:
        assert exported.data.dtype.kind == "f"


@pytest.mark.parametrize("cfa", [False, True])
def test_full_session_process_and_pipeline_replay(tmp_path, cfa):
    source = tmp_path / "session"
    make_session(source, cfa=cfa)
    invoke(["session", "inspect", source])
    before = sorted(source.rglob("*"))
    dry = json.loads(invoke(["calibrate", source, "--dry-run", "--explain"]).stdout)
    assert dry["session_counts"]["light"] == 3 and sorted(source.rglob("*")) == before
    result = tmp_path / "result"
    invoke(["process", source, "--output", result])
    first = load_fits(result / "master.fit")
    assert first.data.ndim == (3 if cfa else 2)
    replay = tmp_path / "replay.fit"
    invoke(["run", result / "master.pipeline.yaml", "--input", source, "--output", replay])
    np.testing.assert_array_equal(first.data, load_fits(replay).data)
    report = json.loads((result / "master.processing.json").read_text())
    assert len(report["steps"]) == 4 and report["pipeline"]["steps"][0]["tool"] == "build_masters"
    assert report["used_frames"] == 3


def test_dataset_pipeline_serialization_and_transitions(lights, tmp_path):
    pipeline = PipelineDefinition(
        steps=[
            PipelineStep(tool="register_frames"),
            PipelineStep(
                tool="stack_frames", params={"method": "mean", "reject_worst_fraction": 0.1}
            ),
        ]
    )
    path = save_pipeline(pipeline, tmp_path / "dataset.yaml")
    assert load_pipeline(path) == pipeline
    result = PipelineExecutor().run(pipeline, lights, tmp_path / "pipeline.fit")
    assert isinstance(result, DatasetPipelineResult)
    assert result.report["used_frames"] == 3
    invalid = PipelineDefinition(steps=[PipelineStep(tool="normalize")])
    with pytest.raises(PipelineError, match="requires an image"):
        PipelineExecutor().run(invalid, lights, tmp_path / "invalid.fit")
    assert not (tmp_path / "invalid.fit").exists()


def test_dataset_agent_plan_is_replayable_and_dry_run_is_read_only(lights, tmp_path):
    result = invoke(["agent", lights, "create a good master", "--dry-run", "--explain"])
    assert "register_frames" in result.stdout and "weighted-sigma-clipped" in result.stdout
    assert "Reason:" in result.stderr
    invoke(["agent", lights, "create a good master", "--output", tmp_path / "agent.fit"])
    assert (tmp_path / "agent.pipeline.yaml").exists()


def test_numerical_modules_import_without_agent_or_pipeline():
    code = (
        "import sys; import astroagent.registration.engine; "
        "import astroagent.calibration.masters; import astroagent.stacking.workflow; "
        "assert not any(k.startswith('astroagent.agent') for k in sys.modules); "
        "assert 'astroagent.pipeline.executor' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_rejection_too_few_frames_and_outputs_not_overwritten(lights, tmp_path):
    out = tmp_path / "out.fit"
    failed = runner.invoke(
        app, ["stack", str(lights), "--reject-worst", "90%", "--output", str(out)]
    )
    assert failed.exit_code == 1 and "usable frames" in failed.stderr
    assert not out.exists()
    invoke(["stack", lights, "--output", out])
    original = out.read_bytes()
    failed = runner.invoke(app, ["stack", str(lights), "--output", str(out)])
    assert failed.exit_code == 1 and out.read_bytes() == original
