import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError
from synthetic import make_session, star_image, star_positions, write_frame

from astroagent.agent.dataset_planner import inspect_dataset
from astroagent.agent.providers import OpenAIPlanner
from astroagent.calibration.masters import build_master
from astroagent.calibration.models import CalibrationPlan, FrameType
from astroagent.calibration.session import inspect_frame
from astroagent.errors import PipelineError
from astroagent.io.fits import load_fits, save_fits
from astroagent.models.dataset import AstroDataset
from astroagent.models.layout import ImageLayout
from astroagent.pipeline.executor import PipelineExecutor
from astroagent.pipeline.models import PipelineDefinition, PipelineStep
from astroagent.registration.engine import RegistrationParams, register_frames
from astroagent.stacking.workflow import StackParams, save_stack, stack_frames
from astroagent.tools.registry import default_registry


def test_invalid_master_frame_is_rejected_instead_of_aborting(tmp_path):
    paths = [
        write_frame(tmp_path / f"bias_{i}.fit", np.full((10, 10), 1000.0), "BIAS") for i in range(5)
    ]
    paths.append(write_frame(tmp_path / "bad.fit", np.full((10, 10), np.nan), "BIAS"))
    master = build_master(
        [inspect_frame(p) for p in paths], FrameType.BIAS, tmp_path / "master.fit"
    )
    assert len(master.input_paths) == 5
    assert str(paths[-1]) in master.rejected
    np.testing.assert_array_equal(load_fits(master.path).data, 1000)


def test_process_skips_unreadable_session_file_and_keeps_warning(tmp_path):
    root = tmp_path / "session"
    make_session(root)
    (root / "lights" / "corrupt.fit").write_bytes(b"broken")
    pipeline = PipelineDefinition(
        steps=[
            PipelineStep(tool="calibrate_frames", params={"debayer": True}),
            PipelineStep(tool="register_frames"),
            PipelineStep(tool="stack_frames"),
        ]
    )
    result = PipelineExecutor().run(pipeline, root, tmp_path / "master.fit")
    assert result.report["used_frames"] == 3
    assert any("corrupt.fit" in w for w in result.report["steps"][0]["report"]["warnings"])


def test_registered_directory_can_move_and_reports_failures(tmp_path):
    inputs = []
    for i in range(2):
        inputs.append(
            save_fits(star_image(star_positions() + [i, i], seed=i), tmp_path / f"light_{i}.fit")
        )
    (tmp_path / "invalid.fit").write_bytes(b"broken")
    inputs.append(tmp_path / "invalid.fit")
    original = tmp_path / "registered"
    registered = register_frames(
        AstroDataset(inputs), original, RegistrationParams(reference="light_0.fit")
    )
    assert len(registered.frames) == 2
    moved = tmp_path / "moved"
    original.rename(moved)
    paths = sorted(moved.glob("*.fit"))
    image, report = stack_frames(AstroDataset(paths, source=moved), StackParams(normalize=False))
    assert report["used_frames"] == 2 and report["rejected_frames"] == 1
    assert any(not f["used"] and f["path"].endswith("invalid.fit") for f in report["frames"])
    assert all(f["weight"] > 0 for f in report["frames"] if f["used"])
    metrics = inspect_dataset(moved)
    assert metrics.registration_statistics["failed_frames"] == 1
    assert metrics.selected_reference is not None
    save_stack(image, report, tmp_path / "master.fits.gz")
    assert (tmp_path / "master.processing.json").exists()


def test_existing_provider_plans_datasets_without_pixels(tmp_path):
    root = tmp_path / "session"
    make_session(root, cfa=True)
    metrics = inspect_dataset(root)
    client = Mock()
    plan = {
        "pipeline": {
            "version": 1,
            "steps": [
                {"tool": "calibrate_frames", "params": {"debayer": True}},
                {"tool": "register_frames"},
                {"tool": "stack_frames"},
            ],
        },
        "reasoning": ["Calibrate before registering."],
    }
    client.responses.create.return_value = SimpleNamespace(
        status="completed", output_text=json.dumps(plan)
    )
    planner = OpenAIPlanner("test-model", client=client)
    result = planner.create_plan("process", metrics, default_registry().describe())
    payload = json.loads(client.responses.create.call_args.kwargs["input"])
    assert payload["dataset_metrics"]["number_of_frames"] == 3
    assert payload["dataset_metrics"]["session_counts"]["bias"] == 5
    assert all("data" not in f for f in payload["dataset_metrics"]["frames"])
    assert result.pipeline.steps[0].params["debayer"] is True


@pytest.mark.parametrize(
    "tool,params",
    [
        ("calibrate_frames", {"cfa_pattern": "bad"}),
        ("debayer_frames", {"pattern": "bad"}),
        ("build_masters", {"cfa_pattern": "bad"}),
    ],
)
def test_pattern_validated_before_execution(tool, params):
    with pytest.raises(PipelineError, match="CFA pattern"):
        PipelineExecutor().validate(
            PipelineDefinition(steps=[PipelineStep(tool=tool, params=params)])
        )
    with pytest.raises(ValidationError):
        CalibrationPlan(cfa_pattern="bad")


def test_cfa_alias_survives_loading(tmp_path):
    path = write_frame(tmp_path / "raw.fit", np.ones((10, 10)), BAYERPATN="RGGB")
    assert load_fits(path).layout == ImageLayout.CFA
