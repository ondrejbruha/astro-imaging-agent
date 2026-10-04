import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml
from synthetic import make_session

from astroagent.agent.autonomous import AgentOptions, AutonomousAgent, evaluation_score
from astroagent.agent.models import PlanResult
from astroagent.agent.prompts import SYSTEM_PROMPT
from astroagent.agent.providers import OpenAIPlanner
from astroagent.analysis.statistics import inspect_image
from astroagent.errors import PipelineError
from astroagent.io.fits import load_fits, save_fits
from astroagent.models.dataset import DatasetMetrics
from astroagent.models.image import AstroImage
from astroagent.pipeline.executor import PipelineExecutor
from astroagent.pipeline.models import PipelineDefinition, PipelineStep
from astroagent.pipeline.serialization import load_pipeline


def plan(*steps, alternatives=()):
    return PlanResult(
        pipeline=PipelineDefinition(steps=list(steps)),
        alternatives=list(alternatives),
        reasoning=["Numerical synthetic experiment."],
    )


class SequencePlanner:
    def __init__(self, plans):
        self.plans = iter(plans)
        self.feedback = []

    def create_plan(self, request, metrics, tools):
        return next(self.plans)

    def revise_plan(self, request, metrics, tools, feedback):
        self.feedback.append(feedback)
        return next(self.plans)


@pytest.fixture
def source(tmp_path):
    rng = np.random.default_rng(42)
    y, x = np.indices((32, 32))
    data = 0.3 + 0.2 * np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 10) + rng.normal(0, 0.006, x.shape)
    return save_fits(AstroImage(data.astype(np.float32)), tmp_path / "source.fit")


def test_iterative_alternatives_feedback_and_exact_replay(source, tmp_path):
    alternate = PipelineDefinition(steps=[PipelineStep(tool="denoise", params={"sigma": 1.5})])
    planner = SequencePlanner(
        [
            plan(PipelineStep(tool="denoise", params={"sigma": 0.4}), alternatives=[alternate]),
            plan(PipelineStep(tool="normalize")),
            plan(),
        ]
    )
    agent = AutonomousAgent(planner)
    prepared = agent.prepare(source, "compare and refine")
    assert sorted(tmp_path.iterdir()) == [source]
    original = source.read_bytes()
    output = tmp_path / "final.fit"
    result = agent.run(prepared, output, options=AgentOptions(quality_drop_tolerance=1))
    attempts = result.report["agent"]["iterations"]
    assert len(attempts) == 3 and sum(a["accepted"] for a in attempts) == 2
    chosen = next(a for a in attempts[:2] if a["accepted"])
    assert chosen["score"] == max(a["score"] for a in attempts[:2])
    assert len(planner.feedback) == 2
    assert planner.feedback[-1]["executed_tools"] == ["denoise", "normalize"]
    assert '"data":' not in json.dumps(planner.feedback[-1])
    assert result.report["agent"]["stop_reason"] == "planner finished"
    replay = PipelineExecutor().execute(
        load_fits(source), load_pipeline(output.with_suffix(".pipeline.yaml"))
    )
    np.testing.assert_array_equal(result.image.data, replay.image.data)
    assert source.read_bytes() == original
    assert len(list((tmp_path / "final.agent").glob("*.proposal.yaml"))) == 3
    assert len(list((tmp_path / "final.agent").glob("*.fit"))) == 3


def test_quality_decline_retains_previous_accepted_result(source, tmp_path, monkeypatch):
    planner = SequencePlanner(
        [
            plan(PipelineStep(tool="normalize")),
            plan(PipelineStep(tool="normalize", params={"upper": 0.5})),
        ]
    )
    monkeypatch.setattr("astroagent.agent.autonomous.evaluation_score", lambda m: m.max)
    agent = AutonomousAgent(planner)
    result = agent.run(
        agent.prepare(source, "evaluate"),
        tmp_path / "kept.fit",
        options=AgentOptions(quality_drop_tolerance=0),
    )
    assert result.report["agent"]["stop_reason"] == "quality decline"
    assert result.image.data.max() == 1
    assert len(result.report["pipeline"]["steps"]) == 1
    assert result.report["agent"]["iterations"][-1]["accepted"] is False


def test_failed_candidate_does_not_prevent_valid_alternative(source, tmp_path):
    planner = SequencePlanner(
        [
            plan(
                PipelineStep(tool="color_adjust"),
                alternatives=[PipelineDefinition(steps=[PipelineStep(tool="normalize")])],
            ),
            plan(),
        ]
    )
    agent = AutonomousAgent(planner)
    result = agent.run(agent.prepare(source, "experiment"), tmp_path / "ok.fit")
    attempts = result.report["agent"]["iterations"]
    assert "requires RGB" in attempts[0]["error"] and attempts[1]["accepted"]


@pytest.mark.parametrize("case", ["repeated", "unchanged", "empty", "failure"])
def test_bounded_stop_conditions(source, tmp_path, case):
    first = plan(PipelineStep(tool="normalize"))
    if case == "empty":
        plans = [plan()]
    elif case == "unchanged":
        plans = [plan(PipelineStep(tool="sharpen", params={"amount": 0}))]
    elif case == "failure":
        plans = [plan(PipelineStep(tool="color_adjust"))]
    else:
        plans = [first, first]
    agent = AutonomousAgent(SequencePlanner(plans))
    prepared = agent.prepare(source, "bounded")
    if case == "failure":
        with pytest.raises(PipelineError, match="All agent candidates failed"):
            agent.run(prepared, tmp_path / "final.fit")
        assert not (tmp_path / "final.fit").exists()
    else:
        result = agent.run(prepared, tmp_path / "final.fit")
        assert (
            result.report["agent"]["stop_reason"]
            == {
                "repeated": "no new successful candidate",
                "unchanged": "no pixel change",
                "empty": "planner finished",
            }[case]
        )


def test_replanning_failure_preserves_result_and_reports_problem(source, tmp_path):
    class FailingPlanner(SequencePlanner):
        def revise_plan(self, *args):
            raise PipelineError("provider unavailable")

    agent = AutonomousAgent(FailingPlanner([plan(PipelineStep(tool="normalize"))]))
    result = agent.run(agent.prepare(source, "complete"), tmp_path / "safe.fit")
    assert "provider unavailable" in result.report["agent"]["stop_reason"]
    assert result.image.data.max() == 1


def test_invalid_initial_plan_and_input_replacement_rejected_before_processing(source, tmp_path):
    agent = AutonomousAgent(
        SequencePlanner([plan(PipelineStep(tool="normalize", params={"upper": 0}))])
    )
    with pytest.raises(PipelineError, match="Step 1"):
        agent.prepare(source, "test")
    agent = AutonomousAgent(SequencePlanner([plan()]))
    prepared = agent.prepare(source, "test")
    with pytest.raises(PipelineError, match="input frames"):
        agent.run(prepared, source, overwrite=True)
    assert list(tmp_path.iterdir()) == [source]


def test_max_candidates_and_iteration_limits(source, tmp_path):
    alternate = PipelineDefinition(steps=[PipelineStep(tool="sharpen")])
    agent = AutonomousAgent(
        SequencePlanner([plan(PipelineStep(tool="normalize"), alternatives=[alternate])])
    )
    prepared = agent.prepare(source, "limits")
    result = agent.run(
        prepared, tmp_path / "limited.fit", options=AgentOptions(max_iterations=1, max_candidates=1)
    )
    assert len(result.report["agent"]["iterations"]) == 1
    assert result.report["agent"]["stop_reason"] == "iteration limit reached"
    with pytest.raises(PipelineError, match="already exists"):
        agent.run(prepared, tmp_path / "limited.fit")


def test_cfa_session_autonomous_separate_rounds_and_replay(tmp_path):
    session = tmp_path / "session"
    make_session(session, cfa=True)
    planner = SequencePlanner(
        [
            plan(PipelineStep(tool="build_masters")),
            plan(PipelineStep(tool="calibrate_frames", params={"debayer": True})),
            plan(PipelineStep(tool="register_frames")),
            plan(PipelineStep(tool="stack_frames")),
            plan(),
        ]
    )
    agent = AutonomousAgent(planner)
    prepared = agent.prepare(session, "do the entire session autonomously")
    output = tmp_path / "master.fit"
    result = agent.run(prepared, output, options=AgentOptions(max_iterations=5))
    assert result.image.channels == 3
    assert result.report["used_frames"] == 3
    assert len(result.report["agent"]["iterations"]) == 4
    assert [s["tool"] for s in result.report["pipeline"]["steps"]] == [
        "build_masters",
        "calibrate_frames",
        "register_frames",
        "stack_frames",
    ]
    assert planner.feedback[2]["executed_tools"][-1] == "register_frames"
    assert planner.feedback[2]["attempts"][-1]["metrics_after"]["registration_statistics"]["frames"]
    replay = PipelineExecutor().run(
        load_pipeline(output.with_suffix(".pipeline.yaml")), session, tmp_path / "replay.fit"
    )
    np.testing.assert_array_equal(result.image.data, replay.value.data)


def test_dataset_stopped_before_master_has_readable_error(tmp_path):
    session = tmp_path / "session"
    make_session(session)
    agent = AutonomousAgent(SequencePlanner([plan(PipelineStep(tool="build_masters"))]))
    with pytest.raises(PipelineError, match="before producing a master"):
        agent.run(
            agent.prepare(session, "build"),
            tmp_path / "unfinished.fit",
            options=AgentOptions(max_iterations=1),
        )


def test_evaluation_score_bounds_and_dataset_quality():
    image = AstroImage(np.linspace(0, 0.8, 1024, dtype=np.float32).reshape(32, 32))
    assert 0 <= evaluation_score(inspect_image(image)) <= 1
    assert evaluation_score(
        DatasetMetrics(
            number_of_frames=2,
            session_counts={},
            frames=[{"quality_score": 0.2}, {"quality_score": 0.8}],
        )
    ) == pytest.approx(0.5)
    assert evaluation_score(DatasetMetrics(number_of_frames=0, session_counts={}, frames=[])) == 0


def test_existing_llm_gets_aia_guide_and_execution_feedback(source, tmp_path):
    client = Mock()
    client.responses.create.side_effect = [
        SimpleNamespace(
            status="completed", output_text=plan(PipelineStep(tool="normalize")).model_dump_json()
        ),
        SimpleNamespace(status="completed", output_text=plan().model_dump_json()),
    ]
    agent = AutonomousAgent(OpenAIPlanner("offline-model", client=client))
    result = agent.run(agent.prepare(source, "improve"), tmp_path / "llm.fit")
    second = client.responses.create.call_args_list[1].kwargs
    feedback = json.loads(second["input"])["execution_feedback"]
    assert feedback["executed_tools"] == ["normalize"]
    assert "aia stack" in second["instructions"] and "aia sharpen" in SYSTEM_PROMPT
    assert "astro stack" not in SYSTEM_PROMPT
    assert result.report["planner"]["model"] == "offline-model"


def test_release_runs_on_pushed_version_tag():
    from pathlib import Path

    config = yaml.load(Path(".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    assert config["on"]["push"]["tags"] == ["v*"]
    assert "release" not in config["on"] and "workflow_dispatch" in config["on"]
    assert config["jobs"]["publish"]["needs"] == "build"


def test_masked_agent_export_uses_current_tones_without_second_stretch(source, tmp_path):
    from astroagent.io.images import load_image

    image = load_fits(source)
    image.data[:2] = np.nan
    masked = save_fits(image, tmp_path / "masked.fit")
    agent = AutonomousAgent(SequencePlanner([plan(PipelineStep(tool="normalize")), plan()]))
    prepared = agent.prepare(masked, "normalize")
    assert all(t.supports_nan for t in agent.available_tools(prepared.value))
    output = tmp_path / "preview.png"
    result = agent.run(prepared, output)
    encoded = load_image(output).data.astype(float) / 65535
    np.testing.assert_allclose(encoded[2:], result.image.data[2:], atol=1 / 65535)
    assert (encoded[:2] == 0).all()
    assert np.isnan(result.image.data[:2]).all()
    assert result.report["export"]["display_transform"] is None


def test_agent_refuses_raw_cfa_single_image(source):
    image = load_fits(source)
    image.header["BAYERPAT"] = "RGGB"
    save_fits(image, source, overwrite=True)
    with pytest.raises(PipelineError, match="session directory"):
        AutonomousAgent().prepare(source, "process")


def test_rules_apply_requested_contrast_colors_and_sharpening(source, tmp_path):
    image = load_fits(source)
    image.data = np.stack([image.data * 0.8, image.data, image.data * 0.6], axis=-1)
    save_fits(image, source, overwrite=True)
    agent = AutonomousAgent()
    prepared = agent.prepare(source, "lokální kontrast, barvy a doostření")
    tools = {s.tool for s in prepared.plan.pipeline.steps}
    assert {"local_contrast", "color_adjust", "sharpen"} <= tools
    result = agent.run(prepared, tmp_path / "edited.tiff")
    assert result.report["agent"]["stop_reason"] == "planner finished"
    assert len(result.report["agent"]["iterations"]) == 1
