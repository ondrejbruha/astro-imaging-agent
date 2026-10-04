import json

import numpy as np
import pytest

from astroagent.agent.executor import AgentExecutor
from astroagent.agent.planner import RuleBasedPlanner
from astroagent.analysis.quality import analyze_image
from astroagent.analysis.statistics import BackgroundMetrics, inspect_image
from astroagent.errors import PipelineError
from astroagent.io import save_fits
from astroagent.models.image import AstroImage
from astroagent.pipeline.executor import PipelineExecutor
from astroagent.pipeline.serialization import load_pipeline
from astroagent.tools.registry import default_registry


def test_rule_planner_selects_ordered_steps(image):
    metrics = inspect_image(image)
    metrics.background = BackgroundMetrics(median=20, sigma=100, gradient_estimate=300)
    metrics.appears_linear = True
    plan = RuleBasedPlanner().create_plan(
        "natural processing", metrics, default_registry().describe()
    )
    assert [step.tool for step in plan.pipeline.steps] == [
        "background_extract",
        "denoise",
        "stretch",
    ]
    assert len(plan.reasoning) == 3
    PipelineExecutor().validate(plan.pipeline)


def test_planner_handles_constant_normalized_and_missing_tools(image):
    constant = analyze_image(AstroImage(np.zeros((8, 8))))
    assert not RuleBasedPlanner().create_plan("process", constant, []).pipeline.steps
    metrics = inspect_image(image)
    plan = RuleBasedPlanner().create_plan("process", metrics, default_registry().describe())
    assert plan.pipeline.steps[0].tool == "normalize"
    metrics.appears_linear = True
    plan = RuleBasedPlanner().create_plan("process", metrics, [])
    assert not plan.pipeline.steps and "unavailable" in plan.reasoning[0]
    metrics.appears_linear = False
    metrics.min, metrics.max = 0, 1
    plan = RuleBasedPlanner().create_plan("process", metrics, default_registry().describe())
    assert not plan.pipeline.steps
    with pytest.raises(PipelineError, match="empty"):
        RuleBasedPlanner().create_plan(" ", metrics, [])


def test_small_image_skips_background(image):
    metrics = inspect_image(image)
    metrics.dimensions = (2, 2)
    metrics.background = BackgroundMetrics(median=20, sigma=0, gradient_estimate=500)
    plan = RuleBasedPlanner().create_plan("process", metrics, default_registry().describe())
    assert "background_extract" not in [step.tool for step in plan.pipeline.steps]
    assert "too small" in plan.reasoning[0]


def test_background_corrected_bounded_image_is_renormalized(image):
    metrics = inspect_image(image)
    metrics.min, metrics.max = 0, 1
    metrics.appears_linear = False
    metrics.background = BackgroundMetrics(median=0.5, sigma=0, gradient_estimate=500)
    result = RuleBasedPlanner().create_plan("process", metrics, default_registry().describe())
    assert [step.tool for step in result.pipeline.steps] == ["background_extract", "normalize"]


def test_agent_prepare_is_readonly_execute_saves_and_replays(tmp_path, star_image):
    input_path = save_fits(star_image, tmp_path / "input.fit")
    agent = AgentExecutor()
    prepared = agent.prepare(input_path, "zpracuj obrázek")
    assert set(path.name for path in tmp_path.iterdir()) == {"input.fit"}
    assert prepared.metrics.background is not None and prepared.metrics.stars.star_count == 4
    result = agent.execute(prepared, tmp_path / "out.fit")
    assert result.report.request == "zpracuj obrázek"
    assert result.report.reasoning
    assert result.report.planner == {"provider": "rules"}
    report = json.loads((tmp_path / "out.processing.json").read_text())
    assert report["metrics_after"]["stars"] is not None
    replay = PipelineExecutor().execute(
        prepared.image, load_pipeline(tmp_path / "out.pipeline.yaml")
    )
    np.testing.assert_array_equal(replay.image.data, result.image.data)
