import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from typer.testing import CliRunner

from astroagent.agent.providers import OpenAIPlanner
from astroagent.cli.main import app

runner = CliRunner()


def test_help_version_and_tool_schemas():
    assert runner.invoke(app, ["--help"]).exit_code == 0
    version = runner.invoke(app, ["--version"])
    assert version.exit_code == 0, version.output
    assert "astro-imaging-agent" in version.stdout
    tools = runner.invoke(app, ["tools"])
    assert tools.exit_code == 0, tools.output
    assert len(json.loads(tools.stdout)) == 4


def test_inspect_human_json_and_analysis(fits_path):
    result = runner.invoke(app, ["inspect", str(fits_path), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["dimensions"] == [32, 32]
    assert "mean:" in runner.invoke(app, ["inspect", str(fits_path)]).stdout
    for command in ["analyze-background", "analyze-stars"]:
        result = runner.invoke(app, [command, str(fits_path)])
        assert result.exit_code == 0, result.output
        assert isinstance(json.loads(result.stdout), dict)


@pytest.mark.parametrize(
    "command,options",
    [
        ("normalize", []),
        ("stretch", ["--method", "asinh", "--strength", "0.6"]),
        ("denoise", ["--sigma", "0.8"]),
        ("background-extract", ["--polynomial-degree", "1"]),
    ],
)
def test_processing_commands_write_all_artifacts(fits_path, tmp_path, command, options):
    output = tmp_path / f"{command}.fit"
    result = runner.invoke(app, [command, str(fits_path), str(output), *options])
    assert result.exit_code == 0, result.output
    assert output.exists() and output.with_suffix(".pipeline.yaml").exists()
    assert output.with_suffix(".processing.json").exists()


def test_default_output_and_raster_command(fits_path, tmp_path):
    result = runner.invoke(app, ["stretch", str(fits_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "input.processed.fit").exists()
    result = runner.invoke(app, ["stretch", str(fits_path), str(tmp_path / "preview.png")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "preview.png").exists()


def test_run_and_verbose(fits_path, tmp_path):
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("version: 1\nsteps:\n  - tool: stretch\n    params: {}\n")
    result = runner.invoke(
        app,
        [
            "-v",
            "run",
            str(pipeline),
            "--input",
            str(fits_path),
            "--output",
            str(tmp_path / "out.fit"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Step 1: stretch" in result.stderr


def test_agent_dry_run_explain_and_execution(fits_path, tmp_path):
    result = runner.invoke(
        app, ["agent", str(fits_path), "zpracuj obrázek", "--dry-run", "--explain"]
    )
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.stdout)["version"] == 1
    assert "Reason:" in result.stderr
    assert list(tmp_path.iterdir()) == [fits_path]
    result = runner.invoke(
        app, ["agent", str(fits_path), "process", "--output", str(tmp_path / "agent.fit")]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "agent.processing.json").exists()


def test_user_errors_and_debug_traceback(tmp_path, fits_path):
    result = runner.invoke(app, ["inspect", str(tmp_path / "missing.fit")])
    assert result.exit_code == 1
    assert "Error:" in result.stderr and "Traceback" not in result.output
    debug = runner.invoke(app, ["-vv", "inspect", str(tmp_path / "missing.fit")])
    assert debug.exit_code == 1 and "Traceback" in debug.stderr
    invalid = runner.invoke(app, ["stretch", str(fits_path), "--strength", "100"])
    assert invalid.exit_code == 1 and "Step 1" in invalid.stderr
    assert not (tmp_path / "input.processed.fit").exists()


def test_llm_provider_requires_model(fits_path):
    result = runner.invoke(
        app, ["agent", str(fits_path), "process", "--provider", "openai", "--dry-run"]
    )
    assert result.exit_code == 1 and "--model" in result.stderr


def test_llm_cli_plan_execution_and_invalid_params(tmp_path, fits_path, monkeypatch):
    plan = {
        "pipeline": {"version": 1, "steps": [{"tool": "normalize", "params": {}}]},
        "reasoning": ["Normalize the range."],
    }
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        status="completed", output_text=json.dumps(plan)
    )
    planner = OpenAIPlanner("test-model", client=client)
    monkeypatch.setattr("astroagent.cli.main.create_planner", lambda *args, **kwargs: planner)
    result = runner.invoke(
        app,
        [
            "agent",
            str(fits_path),
            "process",
            "--provider",
            "openai",
            "--model",
            "test-model",
            "--output",
            str(tmp_path / "llm.fit"),
            "--explain",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "llm.processing.json").read_text())
    assert report["planner"] == {"provider": "openai", "model": "test-model"}
    assert "Normalize the range" in result.stderr
    plan["pipeline"]["steps"][0]["params"] = {"upper": 0, "lower": 1}
    client.responses.create.return_value.output_text = json.dumps(plan)
    result = runner.invoke(
        app,
        [
            "agent",
            str(fits_path),
            "process",
            "--provider",
            "openai",
            "--model",
            "test-model",
            "--output",
            str(tmp_path / "invalid.fit"),
        ],
    )
    assert result.exit_code == 1 and "Step 1" in result.stderr
    assert not (tmp_path / "invalid.fit").exists()
