import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from astroagent.agent.models import PlanResult
from astroagent.agent.planner import RuleBasedPlanner
from astroagent.agent.providers import (
    AnthropicPlanner,
    GeminiPlanner,
    OpenAIPlanner,
    create_planner,
)
from astroagent.analysis.statistics import inspect_image
from astroagent.errors import PipelineError
from astroagent.tools.registry import default_registry

VALID_PLAN = {
    "pipeline": {
        "version": 1,
        "steps": [{"tool": "stretch", "params": {"method": "asinh", "strength": 0.4}}],
    },
    "reasoning": ["A mild stretch preserves the bright endpoint."],
}


def make_client(provider, payload=VALID_PLAN):
    client = Mock()
    if provider == "openai":
        client.responses.create.return_value = SimpleNamespace(
            status="completed", output_text=json.dumps(payload)
        )
    elif provider == "anthropic":
        client.messages.create.return_value = SimpleNamespace(
            stop_reason="tool_use",
            content=[SimpleNamespace(type="tool_use", name="submit_plan", input=payload)],
        )
    else:
        client.models.generate_content.return_value = SimpleNamespace(text=json.dumps(payload))
    return client


@pytest.mark.parametrize(
    "planner_class,provider,method",
    [
        (OpenAIPlanner, "openai", "responses"),
        (AnthropicPlanner, "anthropic", "messages"),
        (GeminiPlanner, "gemini", "models"),
    ],
)
def test_providers_return_valid_plan_without_pixels(image, planner_class, provider, method):
    client = make_client(provider)
    planner = planner_class("explicit-model", client=client)
    result = planner.create_plan(
        "natural processing", inspect_image(image), default_registry().describe()
    )
    assert result == PlanResult.model_validate(VALID_PLAN)
    call = (
        getattr(client, method).create if provider != "gemini" else client.models.generate_content
    )
    kwargs = call.call_args.kwargs
    assert kwargs["model"] == "explicit-model"
    payload = (
        kwargs["input"]
        if provider == "openai"
        else kwargs["messages"][0]["content"]
        if provider == "anthropic"
        else kwargs["contents"]
    )
    parsed = json.loads(payload)
    assert parsed["request"] == "natural processing"
    assert "data" not in parsed["image_metrics"]
    assert {t["name"] for t in parsed["available_tools"]} >= {"normalize", "register_frames"}
    assert "api_key" not in payload
    if provider == "openai":
        assert kwargs["store"] is False


@pytest.mark.parametrize(
    "planner_class,provider",
    [(OpenAIPlanner, "openai"), (AnthropicPlanner, "anthropic"), (GeminiPlanner, "gemini")],
)
def test_provider_invalid_json_and_unknown_tools(image, planner_class, provider):
    client = make_client(provider, {"bad": True})
    with pytest.raises(PipelineError, match="invalid"):
        planner_class("model", client=client).create_plan(
            "process", inspect_image(image), default_registry().describe()
        )
    client = make_client(
        provider, {"pipeline": {"steps": [{"tool": "invented", "params": {}}]}, "reasoning": []}
    )
    with pytest.raises(PipelineError, match="not advertised"):
        planner_class("model", client=client).create_plan(
            "process", inspect_image(image), default_registry().describe()
        )


@pytest.mark.parametrize(
    "planner_class,provider,method",
    [
        (OpenAIPlanner, "openai", "responses"),
        (AnthropicPlanner, "anthropic", "messages"),
        (GeminiPlanner, "gemini", "models"),
    ],
)
def test_provider_api_errors_are_redacted(image, planner_class, provider, method):
    client = make_client(provider)
    call = (
        getattr(client, method).create if provider != "gemini" else client.models.generate_content
    )
    call.side_effect = RuntimeError("sensitive response body sk-secret")
    with pytest.raises(PipelineError, match="planning request failed") as caught:
        planner_class("model", client=client).create_plan("process", inspect_image(image), [])
    assert "sk-secret" not in str(caught.value)


def test_empty_and_incomplete_provider_responses(image):
    metrics = inspect_image(image)
    for planner_class, provider in [
        (OpenAIPlanner, "openai"),
        (AnthropicPlanner, "anthropic"),
        (GeminiPlanner, "gemini"),
    ]:
        client = make_client(provider)
        if provider == "openai":
            client.responses.create.return_value.status = "incomplete"
        elif provider == "anthropic":
            client.messages.create.return_value.stop_reason = "max_tokens"
        else:
            client.models.generate_content.return_value.text = None
        with pytest.raises(PipelineError, match="incomplete|empty"):
            planner_class("model", client=client).create_plan("process", metrics, [])


def test_missing_optional_sdk_and_api_key(monkeypatch, image):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr("astroagent.agent.providers.import_module", missing)
    with pytest.raises(PipelineError, match="Install provider support"):
        OpenAIPlanner("model").create_plan("process", inspect_image(image), [])
    monkeypatch.setattr(
        "astroagent.agent.providers.import_module",
        lambda name: SimpleNamespace(OpenAI=Mock()),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(PipelineError, match="OPENAI_API_KEY"):
        OpenAIPlanner("model").create_plan("process", inspect_image(image), [])


def test_sdk_client_construction_uses_environment_without_real_calls(monkeypatch, image):
    client = make_client("openai")
    constructor = Mock(return_value=client)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    monkeypatch.setattr(
        "astroagent.agent.providers.import_module",
        lambda name: SimpleNamespace(OpenAI=constructor),
    )
    OpenAIPlanner("model", timeout=12).create_plan(
        "process", inspect_image(image), default_registry().describe()
    )
    assert constructor.call_args.kwargs == {
        "api_key": "fake-test-key",
        "timeout": 12,
        "max_retries": 0,
    }


def test_provider_factory_and_input_validation(image):
    assert isinstance(create_planner(), RuleBasedPlanner)
    for name, expected in [
        ("openai", OpenAIPlanner),
        ("anthropic", AnthropicPlanner),
        ("gemini", GeminiPlanner),
    ]:
        assert isinstance(create_planner(name, "model"), expected)
    with pytest.raises(PipelineError, match="--model"):
        create_planner("openai")
    with pytest.raises(PipelineError, match="--model"):
        create_planner("rules", "model")
    with pytest.raises(PipelineError, match="model"):
        OpenAIPlanner("")
    with pytest.raises(PipelineError, match="timeout"):
        OpenAIPlanner("model", timeout=0)
    with pytest.raises(PipelineError, match="empty"):
        OpenAIPlanner("model", client=Mock()).create_plan(" ", inspect_image(image), [])


def test_real_optional_sdk_request_shapes_without_network():
    # These imports are optional so the base-only CI still exercises every adapter using mocks.
    genai = pytest.importorskip("google.genai")
    config = genai.types.GenerateContentConfig(
        system_instruction="prompt", response_mime_type="application/json", max_output_tokens=4096
    )
    assert config.response_mime_type == "application/json"
    options = genai.types.HttpOptions(timeout=1000, retry_options={"attempts": 1})
    assert options.timeout == 1000
    openai = pytest.importorskip("openai")
    assert hasattr(openai.OpenAI(api_key="fake-test-key"), "responses")
    anthropic = pytest.importorskip("anthropic")
    assert hasattr(anthropic.Anthropic(api_key="fake-test-key"), "messages")
