"""Exercise optional SDK serialization and response parsing with HTTP mocks."""

import json

import pytest

from astroagent.agent.providers import AnthropicPlanner, GeminiPlanner, OpenAIPlanner
from astroagent.analysis.statistics import inspect_image
from astroagent.tools.registry import default_registry

PLAN = {
    "pipeline": {"version": 1, "steps": [{"tool": "normalize", "params": {}}]},
    "reasoning": ["Normalize the input range."],
}


def test_openai_sdk_roundtrip_without_network(image):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/v1/responses"
        assert body["text"]["format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": "test-model",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": json.dumps(PLAN), "annotations": []}
                        ],
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        with openai.OpenAI(api_key="fake-test-key", http_client=transport) as client:
            result = OpenAIPlanner("test-model", client=client).create_plan(
                "process", inspect_image(image), default_registry().describe()
            )
    assert result.pipeline.steps[0].tool == "normalize"
    assert len(requests) == 1


def test_anthropic_sdk_roundtrip_without_network(image):
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")

    def respond(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/messages"
        assert body["tool_choice"]["name"] == "submit_plan"
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "content": [
                    {"type": "tool_use", "id": "tool_test", "name": "submit_plan", "input": PLAN}
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        with anthropic.Anthropic(api_key="fake-test-key", http_client=transport) as client:
            result = AnthropicPlanner("test-model", client=client).create_plan(
                "process", inspect_image(image), default_registry().describe()
            )
    assert result.pipeline.steps[0].tool == "normalize"


def test_gemini_sdk_roundtrip_without_network(image):
    genai = pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")

    def respond(request):
        body = json.loads(request.content)
        assert "generateContent" in request.url.path
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": json.dumps(PLAN)}]},
                        "finishReason": "STOP",
                    }
                ],
            },
        )

    with genai.Client(
        api_key="fake-test-key",
        http_options={"client_args": {"transport": httpx.MockTransport(respond)}},
    ) as client:
        result = GeminiPlanner("test-model", client=client).create_plan(
            "process", inspect_image(image), default_registry().describe()
        )
    assert result.pipeline.steps[0].tool == "normalize"
