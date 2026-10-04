import logging
import math
import os
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any, Literal

from pydantic import ValidationError

from astroagent.agent.models import PlanResult
from astroagent.agent.planner import Planner, RuleBasedPlanner
from astroagent.agent.prompts import SYSTEM_PROMPT, planning_payload
from astroagent.analysis.statistics import ImageMetrics
from astroagent.errors import PipelineError
from astroagent.tools.base import ToolDescription

logger = logging.getLogger(__name__)
ProviderName = Literal["rules", "openai", "anthropic", "gemini"]


def _sdk_client(
    module_name: str, constructor: str, provider: str, key_name: str, **kwargs: Any
) -> Any:
    try:
        module = import_module(module_name)
    except ImportError:
        raise PipelineError(
            f"Install provider support: pip install 'astro-imaging-agent[{provider}]' "
            f"(developers: poetry install -E {provider})."
        ) from None
    key = os.environ.get(key_name)
    if not key:
        raise PipelineError(f"Set the {key_name} environment variable to use {provider}.")
    return getattr(module, constructor)(api_key=key, **kwargs)


class LLMPlanner(ABC):
    """Common validated planning boundary for optional SDK-backed providers.

    Providers return JSON or a structured tool response. Schema and tool-name
    validation happen here; PipelineExecutor then validates each tool's parameters.
    API errors omit potentially sensitive SDK response bodies and credentials.
    """

    provider: str

    def __init__(self, model: str, *, client: Any = None, timeout: float = 60.0) -> None:
        """Require an explicit model and allow fake SDK clients for offline tests."""
        if not model.strip():
            raise PipelineError("LLM model must not be empty.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise PipelineError("LLM timeout must be positive.")
        self.model = model
        self.client = client
        self.timeout = timeout

    def create_plan(
        self, request: str, image_metrics: ImageMetrics, available_tools: list[ToolDescription]
    ) -> PlanResult:
        """Fetch one plan and reject malformed responses before any execution."""
        if not request.strip():
            raise PipelineError("Agent request must not be empty.")
        payload = planning_payload(request, image_metrics, available_tools)
        try:
            response = self.request_plan(payload)
        except PipelineError:
            raise
        except Exception as exc:
            logger.debug("%s SDK failed with %s", self.provider, type(exc).__name__)
            raise PipelineError(
                f"{self.provider} planning request failed ({type(exc).__name__}); "
                "check credentials, model access, limits, and connection."
            ) from None
        try:
            plan = (
                PlanResult.model_validate_json(response)
                if isinstance(response, str)
                else PlanResult.model_validate(response)
            )
        except (ValidationError, ValueError) as exc:
            logger.debug("Invalid %s plan: %s", self.provider, type(exc).__name__)
            raise PipelineError(
                f"{self.provider} returned an invalid or incomplete plan."
            ) from None
        available = {tool.name for tool in available_tools}
        if any(step.tool not in available for step in plan.pipeline.steps):
            raise PipelineError(f"{self.provider} proposed a tool that was not advertised.")
        return plan

    @abstractmethod
    def request_plan(self, payload: str) -> str | dict[str, Any]:
        """Call the SDK without executing tools or sending image data."""


class OpenAIPlanner(LLMPlanner):
    """Use OpenAI Responses JSON mode and local Pydantic/tool validation."""

    provider = "openai"

    def request_plan(self, payload: str) -> str:
        """Use a non-stored Responses request and reject incomplete/refused responses."""
        if self.client is None:
            self.client = _sdk_client(
                "openai", "OpenAI", "openai", "OPENAI_API_KEY", timeout=self.timeout, max_retries=0
            )
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=payload,
            text={"format": {"type": "json_object"}},
            max_output_tokens=4096,
            store=False,
        )
        if response.status != "completed" or not response.output_text:
            raise PipelineError("openai returned an incomplete or refused planning response.")
        return str(response.output_text)


class AnthropicPlanner(LLMPlanner):
    """Use a forced Claude submit_plan tool call; no processing tool is called remotely."""

    provider = "anthropic"

    def request_plan(self, payload: str) -> dict[str, Any]:
        """Extract one complete structured tool response from the Messages API."""
        if self.client is None:
            self.client = _sdk_client(
                "anthropic",
                "Anthropic",
                "anthropic",
                "ANTHROPIC_API_KEY",
                timeout=self.timeout,
                max_retries=0,
            )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": payload}],
            tools=[
                {
                    "name": "submit_plan",
                    "description": "Submit a replayable processing plan with short reasons.",
                    "input_schema": PlanResult.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": "submit_plan", "disable_parallel_tool_use": True},
        )
        calls = [
            block
            for block in response.content
            if block.type == "tool_use" and block.name == "submit_plan"
        ]
        if (
            response.stop_reason != "tool_use"
            or len(calls) != 1
            or not isinstance(calls[0].input, dict)
        ):
            raise PipelineError("anthropic returned an incomplete or refused planning response.")
        return dict(calls[0].input)


class GeminiPlanner(LLMPlanner):
    """Use Google GenAI JSON generation and the same local validation boundary."""

    provider = "gemini"

    def request_plan(self, payload: str) -> str:
        """Generate a JSON object using a schema provided in the prompt.

        Parameter dictionaries are extensible, so the provider-specific subset
        of JSON Schema is not used; full validation remains local.
        """
        if self.client is None:
            self.client = _sdk_client(
                "google.genai",
                "Client",
                "gemini",
                "GEMINI_API_KEY",
                http_options={
                    "timeout": int(self.timeout * 1000),
                    "retry_options": {"attempts": 1},
                },
            )
        response = self.client.models.generate_content(
            model=self.model,
            contents=payload,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "max_output_tokens": 4096,
            },
        )
        if not response.text:
            raise PipelineError("gemini returned an empty or refused planning response.")
        # Truncated nonempty responses are rejected by model_validate_json.
        return str(response.text)


def create_planner(
    provider: ProviderName = "rules", model: str | None = None, *, timeout: float = 60.0
) -> Planner:
    """Select a planner without importing optional SDKs until the first request."""
    if provider == "rules":
        if model is not None:
            raise PipelineError("--model requires an LLM --provider.")
        return RuleBasedPlanner()
    if model is None:
        raise PipelineError("Specify --model when selecting an LLM provider.")
    providers: dict[str, type[LLMPlanner]] = {
        "openai": OpenAIPlanner,
        "anthropic": AnthropicPlanner,
        "gemini": GeminiPlanner,
    }
    return providers[provider](model, timeout=timeout)
