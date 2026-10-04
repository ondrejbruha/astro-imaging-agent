from typing import Literal

from pydantic import Field, JsonValue, field_validator

from astroagent.analysis.statistics import ImageMetrics
from astroagent.models.base import SchemaModel


class PipelineStep(SchemaModel):
    """A tool name and JSON-compatible parameters, independent of any planner."""

    tool: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)


class PipelineDefinition(SchemaModel):
    """Versioned, replayable processing definition; empty pipelines are valid."""

    version: Literal[1] = 1
    steps: list[PipelineStep] = Field(default_factory=list, max_length=100)

    @field_validator("version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        """Reject bools and coerced strings as pipeline format versions."""
        if type(value) is not int:
            raise ValueError("Pipeline version must be the integer 1.")
        return value


class StepReport(SchemaModel):
    """Fully resolved parameters and measurements from one executed step."""

    tool: str
    params: dict[str, JsonValue]
    metrics_before: ImageMetrics | None
    metrics_after: ImageMetrics | None
    duration_ms: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class ProcessingReport(SchemaModel):
    """Execution provenance and summaries; timing is informational, not deterministic."""

    version: Literal[1] = 1
    input: str | None
    output: str | None = None
    package_version: str
    dependency_versions: dict[str, str]
    input_sha256: str
    output_sha256: str | None = None
    output_file_sha256: str | None = None
    exported_metrics: ImageMetrics | None = None
    planner: dict[str, str] = Field(default_factory=dict)
    pipeline: PipelineDefinition
    steps: list[StepReport] = Field(default_factory=list)
    metrics_before: ImageMetrics
    metrics_after: ImageMetrics
    warnings: list[str] = Field(default_factory=list)
    request: str | None = None
    reasoning: list[str] = Field(default_factory=list)
