"""Typed local-host input, resource, and provider configuration contracts."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, StrictInt, field_validator, model_validator

from astroagent.calibration.models import FrameType
from astroagent.models.base import SchemaModel
from astroagent.pipeline.models import PipelineDefinition
from astroagent.registration.stars import DetectionParams


class ImageInput(SchemaModel):
    """A single document; HDUs are zero-based and restricted to FITS."""

    kind: Literal["image"]
    path: Path
    hdu: StrictInt | None = Field(default=None, ge=0)


class DatasetInput(SchemaModel):
    """Explicit FITS frame selection; no directory rediscovery is permitted."""

    kind: Literal["dataset"]
    frames: list[Path] = Field(min_length=1, max_length=10000)
    reference: Path | None = None
    purposes: dict[str, FrameType] = Field(default_factory=dict)


InputReference = Annotated[ImageInput | DatasetInput, Field(discriminator="kind")]


class HandshakeParams(SchemaModel):
    """Optional immutable filesystem policy selected by the local host."""

    read_roots: list[Path] = Field(default_factory=list, max_length=64)
    workspace_root: Path | None = None
    memory_mb: StrictInt = Field(default=256, ge=1, le=65536)
    scratch_bytes: StrictInt = Field(default=32 * 1024**3, ge=1)


class EmptyParams(SchemaModel):
    """A method with no parameters; unknown fields are rejected."""


class PipelineParams(SchemaModel):
    """Validate ordinary version-one processing YAML fields and transitions."""

    pipeline: PipelineDefinition
    input_kind: Literal["image", "dataset"] = "image"


class PipelineLoadParams(SchemaModel):
    """Load ordinary YAML from an authorized input location."""

    path: Path
    input_kind: Literal["image", "dataset"] = "image"


class PipelineSaveParams(PipelineParams):
    """Save resolved defaults into a unique job workspace."""

    filename: str = Field(default="pipeline.yaml", pattern=r"^[A-Za-z0-9_.-]{1,128}\.ya?ml$")


class ExecuteParams(SchemaModel):
    """Execute a user-reviewed definition; optional plan ID binds input identity."""

    input: InputReference
    pipeline: PipelineDefinition
    output_name: str = Field(default="output.fit", pattern=r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}$")
    plan_id: str | None = Field(default=None, min_length=1, max_length=128)


class PathParams(SchemaModel):
    """Header-only FITS document listing."""

    path: Path


class InspectParams(SchemaModel):
    """Selected-image inspection with optional quality measurements."""

    input: ImageInput
    quality: bool = False
    detection: DetectionParams = Field(default_factory=DetectionParams)


class DatasetParams(SchemaModel):
    """Header-only session inspection or bounded per-frame quality analysis."""

    input: DatasetInput
    detection: DetectionParams = Field(default_factory=DetectionParams)


class PreviewParams(SchemaModel):
    """Exact full-resolution processing followed by optional crop/output resampling."""

    input: ImageInput
    pipeline: PipelineDefinition
    roi: tuple[StrictInt, StrictInt, StrictInt, StrictInt] | None = None
    scale: float = Field(default=1, gt=0, le=1)
    generation: StrictInt = Field(default=0, ge=0)
    channel: str = Field(default="default", min_length=1, max_length=128)


class ProviderConfig(SchemaModel):
    """Request-local credentials with no persistent environment mutation."""

    provider: Literal["rules", "openai", "anthropic", "gemini"] = "rules"
    model: str | None = Field(default=None, min_length=1, max_length=256)
    timeout: float = Field(default=60, ge=0.1, le=120)
    api_key: SecretStr | None = None

    @model_validator(mode="after")
    def explicit_configuration(self) -> "ProviderConfig":
        """Require a model/key for providers and reject irrelevant rules configuration."""
        if self.provider == "rules":
            if self.model is not None or self.api_key is not None:
                raise ValueError("Rule planning does not use a model or API key.")
        elif (
            self.model is None
            or self.api_key is None
            or not self.api_key.get_secret_value().strip()
        ):
            raise ValueError("Provider configuration requires an explicit model and API key.")
        return self

    @field_validator("model")
    @classmethod
    def nonempty_model(cls, value: str | None) -> str | None:
        """Reject whitespace model identifiers."""
        if value is not None and not value.strip():
            raise ValueError("Model must not be blank.")
        return value


class PlanParams(SchemaModel):
    """Read-only planning; execution always needs a separate execute request."""

    input: InputReference
    request: str = Field(min_length=1, max_length=16000)
    config: ProviderConfig = Field(default_factory=ProviderConfig)


class TestProviderParams(SchemaModel):
    """Explicitly requested minimal planner connection test."""

    config: ProviderConfig


class JobParams(SchemaModel):
    """Control requests identify server-created jobs only."""

    job_id: str = Field(min_length=1, max_length=128)
