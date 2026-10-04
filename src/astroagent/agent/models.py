from pydantic import Field

from astroagent.models.base import SchemaModel
from astroagent.pipeline.models import PipelineDefinition


class PlanResult(SchemaModel):
    """Replayable pipeline and short user-facing reasons, never hidden model reasoning."""

    pipeline: PipelineDefinition
    reasoning: list[str] = Field(default_factory=list)
