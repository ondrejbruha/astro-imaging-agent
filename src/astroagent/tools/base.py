from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from astroagent.analysis.statistics import ImageMetrics, inspect_image
from astroagent.errors import PipelineError
from astroagent.io.metadata import header_metadata
from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage


class ToolDescription(SchemaModel):
    """Provider-independent description and JSON parameter schema."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolResult:
    """New image and diagnostics from one deterministic processing operation."""

    image: AstroImage
    metrics_before: ImageMetrics | None = None
    metrics_after: ImageMetrics | None = None
    warnings: list[str] = field(default_factory=list)


class ImageTool[Params: SchemaModel](ABC):
    """Typed tool abstraction with validated parameters and uniform measurements."""

    name: str
    description: str
    params_model: type[Params]

    def describe(self) -> ToolDescription:
        """Expose a function-calling-compatible schema without any provider SDK."""
        return ToolDescription(
            name=self.name,
            description=self.description,
            parameters=self.params_model.model_json_schema(),
        )

    def execute(self, image: AstroImage, params: Params) -> ToolResult:
        """Validate inputs and return processing diagnostics without mutating input."""
        validated = self.params_model.model_validate(params)
        if not np.isfinite(image.data).all():
            raise PipelineError("Processing requires finite pixels; repair or mask NaN/Inf first.")
        before = inspect_image(image)
        output, warnings = self.process(image, validated)
        if not np.isfinite(output.data).all():
            raise PipelineError(f"Tool '{self.name}' produced nonfinite pixel values.")
        output.header.add_history(f"astro-imaging-agent: {self.name}")
        output.metadata.update(header_metadata(output.header))
        if "SATURATE" not in output.header:
            output.metadata.pop("SATURATE", None)
        return ToolResult(output, before, inspect_image(output), warnings)

    @abstractmethod
    def process(self, image: AstroImage, params: Params) -> tuple[AstroImage, list[str]]:
        """Compute a new image and explicit operation-specific warnings."""
