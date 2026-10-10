from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from astroagent.analysis.statistics import ImageMetrics, inspect_image
from astroagent.errors import PipelineError
from astroagent.execution import ExecutionContext, checkpoint, execution_scope
from astroagent.io.metadata import header_metadata
from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage


class ToolDescription(SchemaModel):
    """Provider-independent description and JSON parameter schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    input_kind: Literal["image", "dataset"] = "image"
    supports_nan: bool = False
    output_kind: Literal["image", "dataset"] = "image"
    changes_pixels: bool = True
    compatible_layouts: list[str] = field(default_factory=lambda: ["mono", "rgb"])
    preview_strategies: list[str] = field(default_factory=list)
    ui_hints: dict[str, Any] = field(default_factory=dict)


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
    supports_nan: bool = False
    compatible_layouts: ClassVar[list[str]] = ["mono", "rgb"]
    preview_strategies: ClassVar[list[str]] = []

    def describe(self) -> ToolDescription:
        """Expose a function-calling-compatible schema without any provider SDK."""
        return ToolDescription(
            name=self.name,
            description=self.description,
            parameters=self.params_model.model_json_schema(),
            supports_nan=self.supports_nan,
            compatible_layouts=self.compatible_layouts,
            preview_strategies=self.preview_strategies,
        )

    @execution_scope
    def execute(
        self, image: AstroImage, params: Params, *, context: ExecutionContext | None = None
    ) -> ToolResult:
        """Validate inputs and return processing diagnostics without mutating input."""
        validated = self.params_model.model_validate(params)
        if image.cfa is not None:
            raise PipelineError("Raw CFA requires calibration and debayering before image editing.")
        if (
            np.isinf(image.data).any()
            or not np.isfinite(image.data).any()
            or (not self.supports_nan and np.isnan(image.data).any())
        ):
            raise PipelineError("Processing requires finite pixels; repair or mask NaN/Inf first.")
        before = inspect_image(image)
        output, warnings = self.process(image, validated)
        checkpoint()
        if (
            np.isinf(output.data).any()
            or not np.isfinite(output.data).any()
            or (np.isnan(output.data).any() and not self.supports_nan)
            or (
                self.supports_nan
                and not np.array_equal(np.isnan(image.data), np.isnan(output.data))
            )
        ):
            raise PipelineError(f"Tool '{self.name}' produced nonfinite pixel values.")
        output.header.add_history(f"astro-imaging-agent: {self.name}")
        output.metadata.update(header_metadata(output.header))
        if "SATURATE" not in output.header:
            output.metadata.pop("SATURATE", None)
        return ToolResult(output, before, inspect_image(output), warnings)

    @abstractmethod
    def process(self, image: AstroImage, params: Params) -> tuple[AstroImage, list[str]]:
        """Compute a new image and explicit operation-specific warnings."""
