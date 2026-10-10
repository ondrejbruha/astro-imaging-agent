"""Processing previews reuse full-resolution tools before display-only cropping/resizing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import Field, StrictInt

from astroagent import __version__
from astroagent.errors import PipelineError, ResourceLimitError
from astroagent.execution import ExecutionContext, checkpoint, emit_progress, execution_scope
from astroagent.io.datasets import write_json
from astroagent.io.images import save_image
from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage
from astroagent.pipeline.executor import PipelineExecutor
from astroagent.pipeline.models import PipelineDefinition


class PreviewOptions(SchemaModel):
    """Output geometry, expressed in zero-based full-resolution x/y/width/height."""

    roi: tuple[StrictInt, StrictInt, StrictInt, StrictInt] | None = None
    scale: float = Field(default=1, gt=0, le=1)
    max_display_pixels: int = Field(default=4_000_000, ge=1)


@dataclass
class PreviewResult:
    """Display artifact and provenance; no scientific output is implicitly stretched."""

    display: Path
    provenance: Path
    approximate: bool


@execution_scope
def create_preview(
    image: AstroImage,
    pipeline: PipelineDefinition,
    output: Path,
    *,
    options: PreviewOptions | None = None,
    executor: PipelineExecutor | None = None,
    input_identity: dict[str, Any] | None = None,
    context: ExecutionContext | None = None,
) -> PreviewResult:
    """Process the full image then crop, so global models/statistics and radii stay correct.

    The display PNG uses separately recorded finite min/max mapping and 8-bit
    quantization. NaNs become black. Resizing is bilinear and approximate; there
    is no accelerated ROI kernel, reduced-input processing, or persistent cache.
    """
    options = options or PreviewOptions()
    executor = executor or PipelineExecutor()
    if output.suffix.lower() != ".png":
        raise PipelineError("Processing preview display output must use a .png filename.")
    executor.validate(pipeline, input_kind="image")
    height, width = image.data.shape[:2]
    x, y, w, h = options.roi or (0, 0, width, height)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise PipelineError("Preview ROI must lie within the selected image.")
    size = (max(1, int(w * options.scale)), max(1, int(h * options.scale)))
    if size[0] * size[1] > options.max_display_pixels:
        raise ResourceLimitError("Preview display exceeds the pixel budget; reduce scale or ROI.")
    result = executor.execute(image, pipeline)
    emit_progress("preview-display")
    data = result.image.data[y : y + h, x : x + w]
    valid = np.isfinite(data)
    if not valid.any():
        raise PipelineError("Preview region has no finite pixels.")
    low, high = float(data[valid].min()), float(data[valid].max())
    mapped = np.where(valid, np.clip((data - low) / max(high - low, 1e-30), 0, 1), 0)
    raster = Image.fromarray(np.asarray(np.rint(mapped * 255), dtype=np.uint8))
    if raster.size != size:
        raster = raster.resize(size, Image.Resampling.BILINEAR)
    checkpoint()
    save_image(AstroImage(np.asarray(raster)), output)
    provenance = output.with_suffix(".preview.json")
    write_json(
        provenance,
        {
            "aia_version": __version__,
            "input_identity": input_identity,
            "input": None if image.path is None else str(image.path),
            "input_hdu": image.input_hdu,
            "pipeline": result.report.pipeline.model_dump(mode="json"),
            "roi": [x, y, w, h],
            "scale": options.scale,
            "strategy": "full-resolution-then-crop-resize",
            "approximate": options.scale != 1,
            "processing_equivalent_to_full_crop": True,
            "display_mapping": {
                "method": "finite-minmax",
                "lower": low,
                "upper": high,
                "bits": 8,
                "nonfinite": "black",
                "resampling": "bilinear",
            },
            "limitations": [
                "Full image computation is required.",
                "Display mapping quantizes scientific samples.",
            ],
        },
    )
    return PreviewResult(output, provenance, options.scale != 1)
