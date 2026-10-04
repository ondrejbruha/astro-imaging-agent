from pathlib import Path
from typing import Any

import numpy as np

from astroagent.errors import PipelineError
from astroagent.io.images import image_format, save_image
from astroagent.models.image import AstroImage


def export_image(image: AstroImage, output: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Preserve scientific FITS/TIFF or create a recorded global asinh display export.

    PNG/JPEG/WebP/BMP map finite min..max through asinh(10*x)/asinh(10), fill
    uncovered pixels black and quantize. RGB uses a common range across channels.
    Scientific arrays remain untouched; display outputs are not calibration inputs.
    """
    kind = image_format(output)
    report: dict[str, Any] = {"format": kind, "display_transform": None}
    if kind not in {"fits", "tiff"}:
        finite = image.data[np.isfinite(image.data)]
        if not finite.size:
            raise PipelineError("Display export requires finite samples.")
        lower, upper = float(finite.min()), float(finite.max())
        data = np.zeros_like(image.data, dtype=np.float32)
        valid = np.isfinite(image.data)
        if upper > lower:
            data[valid] = np.arcsinh(
                10 * (image.data[valid] - lower) / (upper - lower)
            ) / np.arcsinh(10)
        image = image.with_data(np.clip(data, 0, 1))
        report["display_transform"] = {
            "method": "asinh",
            "strength": 10,
            "lower": lower,
            "upper": upper,
            "invalid_fill": 0,
        }
    save_image(image, output, overwrite=overwrite)
    return report
