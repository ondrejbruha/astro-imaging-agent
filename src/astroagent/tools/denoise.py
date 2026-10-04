from typing import Literal

import numpy as np
from pydantic import Field
from scipy.ndimage import gaussian_filter

from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage
from astroagent.tools.base import ImageTool


class DenoiseParams(SchemaModel):
    """Spatial Gaussian standard deviation in pixels; channels are never mixed."""

    method: Literal["gaussian"] = "gaussian"
    sigma: float = Field(default=0.8, gt=0, le=10)


class DenoiseTool(ImageTool[DenoiseParams]):
    """Apply a deterministic Gaussian filter with reflected edges."""

    name = "denoise"
    description = "Reduce pixel noise using Gaussian smoothing (also softens stars)."
    params_model = DenoiseParams

    def process(self, image: AstroImage, params: DenoiseParams) -> tuple[AstroImage, list[str]]:
        """Filter only the spatial axes in float64."""
        sigma = (
            (params.sigma, params.sigma) if image.channels == 1 else (params.sigma, params.sigma, 0)
        )
        data = gaussian_filter(
            np.asarray(image.data, dtype=np.float64), sigma=sigma, mode="reflect"
        )
        return image.with_data(data), [
            "Gaussian smoothing can broaden stars and reduce fine detail."
        ]
