from typing import Literal

import numpy as np
from pydantic import Field

from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage
from astroagent.tools.base import ImageTool


class StretchParams(SchemaModel):
    """Stretch strength in [0,1] and optional black point in input intensity units."""

    method: Literal["linear", "asinh"] = "asinh"
    strength: float = Field(default=0.6, ge=0, le=1)
    black_point: float | None = None


class StretchTool(ImageTool[StretchParams]):
    """Bounded global linear/asinh mapping without channel-specific scaling."""

    name = "stretch"
    description = "Normalize then stretch globally using linear or asinh mapping; output is 0..1."
    params_model = StretchParams

    def process(self, image: AstroImage, params: StretchParams) -> tuple[AstroImage, list[str]]:
        """Map endpoints to 0/1; asinh gain is 10**(3*strength)-1."""
        data = np.asarray(image.data, dtype=np.float64)
        black = float(data.min()) if params.black_point is None else params.black_point
        white = float(data.max())
        if black >= white:
            if params.black_point is not None:
                raise ValueError("black_point must be smaller than the image maximum")
            scaled = np.zeros_like(data)
            warnings = ["Constant image stretched to zero."]
        else:
            scaled = np.clip((data - black) / (white - black), 0, 1)
            warnings = []
            if np.any(data < black):
                warnings.append("Samples below black_point were clipped to zero.")
        gain = 10 ** (3 * params.strength) - 1
        if params.method == "asinh" and gain > 0:
            scaled = np.arcsinh(gain * scaled) / np.arcsinh(gain)
        output = image.with_data(scaled)
        output.saturation_level = 1.0
        output.header["SATURATE"] = 1.0
        output.header["BUNIT"] = "dimensionless"
        output.header["ASTRSTR"] = params.method == "asinh" and gain > 0
        return output, warnings
