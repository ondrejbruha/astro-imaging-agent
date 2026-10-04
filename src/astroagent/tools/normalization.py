import numpy as np
from pydantic import Field, model_validator

from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage
from astroagent.tools.base import ImageTool


class NormalizeParams(SchemaModel):
    """Target interval for a global min/max mapping shared across RGB channels."""

    lower: float = Field(default=0.0, ge=0, le=1)
    upper: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def ordered_range(self) -> "NormalizeParams":
        """Reject empty or inverted target intervals."""
        if self.lower >= self.upper:
            raise ValueError("lower must be smaller than upper")
        return self


class NormalizeTool(ImageTool[NormalizeParams]):
    """Normalize all channel samples using one shared scale."""

    name = "normalize"
    description = "Map the global pixel range into a bounded interval (default 0..1)."
    params_model = NormalizeParams
    supports_nan = True

    def process(self, image: AstroImage, params: NormalizeParams) -> tuple[AstroImage, list[str]]:
        """Use float64 and map constant images to the lower bound with a warning."""
        data = np.asarray(image.data, dtype=np.float64)
        low, high = float(np.nanmin(data)), float(np.nanmax(data))
        warnings = []
        if high == low:
            result = np.full(data.shape, params.lower)
            result[np.isnan(data)] = np.nan
            warnings.append("Constant image mapped to the lower normalization bound.")
        else:
            result = params.lower + (data - low) / (high - low) * (params.upper - params.lower)
        output = image.with_data(result)
        output.saturation_level = params.upper
        output.header["SATURATE"] = params.upper
        output.header["BUNIT"] = "dimensionless"
        return output, warnings
