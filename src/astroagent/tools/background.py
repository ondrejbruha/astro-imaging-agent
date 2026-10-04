import numpy as np
from pydantic import Field

from astroagent.analysis.background import estimate_surface
from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage
from astroagent.tools.base import ImageTool


class BackgroundExtractParams(SchemaModel):
    """Grid divisions per axis, total polynomial degree, and sigma clipping."""

    grid_size: int = Field(default=8, ge=2, le=64, strict=True)
    polynomial_degree: int = Field(default=2, ge=0, le=3, strict=True)
    sigma_clipping_threshold: float = Field(default=3.0, gt=0, le=10)


class BackgroundExtractTool(ImageTool[BackgroundExtractParams]):
    """Subtract independently estimated sky surfaces without clipping residuals."""

    name = "background_extract"
    description = "Subtract sigma-clipped tiled polynomial sky estimates per channel."
    params_model = BackgroundExtractParams

    def process(
        self, image: AstroImage, params: BackgroundExtractParams
    ) -> tuple[AstroImage, list[str]]:
        """Remove sky offset and gradient, preserving negative scientific residuals."""
        data = np.asarray(image.data, dtype=np.float64)
        planes = [data] if image.channels == 1 else [data[..., channel] for channel in range(3)]
        corrected = [
            plane
            - estimate_surface(
                plane,
                grid_size=params.grid_size,
                polynomial_degree=params.polynomial_degree,
                sigma=params.sigma_clipping_threshold,
            )
            for plane in planes
        ]
        result = corrected[0] if image.channels == 1 else np.stack(corrected, axis=-1)
        output = image.with_data(result)
        output.saturation_level = None
        output.header.remove("SATURATE", ignore_missing=True)
        return output, [
            "Extended nebulosity can bias automatic background sampling; inspect the result."
        ]
