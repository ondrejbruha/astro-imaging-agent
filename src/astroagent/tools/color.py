import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from astroagent.errors import PipelineError
from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage
from astroagent.tools.base import ImageTool
from astroagent.tools.detail import bounded_luminance


def rgb_to_hsv(data: NDArray[np.float32]) -> NDArray[np.float32]:
    """Vectorized RGB-to-HSV conversion with hue in degrees and achromatic hue zero."""
    maximum, minimum = data.max(axis=-1), data.min(axis=-1)
    delta = maximum - minimum
    denominator = np.where(delta > 0, delta, 1)
    r, g, b = data[..., 0], data[..., 1], data[..., 2]
    hue = np.select(
        [maximum == r, maximum == g],
        [(g - b) / denominator, (b - r) / denominator + 2],
        default=(r - g) / denominator + 4,
    )
    hue = np.where(delta > 0, np.mod(hue * 60, 360), 0)
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 0)
    return np.stack([hue, saturation, maximum], axis=-1).astype(np.float32)


def hsv_to_rgb(data: NDArray[np.float32]) -> NDArray[np.float32]:
    """Vectorized HSV-to-RGB conversion; callers supply finite normalized S/V."""
    hue, saturation, value = data[..., 0] / 60, data[..., 1], data[..., 2]
    sector = np.floor(hue).astype(int) % 6
    fraction = hue - np.floor(hue)
    p, q, t = (
        value * (1 - saturation),
        value * (1 - saturation * fraction),
        value * (1 - saturation * (1 - fraction)),
    )
    choices = np.stack(
        [
            np.stack([np.asarray(channel, dtype=np.float32) for channel in v], axis=-1)
            for v in [
                (value, t, p),
                (q, value, p),
                (p, value, t),
                (p, q, value),
                (t, p, value),
                (value, p, q),
            ]
        ],
        axis=-2,
    )
    return np.take_along_axis(
        choices, np.broadcast_to(sector[..., None, None], (*sector.shape, 1, 3)), axis=-2
    )[..., 0, :].astype(np.float32)


class ColorAdjustParams(SchemaModel):
    """RGB gains and smooth hue-selective HSV edits; hue/width/shift use degrees."""

    red_gain: float = Field(default=1, gt=0, le=4)
    green_gain: float = Field(default=1, gt=0, le=4)
    blue_gain: float = Field(default=1, gt=0, le=4)
    saturation: float = Field(default=1, ge=0, le=3)
    hue_shift: float = Field(default=0, ge=-180, le=180)
    target_hue: float | None = Field(default=None, ge=0, lt=360)
    hue_width: float = Field(default=40, gt=0, le=180)


class ColorAdjustTool(ImageTool[ColorAdjustParams]):
    """Edit all colors or a smooth circular hue band without photometric claims."""

    name = "color_adjust"
    description = (
        "RGB gains, saturation and hue shift with optional selective hue band; normalized RGB only."
    )
    params_model = ColorAdjustParams
    supports_nan = True
    compatible_layouts = ["rgb"]

    def process(self, image: AstroImage, params: ColorAdjustParams) -> tuple[AstroImage, list[str]]:
        """Use a cosine hue-band mask; achromatic samples remain unselected for selective edits."""
        if image.channels != 3:
            raise PipelineError(
                "Selective color editing requires RGB; debayer OSC or provide RGB data."
            )
        data, _ = bounded_luminance(image)
        mask = np.isnan(data)
        finite = np.where(mask, 0, data)
        hsv = rgb_to_hsv(finite)
        if params.target_hue is None:
            selection = np.ones(data.shape[:2], dtype=np.float32)
        else:
            distance = np.abs((hsv[..., 0] - params.target_hue + 180) % 360 - 180)
            selection = np.where(
                distance < params.hue_width,
                0.5 * (1 + np.cos(np.pi * distance / params.hue_width)),
                0,
            )
            selection *= hsv[..., 1] > 1e-6
        hsv[..., 0] = (hsv[..., 0] + params.hue_shift * selection) % 360
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + (params.saturation - 1) * selection), 0, 1)
        output = hsv_to_rgb(hsv)
        gains = np.array([params.red_gain, params.green_gain, params.blue_gain], dtype=np.float32)
        output *= 1 + (gains - 1) * selection[..., None]
        incomplete = mask.any(axis=-1)
        output[incomplete] = data[incomplete]
        output[mask] = np.nan
        result = image.with_data(np.clip(output, 0, 1))
        result.header["SATURATE"] = 1.0
        result.saturation_level = 1.0
        return result, [
            "Color adjustments are display edits; they are not photometric color calibration."
        ]
