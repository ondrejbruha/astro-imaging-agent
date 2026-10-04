from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy.ndimage import gaussian_filter

from astroagent.errors import PipelineError
from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage
from astroagent.tools.base import ImageTool


def bounded_luminance(image: AstroImage) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Require normalized display samples; preserve NaN coverage and RGB color ratios."""
    data = np.asarray(image.data, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0 or finite.min() < 0 or finite.max() > 1:
        raise PipelineError(
            "Detail/color editing requires samples in 0..1; normalize or stretch first."
        )
    if image.channels == 1:
        luminance = data
    else:
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        valid = np.isfinite(data)
        denominator = valid @ weights
        luminance = np.divide(
            np.where(valid, data, 0) @ weights,
            denominator,
            out=np.full(data.shape[:2], np.nan, dtype=np.float32),
            where=denominator > 0,
        )
    return data, np.asarray(luminance, dtype=np.float32)


def masked_gaussian(data: NDArray[Any], radius: float) -> NDArray[np.float32]:
    """Blur finite neighbors using normalized Gaussian convolution without filling coverage."""
    valid = np.isfinite(data)
    numerator = gaussian_filter(np.where(valid, data, 0).astype(np.float32), radius, mode="reflect")
    denominator = gaussian_filter(valid.astype(np.float32), radius, mode="reflect")
    return np.asarray(
        np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-6),
        dtype=np.float32,
    )


def apply_luminance(
    image: AstroImage, data: NDArray[np.float32], old: NDArray[np.float32], new: NDArray[np.float32]
) -> AstroImage:
    """Scale RGB by a common luminance ratio, clipping only display range endpoints."""
    if image.channels == 1:
        output = new
    else:
        ratio = np.divide(new, old, out=np.ones_like(new), where=old > 1e-8)
        output = data * ratio[..., None]
    output[np.isnan(data)] = np.nan
    result = image.with_data(np.clip(output, 0, 1).astype(np.float32))
    result.header["SATURATE"] = 1.0
    result.saturation_level = 1.0
    return result


class LocalContrastParams(SchemaModel):
    """Large-scale luminance detail boost; Gaussian radius is in image pixels."""

    radius: float = Field(default=12, gt=0, le=200)
    amount: float = Field(default=0.3, ge=0, le=2)
    protect_highlights: bool = True


class LocalContrastTool(ImageTool[LocalContrastParams]):
    """Enhance local luminance contrast while limiting bright-star and black-sky halos."""

    name = "local_contrast"
    description = (
        "Enhance local luminance contrast on normalized mono/RGB data; preserve NaN coverage."
    )
    params_model = LocalContrastParams
    supports_nan = True

    def process(
        self, image: AstroImage, params: LocalContrastParams
    ) -> tuple[AstroImage, list[str]]:
        """Boost differences from a broad Gaussian mean, optionally tapering at 0/1."""
        data, luminance = bounded_luminance(image)
        detail = luminance - masked_gaussian(luminance, params.radius)
        taper = 4 * luminance * (1 - luminance) if params.protect_highlights else 1
        return apply_luminance(
            image, data, luminance, luminance + params.amount * taper * detail
        ), [
            "Local contrast can create halos and amplify noise; inspect stars and faint nebulosity."
        ]


class SharpenParams(SchemaModel):
    """Thresholded luminance unsharp mask; threshold is in normalized intensity units."""

    radius: float = Field(default=1, gt=0, le=10)
    amount: float = Field(default=0.4, ge=0, le=2)
    threshold: float = Field(default=0.01, ge=0, le=1)
    protect_highlights: bool = True


class SharpenTool(ImageTool[SharpenParams]):
    """Apply a modest, noise-thresholded sharpening operation without deconvolution."""

    name = "sharpen"
    description = (
        "Thresholded luminance unsharp masking on normalized images; limit noise and star halos."
    )
    params_model = SharpenParams
    supports_nan = True

    def process(self, image: AstroImage, params: SharpenParams) -> tuple[AstroImage, list[str]]:
        """Soft-threshold Gaussian high-frequency residuals before adding detail."""
        data, luminance = bounded_luminance(image)
        residual = luminance - masked_gaussian(luminance, params.radius)
        detail = np.sign(residual) * np.maximum(np.abs(residual) - params.threshold, 0)
        taper = 1 - luminance**4 if params.protect_highlights else 1
        return apply_luminance(
            image, data, luminance, luminance + params.amount * taper * detail
        ), ["Unsharp masking is a display edit; excessive amount can cause ringing."]
