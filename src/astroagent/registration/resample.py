from typing import Literal

import numpy as np
from scipy.ndimage import affine_transform, maximum_filter

from astroagent.models.image import AstroImage
from astroagent.registration.transform import RegistrationTransform

Interpolation = Literal["nearest", "bilinear", "bicubic"]


def resample_image(
    image: AstroImage,
    transform: RegistrationTransform,
    shape: tuple[int, int],
    *,
    interpolation: Interpolation = "bicubic",
) -> AstroImage:
    """Resample each channel identically; invalid support and outside coverage become NaN.

    Cubic splines prefilter finite-filled data. A conservative dilated invalid
    mask prevents interpolated holes from becoming artificial zero-valued signal.
    Cubic interpolation may overshoot; no intensity or negative-value clipping occurs.
    """
    inverse = np.linalg.inv(np.asarray(transform.matrix, dtype=float))
    order = {"nearest": 0, "bilinear": 1, "bicubic": 3}[interpolation]
    matrix = inverse[:2, :2][::-1, ::-1]
    offset = inverse[:2, 2][::-1]
    planes = [image.data] if image.channels == 1 else [image.data[..., c] for c in range(3)]
    outputs = []
    for plane in planes:
        invalid = ~np.isfinite(plane)
        filled = np.where(invalid, 0, plane).astype(np.float32, copy=False)
        out = affine_transform(
            filled,
            matrix,
            offset,
            output_shape=shape,
            order=order,
            cval=np.nan,
        )
        if invalid.any():
            support = maximum_filter(invalid, size=1 if order == 0 else 2 * order + 1)
            bad = affine_transform(
                support.astype(np.uint8),
                matrix,
                offset,
                output_shape=shape,
                order=0,
                cval=1,
            )
            out[bad != 0] = np.nan
        outputs.append(out)
    return image.with_data(outputs[0] if image.channels == 1 else np.stack(outputs, axis=-1))
