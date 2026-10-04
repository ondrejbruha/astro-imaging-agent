from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astroagent.models.image import AstroImage


@dataclass
class DefectMap:
    """Boolean sensor defects; cold-pixel support is reserved for future flat analysis."""

    hot_pixels: NDArray[np.bool_]
    cold_pixels: NDArray[np.bool_] | None = None


def detect_hot_pixels(dark: AstroImage, *, sigma: float = 8) -> DefectMap:
    """Find high dark-current pixels above median + sigma*1.4826*MAD."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Hot pixel sigma must be positive and finite.")
    data = np.asarray(dark.data, dtype=float)
    median = np.nanmedian(data, axis=(0, 1))
    mad = 1.4826 * np.nanmedian(np.abs(data - median), axis=(0, 1))
    floor = np.finfo(np.float32).eps * np.maximum(1, np.abs(median))
    return DefectMap(np.isfinite(data) & (data > median + sigma * np.maximum(mad, floor)))


def correct_defects(image: AstroImage, defects: DefectMap) -> AstroImage:
    """Replace hot samples by finite neighborhood medians; CFA uses same-phase neighbors."""
    if defects.hot_pixels.shape != image.data.shape:
        raise ValueError("Defect map dimensions differ from the image.")
    output = np.array(image.data, dtype=np.float32, copy=True)
    step = 2 if image.cfa is not None else 1
    offsets = np.array(
        [(y, x) for y in (-step, 0, step) for x in (-step, 0, step) if (y, x) != (0, 0)]
    )
    channels = [None] if image.channels == 1 else list(range(3))
    for channel in channels:
        plane = output if channel is None else output[..., channel]
        hot = defects.hot_pixels if channel is None else defects.hot_pixels[..., channel]
        if hot.any():
            positions = np.argwhere(hot)
            neighbors = positions[:, None, :] + offsets[None, :, :]
            valid = (
                (neighbors[..., 0] >= 0)
                & (neighbors[..., 0] < plane.shape[0])
                & (neighbors[..., 1] >= 0)
                & (neighbors[..., 1] < plane.shape[1])
            )
            rows = np.clip(neighbors[..., 0], 0, plane.shape[0] - 1)
            columns = np.clip(neighbors[..., 1], 0, plane.shape[1] - 1)
            values = np.where(valid & ~hot[rows, columns], plane[rows, columns], np.nan)
            plane[hot] = np.nanmedian(values, axis=1)
    result = image.with_data(output)
    result.header.add_history(f"Cosmetic correction: {int(defects.hot_pixels.sum())} hot samples")
    return result
