import warnings
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import numpy as np
from astropy.stats import sigma_clip
from numpy.typing import NDArray
from pydantic import Field

from astroagent.errors import PipelineError
from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage
from astroagent.stacking.normalization import normalization_coefficients, normalization_statistics

StackMethod = Literal[
    "mean", "median", "weighted", "weighted-mean", "sigma-clipped", "weighted-sigma-clipped"
]


class CombineParams(SchemaModel):
    """NaN-aware pixel combination; working memory is bounded by rows and a tile budget."""

    method: StackMethod = "weighted-sigma-clipped"
    sigma_low: float = Field(default=3, gt=0)
    sigma_high: float = Field(default=3, gt=0)
    max_iterations: int = Field(default=5, ge=1, le=100)
    tile_rows: int = Field(default=128, ge=1)
    memory_mb: int = Field(default=256, ge=1)


def combine_pixels(
    frames: NDArray[Any],
    params: CombineParams,
    weights: NDArray[Any] | None = None,
) -> NDArray[np.float32]:
    """Combine an N x tile array, renormalizing weights for each valid pixel.

    Sigma clipping uses median and 1.4826*MAD, excluding NaN/Inf. Zero MAD
    rejects all nonmedian samples; very small N can overreject real variation.
    Fully masked pixels stay NaN. Accumulation uses float64, output float32.
    """
    if frames.ndim < 2 or frames.shape[0] < 1:
        raise PipelineError("Pixel combination requires at least one frame.")
    weights = np.ones(frames.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    if (
        weights.shape != (frames.shape[0],)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise PipelineError("Invalid stacking weights.")
    valid = np.isfinite(frames)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if "sigma-clipped" in params.method:
            clipped = sigma_clip(
                np.ma.array(frames, mask=~valid),
                sigma_lower=params.sigma_low,
                sigma_upper=params.sigma_high,
                maxiters=params.max_iterations,
                cenfunc="median",
                stdfunc="mad_std",
                axis=0,
            )
            valid &= ~np.ma.getmaskarray(clipped)
        if params.method == "median":
            return np.asarray(
                np.nanmedian(np.where(valid, frames, np.nan), axis=0), dtype=np.float32
            )
        if not params.method.startswith("weighted"):
            weights = np.ones(len(weights))
        expanded = weights.reshape((-1,) + (1,) * (frames.ndim - 1))
        denominator = np.sum(valid * expanded, axis=0, dtype=np.float64)
        numerator = np.sum(np.where(valid, frames, 0) * expanded, axis=0, dtype=np.float64)
        return np.asarray(
            np.divide(
                numerator,
                denominator,
                out=np.full(denominator.shape, np.nan),
                where=denominator > 0,
            ),
            dtype=np.float32,
        )


def combine_images(
    images: Iterable[AstroImage],
    count: int,
    params: CombineParams,
    *,
    weights: NDArray[Any] | None = None,
    normalize: bool = False,
    reference_index: int = 0,
) -> tuple[AstroImage, list[dict[str, list[float]]]]:
    """Spool each input once to disk, then stack row tiles without holding the cube in RAM.

    Disk usage is N*H*W*C*4 bytes. RAM includes one input/output image, a tile,
    and clipping temporaries. The budget is approximate, not a hard RSS limit.
    """
    if count < 1 or not 0 <= reference_index < count:
        raise PipelineError("Invalid frame count or normalization reference index.")
    template: AstroImage | None = None
    statistics = []
    coefficients: list[dict[str, list[float]]] = []
    with TemporaryDirectory(prefix="astro-stack-") as directory:
        cube = None
        seen = 0
        try:
            for i, image in enumerate(images):
                if i >= count:
                    raise PipelineError("More frames supplied than the declared count.")
                if template is None:
                    template = image.with_data(np.empty((1, 1), dtype=np.float32))
                    shape = image.data.shape
                    cube = np.lib.format.open_memmap(
                        Path(directory) / "frames.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(count, *shape),
                    )
                if image.data.shape != shape:
                    raise PipelineError(
                        "Stacking requires equal image dimensions and channel layouts."
                    )
                assert cube is not None
                cube[i] = image.data
                if normalize:
                    statistics.append(normalization_statistics(image.data))
                seen += 1
            if seen != count or template is None or cube is None:
                raise PipelineError("Fewer frames supplied than the declared count.")
            channels = 1 if len(shape) == 2 else 3
            for i in range(count):
                scale, offset = (
                    normalization_coefficients(statistics[i], statistics[reference_index])
                    if normalize
                    else ([1.0] * channels, [0.0] * channels)
                )
                coefficients.append({"scale": scale, "offset": offset})
            output = np.empty(shape, dtype=np.float32)
            rows = max(
                1,
                min(
                    params.tile_rows,
                    params.memory_mb * 1024**2 // (count * shape[1] * channels * 32),
                ),
            )
            for y in range(0, shape[0], rows):
                tile = np.array(cube[:, y : y + rows], copy=True)
                if normalize:
                    tile_scale = np.array([c["scale"] for c in coefficients], dtype=np.float32)
                    tile_offset = np.array([c["offset"] for c in coefficients], dtype=np.float32)
                    dims = (count, 1, 1) if channels == 1 else (count, 1, 1, 3)
                    tile *= tile_scale.reshape(dims)
                    tile += tile_offset.reshape(dims)
                output[y : y + rows] = combine_pixels(tile, params, weights)
            return template.with_data(output), coefficients
        finally:
            if cube is not None:
                cube.flush()
                cube._mmap.close()  # type: ignore[attr-defined]
