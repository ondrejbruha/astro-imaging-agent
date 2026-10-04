from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy.io.fits import Header
from numpy.typing import NDArray


@dataclass
class AstroImage:
    """A mono (H, W) or channel-last RGB (H, W, 3) image and its FITS header.

    Arrays are owned by the caller; tools return new arrays and copied metadata.
    FITS storage layout is converted at the I/O boundary only.
    """

    data: NDArray[Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None
    header: Header = field(default_factory=Header)
    saturation_level: float | None = None
    storage_channel_axis: int = 0

    def __post_init__(self) -> None:
        """Validate shape and real numeric data without copying large arrays."""
        if self.data.ndim not in (2, 3) or (self.data.ndim == 3 and self.data.shape[-1] != 3):
            raise ValueError("Images must have shape (height, width) or (height, width, 3).")
        if 0 in self.data.shape or self.data.dtype.kind not in "uif":
            raise ValueError("Images must contain nonempty real numeric data.")

    @property
    def channels(self) -> int:
        """Return one for monochrome images and three for RGB images."""
        return 1 if self.data.ndim == 2 else 3

    def with_data(self, data: NDArray[Any]) -> "AstroImage":
        """Return a new image with independent metadata and header copies."""
        return AstroImage(
            data=data,
            metadata=deepcopy(self.metadata),
            path=self.path,
            header=self.header.copy(),
            saturation_level=self.saturation_level,
            storage_channel_axis=self.storage_channel_axis,
        )
