from enum import StrEnum

import numpy as np
from pydantic import field_validator

from astroagent.models.base import SchemaModel


class ImageLayout(StrEnum):
    """Scientific sampling layout; CFA must be calibrated before demosaicing."""

    MONO = "mono"
    RGB = "rgb"
    CFA = "cfa"


class CFAMetadata(SchemaModel):
    """Bayer tile and sensor-origin offsets, interpreted modulo two."""

    pattern: str
    x_offset: int = 0
    y_offset: int = 0

    @field_validator("pattern")
    @classmethod
    def valid_pattern(cls, value: str) -> str:
        """Accept only the four supported Bayer arrangements; never infer a pattern."""
        value = value.strip().upper()
        if value not in {"RGGB", "BGGR", "GRBG", "GBRG"}:
            raise ValueError("CFA pattern must be RGGB, BGGR, GRBG, or GBRG.")
        return value

    def tile(self) -> np.ndarray:
        """Return the effective color tile at image pixel (0, 0)."""
        tile = np.array(list(self.pattern)).reshape(2, 2)
        return np.roll(tile, (-self.y_offset % 2, -self.x_offset % 2), axis=(0, 1))
