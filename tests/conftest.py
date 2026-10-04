from pathlib import Path

import numpy as np
import pytest
from astropy.io.fits import Header

from astroagent.io.fits import save_fits
from astroagent.models.image import AstroImage


@pytest.fixture
def image() -> AstroImage:
    header = Header(
        {
            "OBJECT": "Synthetic nebula",
            "EXPTIME": 120.0,
            "BUNIT": "adu",
            "CRPIX1": 16.0,
            "CRPIX2": 16.0,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
        }
    )
    header.add_comment("Original observer comment")
    header.add_history("Original processing history")
    return AstroImage(
        np.arange(1024, dtype=np.uint16).reshape(32, 32),
        metadata={"OBJECT": "Synthetic nebula"},
        header=header,
    )


@pytest.fixture
def fits_path(tmp_path: Path, image: AstroImage) -> Path:
    return save_fits(image, tmp_path / "input.fit")


@pytest.fixture
def star_image() -> AstroImage:
    rng = np.random.default_rng(420)
    y, x = np.indices((96, 96))
    data = 100 + 0.04 * x + 0.02 * y + rng.normal(0, 0.4, x.shape)
    for sy, sx in [(20, 20), (20, 70), (60, 30), (70, 70)]:
        data += 150 * np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * 1.3**2))
    return AstroImage(data)
