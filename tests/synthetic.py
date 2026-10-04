from pathlib import Path

import numpy as np
from astropy.io.fits import Header

from astroagent.io.fits import save_fits
from astroagent.models.image import AstroImage


def star_positions(seed=12):
    rng = np.random.default_rng(seed)
    return np.array(
        [
            (x + rng.uniform(-2, 2), y + rng.uniform(-2, 2))
            for y in range(16, 96, 18)
            for x in range(16, 112, 18)
        ]
    )


def star_image(points=None, *, seed=0, noise=1.0, sigma=1.3, rgb=False, shape=(112, 128)):
    points = star_positions() if points is None else points
    yy, xx = np.indices(shape)
    data = np.full(shape, 100.0)
    for i, (x, y) in enumerate(points):
        data += (500 + 35 * i) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    data += np.random.default_rng(seed).normal(0, noise, shape)
    if rgb:
        data = np.stack([data, data * 0.7, data * 0.4], axis=-1)
    return AstroImage(
        data.astype(np.float32),
        header=Header(
            {"OBJECT": "Synthetic stars", "EXPTIME": 30, "GAIN": 100, "OFFSET": 10, "CCD-TEMP": -10}
        ),
    )


def write_frame(path: Path, data, kind="LIGHT", **cards):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = Header(
        {"IMAGETYP": kind, "EXPTIME": 30, "GAIN": 100, "OFFSET": 10, "CCD-TEMP": -10, **cards}
    )
    return save_fits(AstroImage(np.asarray(data, dtype=np.float32), header=header), path)


def make_session(root: Path, *, cfa=False):
    rng = np.random.default_rng(91)
    shape = (112, 128)
    yy, xx = np.indices(shape)
    flat = 1 - 0.25 * ((xx - 64) ** 2 + (yy - 56) ** 2) / (64**2 + 56**2)
    flat /= np.median(flat)
    cards = {"BAYERPAT": "RGGB"} if cfa else {}
    for kind, level, exposure in [("BIAS", 1000, 0), ("DARK", 1008, 30), ("FLAT", 20000, 1)]:
        for i in range(5):
            data = np.full(shape, level) if kind != "FLAT" else 1000 + 20000 * flat
            write_frame(
                root / kind.lower() / f"{kind.lower()}_{i}.fit",
                data + rng.normal(0, 0.1, shape),
                kind,
                EXPTIME=exposure,
                **cards,
            )
    for i, shift in enumerate([(0, 0), (2.4, -1.3), (-1.7, 1.8)]):
        true = star_image(star_positions() + shift, seed=i).data
        write_frame(root / "lights" / f"light_{i}.fit", true * flat + 1008, **cards)
    return flat
