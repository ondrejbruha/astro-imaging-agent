"""Create a small deterministic mono or OSC session for CLI demonstrations."""

import argparse
from pathlib import Path

import numpy as np
from astropy.io.fits import Header

from astroagent.io.fits import save_fits
from astroagent.models.image import AstroImage


def main() -> None:
    """Simulate calibrated truth, vignetting, bias, dark current, stars and camera noise."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--osc", action="store_true", help="Simulate raw RGGB CFA samples.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new session directory.")
    rng = np.random.default_rng(0)
    yy, xx = np.indices((112, 128))
    response = 1 - 0.3 * ((xx - 64) ** 2 + (yy - 56) ** 2) / (64**2 + 56**2)
    response /= np.median(response)
    positions = [
        (x + rng.uniform(-2, 2), y + rng.uniform(-2, 2))
        for y in range(16, 96, 18)
        for x in range(16, 112, 18)
    ]
    for kind, count, exposure in (
        ("bias", 10, 0),
        ("dark", 10, 30),
        ("flat", 10, 1),
        ("light", 8, 30),
    ):
        directory = args.output / ("lights" if kind == "light" else kind)
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            if kind == "bias":
                data = np.full(xx.shape, 1000.0)
            elif kind == "dark":
                data = np.full(xx.shape, 1008.0)
            elif kind == "flat":
                data = 1000 + 20000 * response
            else:
                dx, dy = rng.uniform(-3, 3, 2)
                truth = np.full(xx.shape, 100.0)
                for j, (x, y) in enumerate(positions):
                    truth += (500 + 35 * j) * np.exp(
                        -((xx - x - dx) ** 2 + (yy - y - dy) ** 2) / (2 * 1.3**2)
                    )
                data = 1008 + truth * response
            data += rng.normal(0, 0.5, xx.shape)
            header = Header(
                {
                    "IMAGETYP": kind.upper(),
                    "EXPTIME": exposure,
                    "GAIN": 100,
                    "OFFSET": 10,
                    "CCD-TEMP": -10,
                    "OBJECT": "Synthetic session",
                }
            )
            if args.osc:
                header["BAYERPAT"] = "RGGB"
                # Different color responses demonstrate CFA-phase flat normalization.
                if kind in ("light", "flat"):
                    data[::2, ::2] = 1000 + (data[::2, ::2] - 1000) * 1.2
                    data[1::2, 1::2] = 1000 + (data[1::2, 1::2] - 1000) * 0.8
            save_fits(
                AstroImage(data.astype(np.float32), header=header),
                directory / f"{kind}_{i:03d}.fit",
            )
    print(args.output)


if __name__ == "__main__":
    main()
