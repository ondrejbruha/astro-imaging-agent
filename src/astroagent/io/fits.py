import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from astropy.io import fits

from astroagent.errors import ImageIOError
from astroagent.io.metadata import header_metadata
from astroagent.models.image import AstroImage


def load_fits(path: Path | str) -> AstroImage:
    """Read the first image HDU, rejecting ambiguous RGB cubes.

    An RGB cube must have exactly one end axis of length three; its two spatial
    dimensions must not also suggest an alternative RGB layout.
    """
    source = Path(path)
    try:
        with fits.open(source, memmap=False) as hdus:
            for hdu in hdus:
                if not isinstance(hdu, fits.PrimaryHDU | fits.ImageHDU | fits.CompImageHDU):
                    continue
                if hdu.data is None:
                    continue
                data = np.array(hdu.data, copy=True)
                header = hdus[0].header.copy()
                if hdu is not hdus[0]:
                    header.extend(hdu.header, update=True)
                if data.ndim == 3:
                    first, last = data.shape[0] == 3, data.shape[-1] == 3
                    declared_axis = header.get("ASTRCHAX")
                    if declared_axis is not None:
                        if declared_axis not in (0, 2) or data.shape[declared_axis] != 3:
                            raise ImageIOError("Invalid ASTRCHAX RGB channel axis declaration.")
                        channel_axis = int(declared_axis)
                    elif first == last:
                        raise ImageIOError(
                            "FITS RGB cube has an ambiguous or unsupported channel axis."
                        )
                    else:
                        channel_axis = 0 if first else 2
                    if channel_axis == 0:
                        data = np.moveaxis(data, 0, -1)
                elif data.ndim != 2:
                    raise ImageIOError("FITS file does not contain a supported 2D or RGB image.")
                saturation = header.get("SATURATE")
                level = None if saturation is None else float(saturation)
                image = AstroImage(data, header_metadata(header), source, header, level)
                if data.ndim == 3:
                    image.storage_channel_axis = channel_axis
                return image
    except ImageIOError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ImageIOError(f"Cannot read FITS file '{source}': {exc}") from exc
    raise ImageIOError("FITS file does not contain a supported image HDU.")


def save_fits(image: AstroImage, path: Path | str, *, overwrite: bool = False) -> Path:
    """Atomically save mono or channel-first RGB FITS with scientific metadata.

    Structural/scaling/checksum cards are regenerated for the new array.
    Existing files require explicit overwrite permission.
    """
    target = Path(path)
    if target.exists() and not overwrite:
        raise ImageIOError(f"Output already exists: {target}. Use --overwrite to replace it.")
    temporary: Path | None = None
    try:
        header = image.header.copy()
        for key in ("BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM", "EXTNAME", "EXTVER"):
            header.remove(key, ignore_missing=True, remove_all=True)
        data = image.data
        if image.channels == 3:
            if image.storage_channel_axis == 0:
                data = np.moveaxis(data, -1, 0)
            header["ASTRCHAX"] = image.storage_channel_axis
        suffix = ".fits.gz" if target.suffix.lower() == ".gz" else ".fits"
        with NamedTemporaryFile(dir=target.parent, suffix=suffix, delete=False) as stream:
            temporary = Path(stream.name)
        fits.PrimaryHDU(data=data, header=header).writeto(temporary, overwrite=True, checksum=True)
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()
        return target
    except (OSError, ValueError) as exc:
        raise ImageIOError(f"Cannot save FITS file '{target}': {exc}") from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
