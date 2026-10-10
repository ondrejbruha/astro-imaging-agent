import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from astropy.io import fits

from astroagent.errors import ImageIOError
from astroagent.io.metadata import header_metadata
from astroagent.models.image import AstroImage


def list_hdus(path: Path | str) -> list[dict[str, Any]]:
    """List zero-based HDUs using headers only, including unsupported layouts.

    Dimensions follow NumPy axis order. BITPIX describes on-disk precision;
    BSCALE/BZERO describe scaling applied on load, not a processing dtype.
    """
    entries = []
    try:
        with fits.open(path, memmap=True, lazy_load_hdus=True) as hdus:
            for index, item in enumerate(hdus):
                header = item.header
                dimensions = [
                    int(header[f"NAXIS{axis}"])
                    for axis in range(int(header.get("NAXIS", 0)), 0, -1)
                ]
                image = isinstance(item, fits.PrimaryHDU | fits.ImageHDU | fits.CompImageHDU)
                supported = image and len(dimensions) == 2 and all(dimensions)
                if image and len(dimensions) == 3 and all(dimensions):
                    axis = header.get("ASTRCHAX", hdus[0].header.get("ASTRCHAX"))
                    supported = (
                        axis in (0, 2) and dimensions[axis] == 3
                        if axis is not None
                        else (dimensions[0] == 3) != (dimensions[-1] == 3)
                    )
                entries.append(
                    {
                        "index": index,
                        "name": str(item.name),
                        "version": header.get("EXTVER"),
                        "type": type(item).__name__,
                        "dimensions": dimensions,
                        "bitpix": header.get("BITPIX"),
                        "storage_dtype": {
                            8: "uint8",
                            16: "int16",
                            32: "int32",
                            64: "int64",
                            -32: "float32",
                            -64: "float64",
                        }.get(header.get("BITPIX")),
                        "bscale": header.get("BSCALE", 1),
                        "bzero": header.get("BZERO", 0),
                        "supported": supported,
                        "reason": None
                        if supported
                        else "Empty, non-image, or unsupported/ambiguous layout.",
                    }
                )
    except (OSError, ValueError, TypeError) as exc:
        raise ImageIOError("Cannot list FITS HDUs; check file readability and format.") from exc
    return entries


def load_fits(path: Path | str, *, hdu: int | None = None) -> AstroImage:
    """Read the first image HDU, rejecting ambiguous RGB cubes.

    An RGB cube must have exactly one end axis of length three; its two spatial
    dimensions must not also suggest an alternative RGB layout.
    """
    source = Path(path)
    if hdu is not None and (type(hdu) is not int or hdu < 0):
        raise ImageIOError("HDU index must be a nonnegative zero-based integer.")
    try:
        with fits.open(source, memmap=False) as hdus:
            if hdu is not None and hdu >= len(hdus):
                raise ImageIOError("Selected HDU index is outside the FITS file.")
            for index, item in enumerate(hdus):
                if hdu is not None and index != hdu:
                    continue
                if not isinstance(item, fits.PrimaryHDU | fits.ImageHDU | fits.CompImageHDU):
                    continue
                if item.data is None:
                    continue
                data = np.array(item.data, copy=True)
                header = hdus[0].header.copy()
                if item is not hdus[0]:
                    header.extend(item.header, update=True)
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
                if "BAYERPAT" not in header and "BAYERPATN" in header:
                    header["BAYERPAT"] = header["BAYERPATN"]
                saturation = header.get("SATURATE")
                level = None if saturation is None else float(saturation)
                if level is None and data.dtype.kind in "ui":
                    level = float(np.iinfo(data.dtype).max)
                image = AstroImage(data, header_metadata(header), source, header, level)
                image.input_hdu = index
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
