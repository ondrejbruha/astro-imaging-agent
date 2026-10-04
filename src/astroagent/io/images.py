import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import png
import tifffile
from astropy.io.fits import Header
from PIL import Image, ImageOps

from astroagent.errors import ImageIOError
from astroagent.io.fits import load_fits, save_fits
from astroagent.models.image import AstroImage

FITS_SUFFIXES = frozenset({".fit", ".fits", ".fts"})
RASTER_SUFFIXES = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def image_format(path: Path | str) -> str:
    """Resolve supported formats from suffixes, including gzip-compressed FITS."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".gz" and source.with_suffix("").suffix.lower() in FITS_SUFFIXES:
        return "fits"
    if suffix in FITS_SUFFIXES:
        return "fits"
    if suffix in {".tif", ".tiff"}:
        return "tiff"
    if suffix in {".jpg", ".jpeg"}:
        return "jpeg"
    if suffix in RASTER_SUFFIXES:
        return suffix[1:]
    raise ImageIOError(
        f"Unsupported image extension '{suffix}'. Supported: FITS, TIFF, PNG, JPEG, WebP, BMP."
    )


def load_image(path: Path | str) -> AstroImage:
    """Load FITS, precision-preserving TIFF/PNG, or standard 8-bit raster images.

    Multi-page TIFF, animations, alpha channels, and ambiguous non-RGB cubes are
    rejected instead of silently flattening them or discarding data.
    """
    source = Path(path)
    format_name = image_format(source)
    if format_name == "fits":
        return load_fits(source)
    try:
        metadata: dict[str, Any] = {"format": format_name}
        header = Header()
        if format_name == "tiff":
            with tifffile.TiffFile(source) as file:
                if len(file.series) != 1:
                    raise ImageIOError("Multi-series TIFF images are not supported.")
                series = file.series[0]
                data = series.asarray()
                if data.ndim == 3 and "S" in series.axes:
                    data = np.moveaxis(data, series.axes.index("S"), -1)
                if series.axes not in {"YX", "YXS", "SYX"}:
                    raise ImageIOError("TIFF must contain a single mono or RGB image, not a stack.")
                description = getattr(file.pages[0], "description", "")
                if description:
                    metadata["description"] = description
                    try:
                        stored = json.loads(description)
                    except (ValueError, TypeError):
                        stored = None
                    if isinstance(stored, dict) and stored.get("astroagent") == 1:
                        metadata = stored["metadata"]
                        header = Header.fromstring(stored["fits_header"], sep="\n")
        elif format_name == "png":
            width, height, rows, info = png.Reader(filename=str(source)).read()
            if info["alpha"]:
                raise ImageIOError(
                    "PNG alpha channels are not supported; provide mono or RGB data."
                )
            dtype = np.uint16 if info["bitdepth"] == 16 else np.uint8
            data = np.vstack([np.asarray(row, dtype=dtype) for row in rows])
            if info.get("palette") is not None:
                palette = np.asarray(info["palette"], dtype=np.uint8)
                if palette.shape[1] != 3:
                    raise ImageIOError("Transparent palette PNG is not supported.")
                data = palette[data]
            elif info["planes"] == 3:
                data = data.reshape(height, width, 3)
            elif info["bitdepth"] < 8:
                data = np.rint(data.astype(float) * 255 / (2 ** info["bitdepth"] - 1)).astype(
                    np.uint8
                )
            metadata["bit_depth"] = info["bitdepth"]
        else:
            with Image.open(source) as original:
                if getattr(original, "n_frames", 1) != 1:
                    raise ImageIOError("Animated or multi-frame raster images are not supported.")
                if "A" in original.getbands() or "transparency" in original.info:
                    raise ImageIOError(
                        "Alpha channels are not supported; provide mono or RGB data."
                    )
                oriented = ImageOps.exif_transpose(original)
                if oriented.mode not in {"L", "RGB"}:
                    oriented = oriented.convert("RGB")
                data = np.array(oriented)
                metadata["exif"] = {
                    str(key): str(value) for key, value in original.getexif().items()
                }
        level = header.get("SATURATE")
        return AstroImage(
            data,
            metadata=metadata,
            path=source,
            header=header,
            saturation_level=None if level is None else float(level),
        )
    except ImageIOError:
        raise
    except (OSError, ValueError, TypeError, KeyError, png.Error) as exc:
        raise ImageIOError(f"Cannot read image '{source}': {exc}") from exc


def _integer_export(image: AstroImage, bits: int) -> np.ndarray[Any, Any]:
    data = image.data
    maximum = 2**bits - 1
    dtype = np.uint16 if bits == 16 else np.uint8
    if not np.isfinite(data).all():
        raise ImageIOError("Raster export requires finite pixel values.")
    if data.dtype.kind == "f":
        if data.min() < 0 or data.max() > 1:
            raise ImageIOError(
                "Raster export requires floats in 0..1; add normalize or stretch first."
            )
        return np.asarray(np.rint(data * maximum), dtype=dtype)
    if data.dtype.kind == "u" and data.dtype.itemsize <= 2:
        source_maximum = np.iinfo(data.dtype).max
        return np.asarray(np.rint(data.astype(np.float64) * maximum / source_maximum), dtype=dtype)
    raise ImageIOError("Raster export requires uint8/uint16 or normalized float data.")


def save_image(image: AstroImage, path: Path | str, *, overwrite: bool = False) -> Path:
    """Save by extension, preserving float TIFF/FITS and quantizing bounded exports.

    PNG uses 16-bit channels for float/uint16 inputs, otherwise 8-bit. JPEG, WebP,
    and BMP use 8 bits; JPEG is lossy, WebP is lossless. TIFF embeds metadata and
    the complete FITS header. Other raster exports retain provenance in sidecars.
    """
    target = Path(path)
    format_name = image_format(target)
    if format_name == "fits":
        return save_fits(image, target, overwrite=overwrite)
    if target.exists() and not overwrite:
        raise ImageIOError(f"Output already exists: {target}. Use --overwrite to replace it.")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=target.parent, suffix=target.suffix, delete=False) as stream:
            temporary = Path(stream.name)
        if format_name == "tiff":
            description = json.dumps(
                {
                    "astroagent": 1,
                    "metadata": image.metadata,
                    "fits_header": image.header.tostring(sep="\n", endcard=False, padding=False),
                },
                ensure_ascii=True,
                allow_nan=False,
            )
            tifffile.imwrite(
                temporary,
                image.data,
                photometric="rgb" if image.channels == 3 else "minisblack",
                description=description,
                metadata=None,
            )
        elif format_name == "png":
            bits = 8 if image.data.dtype == np.uint8 else 16
            data = _integer_export(image, bits)
            with temporary.open("wb") as stream:
                writer = png.Writer(
                    width=data.shape[1],
                    height=data.shape[0],
                    greyscale=image.channels == 1,
                    bitdepth=bits,
                )
                writer.write(stream, data.reshape(data.shape[0], -1).tolist())
        else:
            data = _integer_export(image, 8)
            options: dict[str, Any] = {}
            if format_name == "jpeg":
                options = {"quality": 95, "subsampling": 0}
            elif format_name == "webp":
                options = {"lossless": True}
            Image.fromarray(data).save(temporary, format=format_name.upper(), **options)
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()
        return target
    except (OSError, ValueError, TypeError, png.Error) as exc:
        raise ImageIOError(f"Cannot save image '{target}': {exc}") from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
