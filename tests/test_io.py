from pathlib import Path

import numpy as np
import pytest
import tifffile
from astropy.io import fits
from PIL import Image

from astroagent.errors import ImageIOError
from astroagent.io import load_fits, load_image, save_fits, save_image
from astroagent.io.images import image_format
from astroagent.models.image import AstroImage


@pytest.mark.parametrize("extension", ["fit", "fits", "fts", "fits.gz"])
def test_fits_roundtrip_and_metadata(tmp_path, image, extension):
    path = save_fits(image, tmp_path / f"result.{extension}")
    loaded = load_fits(path)
    np.testing.assert_array_equal(loaded.data, image.data)
    assert loaded.header["OBJECT"] == "Synthetic nebula"
    assert loaded.header["EXPTIME"] == 120
    assert loaded.header["CRPIX1"] == 16
    assert "Original observer comment" in loaded.header["COMMENT"]
    assert "Original processing history" in loaded.header["HISTORY"]
    if extension.endswith("gz"):
        assert path.read_bytes()[:2] == b"\x1f\x8b"


@pytest.mark.parametrize("shape,axis", [((3, 8, 9), 0), ((8, 9, 3), 2)])
def test_rgb_fits_keeps_storage_axis(tmp_path, shape, axis):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    fits.writeto(tmp_path / "rgb.fit", data)
    loaded = load_fits(tmp_path / "rgb.fit")
    assert loaded.data.shape == (8, 9, 3)
    np.testing.assert_array_equal(loaded.data, np.moveaxis(data, axis, -1))
    save_fits(loaded, tmp_path / "out.fit")
    with fits.open(tmp_path / "out.fit") as hdus:
        np.testing.assert_array_equal(hdus[0].data, data)
        assert hdus[0].header["ASTRCHAX"] == axis


def test_own_rgb_declaration_resolves_three_pixel_dimension(tmp_path):
    image = AstroImage(np.zeros((5, 3, 3)))
    np.testing.assert_array_equal(
        load_fits(save_fits(image, tmp_path / "rgb.fit")).data, image.data
    )


@pytest.mark.parametrize("shape", [(3, 8, 3), (4, 8, 9), (5,), (2, 4, 5, 6)])
def test_unsupported_fits_shapes(tmp_path, shape):
    fits.writeto(tmp_path / "bad.fit", np.zeros(shape))
    with pytest.raises(ImageIOError, match="ambiguous|supported"):
        load_fits(tmp_path / "bad.fit")


def test_invalid_channel_declaration(tmp_path):
    fits.writeto(tmp_path / "bad.fit", np.zeros((3, 8, 9)), header=fits.Header({"ASTRCHAX": 1}))
    with pytest.raises(ImageIOError, match="ASTRCHAX"):
        load_fits(tmp_path / "bad.fit")


def test_extension_hdu_and_primary_metadata(tmp_path):
    hdus = fits.HDUList(
        [
            fits.PrimaryHDU(header=fits.Header({"OBSERVER": "Observer"})),
            fits.ImageHDU(np.ones((5, 6)), header=fits.Header({"FILTER": "Ha"})),
        ]
    )
    # FITS strings are ASCII; unicode owner metadata lives in package NOTICE.
    hdus[0].header["OBSERVER"] = "Observer"
    hdus.writeto(tmp_path / "extension.fit")
    loaded = load_fits(tmp_path / "extension.fit")
    assert loaded.header["OBSERVER"] == "Observer"
    assert loaded.header["FILTER"] == "Ha"
    save_fits(loaded, tmp_path / "out.fit")
    with fits.open(tmp_path / "out.fit") as result:
        assert result[0].header["SIMPLE"]


def test_no_image_or_corrupt_fits(tmp_path):
    fits.PrimaryHDU().writeto(tmp_path / "empty.fit")
    with pytest.raises(ImageIOError, match="supported image HDU"):
        load_fits(tmp_path / "empty.fit")
    (tmp_path / "corrupt.fit").write_text("not fits")
    with pytest.raises(ImageIOError, match="Cannot read"):
        load_fits(tmp_path / "corrupt.fit")


def test_scaled_integer_to_float_header_is_not_double_scaled(tmp_path):
    header = fits.Header({"BSCALE": 2.0, "BZERO": 100.0, "OBJECT": "Target"})
    hdu = fits.PrimaryHDU(np.arange(20, dtype=np.int16).reshape(4, 5))
    hdu.header.update(header)
    hdu.writeto(tmp_path / "scaled.fit")
    image = load_fits(tmp_path / "scaled.fit")
    expected = image.data.copy()
    save_fits(image, tmp_path / "out.fit")
    np.testing.assert_array_equal(load_fits(tmp_path / "out.fit").data, expected)
    assert load_fits(tmp_path / "out.fit").header["OBJECT"] == "Target"


@pytest.mark.parametrize("format_name", ["fit", "tiff", "png", "jpg", "webp", "bmp"])
def test_io_refuses_overwrite(tmp_path, image, format_name):
    path = save_image(image, tmp_path / f"out.{format_name}")
    old_bytes = path.read_bytes()
    with pytest.raises(ImageIOError, match="exists"):
        save_image(image, path)
    assert path.read_bytes() == old_bytes
    save_image(image, path, overwrite=True)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("rgb", [False, True])
def test_tiff_precision_and_header(tmp_path, image, dtype, rgb):
    data = np.arange(120).reshape(10, 12).astype(dtype)
    if rgb:
        data = np.stack([data, data * 2, data * 3], axis=-1)
    original = image.with_data(data)
    result = load_image(save_image(original, tmp_path / "out.tiff"))
    assert result.data.dtype == dtype
    np.testing.assert_array_equal(result.data, data)
    assert result.header["OBJECT"] == image.header["OBJECT"]
    assert result.metadata == original.metadata


def test_planar_rgb_tiff_and_rejected_stack(tmp_path):
    data = np.zeros((3, 10, 12), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "planar.tiff", data, photometric="rgb", planarconfig="separate")
    assert load_image(tmp_path / "planar.tiff").data.shape == (10, 12, 3)
    tifffile.imwrite(tmp_path / "stack.tiff", np.zeros((4, 10, 12)), photometric="minisblack")
    with pytest.raises(ImageIOError, match="stack"):
        load_image(tmp_path / "stack.tiff")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("rgb", [False, True])
def test_png_preserves_8_and_16_bit_mono_and_rgb(tmp_path, dtype, rgb):
    data = np.random.default_rng(7).integers(
        0, np.iinfo(dtype).max, size=(12, 15, 3) if rgb else (12, 15), dtype=dtype
    )
    decoded = load_image(save_image(AstroImage(data), tmp_path / "image.png"))
    assert decoded.data.dtype == dtype
    np.testing.assert_array_equal(decoded.data, data)


@pytest.mark.parametrize("extension", ["png", "jpg", "webp", "bmp"])
def test_bounded_float_exports(tmp_path, extension):
    data = np.linspace(0, 1, 20 * 30 * 3).reshape(20, 30, 3)
    loaded = load_image(save_image(AstroImage(data), tmp_path / f"out.{extension}"))
    assert loaded.data.shape == data.shape
    maximum = np.iinfo(loaded.data.dtype).max
    tolerance = 0.03 if extension == "jpg" else 1 / maximum
    np.testing.assert_allclose(loaded.data / maximum, data, atol=tolerance)


@pytest.mark.parametrize(
    "data", [np.ones((5, 5)) * 100, np.full((5, 5), np.nan), np.ones((5, 5), dtype=np.int32)]
)
def test_raster_requires_explicit_normalization(tmp_path, data):
    with pytest.raises(ImageIOError, match="export requires"):
        save_image(AstroImage(data), tmp_path / "out.png")
    assert not (tmp_path / "out.png").exists()


def test_palette_png_and_alpha_rejection(tmp_path):
    palette = Image.new("P", (5, 6))
    palette.putpalette([10, 20, 30] * 256)
    palette.save(tmp_path / "palette.png")
    assert load_image(tmp_path / "palette.png").data.shape == (6, 5, 3)
    Image.new("RGBA", (5, 6)).save(tmp_path / "alpha.png")
    with pytest.raises(ImageIOError, match="alpha"):
        load_image(tmp_path / "alpha.png")


def test_missing_bad_and_unsupported_raster(tmp_path):
    with pytest.raises(ImageIOError, match="extension"):
        image_format(Path("image.raw"))
    with pytest.raises(ImageIOError, match="Cannot read"):
        load_image(tmp_path / "missing.png")
    (tmp_path / "corrupt.tiff").write_text("bad")
    with pytest.raises(ImageIOError, match="Cannot read"):
        load_image(tmp_path / "corrupt.tiff")


def test_invalid_astroimage_shapes():
    for data in [
        np.zeros((5,)),
        np.zeros((3, 4, 4)),
        np.zeros((0, 5)),
        np.zeros((4, 5), dtype=complex),
    ]:
        with pytest.raises(ValueError):
            AstroImage(data)
