import json

import numpy as np
import pytest
from astropy.io import fits
from pydantic import ValidationError

from astroagent.analysis.hfr import HFRParams, measure_hfr
from astroagent.errors import ImageIOError, PipelineError
from astroagent.io.fits import list_hdus, load_fits
from astroagent.io.images import load_image
from astroagent.models.image import AstroImage
from astroagent.pipeline.executor import PipelineExecutor
from astroagent.pipeline.models import PipelineDefinition, PipelineStep
from astroagent.preview import PreviewOptions, create_preview
from astroagent.registration.stars import DetectedStar, StarCatalog


@pytest.mark.parametrize("sigma", [1, 1.5, 2, 2.5])
@pytest.mark.parametrize("background", [0, 1000])
def test_hfr_gaussian_curve_of_growth(sigma, background):
    yy, xx = np.indices((64, 64))
    cx, cy = 31.25, 30.8
    data = background + 1000 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    catalog = StarCatalog(
        image_width=64, image_height=64, stars=[DetectedStar(x=cx, y=cy, flux=1000)]
    )
    measure_hfr(AstroImage(data), catalog, HFRParams())
    # Pixel area interpolation broadens narrow undersampled sources slightly.
    assert catalog.stars[0].hfr == pytest.approx(sigma * np.sqrt(2 * np.log(2)), abs=0.13)


@pytest.mark.parametrize("case", ["mask", "saturated", "border", "blend", "empty"])
def test_hfr_unreliable_sources_are_null(case):
    yy, xx = np.indices((64, 64))
    data = 100 + 1000 * np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / 8)
    star = DetectedStar(x=32, y=32, flux=1000)
    catalog = StarCatalog(image_width=64, image_height=64, stars=[star])
    image = AstroImage(data)
    if case == "mask":
        image.data[32, 33] = np.nan
    elif case == "saturated":
        image.saturation_level = 900
    elif case == "border":
        star.x = 2
    elif case == "blend":
        catalog.stars.append(DetectedStar(x=34, y=32, flux=500))
    else:
        image.data[:] = 100
    measure_hfr(image, catalog, HFRParams())
    assert star.hfr is None and star.hfr_warning
    assert any("No reliable" in warning for warning in catalog.warnings)


def test_hfr_validation_and_old_catalog():
    with pytest.raises(ValidationError):
        HFRParams(background_inner=6)
    assert DetectedStar.model_validate({"x": 1, "y": 1, "flux": 1}).hfr is None


def test_selected_hdu_scaling_metadata_and_provenance(tmp_path):
    path = tmp_path / "multi.fit"
    primary = fits.PrimaryHDU(header=fits.Header({"OBJECT": "Primary object"}))
    first = fits.ImageHDU(np.ones((16, 16)), name="FIRST")
    selected = fits.ImageHDU(np.arange(256, dtype=np.int16).reshape(16, 16), name="SCIENCE")
    selected.header["BSCALE"] = 2
    selected.header["BZERO"] = 100
    selected.header["CRPIX1"] = 8
    fits.HDUList([primary, first, selected, fits.BinTableHDU.from_columns([])]).writeto(path)
    entries = list_hdus(path)
    assert [entry["supported"] for entry in entries] == [False, True, True, False]
    assert entries[2]["bitpix"] == 16
    image = load_image(path, hdu=2)
    assert image.input_hdu == 2 and image.header["OBJECT"] == "Primary object"
    assert image.header["CRPIX1"] == 8
    np.testing.assert_array_equal(image.data, np.arange(256).reshape(16, 16) * 2 + 100)
    result = PipelineExecutor().run(PipelineDefinition(), path, tmp_path / "out.fit", hdu=2)
    assert result.report.input_hdu == 2
    np.testing.assert_array_equal(load_fits(tmp_path / "out.fit").data, image.data)
    assert load_fits(path).input_hdu == 1
    for index in [0, 3, 4, -1, True]:
        with pytest.raises(ImageIOError):
            load_fits(path, hdu=index)


def test_hdu_listing_does_not_read_pixels(tmp_path, monkeypatch):
    path = tmp_path / "header.fit"
    fits.writeto(path, np.ones((16, 16)))

    def fail(*args, **kwargs):
        raise AssertionError("pixels read")

    monkeypatch.setattr(fits.PrimaryHDU, "data", property(fail))
    assert list_hdus(path)[0]["dimensions"] == [16, 16]


def test_preview_global_processing_crop_and_input_immutability(tmp_path):
    yy, xx = np.indices((64, 64))
    image = AstroImage(100 + xx + 2 * yy + np.random.default_rng(4).normal(0, 0.1, (64, 64)))
    before = image.data.copy()
    pipeline = PipelineDefinition(
        steps=[PipelineStep(tool="normalize"), PipelineStep(tool="denoise")]
    )
    options = PreviewOptions(roi=(10, 10, 24, 24))
    result = create_preview(image, pipeline, tmp_path / "preview.png", options=options)
    full = PipelineExecutor().execute(image, pipeline).image.data[10:34, 10:34]
    mapped = np.rint((full - full.min()) / (full.max() - full.min()) * 255).astype(np.uint8)
    from PIL import Image

    np.testing.assert_array_equal(np.asarray(Image.open(result.display)), mapped)
    np.testing.assert_array_equal(image.data, before)
    provenance = json.loads(result.provenance.read_text())
    assert provenance["processing_equivalent_to_full_crop"] is True
    assert not result.approximate
    assert "display_mapping" in provenance
    approximate = create_preview(
        image, pipeline, tmp_path / "small.png", options=PreviewOptions(scale=0.5)
    )
    assert approximate.approximate
    before_export = result.display.read_bytes()
    with pytest.raises(ImageIOError, match="already exists"):
        create_preview(image, pipeline, result.display, options=options)
    assert result.display.read_bytes() == before_export
    with pytest.raises(PipelineError):
        create_preview(
            image, pipeline, tmp_path / "invalid.png", options=PreviewOptions(roi=(-1, 0, 1, 1))
        )
    with pytest.raises(PipelineError):
        create_preview(
            image,
            PipelineDefinition(steps=[PipelineStep(tool="stack_frames")]),
            tmp_path / "stack.png",
        )
    with pytest.raises(PipelineError, match=".png filename"):
        create_preview(image, pipeline, tmp_path / "display.fit")
    assert not (tmp_path / "display.fit").exists()


@pytest.mark.parametrize("channel_axis", [0, 2])
@pytest.mark.parametrize("compressed", [False, True])
def test_selected_rgb_extension_layout_compression_and_wcs(tmp_path, channel_axis, compressed):
    shape = (3, 32, 40) if channel_axis == 0 else (32, 40, 3)
    pixels = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    extension = (fits.CompImageHDU if compressed else fits.ImageHDU)(pixels, name="SCIENCE")
    extension.header["ASTRCHAX"] = channel_axis
    extension.header["CTYPE1"] = "RA---TAN"
    extension.header["CTYPE2"] = "DEC--TAN"
    extension.header["CRPIX1"] = 17
    extension.header["CRPIX2"] = 19
    path = tmp_path / "rgb.fit"
    fits.HDUList([fits.PrimaryHDU(np.ones((32, 40))), extension]).writeto(path)
    entries = list_hdus(path)
    assert entries[1]["supported"] and entries[1]["storage_dtype"] == "int16"
    image = load_fits(path, hdu=1)
    expected = np.moveaxis(pixels, 0, -1) if channel_axis == 0 else pixels
    np.testing.assert_array_equal(image.data, expected)
    assert image.storage_channel_axis == channel_axis
    assert image.header["CRPIX1"] == 17 and image.header["CTYPE1"] == "RA---TAN"


def test_selected_float_hdu_nan_masks_and_nonfits_selection(tmp_path):
    path = tmp_path / "float.fit"
    selected = np.arange(256, dtype=np.float32).reshape(16, 16)
    selected[4, 5] = np.nan
    fits.HDUList([fits.PrimaryHDU(np.ones((16, 16))), fits.ImageHDU(selected)]).writeto(path)
    image = load_image(path, hdu=1)
    np.testing.assert_array_equal(image.data, selected)
    with pytest.raises(ImageIOError, match="only for FITS"):
        load_image(tmp_path / "image.tiff", hdu=1)


def test_hfr_empty_catalog_and_quality_weight_compatibility():
    from astroagent.analysis.frame_quality import FrameQualityMetrics, score_frames

    catalog = StarCatalog(image_width=32, image_height=32)
    measure_hfr(AstroImage(np.ones((32, 32))), catalog, HFRParams())
    assert catalog.warnings == ["No reliable HFR measurements are available."]
    base = FrameQualityMetrics(path="a", star_count=10, background_median=100, background_sigma=1)
    other = base.model_copy(update={"path": "b", "median_hfr": 100})
    scored = score_frames([base, other])
    assert scored[0].quality_score == scored[1].quality_score
