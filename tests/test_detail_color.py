import numpy as np
import pytest
from astropy.io.fits import Header
from typer.testing import CliRunner

from astroagent.cli.main import app
from astroagent.errors import PipelineError
from astroagent.io.fits import load_fits, save_fits
from astroagent.models.image import AstroImage
from astroagent.pipeline.executor import PipelineExecutor
from astroagent.pipeline.serialization import load_pipeline
from astroagent.tools.color import ColorAdjustParams, ColorAdjustTool, hsv_to_rgb, rgb_to_hsv
from astroagent.tools.detail import (
    LocalContrastParams,
    LocalContrastTool,
    SharpenParams,
    SharpenTool,
)
from astroagent.tools.normalization import NormalizeParams, NormalizeTool
from astroagent.tools.stretch import StretchParams, StretchTool


def display_image(rgb=False):
    y, x = np.indices((48, 48))
    data = 0.2 + 0.25 * np.exp(-((x - 24) ** 2 + (y - 24) ** 2) / 20)
    if rgb:
        data = np.stack([data * 0.8, data, data * 0.6], axis=-1)
    return AstroImage(data.astype(np.float32), header=Header({"OBJECT": "Synthetic detail"}))


@pytest.mark.parametrize("rgb", [False, True])
@pytest.mark.parametrize(
    "tool,params",
    [
        (LocalContrastTool(), LocalContrastParams(radius=6, amount=0.6)),
        (SharpenTool(), SharpenParams(radius=1.5, amount=0.6, threshold=0)),
    ],
)
def test_detail_boosts_feature_preserves_ratios_and_metadata(rgb, tool, params):
    image = display_image(rgb)
    before = image.data.copy()
    output = tool.execute(image, params).image
    assert output.data[24, 24].mean() > before[24, 24].mean()
    assert output.header["OBJECT"] == image.header["OBJECT"]
    assert tool.name in str(output.header["HISTORY"])
    assert output.data.min() >= 0 and output.data.max() <= 1
    np.testing.assert_array_equal(image.data, before)
    if rgb:
        np.testing.assert_allclose(output.data[..., 0] / output.data[..., 1], 0.8, atol=2e-6)


def test_sharpen_threshold_protects_small_noise():
    rng = np.random.default_rng(2)
    image = AstroImage((0.3 + rng.normal(0, 0.001, (32, 32))).astype(np.float32))
    output = SharpenTool().execute(image, SharpenParams(threshold=0.02)).image
    np.testing.assert_array_equal(output.data, image.data)


@pytest.mark.parametrize(
    "tool,params",
    [
        (LocalContrastTool(), LocalContrastParams()),
        (SharpenTool(), SharpenParams()),
        (ColorAdjustTool(), ColorAdjustParams(saturation=1.1)),
        (NormalizeTool(), NormalizeParams()),
        (StretchTool(), StretchParams()),
    ],
)
def test_display_tools_preserve_full_and_partial_nan_masks(tool, params):
    image = display_image(rgb=True)
    image.data[:3] = np.nan
    image.data[20, 20, 0] = np.nan
    output = tool.execute(image, params).image
    np.testing.assert_array_equal(np.isnan(output.data), np.isnan(image.data))
    assert np.isfinite(output.data[20, 20, 1:]).all()


@pytest.mark.parametrize(
    "tool,params",
    [
        (LocalContrastTool(), LocalContrastParams()),
        (SharpenTool(), SharpenParams()),
        (ColorAdjustTool(), ColorAdjustParams()),
    ],
)
def test_display_tools_require_bounded_data(tool, params):
    image = display_image(rgb=True)
    image.data *= 100
    with pytest.raises(PipelineError, match="normalize or stretch"):
        tool.execute(image, params)


def test_hsv_roundtrip_known_and_random_colors():
    known = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.2, 0.2, 0.2]]], dtype=np.float32)
    hsv = rgb_to_hsv(known)
    np.testing.assert_allclose(hsv[0, :3, 0], [0, 120, 240])
    rng = np.random.default_rng(10)
    rgb = rng.uniform(0, 1, (32, 32, 3)).astype(np.float32)
    np.testing.assert_allclose(hsv_to_rgb(rgb_to_hsv(rgb)), rgb, atol=5e-7)
    np.testing.assert_allclose(hsv_to_rgb(hsv), known, atol=5e-7)


def test_selective_color_wraps_red_hue_and_preserves_other_colors():
    hsv = np.array(
        [[[359, 0.5, 0.8], [1, 0.5, 0.8], [120, 0.5, 0.8], [0, 0, 0.4]]], dtype=np.float32
    )
    image = AstroImage(hsv_to_rgb(hsv))
    result = (
        ColorAdjustTool()
        .execute(image, ColorAdjustParams(target_hue=0, hue_width=20, saturation=1.5))
        .image
    )
    after = rgb_to_hsv(result.data)
    assert after[0, 0, 1] > 0.74 and after[0, 1, 1] > 0.74
    np.testing.assert_allclose(result.data[0, 2:], image.data[0, 2:], atol=1e-7)


def test_color_gains_hue_rotation_and_mono_rejection():
    image = AstroImage(np.full((4, 4, 3), [0.8, 0.4, 0.4], dtype=np.float32))
    result = ColorAdjustTool().execute(image, ColorAdjustParams(hue_shift=120)).image
    np.testing.assert_allclose(result.data, np.full((4, 4, 3), [0.4, 0.8, 0.4]), atol=1e-7)
    gained = ColorAdjustTool().execute(image, ColorAdjustParams(red_gain=0.5)).image
    np.testing.assert_allclose(gained.data[..., 0], 0.4)
    with pytest.raises(PipelineError, match="requires RGB"):
        ColorAdjustTool().execute(display_image(), ColorAdjustParams())


@pytest.mark.parametrize(
    "command,options",
    [
        ("local-contrast", ["--radius", "6", "--amount", ".4"]),
        ("sharpen", ["--threshold", ".002"]),
        ("color-adjust", ["--target-hue", "120", "--saturation", "1.2"]),
    ],
)
def test_new_cli_commands_and_offline_replay(tmp_path, command, options):
    source = save_fits(display_image(rgb=True), tmp_path / "source.fit")
    output = tmp_path / "edited.fit"
    invocation = CliRunner().invoke(app, [command, str(source), str(output), *options])
    assert invocation.exit_code == 0, invocation.output
    replay = PipelineExecutor().execute(
        load_fits(source), load_pipeline(output.with_suffix(".pipeline.yaml"))
    )
    np.testing.assert_array_equal(replay.image.data, load_fits(output).data)


@pytest.mark.parametrize(
    "tool,params",
    [
        (NormalizeTool(), NormalizeParams()),
        (StretchTool(), StretchParams()),
    ],
)
def test_constant_with_nan_remains_masked(tool, params):
    data = np.full((12, 12), 4, dtype=np.float32)
    data[0] = np.nan
    result = tool.execute(AstroImage(data), params).image
    assert np.isnan(result.data[0]).all() and (result.data[1:] == 0).all()


def test_image_edits_refuse_raw_cfa():
    image = display_image()
    image.header["BAYERPAT"] = "RGGB"
    with pytest.raises(PipelineError, match="calibration and debayering"):
        NormalizeTool().execute(image, NormalizeParams())


def test_declared_osc_without_pattern_cannot_be_treated_as_mono():
    image = display_image()
    image.header["COLORTYP"] = "OSC"
    with pytest.raises(ValueError, match="CFA pattern is required"):
        NormalizeTool().execute(image, NormalizeParams())


def test_cli_restores_host_logging_handlers(tmp_path, fits_path, caplog):
    import logging

    root = logging.getLogger()
    handlers = list(root.handlers)
    result = CliRunner().invoke(app, ["inspect", str(fits_path)])
    assert result.exit_code == 0
    assert root.handlers == handlers
    logging.getLogger("astroagent").warning("Host logging still active")
    assert "Host logging still active" in caplog.text
