import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io.fits import Header
from synthetic import make_session, write_frame

from astroagent.calibration.debayer import debayer_image
from astroagent.calibration.defects import correct_defects, detect_hot_pixels
from astroagent.calibration.engine import calibrate_image
from astroagent.calibration.masters import (
    build_master,
    build_masters,
    choose_master,
    normalize_flat,
)
from astroagent.calibration.models import CalibrationPlan, FrameType
from astroagent.calibration.session import (
    classify_frame,
    discover_session,
    group_frames,
    inspect_frame,
    normalize_metadata,
    validate_session,
)
from astroagent.calibration.workflow import calibrate_frames
from astroagent.errors import PipelineError
from astroagent.io.fits import load_fits
from astroagent.models.dataset import AstroDataset
from astroagent.models.image import AstroImage
from astroagent.models.layout import CFAMetadata, ImageLayout
from astroagent.stacking.stack import CombineParams


def image(data, **header):
    return AstroImage(
        np.asarray(data, dtype=np.float32),
        header=Header({"EXPTIME": 30, "GAIN": 100, "OFFSET": 10, "CCD-TEMP": -10, **header}),
    )


@pytest.mark.parametrize(
    "header,path,kind",
    [
        ({"IMAGETYP": "Light Frame"}, "darks/a.fit", FrameType.LIGHT),
        ({"FRAME": "dark"}, "a.fit", FrameType.DARK),
        ({}, "flats/a.fit", FrameType.FLAT),
        ({}, "bias_000.fit", FrameType.BIAS),
        ({}, "starfield.fit", FrameType.UNKNOWN),
        ({"IMAGETYP": "ambiguous"}, "lights/a.fit", FrameType.UNKNOWN),
        ({}, "dark_flat_001.fit", FrameType.DARK_FLAT),
        ({}, "light_dark.fit", FrameType.UNKNOWN),
    ],
)
def test_conservative_classification(header, path, kind):
    assert classify_frame(Header(header), Path(path)) == kind


def test_metadata_aliases_layout_and_cfa_offsets():
    header = Header(
        {
            "NAXIS": 2,
            "NAXIS1": 20,
            "NAXIS2": 10,
            "BITPIX": 16,
            "EXPOSURE": 120,
            "FRAME": "object",
            "SET-TEMP": -15,
            "CCD-GAIN": 20,
            "FILTERID": "Ha",
            "BAYERPAT": "bggr",
            "XBAYROFF": 1,
            "YBAYROFF": 1,
            "XBINNING": 2,
            "YBINNING": 2,
        }
    )
    info = normalize_metadata(header, Path("a.fit"))
    assert info.exposure == 120 and info.gain == 20 and info.temperature == -15
    assert info.layout == ImageLayout.CFA and info.filter_name == "Ha"
    assert info.binning == (2, 2)
    assert info.cfa.tile().tolist() == [["R", "G"], ["G", "B"]]
    assert CFAMetadata(pattern="RGGB").tile().tolist() == [["R", "G"], ["G", "B"]]
    with pytest.raises(ValueError, match="CFA"):
        CFAMetadata(pattern="unknown")
    del header["BAYERPAT"]
    header["COLORTYP"] = "OSC"
    with pytest.raises(PipelineError, match="CFA pattern"):
        normalize_metadata(header, Path("a.fit"))


def test_grouping_exposure_filter_gain_and_cfa(tmp_path):
    data = np.full((8, 10), 10)
    paths = [
        write_frame(tmp_path / f"dark_{i}.fit", data, "DARK", EXPTIME=exposure)
        for i, exposure in enumerate([30, 30, 120])
    ]
    frames = [inspect_frame(p) for p in paths]
    assert [len(g) for g in group_frames(frames, FrameType.DARK)] == [2, 1]
    paths = [
        write_frame(tmp_path / f"flat_{i}.fit", data, "FLAT", FILTER=filter_name)
        for i, filter_name in enumerate(["R", "R", "Ha"])
    ]
    assert [len(g) for g in group_frames([inspect_frame(p) for p in paths], FrameType.FLAT)] == [
        2,
        1,
    ]


def test_synthetic_calibration_reconstruction_and_vignetting():
    yy, xx = np.indices((40, 50))
    truth = 100 + 2 * xx + 3 * yy
    flat = 0.65 + 0.35 * np.exp(-((xx - 25) ** 2 + (yy - 20) ** 2) / 500)
    flat /= np.median(flat)
    bias = np.full(truth.shape, 1000)
    dark = np.full(truth.shape, 8)
    rng = np.random.default_rng(4)
    raw = truth * flat + dark + bias + rng.normal(0, 0.05, truth.shape)
    result, warnings, quality = calibrate_image(
        image(raw, OBJECT="M31"),
        CalibrationPlan(),
        bias=image(bias),
        dark=image(dark, DCBIAS=False),
        flat=image(flat),
    )
    assert np.sqrt(np.mean((result.data - truth) ** 2)) < 0.07
    assert result.data.dtype == np.float32 and result.header["CALIBRAT"]
    assert result.header["OBJECT"] == "M31" and "Flat calibrated" in str(result.header["HISTORY"])
    uniform, _, _ = calibrate_image(
        image(200 * flat + 1008),
        CalibrationPlan(),
        bias=image(bias),
        dark=image(dark, DCBIAS=False),
        flat=image(flat),
    )
    assert uniform.data.std() < 0.001
    assert quality["nan_fraction_after"] == 0
    assert warnings == []


@pytest.mark.parametrize("contains_bias", [True, False])
def test_no_double_bias_subtraction_and_negative_values(contains_bias):
    bias = image(np.full((8, 8), 1000))
    dark = image(np.full((8, 8), 1008 if contains_bias else 8), DCBIAS=contains_bias)
    light = image(np.full((8, 8), 1005))
    result, _, _ = calibrate_image(light, CalibrationPlan(), bias=bias, dark=dark)
    np.testing.assert_array_equal(result.data, -3)
    np.testing.assert_array_equal(light.data, 1005)


@pytest.mark.parametrize("contains_bias", [True, False])
def test_exposure_scaling_and_temperature_warning(contains_bias):
    bias = image(np.full((8, 8), 1000))
    dark = image(
        np.full((8, 8), 1008 if contains_bias else 8),
        DCBIAS=contains_bias,
        EXPTIME=30,
        **{"CCD-TEMP": -5},
    )
    light = image(np.full((8, 8), 1116), EXPTIME=60)
    with pytest.raises(PipelineError, match="exposure"):
        calibrate_image(light, CalibrationPlan(), bias=bias, dark=dark)
    result, warnings, quality = calibrate_image(
        light, CalibrationPlan(dark_scaling=True), bias=bias, dark=dark
    )
    np.testing.assert_array_equal(result.data, 100)
    assert quality["dark_scale"] == 2
    assert any("5.0 C" in w for w in warnings)
    if contains_bias:
        with pytest.raises(PipelineError, match="requires a master bias"):
            calibrate_image(light, CalibrationPlan(dark_scaling=True), dark=dark)


def test_foreign_dark_requires_bias_content_choice():
    raw = image(np.full((8, 8), 20))
    dark = image(np.full((8, 8), 2))
    with pytest.raises(PipelineError, match="bias content"):
        calibrate_image(raw, CalibrationPlan(), dark=dark)
    result, _, _ = calibrate_image(raw, CalibrationPlan(dark_contains_bias=True), dark=dark)
    np.testing.assert_array_equal(result.data, 18)


def test_invalid_flat_mask_and_relative_threshold():
    raw = image(np.full((10, 10), 10))
    response = np.ones((10, 10))
    response[1, 1] = 0
    response[2, 2] = 0.01
    response[3, 3] = np.nan
    result, _, quality = calibrate_image(raw, CalibrationPlan(), flat=image(response))
    assert np.isnan(result.data[1, 1]) and np.isnan(result.data[2, 2])
    assert quality["nan_fraction_after"] == pytest.approx(0.03)
    response[:5] = 0
    with pytest.raises(PipelineError, match="invalid pixels"):
        calibrate_image(raw, CalibrationPlan(), flat=image(response))


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
@pytest.mark.parametrize("offset", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_cfa_debayer_channel_order_and_offset(pattern, offset):
    cfa = CFAMetadata(pattern=pattern, x_offset=offset[0], y_offset=offset[1])
    yy, xx = np.indices((12, 14))
    colors = cfa.tile()[yy % 2, xx % 2]
    raw = np.zeros((12, 14))
    for color, value in zip("RGB", [10, 20, 40], strict=True):
        raw[colors == color] = value
    source = image(raw, BAYERPAT=pattern, XBAYROFF=offset[0], YBAYROFF=offset[1])
    output = debayer_image(source)
    assert output.layout == ImageLayout.RGB and output.data.shape == (12, 14, 3)
    np.testing.assert_allclose(output.data, np.broadcast_to([10, 20, 40], output.data.shape))
    assert output.header["DEBAYER"] and "BAYERPAT" not in output.header
    assert "BAYERPAT" in source.header
    with pytest.raises(PipelineError, match="before debayering"):
        calibrate_image(output, CalibrationPlan())


def test_cfa_flat_normalization_and_nan_debayer():
    raw = np.empty((10, 12))
    for (y, x), level in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [100, 150, 200, 300], strict=True):
        raw[y::2, x::2] = level
    source = image(raw, BAYERPAT="RGGB")
    normalized = normalize_flat(source)
    np.testing.assert_array_equal(normalized.data, 1)
    source.data[4, 4] = np.nan
    rgb = debayer_image(source)
    assert np.isnan(rgb.data[4, 4]).all() and np.isfinite(rgb.data[5, 5]).all()
    with pytest.raises(PipelineError, match="CFA pattern"):
        debayer_image(image(np.ones((10, 10))))


def test_hot_pixels_and_cfa_phase_correction():
    dark = np.zeros((14, 14))
    dark[6, 6] = 10000
    defects = detect_hot_pixels(image(dark))
    assert defects.hot_pixels.sum() == 1
    raw = np.full((14, 14), 10.0)
    raw[::2, ::2] = 40
    raw[6, 6] = 10000
    corrected = correct_defects(image(raw, BAYERPAT="RGGB"), defects)
    assert corrected.data[6, 6] == 40


def test_master_bias_dark_flat_and_session_report(tmp_path):
    root = tmp_path / "session"
    make_session(root)
    session = discover_session(root)
    assert session.counts()["light"] == 3 and session.counts()["bias"] == 5
    masters = build_masters(session, tmp_path / "masters")
    assert len(masters) == 3
    by_kind = {m.info.frame_type: m for m in masters}
    bias = load_fits(by_kind[FrameType.BIAS].path)
    dark = load_fits(by_kind[FrameType.DARK].path)
    flat = load_fits(by_kind[FrameType.FLAT].path)
    assert np.median(bias.data) == pytest.approx(1000, abs=0.02)
    assert np.median(dark.data) == pytest.approx(8, abs=0.02)
    assert dark.header["DCBIAS"] is False and dark.header["NCOMBINE"] == 5
    assert np.median(flat.data) == pytest.approx(1, abs=1e-6)
    output = calibrate_frames(
        AstroDataset([f.path for f in session.frames], masters=masters), tmp_path / "calibrated"
    )
    assert len(output.frames) == 3
    calibrated = load_fits(output.frames[0])
    assert calibrated.header["CALIBRAT"]
    report = json.loads((tmp_path / "calibrated" / "calibration.processing.json").read_text())
    assert len(report["frames"]) == 3 and report["frames"][0]["plan"]["master_dark"]
    assert report["frames"][0]["quality"]["nan_fraction_after"] == 0


def test_filter_matching_and_incompatible_master_rejection(tmp_path):
    paths = [
        write_frame(tmp_path / f"flat_{i}.fit", np.full((10, 10), 200), "FLAT", FILTER=f)
        for i, f in enumerate(["R", "Ha"])
    ]
    masters = [
        build_master([inspect_frame(p)], FrameType.FLAT, tmp_path / f"master_{i}.fit")
        for i, p in enumerate(paths)
    ]
    light = inspect_frame(write_frame(tmp_path / "light.fit", np.ones((10, 10)), FILTER="Ha"))
    assert choose_master(light, masters, FrameType.FLAT) == masters[1]
    other = image(np.ones((10, 10)), FILTER="OIII")
    with pytest.raises(PipelineError, match="Incompatible"):
        calibrate_image(image(np.ones((10, 10)), FILTER="Ha"), CalibrationPlan(), flat=other)


def test_inconsistent_cfa_and_multiple_dark_groups(tmp_path):
    root = tmp_path / "session"
    write_frame(root / "lights" / "a.fit", np.ones((10, 10)), BAYERPAT="RGGB")
    write_frame(root / "lights" / "b.fit", np.ones((10, 10)), BAYERPAT="BGGR")
    with pytest.raises(PipelineError, match="inconsistent"):
        validate_session(discover_session(root))
    _ = [
        write_frame(tmp_path / "darks" / f"dark_{i}.fit", np.full((10, 10), 10), "DARK", EXPTIME=e)
        for i, e in enumerate([30, 120])
    ]
    session = discover_session(tmp_path / "darks")
    masters = build_masters(session, tmp_path / "masters", params=CombineParams(method="mean"))
    assert len(masters) == 2 and {m.info.exposure for m in masters} == {30, 120}
    assert all(m.contains_bias for m in masters)
