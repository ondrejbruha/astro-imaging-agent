import json

import numpy as np
from astropy.io.fits import Header
from synthetic import make_session
from typer.testing import CliRunner

from astroagent.calibration.engine import calibrate_image
from astroagent.calibration.models import CalibrationPlan
from astroagent.calibration.workflow import debayer_frames
from astroagent.cli.main import app
from astroagent.io.fits import load_fits, save_fits
from astroagent.models.dataset import AstroDataset
from astroagent.models.image import AstroImage


def test_explicit_pattern_applies_to_foreign_monochannel_masters_without_mutation():
    header = Header({"COLORTYP": "OSC", "EXPTIME": 30})
    image = AstroImage(np.full((16, 16), 120, dtype=np.float32), header=header)
    bias = AstroImage(np.full((16, 16), 10, dtype=np.float32), header=header.copy())
    dark = AstroImage(np.full((16, 16), 5, dtype=np.float32), header=header.copy())
    dark.header["DCBIAS"] = False
    flat = AstroImage(np.full((16, 16), 1, dtype=np.float32), header=header.copy())
    result, _, _ = calibrate_image(
        image, CalibrationPlan(cfa_pattern="RGGB"), bias=bias, dark=dark, flat=flat
    )
    np.testing.assert_allclose(result.data, 105)
    assert result.cfa.pattern == "RGGB"
    assert all("BAYERPAT" not in v.header for v in (image, bias, dark, flat))


def test_dataset_debayer_override_supersedes_unknown_osc_header(tmp_path):
    image = AstroImage(np.full((16, 16), 0.4, dtype=np.float32), header=Header({"COLORTYP": "OSC"}))
    source = save_fits(image, tmp_path / "cfa.fit")
    result = debayer_frames(AstroDataset([source]), tmp_path / "rgb", pattern="BGGR")
    rgb = load_fits(result.frames[0])
    assert rgb.channels == 3 and rgb.header["CFASRC"] == "BGGR"
    np.testing.assert_allclose(rgb.data, 0.4, atol=1e-6)


def test_independent_master_cli_explicit_calibration_and_debayer(tmp_path):
    session = tmp_path / "session"
    make_session(session, cfa=True)
    runner = CliRunner()

    def invoke(args):
        result = runner.invoke(app, list(map(str, args)))
        assert result.exit_code == 0, result.output

    bias, dark, flat = [tmp_path / f"master-{name}.fit" for name in ("bias", "dark", "flat")]
    invoke(["master", "bias", session / "bias", "--output", bias])
    invoke(["master", "dark", session / "dark", "--bias", bias, "--output", dark])
    invoke(["master", "flat", session / "flat", "--bias", bias, "--output", flat])
    assert load_fits(dark).header["DCBIAS"] is False
    normalized = load_fits(flat).data
    for y in range(2):
        for x in range(2):
            np.testing.assert_allclose(np.nanmedian(normalized[y::2, x::2]), 1, atol=1e-6)
    output = tmp_path / "calibrated"
    invoke(
        [
            "calibrate",
            session / "lights",
            "--bias",
            bias,
            "--dark",
            dark,
            "--flat",
            flat,
            "--no-debayer",
            "--output",
            output,
        ]
    )
    report = json.loads((output / "calibration.processing.json").read_text())
    assert all(f["plan"]["master_dark"] == str(dark) for f in report["frames"])
    calibrated = next(output.glob("*.fit"))
    assert load_fits(calibrated).header["CALIBRAT"] is True
    rgb = tmp_path / "rgb.fit"
    invoke(["debayer", calibrated, "--output", rgb])
    assert load_fits(rgb).channels == 3 and load_fits(rgb).header["DEBAYER"] is True
