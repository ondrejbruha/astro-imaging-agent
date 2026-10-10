from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from astroagent.calibration.debayer import debayer_image
from astroagent.calibration.masters import build_master, build_masters
from astroagent.calibration.models import CalibrationPlan, FrameType
from astroagent.calibration.session import inspect_frame, inspect_session_frames
from astroagent.calibration.workflow import calibrate_frames, debayer_frames
from astroagent.execution import ExecutionCancelled, ExecutionContext, Progress, current_context
from astroagent.io.fits import load_fits
from astroagent.models.dataset import AstroDataset
from astroagent.models.image import AstroImage
from astroagent.models.layout import CFAMetadata
from astroagent.pipeline.executor import PipelineExecutor
from astroagent.pipeline.models import PipelineDefinition, PipelineStep
from astroagent.registration.engine import analyze_frames, register_frames
from astroagent.stacking.stack import CombineParams, combine_images
from astroagent.tools.base import ImageTool
from astroagent.tools.normalization import NormalizeParams
from astroagent.tools.registry import ToolRegistry


def cancel_phase(context, phase, *, completed=None):
    events = []

    def progress(item):
        events.append(item)
        if item.phase == phase and (completed is None or item.completed_units == completed):
            context.cancellation.cancel()

    context.progress = progress
    return events


def test_custom_tool_context_and_between_step_cancellation(image):
    class Offset(ImageTool[NormalizeParams]):
        name = "offset"
        description = "Custom tool with the original process signature"
        params_model = NormalizeParams

        def process(self, image, params):
            assert current_context() is context
            calls.append(True)
            return image.with_data(image.data + 1), []

    calls = []
    context = ExecutionContext()
    events = cancel_phase(context, "pipeline", completed=1)
    executor = PipelineExecutor(ToolRegistry([Offset()]))
    before = image.data.copy()
    with pytest.raises(ExecutionCancelled):
        executor.execute(
            image, PipelineDefinition(steps=[PipelineStep(tool="offset")] * 2), context=context
        )
    assert calls == [True]
    assert events[-1] == Progress("pipeline", 1, 2, "step", 1)
    assert current_context() is None
    np.testing.assert_array_equal(image.data, before)


def test_cancel_before_loading_and_no_outputs(tmp_path, fits_path):
    context = ExecutionContext()
    context.cancellation.cancel()
    with pytest.raises(ExecutionCancelled):
        PipelineExecutor().run(
            PipelineDefinition(), fits_path, tmp_path / "out.fit", context=context
        )
    assert not (tmp_path / "out.fit").exists()


def test_stack_tile_cancellation_and_configured_scratch_cleanup(tmp_path):
    images = [AstroImage(np.ones((16, 16)) * i) for i in range(3)]
    context = ExecutionContext(scratch_directory=tmp_path)
    events = cancel_phase(context, "stack-combining", completed=1)
    with pytest.raises(ExecutionCancelled):
        combine_images(iter(images), 3, CombineParams(method="mean", tile_rows=2), context=context)
    assert not list(tmp_path.iterdir())
    assert any(event.phase == "stack-spooling" and event.completed_units == 3 for event in events)


def test_spool_budget_and_reference_numerics(tmp_path):
    images = [AstroImage(np.arange(256).reshape(16, 16) + i) for i in range(3)]
    with pytest.raises(Exception, match="scratch byte budget"):
        combine_images(
            images,
            3,
            CombineParams(method="mean"),
            context=ExecutionContext(scratch_directory=tmp_path, scratch_bytes=1),
        )
    result, _ = combine_images(
        images,
        3,
        CombineParams(method="mean", tile_rows=2),
        context=ExecutionContext(scratch_directory=tmp_path),
    )
    np.testing.assert_array_equal(result.data, np.mean([image.data for image in images], axis=0))
    assert not list(tmp_path.iterdir())


def test_debayer_channel_checkpoint(image):
    context = ExecutionContext()
    cancel_phase(context, "debayer-channels", completed=1)
    with pytest.raises(ExecutionCancelled):
        debayer_image(image, CFAMetadata(pattern="RGGB"), context=context)


def test_calibration_cancel_is_not_excluded_frame(tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / f"light-{i}.fit"
        fits.writeto(path, np.ones((16, 16)) * 100, header=fits.Header({"IMAGETYP": "LIGHT"}))
        paths.append(path)
    context = ExecutionContext()
    cancel_phase(context, "calibration-kernel")
    with pytest.raises(ExecutionCancelled):
        calibrate_frames(
            AstroDataset(paths), tmp_path / "output", CalibrationPlan(), context=context
        )
    assert not list((tmp_path / "output").glob("*.fit"))
    assert not (tmp_path / "output" / "calibration.processing.json").exists()


def test_analysis_cancel_is_not_failed_frame(tmp_path, monkeypatch):
    from astroagent.registration import engine

    def measure(*args, **kwargs):
        raise ExecutionCancelled("Stop")

    path = tmp_path / "image.fit"
    fits.writeto(path, np.ones((16, 16)))
    monkeypatch.setattr(engine, "measure_frame", measure)
    with pytest.raises(ExecutionCancelled):
        analyze_frames(AstroDataset([path]))


def test_pre_cancelled_registration_and_debayer_do_not_create_outputs(tmp_path):
    dataset = AstroDataset([Path("absent.fit")])
    for operation in (register_frames, debayer_frames):
        context = ExecutionContext()
        context.cancellation.cancel()
        with pytest.raises(ExecutionCancelled):
            operation(dataset, tmp_path / "output", context=context)
        assert not (tmp_path / "output").exists()


def test_session_override_metadata_and_checkpoint(tmp_path):
    path = tmp_path / "unknown.fit"
    fits.writeto(
        path,
        np.ones((16, 16)),
        header=fits.Header(
            {
                "DATE-OBS": "2026-10-10T19:00:00",
                "INSTRUME": "Test camera",
                "TELESCOP": "Test scope",
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRPIX1": 8,
                "CRPIX2": 8,
                "CRVAL1": 10,
                "CRVAL2": 20,
                "CDELT1": -1 / 3600,
                "CDELT2": 1 / 3600,
            }
        ),
    )
    frame = inspect_session_frames([path], purposes={str(path): "light"}).frames[0]
    assert frame.frame_type == FrameType.LIGHT
    assert frame.purpose_source == "explicit-override"
    assert frame.capture_time == "2026-10-10T19:00:00"
    assert frame.camera == "Test camera" and frame.telescope == "Test scope"
    assert frame.pixel_scale_arcsec == pytest.approx((1, 1))
    assert "IMAGETYP" not in load_fits(path).header
    context = ExecutionContext()
    cancel_phase(context, "session-inspection", completed=1)
    with pytest.raises(ExecutionCancelled):
        inspect_session_frames([path, path], context=context)


def test_master_quality_spool_and_group_cancellation(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"bias-{index}.fit"
        fits.writeto(
            path, np.ones((16, 16)) * (100 + index), header=fits.Header({"IMAGETYP": "BIAS"})
        )
        paths.append(path)
    frames = [inspect_frame(path) for path in paths]
    for phase in ["master-analysis", "stack-spooling", "stack-combining"]:
        context = ExecutionContext(scratch_directory=tmp_path)
        cancel_phase(context, phase, completed=1)
        with pytest.raises(ExecutionCancelled):
            build_master(
                frames,
                FrameType.BIAS,
                tmp_path / "master.fit",
                context=context,
                params=CombineParams(method="mean", tile_rows=2),
            )
        assert not (tmp_path / "master.fit").exists()
        assert not list(tmp_path.glob("astro-stack-*"))
    context = ExecutionContext()
    cancel_phase(context, "building-masters")
    with pytest.raises(ExecutionCancelled):
        build_masters(inspect_session_frames(paths), tmp_path / "masters", context=context)
    assert not list((tmp_path / "masters").glob("*.fit"))


def test_registration_cancel_inside_per_frame_catch(tmp_path, monkeypatch):
    from synthetic import star_image, star_positions

    from astroagent.io.fits import save_fits
    from astroagent.registration import engine

    paths = [
        save_fits(star_image(star_positions() + [index, index]), tmp_path / f"light-{index}.fit")
        for index in range(2)
    ]

    def stop(*args, **kwargs):
        raise ExecutionCancelled("Cancelled during resampling.")

    monkeypatch.setattr(engine, "resample_image", stop)
    with pytest.raises(ExecutionCancelled):
        register_frames(AstroDataset(paths, reference=paths[0]), tmp_path / "registered")
    assert not (tmp_path / "registered" / "registration.json").exists()
    assert len(list((tmp_path / "registered").glob("*.fit"))) == 1


def test_existing_file_alias_collision_and_context_budget_validation(tmp_path, fits_path):
    import os

    output = tmp_path / "alias.fit"
    os.link(fits_path, output)
    before = fits_path.read_bytes()
    with pytest.raises(Exception, match="input image"):
        PipelineExecutor().run(PipelineDefinition(), fits_path, output, overwrite=True)
    assert fits_path.read_bytes() == before
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            ExecutionContext(memory_mb=value)
