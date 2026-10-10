from typing import Any

import numpy as np

from astroagent.analysis.statistics import inspect_image
from astroagent.calibration.defects import correct_defects, detect_hot_pixels
from astroagent.calibration.models import CalibrationPlan, FrameInfo
from astroagent.calibration.session import compatible, normalize_metadata
from astroagent.errors import PipelineError
from astroagent.execution import ExecutionContext, checkpoint, emit_progress, execution_scope
from astroagent.io.metadata import header_metadata
from astroagent.models.image import AstroImage
from astroagent.models.layout import CFAMetadata


def _info(image: AstroImage) -> FrameInfo:
    from pathlib import Path

    header = image.header.copy()
    header["NAXIS"] = image.data.ndim
    header["NAXIS1"] = image.data.shape[1]
    header["NAXIS2"] = image.data.shape[0]
    header["BITPIX"] = -32
    if image.channels == 3:
        header["NAXIS3"] = 3
        header["ASTRCHAX"] = 0
    return normalize_metadata(header, image.path or Path("memory.fit"))


@execution_scope
def calibrate_image(
    image: AstroImage,
    plan: CalibrationPlan,
    *,
    bias: AstroImage | None = None,
    dark: AstroImage | None = None,
    flat: AstroImage | None = None,
    context: ExecutionContext | None = None,
) -> tuple[AstroImage, list[str], dict[str, Any]]:
    """Apply bias/dark subtraction and CFA-aware flat division in floating point.

    Bias-containing darks replace bias subtraction, never duplicate it. Scaling
    such darks requires a master bias to isolate thermal current. Negative data
    remain valid; unsafe flat response becomes NaN. Foreign darks require DCBIAS
    metadata or an explicit plan.dark_contains_bias choice.
    """
    if image.channels == 3 and image.header.get("DEBAYER", False):
        raise PipelineError("Calibrate raw CFA before debayering; this image is already debayered.")
    if plan.cfa_pattern is not None:
        pattern = CFAMetadata(pattern=plan.cfa_pattern).pattern

        def apply_pattern(value: AstroImage) -> AstroImage:
            if value.channels == 1:
                value = value.with_data(value.data)
                value.header["BAYERPAT"] = pattern
            return value

        image = apply_pattern(image)
        bias = apply_pattern(bias) if bias is not None else None
        dark = apply_pattern(dark) if dark is not None else None
        flat = apply_pattern(flat) if flat is not None else None
    info = _info(image)
    messages: list[str] = []
    before = inspect_image(image)
    for name, master in (("bias", bias), ("dark", dark), ("flat", flat)):
        if master is None:
            continue
        if master.data.shape != image.data.shape or not compatible(
            info, _info(master), flat=name == "flat"
        ):
            raise PipelineError(
                f"Incompatible master {name} dimensions, CFA, filter or camera settings."
            )
        for key in ("gain", "offset"):
            if getattr(info, key) is None or getattr(_info(master), key) is None:
                messages.append(f"Cannot verify master {name} {key}: missing metadata.")
    emit_progress("calibration-kernel")
    data = np.array(image.data, dtype=np.float32, copy=True)
    history: list[str] = []
    scale = 1.0
    contains_bias = False
    if dark is not None:
        content = (
            plan.dark_contains_bias
            if plan.dark_contains_bias is not None
            else dark.header.get("DCBIAS")
        )
        if not isinstance(content, bool):
            raise PipelineError(
                "Master dark bias content is unknown; provide DCBIAS or "
                "--dark-contains-bias/--dark-no-bias."
            )
        contains_bias = content
        dark_info = _info(dark)
        if plan.dark_scaling:
            if info.exposure is None or dark_info.exposure is None or dark_info.exposure <= 0:
                raise PipelineError("Dark scaling requires valid light and dark EXPTIME metadata.")
            scale = info.exposure / dark_info.exposure
            messages.append(
                "Exposure dark scaling assumes linear dark current; amp glow and "
                "sensor nonlinearity can invalidate it."
            )
        elif info.exposure is not None and dark_info.exposure is not None:
            if not np.isclose(info.exposure, dark_info.exposure, rtol=1e-4, atol=1e-3):
                raise PipelineError(
                    "Dark exposure differs from light exposure; enable --scale-dark explicitly."
                )
        else:
            messages.append("Cannot verify dark exposure: missing metadata.")
        if info.temperature is not None and dark_info.temperature is not None:
            difference = abs(info.temperature - dark_info.temperature)
            if difference > plan.temperature_threshold:
                messages.append(
                    f"Dark temperature differs from light temperature by {difference:.1f} C."
                )
        else:
            messages.append("Cannot verify dark temperature: missing metadata.")
        if contains_bias:
            if plan.dark_scaling:
                if bias is None:
                    raise PipelineError("Scaling a bias-containing dark requires a master bias.")
                data -= np.asarray(bias.data, dtype=np.float32)
                data -= (np.asarray(dark.data, dtype=np.float32) - bias.data) * scale
                history.append("Bias calibrated; bias removed from dark before exposure scaling")
            else:
                data -= np.asarray(dark.data, dtype=np.float32)
                history.append("Bias included in master dark; no separate bias subtraction")
        else:
            if bias is not None:
                data -= np.asarray(bias.data, dtype=np.float32)
                history.append("Bias calibrated")
            else:
                messages.append(
                    "Bias-subtracted dark supplied without master bias; light readout "
                    "offset remains."
                )
            data -= np.asarray(dark.data, dtype=np.float32) * scale
        history.append(f"Dark calibrated; scale={scale:.9g}")
    elif bias is not None:
        data -= np.asarray(bias.data, dtype=np.float32)
        history.append("Bias calibrated")
    result = image.with_data(data)
    if plan.cosmetic_correction:
        if dark is None:
            raise PipelineError("Cosmetic correction requires a master dark.")
        result = correct_defects(result, detect_hot_pixels(dark, sigma=plan.hot_pixel_sigma))
        data = result.data
    checkpoint()
    if flat is not None:
        response = np.asarray(flat.data, dtype=np.float32)
        median = np.nanmedian(response, axis=(0, 1))
        if not np.isfinite(median).all() or (median <= 0).any():
            raise PipelineError("Master flat has no positive finite median response.")
        valid = np.isfinite(response) & (response >= plan.flat_min_fraction * median)
        fraction = float(1 - valid.mean())
        if fraction > plan.max_invalid_flat_fraction:
            raise PipelineError(f"Master flat contains {fraction:.1%} invalid pixels.")
        data = np.divide(data, response, out=np.full_like(data, np.nan), where=valid)
        result = result.with_data(data)
        history.append(
            f"Flat calibrated; response below {plan.flat_min_fraction} of median is invalid"
        )
    result.header["CALIBRAT"] = True
    for item in history:
        result.header.add_history(item)
    result.metadata.update(header_metadata(result.header))
    after = inspect_image(result)
    if after.nonfinite_samples / result.data.size > 0.05:
        messages.append("Calibration produced more than 5% invalid samples.")
    if after.standard_deviation > 20 * max(
        before.standard_deviation, float(np.finfo(np.float32).eps)
    ):
        messages.append("Calibration increased standard deviation more than twentyfold.")
    if (before.fraction_of_saturated_pixels or 0) > 0.01:
        messages.append("Input saturation exceeds 1%; calibration cannot recover clipped signal.")
    return (
        result,
        messages,
        {
            "before": before.model_dump(mode="json"),
            "after": after.model_dump(mode="json"),
            "nan_fraction_before": before.nonfinite_samples / image.data.size,
            "nan_fraction_after": after.nonfinite_samples / result.data.size,
            "dark_scale": scale,
            "dark_contains_bias": contains_bias,
        },
    )
