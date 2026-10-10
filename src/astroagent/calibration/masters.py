import logging
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from astroagent.analysis.statistics import inspect_image
from astroagent.calibration.engine import calibrate_image
from astroagent.calibration.models import (
    AstroSession,
    CalibrationPlan,
    FrameInfo,
    FrameType,
    MasterFrame,
)
from astroagent.calibration.session import compatible, group_frames, inspect_frame
from astroagent.errors import AstroError, PipelineError
from astroagent.execution import ExecutionContext, checkpoint, emit_progress, execution_scope
from astroagent.io.datasets import frame_name, prepare_directory, write_json
from astroagent.io.fits import load_fits, save_fits
from astroagent.models.image import AstroImage
from astroagent.stacking.stack import CombineParams, combine_images

logger = logging.getLogger(__name__)


def normalize_flat(image: AstroImage) -> AstroImage:
    """Normalize mono/RGB channels or each of four CFA phases by a positive finite median."""
    data = np.array(image.data, dtype=np.float32, copy=True)
    if image.cfa is not None:
        planes = [data[y::2, x::2] for y in range(2) for x in range(2)]
    else:
        planes = [data] if image.channels == 1 else [data[..., c] for c in range(3)]
    for plane in planes:
        median = float(np.nanmedian(plane))
        if not np.isfinite(median) or median <= 0:
            raise PipelineError(
                "Flat normalization requires positive finite median signal in every phase/channel."
            )
        plane /= median
    return image.with_data(data)


def choose_master(
    info: FrameInfo,
    masters: list[MasterFrame],
    kind: FrameType,
    *,
    dark_scaling: bool = False,
) -> MasterFrame | None:
    """Select compatible sampling/settings, filter-specific flats and closest dark temperature."""
    candidates = [
        m
        for m in masters
        if m.info.frame_type == kind and compatible(info, m.info, flat=kind == FrameType.FLAT)
    ]
    if kind in (FrameType.DARK, FrameType.DARK_FLAT) and not dark_scaling:
        candidates = [
            m
            for m in candidates
            if info.exposure is not None
            and m.info.exposure is not None
            and np.isclose(info.exposure, m.info.exposure, rtol=1e-4, atol=1e-3)
        ]
    if not candidates:
        return None

    def distance(master: MasterFrame) -> tuple[float, float, str]:
        exposure = (
            abs((info.exposure or 0) - (master.info.exposure or 0))
            if kind in (FrameType.DARK, FrameType.DARK_FLAT)
            else 0
        )
        temperature = (
            abs(info.temperature - master.info.temperature)
            if info.temperature is not None and master.info.temperature is not None
            else 1e6
        )
        return exposure, temperature, str(master.path)

    return min(candidates, key=distance)


@execution_scope
def build_master(
    frames: list[FrameInfo],
    kind: FrameType,
    output: Path,
    *,
    params: CombineParams | None = None,
    bias: AstroImage | None = None,
    dark: AstroImage | None = None,
    overwrite: bool = False,
    context: ExecutionContext | None = None,
) -> MasterFrame:
    """Combine compatible calibration frames with conservative frame and pixel rejection.

    Bias is subtracted from each dark before combination. Flats are corrected
    then normalized independently in each CFA phase, and the final master is
    normalized again. Frame median/noise outliers beyond eight MAD are rejected
    only for groups of at least five frames. No bias normalization is applied.
    """
    if not frames or len(group_frames(frames, kind)) != 1:
        raise PipelineError(
            f"Master {kind.value} requires a nonempty compatible calibration group."
        )
    params = CombineParams(method="sigma-clipped") if params is None else params
    if params.method not in ("mean", "median", "sigma-clipped"):
        raise PipelineError(
            "Calibration masters support mean, median or sigma-clipped combination."
        )
    if output.resolve() in {f.path.resolve() for f in frames}:
        raise PipelineError("Master output must not replace an input frame.")
    report_path = output.with_suffix(".processing.json")
    if not overwrite and any(p.exists() for p in (output, report_path)):
        raise PipelineError(f"Output already exists: {output} or its report.")
    started = perf_counter()
    logger.info("Building master %s from %d frames", kind.value, len(frames))
    measurements: list[dict[str, Any]] = []
    rejected: dict[str, list[str]] = {}
    # Header-only discovery and a quality pass keep memory independent of N.
    emit_progress("master-analysis", 0, len(frames), "frame")
    for index, info in enumerate(frames, 1):
        checkpoint()
        try:
            stats = inspect_image(load_fits(info.path))
        except (AstroError, OSError, ValueError) as exc:
            rejected[str(info.path)] = [f"frame quality unavailable: {exc}"]
            emit_progress("master-analysis", index, len(frames), "frame")
            continue
        measurements.append(
            {
                "path": str(info.path),
                "median": stats.median,
                "noise": stats.standard_deviation,
                "saturation": stats.fraction_of_saturated_pixels,
                "median_adu_fraction": (
                    stats.median / stats.saturation_level if stats.saturation_level else None
                ),
                "dynamic_range": stats.percentile_99 - stats.percentile_1,
            }
        )
        emit_progress("master-analysis", index, len(frames), "frame")
    messages: list[str] = []
    for m in measurements:
        if kind == FrameType.FLAT:
            if (m["saturation"] or 0) > 0.1:
                rejected[m["path"]] = ["flat saturation exceeds 10%"]
            if m["median_adu_fraction"] is not None and not 0.05 <= m["median_adu_fraction"] <= 0.9:
                messages.append(f"Flat {m['path']} median ADU is outside 5..90% of saturation.")
    if len(measurements) >= 5:
        for key in ("median", "noise"):
            values = np.array([m[key] for m in measurements])
            center = np.median(values)
            spread = 1.4826 * np.median(np.abs(values - center))
            if spread > np.finfo(np.float32).eps * max(1, abs(center)):
                for m, value in zip(measurements, values, strict=True):
                    if abs(value - center) > 8 * spread:
                        rejected.setdefault(m["path"], []).append(f"{key} exceeds eight MAD")
    used = [f for f in frames if str(f.path) not in rejected]
    for path, reasons in rejected.items():
        logger.warning("Rejected %s: %s", path, "; ".join(reasons))
    for message in messages:
        logger.warning("%s", message)
    if not used:
        raise PipelineError("No usable calibration frames remain after rejection.")

    def corrected() -> Iterator[AstroImage]:
        for info in used:
            checkpoint()
            image = load_fits(info.path)
            if info.cfa is not None:
                image.header["BAYERPAT"] = info.cfa.pattern
                image.header["XBAYROFF"] = info.cfa.x_offset
                image.header["YBAYROFF"] = info.cfa.y_offset
            if kind != FrameType.BIAS and (bias is not None or dark is not None):
                image, warnings, _ = calibrate_image(image, CalibrationPlan(), bias=bias, dark=dark)
                messages.extend(warnings)
            if kind == FrameType.FLAT:
                image = normalize_flat(image)
            yield image

    master, _ = combine_images(corrected(), len(used), params)
    if kind == FrameType.FLAT:
        master = normalize_flat(master)
    master.header.remove("CALIBRAT", ignore_missing=True)
    master.header["MASTER"] = kind.value.upper()
    master.header["IMAGETYP"] = kind.value.upper()
    master.header["NCOMBINE"] = len(used)
    if kind in (FrameType.DARK, FrameType.DARK_FLAT):
        master.header["DCBIAS"] = bias is None
    master.header.add_history(f"Master {kind.value}: {len(used)} frames, {params.method}")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint()
    save_fits(master, output, overwrite=overwrite)
    result = MasterFrame(
        path=output,
        info=inspect_frame(output),
        contains_bias=bias is None and kind in (FrameType.DARK, FrameType.DARK_FLAT),
        input_paths=[f.path for f in used],
        rejected=rejected,
    )
    write_json(
        report_path,
        {
            **result.model_dump(mode="json"),
            "params": params.model_dump(mode="json"),
            "bias": str(bias.path) if bias is not None else None,
            "dark": str(dark.path) if dark is not None else None,
            "measurements": measurements,
            "warnings": list(dict.fromkeys(messages)),
            "duration_ms": (perf_counter() - started) * 1000,
        },
        overwrite=overwrite,
    )
    return result


@execution_scope
def build_masters(
    session: AstroSession,
    output: Path,
    *,
    params: CombineParams | None = None,
    overwrite: bool = False,
    context: ExecutionContext | None = None,
    initial_masters: list[MasterFrame] | None = None,
) -> list[MasterFrame]:
    """Build separate sensor, exposure/temperature and filter groups in dependency order."""
    prepare_directory(output, [f.path for f in session.frames], overwrite=overwrite)
    results: list[MasterFrame] = list(initial_masters or [])
    for kind in (FrameType.BIAS, FrameType.DARK, FrameType.DARK_FLAT, FrameType.FLAT):
        for index, group in enumerate(group_frames(session.of_type(kind), kind), 1):
            checkpoint()
            emit_progress("building-masters")
            info = group[0]
            bias = choose_master(info, results, FrameType.BIAS)
            dark = None
            if kind == FrameType.FLAT:
                dark = choose_master(info, results, FrameType.DARK_FLAT) or choose_master(
                    info, results, FrameType.DARK
                )
            label = kind.value
            if kind in (FrameType.DARK, FrameType.DARK_FLAT):
                label += f"-{info.exposure if info.exposure is not None else 'unknown'}s"
            if kind == FrameType.FLAT:
                label += f"-{info.filter_name or 'none'}"
            label = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
            target = output / frame_name(Path(f"master-{label}.fit"), index)
            result = build_master(
                group,
                kind,
                target,
                params=params,
                bias=load_fits(bias.path) if bias is not None and kind != FrameType.BIAS else None,
                dark=load_fits(dark.path) if dark is not None else None,
                overwrite=overwrite,
            )
            results.append(result)
    write_json(
        output / "calibration-masters.json",
        {
            "masters": [m.model_dump(mode="json") for m in results],
            "session_counts": session.counts(),
            "params": (params or CombineParams(method="sigma-clipped")).model_dump(mode="json"),
        },
        overwrite=overwrite,
    )
    return results
