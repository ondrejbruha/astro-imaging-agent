import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from astroagent.calibration.debayer import debayer_image
from astroagent.calibration.engine import calibrate_image
from astroagent.calibration.masters import build_masters, choose_master
from astroagent.calibration.models import AstroSession, CalibrationPlan, FrameType, MasterFrame
from astroagent.calibration.session import inspect_frame, inspect_session_frames, validate_session
from astroagent.errors import AstroError, PipelineError
from astroagent.io.datasets import frame_name, prepare_directory, write_json
from astroagent.io.fits import load_fits, save_fits
from astroagent.models.dataset import AstroDataset
from astroagent.models.layout import CFAMetadata
from astroagent.stacking.stack import CombineParams

logger = logging.getLogger(__name__)


def plan_calibration(session: AstroSession, plan: CalibrationPlan) -> dict[str, Any]:
    """Describe dependency ordering without creating output or inventing unavailable masters."""
    warnings = validate_session(session)
    counts = session.counts()
    steps = []
    for kind, name in (
        (FrameType.BIAS, "master bias"),
        (FrameType.DARK, "master dark"),
        (FrameType.DARK_FLAT, "master dark flat"),
        (FrameType.FLAT, "master flat"),
    ):
        if counts[kind.value]:
            steps.append(f"Build {name} from {counts[kind.value]} frames")
    steps.append("Calibrate LIGHT: bias/dark subtraction, then flat division when available")
    if any(f.cfa is not None for f in session.of_type(FrameType.LIGHT)):
        steps.append(
            "Debayer raw CFA after calibration using recorded pattern and offsets / bilinear"
        )
    return {
        "session_counts": counts,
        "frames": [f.model_dump(mode="json") for f in session.frames],
        "plan": plan.model_dump(mode="json"),
        "steps": steps,
        "warnings": warnings,
    }


def calibrate_frames(
    dataset: AstroDataset,
    output: Path,
    plan: CalibrationPlan | None = None,
    *,
    debayer: bool = True,
    overwrite: bool = False,
    combine: CombineParams | None = None,
    progress: Callable[[], None] | None = None,
) -> AstroDataset:
    """Discover/build missing masters, select compatible corrections and calibrate each light."""
    plan = CalibrationPlan() if plan is None else plan
    started = perf_counter()
    session = inspect_session_frames(dataset.frames, cfa_pattern=plan.cfa_pattern)
    warnings = validate_session(session)
    prepare_directory(output, dataset.frames, overwrite=overwrite)
    masters = list(dataset.masters)
    explicit: dict[FrameType, MasterFrame] = {}
    for kind, path in (
        (FrameType.BIAS, plan.master_bias),
        (FrameType.DARK, plan.master_dark),
        (FrameType.FLAT, plan.master_flat),
    ):
        if path is not None:
            info = inspect_frame(path, cfa_pattern=plan.cfa_pattern)
            info = info.model_copy(update={"frame_type": kind})
            image = load_fits(path)
            explicit[kind] = MasterFrame(
                path=path, info=info, contains_bias=bool(image.header.get("DCBIAS", False))
            )
    if not masters:
        missing = session.model_copy(
            update={
                "frames": [
                    f
                    for f in session.frames
                    if f.frame_type not in explicit or f.frame_type == FrameType.LIGHT
                ]
            }
        )
        if any(
            missing.of_type(k)
            for k in (FrameType.BIAS, FrameType.DARK, FrameType.DARK_FLAT, FrameType.FLAT)
        ):
            # Explicit bias/dark is also made available to dependent master builders.
            masters = build_masters(
                missing,
                output / "masters",
                params=combine,
                overwrite=overwrite,
                initial_masters=list(explicit.values()),
            )
    result = AstroDataset([], source=dataset.source, masters=masters, reports=dict(dataset.reports))
    cached_master = lru_cache(maxsize=3)(load_fits)
    reports = []
    lights = session.of_type(FrameType.LIGHT)
    for index, info in enumerate(lights, 1):
        tick = perf_counter()
        record: dict[str, Any] = {"path": str(info.path), "output": None, "warnings": []}
        try:
            selected = {}
            for kind in (FrameType.BIAS, FrameType.DARK, FrameType.FLAT):
                candidate = explicit.get(kind) or choose_master(
                    info, masters, kind, dark_scaling=plan.dark_scaling
                )
                available = kind in explicit or any(m.info.frame_type == kind for m in masters)
                if candidate is None and available:
                    raise PipelineError(
                        f"No compatible {kind.value} frame found for {info.path.name}, "
                        f"filter {info.filter_name or 'none'}."
                    )
                selected[kind] = candidate
            per_frame = plan.model_copy(
                update={
                    **{
                        f"master_{kind.value}": master.path if master else None
                        for kind, master in selected.items()
                    },
                }
            )
            image = load_fits(info.path)
            if info.cfa is not None:
                image.header["BAYERPAT"] = info.cfa.pattern
                image.header["XBAYROFF"] = info.cfa.x_offset
                image.header["YBAYROFF"] = info.cfa.y_offset
            image, messages, quality = calibrate_image(
                image,
                per_frame,
                bias=cached_master(per_frame.master_bias) if per_frame.master_bias else None,
                dark=cached_master(per_frame.master_dark) if per_frame.master_dark else None,
                flat=cached_master(per_frame.master_flat) if per_frame.master_flat else None,
            )
            cfa = image.cfa
            if debayer and cfa is not None:
                image = debayer_image(image, cfa)
            target = output / frame_name(
                info.path, index, suffix="_cal_rgb" if debayer and cfa else "_cal"
            )
            save_fits(image, target, overwrite=overwrite)
            result.frames.append(target)
            record.update(
                {
                    "output": str(target),
                    "warnings": messages,
                    "quality": quality,
                    "plan": per_frame.model_dump(mode="json"),
                    "debayer": {
                        "enabled": debayer and cfa is not None,
                        "cfa": cfa.model_dump() if cfa else None,
                        "method": "bilinear",
                    },
                }
            )
            logger.info("Calibrating lights %d/%d: %s", index, len(lights), info.path.name)
        except (AstroError, ValueError, OSError) as exc:
            record["warnings"] = [str(exc)]
            logger.warning("Excluded light %s: %s", info.path, exc)
        record["duration_ms"] = (perf_counter() - tick) * 1000
        reports.append(record)
        if progress is not None:
            progress()
    report = {
        "session_counts": session.counts(),
        "input_paths": [str(p) for p in dataset.frames],
        "masters": [m.model_dump(mode="json") for m in [*masters, *explicit.values()]],
        "params": plan.model_dump(mode="json"),
        "debayer": debayer,
        "combine": (combine or CombineParams(method="sigma-clipped")).model_dump(mode="json"),
        "frames": reports,
        "warnings": warnings,
        "duration_ms": (perf_counter() - started) * 1000,
    }
    result.reports["calibration"] = report
    write_json(output / "calibration.processing.json", report, overwrite=overwrite)
    if not result.frames:
        detail = "; ".join(r["warnings"][0] for r in reports if r["warnings"])
        raise PipelineError(f"No usable calibrated light frames remain. {detail}")
    return result


def debayer_frames(
    dataset: AstroDataset,
    output: Path,
    *,
    pattern: str | None = None,
    overwrite: bool = False,
) -> AstroDataset:
    """Demosaic an existing CFA dataset without calibration or hidden pattern guessing."""
    prepare_directory(output, dataset.frames, overwrite=overwrite)
    result = AstroDataset(
        [], source=dataset.source, masters=dataset.masters, reports=dict(dataset.reports)
    )
    for index, path in enumerate(dataset.frames, 1):
        image = load_fits(path)
        cfa = (
            CFAMetadata(
                pattern=pattern,
                x_offset=int(image.header.get("XBAYROFF", 0)),
                y_offset=int(image.header.get("YBAYROFF", 0)),
            )
            if pattern
            else image.cfa
        )
        image = debayer_image(image, cfa)
        target = output / frame_name(path, index, suffix="_rgb")
        save_fits(image, target, overwrite=overwrite)
        result.frames.append(target)
    write_json(
        output / "debayer.json",
        {
            "inputs": [str(p) for p in dataset.frames],
            "outputs": [str(p) for p in result.frames],
            "pattern": pattern,
            "method": "bilinear",
        },
        overwrite=overwrite,
    )
    return result
