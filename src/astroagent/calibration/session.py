import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from astroagent.calibration.models import AstroSession, FrameInfo, FrameType
from astroagent.errors import PipelineError
from astroagent.execution import ExecutionContext, checkpoint, emit_progress, execution_scope
from astroagent.io.datasets import discover_fits
from astroagent.models.layout import CFAMetadata, ImageLayout

_TYPE_NAMES = {
    "light": FrameType.LIGHT,
    "lights": FrameType.LIGHT,
    "light frame": FrameType.LIGHT,
    "object": FrameType.LIGHT,
    "science": FrameType.LIGHT,
    "dark": FrameType.DARK,
    "darks": FrameType.DARK,
    "dark frame": FrameType.DARK,
    "flat": FrameType.FLAT,
    "flats": FrameType.FLAT,
    "flat field": FrameType.FLAT,
    "flat frame": FrameType.FLAT,
    "bias": FrameType.BIAS,
    "biases": FrameType.BIAS,
    "bias frame": FrameType.BIAS,
    "zero": FrameType.BIAS,
    "darkflat": FrameType.DARK_FLAT,
    "dark flat": FrameType.DARK_FLAT,
    "dark_flat": FrameType.DARK_FLAT,
}


def classify_frame(header: Header, path: Path) -> FrameType:
    """Prefer explicit FITS purpose, then directory tokens, then conservative filename tokens."""
    for key in ("IMAGETYP", "FRAME", "OBSTYPE", "IMAGETYPE"):
        value = header.get(key)
        if value is not None:
            return _TYPE_NAMES.get(str(value).strip().casefold(), FrameType.UNKNOWN)
    for part in reversed(path.parent.parts):
        if part.casefold() in _TYPE_NAMES:
            return _TYPE_NAMES[part.casefold()]
    tokens = re.split(r"[_\- .]+", path.stem.casefold())
    found = {_TYPE_NAMES[token] for token in tokens if token in _TYPE_NAMES}
    if tokens[:2] == ["dark", "flat"]:
        return FrameType.DARK_FLAT
    return next(iter(found)) if len(found) == 1 else FrameType.UNKNOWN


def _number(header: Header, *keys: str) -> float | None:
    for key in keys:
        value = header.get(key)
        if value is not None:
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise PipelineError(f"Invalid FITS {key}: {value!r}.") from exc
            if not math.isfinite(number):
                raise PipelineError(f"Nonfinite FITS {key}.")
            return number
    return None


def normalize_metadata(header: Header, path: Path, *, cfa_pattern: str | None = None) -> FrameInfo:
    """Normalize common camera aliases while retaining unknown values explicitly."""
    naxis = int(header.get("NAXIS", 0))
    if naxis not in (2, 3):
        raise PipelineError(f"Unsupported image dimensions in {path}.")
    width, height = int(header["NAXIS1"]), int(header["NAXIS2"])
    layout = ImageLayout.MONO
    if naxis == 3:
        first, last = int(header["NAXIS3"]) == 3, width == 3
        axis = header.get("ASTRCHAX")
        if axis is None:
            if first == last:
                raise PipelineError(f"Ambiguous RGB cube in {path}.")
            axis = 0 if first else 2
        if axis == 2 and last:
            width, height = height, int(header["NAXIS3"])
        elif axis != 0 or not first:
            raise PipelineError(f"Invalid RGB channel axis in {path}.")
        layout = ImageLayout.RGB
    pattern = cfa_pattern or header.get("BAYERPAT", header.get("BAYERPATN"))
    cfa = None
    if layout != ImageLayout.RGB and not header.get("DEBAYER", False):
        if pattern is not None:
            cfa = CFAMetadata(
                pattern=str(pattern),
                x_offset=int(header.get("XBAYROFF", 0)),
                y_offset=int(header.get("YBAYROFF", 0)),
            )
            layout = ImageLayout.CFA
        elif str(header.get("COLORTYP", "")).casefold() in {"osc", "cfa", "bayer"}:
            raise PipelineError(f"CFA pattern is required for {path}; use --cfa-pattern.")
    filter_name = header.get("FILTER", header.get("FILTERID"))
    sources = {}
    capture_time = header.get("DATE-OBS")
    camera = header.get("INSTRUME", header.get("CAMERA"))
    telescope = header.get("TELESCOP")
    for key, value, card in (
        ("capture_time", capture_time, "DATE-OBS"),
        ("camera", camera, "INSTRUME/CAMERA"),
        ("telescope", telescope, "TELESCOP"),
    ):
        if value is not None:
            sources[key] = card
    pixel_scale = None
    try:
        wcs = WCS(header, naxis=2)
        if wcs.has_celestial:
            scales = np.asarray(proj_plane_pixel_scales(wcs.celestial), dtype=float) * 3600
            if np.isfinite(scales).all() and (scales > 0).all():
                pixel_scale = (float(scales[0]), float(scales[1]))
                sources["pixel_scale_arcsec"] = "celestial WCS projected plane"
    except (ValueError, TypeError):
        pass
    return FrameInfo(
        path=path,
        frame_type=classify_frame(header, path),
        exposure=_number(header, "EXPTIME", "EXPOSURE", "EXP_TIME"),
        gain=_number(header, "GAIN", "CCD-GAIN"),
        offset=_number(header, "OFFSET", "BLKLEVEL"),
        temperature=_number(header, "CCD-TEMP", "SENSOR-T", "SET-TEMP"),
        filter_name=str(filter_name).strip() if filter_name is not None else None,
        cfa=cfa,
        layout=layout,
        width=width,
        height=height,
        binning=(
            int(header.get("XBINNING", header.get("BINX", 1))),
            int(header.get("YBINNING", header.get("BINY", 1))),
        ),
        bitpix=int(header["BITPIX"]),
        master="MASTER" in header,
        capture_time=None if capture_time is None else str(capture_time),
        camera=None if camera is None else str(camera),
        telescope=None if telescope is None else str(telescope),
        pixel_scale_arcsec=pixel_scale,
        metadata_sources=sources,
    )


def inspect_frame(
    path: Path,
    *,
    cfa_pattern: str | None = None,
    frame_type: FrameType | None = None,
    hdu: int | None = None,
) -> FrameInfo:
    """Inspect the first image HDU header without reading full pixel arrays."""
    with fits.open(path, memmap=True) as hdus:
        if hdu is not None and (type(hdu) is not int or hdu < 0 or hdu >= len(hdus)):
            raise PipelineError("Selected HDU index is outside the FITS file.")
        for index, item in enumerate(hdus):
            if hdu is not None and index != hdu:
                continue
            if (
                isinstance(item, fits.PrimaryHDU | fits.ImageHDU | fits.CompImageHDU)
                and item.header.get("NAXIS", 0) >= 2
            ):
                header = hdus[0].header.copy()
                header.extend(item.header, update=True)
                info = normalize_metadata(header, path, cfa_pattern=cfa_pattern)
                if frame_type is not None:
                    info.frame_type = FrameType(frame_type)
                    info.purpose_source = "explicit-override"
                info.input_hdu = index
                return info
    raise PipelineError(f"No image HDU in {path}.")


def discover_session(source: Path | str, *, cfa_pattern: str | None = None) -> AstroSession:
    """Recursively discover a session, reporting unreadable and unknown files."""
    return inspect_session_frames(discover_fits(source, recursive=True), cfa_pattern=cfa_pattern)


@execution_scope
def inspect_session_frames(
    paths: list[Path],
    *,
    cfa_pattern: str | None = None,
    purposes: dict[str, str] | None = None,
    context: ExecutionContext | None = None,
) -> AstroSession:
    """Inspect explicit dataset paths, warning on invalid files and refusing CFA ambiguity."""
    session = AstroSession(frames=[])
    emit_progress("session-inspection", 0, len(paths), "frame")
    for index, path in enumerate(paths, 1):
        checkpoint()
        try:
            purpose = (purposes or {}).get(str(path))
            info = inspect_frame(
                path,
                cfa_pattern=cfa_pattern,
                frame_type=None if purpose is None else FrameType(purpose),
            )
        except (OSError, ValueError, PipelineError) as exc:
            # Unknown CFA is a required user choice, not a reason to silently skip a light.
            if "CFA" in str(exc):
                raise PipelineError(str(exc)) from exc
            session.warnings.append(f"Cannot inspect {path}: {exc}")
            emit_progress("session-inspection", index, len(paths), "frame")
            continue
        session.frames.append(info)
        if info.frame_type == FrameType.UNKNOWN:
            session.warnings.append(f"Unknown frame type: {path}.")
        emit_progress("session-inspection", index, len(paths), "frame")
    return session


def cfa_key(info: FrameInfo) -> tuple[Any, ...]:
    """Represent the effective CFA color phase for compatibility grouping."""
    return () if info.cfa is None else tuple(info.cfa.tile().ravel())


def group_key(info: FrameInfo, kind: FrameType) -> tuple[Any, ...]:
    """Use exact known compatibility values; dark temperature is rounded to 0.1 C."""
    base = (
        info.width,
        info.height,
        info.layout.value,
        info.binning,
        info.gain,
        info.offset,
        cfa_key(info),
    )
    if kind in (FrameType.DARK, FrameType.DARK_FLAT):
        return (
            *base,
            info.exposure,
            None if info.temperature is None else round(info.temperature, 1),
        )
    if kind == FrameType.FLAT:
        return (*base, info.filter_name)
    return base


def group_frames(frames: list[FrameInfo], kind: FrameType) -> list[list[FrameInfo]]:
    """Partition masters by compatible sensor sampling and acquisition settings."""
    groups: dict[tuple[Any, ...], list[FrameInfo]] = defaultdict(list)
    for info in frames:
        groups[group_key(info, kind)].append(info)
    return list(groups.values())


def compatible(target: FrameInfo, candidate: FrameInfo, *, flat: bool = False) -> bool:
    """Require sampling identity and reject conflicting known camera settings."""
    if (target.width, target.height, target.layout, target.binning, cfa_key(target)) != (
        candidate.width,
        candidate.height,
        candidate.layout,
        candidate.binning,
        cfa_key(candidate),
    ):
        return False
    if any(
        a is not None and b is not None and a != b
        for a, b in (
            (target.gain, candidate.gain),
            (target.offset, candidate.offset),
        )
    ):
        return False
    return not flat or target.filter_name == candidate.filter_name


def validate_session(session: AstroSession) -> list[str]:
    """Check light consistency and explain suspicious or missing acquisition metadata."""
    warnings = list(session.warnings)
    lights = session.of_type(FrameType.LIGHT)
    if not lights:
        raise PipelineError(
            "No LIGHT frames found; classify frames using headers or lights/ directory."
        )
    if len({cfa_key(f) for f in lights}) > 1:
        raise PipelineError("CFA patterns are inconsistent across light frames.")
    if len({(f.width, f.height, f.binning, f.layout) for f in lights}) > 1:
        warnings.append(
            "Light dimensions, binning or channel layouts differ; process compatible "
            "groups separately."
        )
    for key in ("gain", "offset", "exposure", "temperature", "bitpix", "filter_name"):
        values = {getattr(f, key) for f in lights}
        if None in values:
            warnings.append(f"Some light {key} metadata is unavailable.")
        if len(values - {None}) > 1:
            warnings.append(f"Light {key} values differ; compatibility will be checked per frame.")
    for kind in (FrameType.BIAS, FrameType.DARK, FrameType.FLAT):
        for info in session.of_type(kind):
            if not any(compatible(light, info, flat=kind == FrameType.FLAT) for light in lights):
                warnings.append(
                    f"{kind.value} {info.path} has no compatible light sampling/settings/filter."
                )
    return warnings
