import json
from pathlib import Path
from typing import Any

from astroagent.errors import PipelineError
from astroagent.io.artifacts import write_text
from astroagent.io.images import FITS_SUFFIXES


def discover_fits(source: Path | str, *, recursive: bool = False) -> list[Path]:
    """Find FITS frames in stable path order; sidecars and raster previews are excluded."""
    root = Path(source)
    if root.is_file():
        paths = [root]
    elif root.is_dir():
        paths = list(root.rglob("*") if recursive else root.iterdir())
    else:
        raise PipelineError(f"Input does not exist: {root}")
    frames = sorted(
        (
            p
            for p in paths
            if p.is_file()
            and (
                p.suffix.lower() in FITS_SUFFIXES
                or (p.suffix.lower() == ".gz" and p.with_suffix("").suffix.lower() in FITS_SUFFIXES)
            )
        ),
        key=lambda p: str(p).casefold(),
    )
    if not frames:
        raise PipelineError(f"No FITS frames found in {root}.")
    return frames


def write_json(path: Path, report: dict[str, Any], *, overwrite: bool = False) -> Path:
    """Write strict, atomic JSON provenance without nonfinite floating values."""
    return write_text(
        path, json.dumps(report, indent=2, allow_nan=False) + "\n", overwrite=overwrite
    )


def prepare_directory(output: Path, inputs: list[Path], *, overwrite: bool) -> None:
    """Refuse input replacement and existing output contents before dataset processing."""
    if any(p.resolve().is_relative_to(output.resolve()) for p in inputs):
        raise PipelineError("Output directory must not contain input frames.")
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not overwrite)):
        raise PipelineError(f"Output already exists: {output}. Use --overwrite to replace it.")
    output.mkdir(parents=True, exist_ok=True)


def frame_name(path: Path, index: int, *, suffix: str = "") -> str:
    """Prefix a stable ordinal to prevent collisions between nested session filenames."""
    name = path.with_suffix("").stem if path.suffix.lower() == ".gz" else path.stem
    return f"{index:04d}_{name}{suffix}.fit"
