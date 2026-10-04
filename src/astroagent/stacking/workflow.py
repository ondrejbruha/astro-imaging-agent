import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pydantic import Field

from astroagent.analysis.frame_quality import FrameQualityMetrics
from astroagent.errors import PipelineError
from astroagent.io.datasets import write_json
from astroagent.io.export import export_image
from astroagent.io.fits import load_fits
from astroagent.io.images import image_format
from astroagent.io.metadata import header_metadata
from astroagent.models.dataset import AstroDataset
from astroagent.models.image import AstroImage
from astroagent.registration.engine import RegistrationResult, analyze_frames
from astroagent.stacking.rejection import RejectionParams, reject_frames
from astroagent.stacking.stack import CombineParams, combine_images
from astroagent.stacking.weighting import QualityWeight, UniformWeight, normalized_weights


class StackParams(CombineParams):
    """Frame rejection and normalization above the tiled pixel-combination algorithm."""

    normalize: bool = True
    min_frames: int = Field(default=2, ge=2)
    rejection: RejectionParams = Field(default_factory=RejectionParams)


def restore_registration(dataset: AstroDataset) -> None:
    """Restore original quality and residuals from a persisted registered directory."""
    if dataset.source is None or not dataset.source.is_dir():
        return
    path = dataset.source / "registration.json"
    if not path.exists():
        return
    document = json.loads(path.read_text(encoding="utf-8"))
    dataset.reference = Path(document["reference"])
    dataset.registrations = [RegistrationResult.model_validate(r) for r in document["frames"]]
    dataset.qualities = [FrameQualityMetrics.model_validate(q) for q in document["qualities"]]
    actual = {p.name: p for p in dataset.frames}
    for metric in dataset.qualities:
        if Path(metric.path).name in actual:
            metric.path = str(actual[Path(metric.path).name])
    for registration in dataset.registrations:
        if registration.output and Path(registration.output).name in actual:
            registration.output = str(actual[Path(registration.output).name])
    dataset.reports["registration"] = document


def stack_frames(
    dataset: AstroDataset,
    params: StackParams | None = None,
) -> tuple[AstroImage, dict[str, Any]]:
    """Reject frame outliers and combine aligned mono/RGB data with complete provenance."""
    params = StackParams() if params is None else params
    tick = perf_counter()
    if not dataset.qualities:
        restore_registration(dataset)
    if not dataset.qualities:
        analyze_frames(dataset)
    by_path = {m.path: m for m in dataset.qualities}
    residuals = {r.output: r.residual_rms for r in dataset.registrations if r.output is not None}
    rejected = reject_frames(dataset.qualities, params.rejection, residuals)
    for path in dataset.frames:
        if str(path) not in by_path:
            rejected[str(path)] = ["frame analysis failed"]
    used = [p for p in dataset.frames if str(p) not in rejected]
    if len(used) < params.min_frames:
        raise PipelineError(
            f"Only {len(used)} usable frames remain; stacking requires {params.min_frames}."
        )
    metrics = [by_path[str(p)] for p in used]
    weights = normalized_weights(
        metrics, QualityWeight() if params.method.startswith("weighted") else UniformWeight()
    )
    reference_index = 0
    for r in dataset.registrations:
        if (
            dataset.reference is not None
            and r.path == str(dataset.reference)
            and r.output
            and Path(r.output) in used
        ):
            reference_index = used.index(Path(r.output))
    image, normalization = combine_images(
        (load_fits(p) for p in used),
        len(used),
        params,
        weights=weights,
        normalize=params.normalize,
        reference_index=reference_index,
    )
    # A registered frame already has reference WCS; copy acquisition context from the reference.
    header_source = used[reference_index]
    template = load_fits(header_source)
    image.header = template.header.copy()
    image.header["NCOMBINE"] = len(used)
    image.header["IMAGETYP"] = "MASTER LIGHT"
    image.header["MASTER"] = "LIGHT"
    image.header.add_history(
        f"Registered {len(used)} frames"
        if dataset.registrations
        else f"Combined {len(used)} frames; input alignment assumed"
    )
    if dataset.reference is not None:
        image.header.add_history(f"Reference frame: {dataset.reference.name}")
    image.header.add_history(f"Stack method: {params.method}")
    image.saturation_level = None
    image.header.remove("SATURATE", ignore_missing=True)
    image.metadata.update(header_metadata(image.header))
    image.metadata.pop("SATURATE", None)
    frames = []
    weight_map = dict(zip(map(str, used), map(float, weights), strict=True))
    normalization_map = dict(zip(map(str, used), normalization, strict=True))
    for path in dataset.frames:
        metric = by_path.get(str(path))
        frames.append(
            {
                "path": str(path),
                "quality": metric.quality_score if metric else None,
                "metrics": metric.model_dump(mode="json") if metric else None,
                "weight": weight_map.get(str(path), 0),
                "used": str(path) not in rejected,
                "reasons": rejected.get(str(path), []),
                "normalization": normalization_map.get(str(path)),
            }
        )
    for registration in dataset.registrations:
        if not registration.success:
            frames.append(
                {
                    "path": registration.path,
                    "quality": None,
                    "weight": 0,
                    "used": False,
                    "reasons": [registration.warning or "registration failed"],
                }
            )
    rms = [
        r.residual_rms
        for r in dataset.registrations
        if r.success and r.output in weight_map and r.residual_rms is not None
    ]
    report = {
        "input_frames": len(frames),
        "used_frames": len(used),
        "rejected_frames": len(frames) - len(used),
        "reference": str(dataset.reference) if dataset.reference is not None else None,
        "registration": {
            "median_residual_rms": float(np.median(rms)) if rms else None,
            "frames": [r.model_dump(mode="json") for r in dataset.registrations],
        },
        "stack": params.model_dump(mode="json"),
        "frames": frames,
        "upstream": dataset.reports,
        "duration_ms": (perf_counter() - tick) * 1000,
        "warnings": (
            ["Sigma clipping with fewer than five frames is statistically fragile."]
            if len(used) < 5 and "sigma-clipped" in params.method
            else []
        ),
    }
    return image, report


def save_stack(
    image: AstroImage,
    report: dict[str, Any],
    output: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write the master in the chosen format and a strict processing sidecar."""
    base = output.with_suffix("") if output.suffix.lower() == ".gz" else output
    report_path = base.with_suffix(".processing.json")
    image_format(output)
    if not overwrite and any(p.exists() for p in (output, report_path)):
        raise PipelineError(f"Output already exists: {output} or its processing report.")
    original_paths = [f["path"] for f in report["frames"]]
    original_paths += [r["path"] for r in report.get("registration", {}).get("frames", [])]
    if any(Path(path).resolve() == output.resolve() for path in original_paths):
        raise PipelineError("Stack output must not replace an input frame.")
    output.parent.mkdir(parents=True, exist_ok=True)
    report["output"] = str(output)
    report["export"] = export_image(image, output, overwrite=overwrite)
    write_json(report_path, report, overwrite=overwrite)
