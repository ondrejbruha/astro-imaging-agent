import logging
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
from pydantic import Field, model_validator

from astroagent.analysis.frame_quality import FrameQualityMetrics, measure_frame, score_frames
from astroagent.errors import AstroError, PipelineError
from astroagent.io.datasets import frame_name, prepare_directory, write_json
from astroagent.io.fits import load_fits, save_fits
from astroagent.io.wcs import copy_reference_wcs
from astroagent.models.base import SchemaModel
from astroagent.models.dataset import AstroDataset
from astroagent.registration.matching import match_stars
from astroagent.registration.reference import select_reference
from astroagent.registration.resample import Interpolation, resample_image
from astroagent.registration.stars import DetectionParams
from astroagent.registration.transform import (
    RegistrationTransform,
    TransformModel,
    ransac_transform,
)

logger = logging.getLogger(__name__)


class RegistrationParams(SchemaModel):
    """Registration eligibility, geometric fitting, interpolation and deterministic seed."""

    reference: str = "auto"
    interpolation: Interpolation = "bicubic"
    model: TransformModel = "similarity"
    detection: DetectionParams = Field(default_factory=DetectionParams)
    max_matching_stars: int = Field(default=100, ge=6, le=500)
    match_radius: float = Field(default=2, gt=0)
    triangle_tolerance: float = Field(default=0.015, gt=0, lt=0.2)
    ransac_trials: int = Field(default=500, ge=1, le=10000)
    min_matches: int = Field(default=6, ge=3)
    min_inliers: int = Field(default=6, ge=3)
    min_inlier_fraction: float = Field(default=0.5, gt=0, le=1)
    max_residual_rms: float = Field(default=1, gt=0)
    min_scale: float = Field(default=0.8, gt=0)
    max_scale: float = Field(default=1.2, gt=0)
    random_seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def scale_range(self) -> "RegistrationParams":
        """Require the permitted scale interval to be ordered."""
        if self.max_scale < self.min_scale:
            raise ValueError("max_scale must be at least min_scale.")
        return self


class RegistrationResult(SchemaModel):
    """Per-frame registration diagnostics, including excluded failures."""

    path: str
    output: str | None = None
    success: bool
    matched_stars: int = 0
    matching_candidates: int = 0
    inliers: int = 0
    residual_rms: float | None = None
    transform: RegistrationTransform | None = None
    warning: str | None = None
    duration_ms: float = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)


def analyze_frames(dataset: AstroDataset, detection: DetectionParams | None = None) -> AstroDataset:
    """Analyze one disk frame at a time; record unreadable frames without aborting the dataset."""
    started = perf_counter()
    metrics: list[FrameQualityMetrics] = []
    failures = []
    for path in dataset.frames:
        try:
            frame, catalog = measure_frame(load_fits(path), path=path, detection=detection)
        except (AstroError, OSError, ValueError) as exc:
            failures.append({"path": str(path), "warning": str(exc)})
            logger.warning("Cannot analyze %s: %s", path, exc)
            continue
        dataset.catalogs[str(path)] = catalog
        metrics.append(frame)
    if not metrics:
        raise PipelineError(
            "No usable frames could be analyzed. Debayer raw CFA before registration."
        )
    dataset.qualities = score_frames(metrics)
    dataset.reports["analysis"] = {
        "frames": [m.model_dump(mode="json") for m in dataset.qualities],
        "failed": failures,
        "duration_ms": (perf_counter() - started) * 1000,
        "detection": (detection or DetectionParams()).model_dump(mode="json"),
    }
    return dataset


def register_frames(
    dataset: AstroDataset,
    output: Path,
    params: RegistrationParams | None = None,
    *,
    overwrite: bool = False,
    progress: Callable[[], None] | None = None,
) -> AstroDataset:
    """Select reference, match triangles, robustly fit and resample to reference coverage."""
    params = RegistrationParams() if params is None else params
    prepare_directory(output, dataset.frames, overwrite=overwrite)
    started = perf_counter()
    if not dataset.catalogs or not dataset.qualities:
        analyze_frames(dataset, params.detection)
    if params.reference == "auto":
        reference = Path(select_reference(dataset.qualities, min_stars=params.min_matches).path)
    else:
        candidates = [
            p
            for p in dataset.frames
            if str(p) == params.reference
            or p.name == params.reference
            or p.resolve() == Path(params.reference).resolve()
        ]
        if len(candidates) != 1:
            raise PipelineError("Explicit reference must identify exactly one input frame.")
        reference = candidates[0]
    reference_image = load_fits(reference)
    reference_catalog = dataset.catalogs.get(str(reference))
    if reference_catalog is None or len(reference_catalog.stars) < params.min_matches:
        raise PipelineError("Reference frame has too few usable detected stars.")
    result = AstroDataset(
        [],
        source=dataset.source,
        reference=reference,
        masters=dataset.masters,
        reports=dict(dataset.reports),
    )
    metrics = {m.path: m for m in dataset.qualities}
    for index, path in enumerate(dataset.frames, 1):
        tick = perf_counter()
        item = RegistrationResult(path=str(path), success=False)
        try:
            if str(path) not in metrics:
                raise PipelineError("Frame analysis failed; excluded from registration.")
            if path == reference:
                image = reference_image.with_data(
                    np.asarray(reference_image.data, dtype=np.float32)
                )
                transform = RegistrationTransform(
                    matrix=np.eye(3).tolist(),
                    inlier_count=len(reference_catalog.stars),
                    residual_rms=0,
                )
                item.matched_stars = len(reference_catalog.stars)
            else:
                image = load_fits(path)
                matching_started = perf_counter()
                matches = match_stars(
                    reference_catalog,
                    dataset.catalogs[str(path)],
                    max_stars=params.max_matching_stars,
                    match_radius=params.match_radius,
                    triangle_tolerance=params.triangle_tolerance,
                    trials=params.ransac_trials,
                    random_seed=params.random_seed,
                    min_scale=params.min_scale,
                    max_scale=params.max_scale,
                )
                item.matched_stars = len(matches.target)
                item.matching_candidates = matches.triangle_candidates
                item.timings_ms["matching"] = (perf_counter() - matching_started) * 1000
                if item.matched_stars < params.min_matches:
                    raise PipelineError(
                        f"Only {item.matched_stars} matching stars were found in {path.name}."
                    )
                fitting_started = perf_counter()
                transform, _ = ransac_transform(
                    matches.target,
                    matches.reference,
                    model=params.model,
                    threshold=params.match_radius,
                    trials=params.ransac_trials,
                    random_seed=params.random_seed,
                )
                item.timings_ms["fitting"] = (perf_counter() - fitting_started) * 1000
            item.transform = transform
            item.inliers = transform.inlier_count
            item.residual_rms = transform.residual_rms
            scales = np.linalg.svd(np.asarray(transform.matrix)[:2, :2], compute_uv=False)
            if (
                item.inliers < params.min_inliers
                or item.inliers / item.matched_stars < params.min_inlier_fraction
                or transform.residual_rms > params.max_residual_rms
                or scales.min() < params.min_scale
                or scales.max() > params.max_scale
            ):
                raise PipelineError(
                    "Registration failed the inlier, residual or scale quality limits."
                )
            if path != reference:
                resampling_started = perf_counter()
                image = resample_image(
                    image,
                    transform,
                    reference_image.data.shape[:2],
                    interpolation=params.interpolation,
                )
                item.timings_ms["resampling"] = (perf_counter() - resampling_started) * 1000
            # All registered pixels share reference astrometry; preserve target acquisition cards.
            copy_reference_wcs(image.header, reference_image.header)
            image.storage_channel_axis = reference_image.storage_channel_axis
            image.header["REGISTER"] = True
            image.header.add_history(
                f"Registered to {reference.name}; "
                f"{params.model}/{params.interpolation}; RMS {transform.residual_rms:.6g}"
            )
            target = output / frame_name(path, index)
            save_fits(image, target, overwrite=overwrite)
            item.success, item.output = True, str(target)
            result.frames.append(target)
            result.qualities.append(metrics[str(path)].model_copy(update={"path": str(target)}))
            logger.info("Registered %d/%d: %s", index, len(dataset.frames), path.name)
        except (AstroError, ValueError, np.linalg.LinAlgError) as exc:
            item.warning = str(exc)
            logger.warning("Frame %s could not be registered and was excluded: %s", path.name, exc)
        item.duration_ms = (perf_counter() - tick) * 1000
        result.registrations.append(item)
        if progress is not None:
            progress()
    report = {
        "reference": str(reference),
        "params": params.model_dump(mode="json"),
        "frames": [r.model_dump(mode="json") for r in result.registrations],
        "qualities": [q.model_dump(mode="json") for q in result.qualities],
        "duration_ms": (perf_counter() - started) * 1000,
    }
    result.reports["registration"] = report
    write_json(output / "registration.json", report, overwrite=overwrite)
    if not result.frames:
        raise PipelineError("No frames could be registered.")
    return result
