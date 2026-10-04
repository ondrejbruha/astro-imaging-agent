from pathlib import Path

from pydantic import JsonValue

from astroagent.agent.models import PlanResult
from astroagent.calibration.models import FrameType
from astroagent.calibration.session import discover_session, validate_session
from astroagent.errors import PipelineError
from astroagent.models.dataset import AstroDataset, DatasetMetrics
from astroagent.pipeline.models import PipelineDefinition, PipelineStep
from astroagent.registration.engine import analyze_frames
from astroagent.registration.reference import select_reference
from astroagent.stacking.workflow import restore_registration


def inspect_dataset(source: Path) -> DatasetMetrics:
    """Expose session metadata and prepared-frame quality to a planner without image arrays."""
    session = discover_session(source)
    lights = session.of_type(FrameType.LIGHT)
    has_calibration = any(
        session.of_type(k) for k in (FrameType.BIAS, FrameType.DARK, FrameType.FLAT)
    )
    warnings = validate_session(session) if lights else list(session.warnings)
    frames = [f.model_dump(mode="json") for f in session.frames]
    reference = None
    registration_statistics = {}
    if not has_calibration and not any(f.cfa for f in session.frames):
        dataset = AstroDataset([f.path for f in (lights or session.frames)], source=source)
        restore_registration(dataset)
        if not dataset.qualities:
            analyze_frames(dataset)
        frames = [q.model_dump(mode="json") for q in dataset.qualities]
        reference = select_reference(dataset.qualities).path
        if dataset.registrations:
            registration_statistics = {
                "frames": [r.model_dump(mode="json") for r in dataset.registrations],
                "failed_frames": sum(not r.success for r in dataset.registrations),
            }
    return DatasetMetrics(
        number_of_frames=len(lights) if lights else len(session.frames),
        session_counts=session.counts(),
        frames=frames,
        selected_reference=reference,
        registration_statistics=registration_statistics,
        warnings=warnings,
    )


class DatasetRulePlanner:
    """Choose a transparent session workflow from metadata; numerical work stays in tools."""

    def create_plan(self, request: str, metrics: DatasetMetrics) -> PlanResult:
        """Prefer compatible preprocessing, best-reference registration and weighted clipping."""
        if not request.strip():
            raise PipelineError("Agent request must not be empty.")
        steps = []
        reasons = []
        has_calibration = any(
            metrics.session_counts.get(k, 0) for k in ("bias", "dark", "flat", "dark_flat")
        )
        cfa = any(f.get("cfa") for f in metrics.frames)
        if has_calibration:
            steps += [PipelineStep(tool="build_masters"), PipelineStep(tool="calibrate_frames")]
            reasons.append(
                "Build compatible bias/dark/filter-flat groups; calibrate original "
                "sensor samples before debayering."
            )
        if cfa:
            steps.append(PipelineStep(tool="debayer_frames"))
            reasons.append(
                "Bayer metadata is explicit; bilinear reconstruction produces RGB for "
                "star detection."
            )
        steps.append(PipelineStep(tool="register_frames", params={"reference": "auto"}))
        reasons.append(
            f"Select a sharp, round, low-noise reference from "
            f"{metrics.number_of_frames} frames and robustly register; exclude failed fits."
        )
        # Conservative quality cut uses PSF widths only when available before execution.
        widths = sorted(
            float(f["median_fwhm"]) for f in metrics.frames if f.get("median_fwhm") is not None
        )
        rejection: dict[str, JsonValue] = {}
        if widths:
            median = widths[len(widths) // 2]
            bad = sum(v > 1.5 * median for v in widths)
            if bad:
                rejection = {"max_fwhm": 1.5 * median}
                reasons.append(
                    f"Reject {bad} frames with FWHM more than 1.5 times the dataset median."
                )
        steps.append(
            PipelineStep(
                tool="stack_frames",
                params={"method": "weighted-sigma-clipped", "rejection": rejection},
            )
        )
        reasons.append(
            "Normalize intensities, weight by dataset quality and clip pixel outliers "
            "before averaging."
        )
        return PlanResult(pipeline=PipelineDefinition(steps=steps), reasoning=reasons)
