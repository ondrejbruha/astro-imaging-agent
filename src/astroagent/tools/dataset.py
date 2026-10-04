from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from astroagent.calibration.masters import build_master, build_masters
from astroagent.calibration.models import CalibrationPlan, FrameType
from astroagent.calibration.session import inspect_frame, inspect_session_frames
from astroagent.calibration.workflow import calibrate_frames, debayer_frames
from astroagent.io.datasets import discover_fits
from astroagent.io.fits import load_fits
from astroagent.models.base import SchemaModel
from astroagent.models.dataset import AstroDataset
from astroagent.models.image import AstroImage
from astroagent.models.layout import CFAMetadata
from astroagent.registration.engine import RegistrationParams, register_frames
from astroagent.stacking.stack import CombineParams
from astroagent.stacking.workflow import StackParams, stack_frames
from astroagent.tools.base import ToolDescription


@dataclass
class DatasetContext:
    """Explicit intermediate-artifact destination supplied by the deterministic executor."""

    output: Path
    overwrite: bool = False


@dataclass
class DatasetToolResult:
    """A dataset or combined image and JSON-safe operation diagnostics."""

    value: AstroDataset | AstroImage
    report: dict[str, Any]


class DatasetTool[Params: SchemaModel](ABC):
    """Typed disk-backed dataset operation, distinct from a single-image numerical tool."""

    name: str
    description: str
    params_model: type[Params]
    output_kind: Literal["dataset", "image"] = "dataset"

    def describe(self) -> ToolDescription:
        """Expose validated parameters with no planner/provider dependency."""
        return ToolDescription(
            name=self.name,
            description=self.description,
            parameters=self.params_model.model_json_schema(),
            input_kind="dataset",
        )

    @abstractmethod
    def execute(
        self, dataset: AstroDataset, params: Params, context: DatasetContext
    ) -> DatasetToolResult:
        """Execute using explicit data and artifact context."""


class RegisterFramesTool(DatasetTool[RegistrationParams]):
    """Register frame datasets to the best or explicitly selected reference."""

    name = "register_frames"
    description = (
        "Detect/match stars, fit a robust transform and register prepared mono/RGB FITS frames."
    )
    params_model = RegistrationParams

    def execute(
        self, dataset: AstroDataset, params: RegistrationParams, context: DatasetContext
    ) -> DatasetToolResult:
        """Delegate all numerical registration to the registration engine."""
        output = register_frames(dataset, context.output, params, overwrite=context.overwrite)
        return DatasetToolResult(output, output.reports["registration"])


class PipelineStackParams(StackParams):
    """Flat aliases preserve the convenient declarative frame-rejection syntax."""

    reject_worst_fraction: float = Field(default=0, ge=0, lt=1)
    min_quality: float | None = Field(default=None, ge=0, le=1)


class StackFramesTool(DatasetTool[PipelineStackParams]):
    """Combine already aligned frames with optional quality and pixel rejection."""

    name = "stack_frames"
    description = (
        "Combine aligned frames with quality weights, optional rejection, "
        "normalization and sigma clipping."
    )
    params_model = PipelineStackParams
    output_kind = "image"

    def execute(
        self, dataset: AstroDataset, params: PipelineStackParams, context: DatasetContext
    ) -> DatasetToolResult:
        """Delegate combination without making the image depend on an agent."""
        resolved = StackParams.model_validate(
            params.model_dump(exclude={"reject_worst_fraction", "min_quality"})
        )
        if params.reject_worst_fraction:
            resolved.rejection.reject_worst_fraction = params.reject_worst_fraction
        if params.min_quality is not None:
            resolved.rejection.min_quality = params.min_quality
        image, report = stack_frames(dataset, resolved)
        return DatasetToolResult(image, report)


class BuildMastersParams(CombineParams):
    """Master-building parameters; session can explicitly override pipeline input paths."""

    method: Literal["mean", "median", "sigma-clipped"] = "sigma-clipped"
    session: str | None = None
    cfa_pattern: str | None = None

    @field_validator("cfa_pattern")
    @classmethod
    def explicit_cfa(cls, value: str | None) -> str | None:
        """Reject unsupported pattern overrides before master construction."""
        return CFAMetadata(pattern=value).pattern if value is not None else None


class BuildMastersTool(DatasetTool[BuildMastersParams]):
    """Build compatible master groups while carrying original lights to the next step."""

    name = "build_masters"
    description = (
        "Group calibration frames by camera/exposure/filter and build bias, dark and flat masters."
    )
    params_model = BuildMastersParams

    def execute(
        self, dataset: AstroDataset, params: BuildMastersParams, context: DatasetContext
    ) -> DatasetToolResult:
        """Build masters and preserve explicit dataset state for calibration."""
        if params.session is not None:
            dataset = AstroDataset(
                discover_fits(params.session, recursive=True), source=Path(params.session)
            )
        session = inspect_session_frames(dataset.frames, cfa_pattern=params.cfa_pattern)
        combine = CombineParams.model_validate(
            params.model_dump(exclude={"session", "cfa_pattern"})
        )
        dataset.masters = build_masters(
            session, context.output, params=combine, overwrite=context.overwrite
        )
        report = {
            "masters": [m.model_dump(mode="json") for m in dataset.masters],
            "session_counts": session.counts(),
            "warnings": session.warnings,
        }
        dataset.reports["masters"] = report
        return DatasetToolResult(dataset, report)


class BuildSingleMasterParams(CombineParams):
    """Independently executable master-building input corrections."""

    method: Literal["mean", "median", "sigma-clipped"] = "sigma-clipped"
    bias: Path | None = None
    dark: Path | None = None


class BuildSingleMasterTool(DatasetTool[BuildSingleMasterParams]):
    """Build one compatible master while preserving the session dataset."""

    params_model = BuildSingleMasterParams

    def __init__(self, kind: FrameType) -> None:
        """Bind a calibration frame purpose without introducing executor branches."""
        self.kind = kind
        self.name = f"build_master_{kind.value}"
        self.description = (
            f"Build a compatible master {kind.value} using deterministic pixel rejection."
        )

    def execute(
        self, dataset: AstroDataset, params: BuildSingleMasterParams, context: DatasetContext
    ) -> DatasetToolResult:
        """Build a purpose-specific master and retain its provenance in dataset state."""
        frames = [inspect_frame(p) for p in dataset.frames]
        frames = [f for f in frames if f.frame_type == self.kind]
        context.output.mkdir(parents=True, exist_ok=True)
        master = build_master(
            frames,
            self.kind,
            context.output / f"master-{self.kind.value}.fit",
            params=CombineParams.model_validate(params.model_dump(exclude={"bias", "dark"})),
            bias=load_fits(params.bias) if params.bias else None,
            dark=load_fits(params.dark) if params.dark else None,
            overwrite=context.overwrite,
        )
        dataset.masters.append(master)
        return DatasetToolResult(dataset, master.model_dump(mode="json"))


class CalibrateFramesParams(CalibrationPlan):
    """Calibration options, with debayering optionally separated into a later tool."""

    debayer: bool = False
    combine: CombineParams = Field(default_factory=lambda: CombineParams(method="sigma-clipped"))


class CalibrateFramesTool(DatasetTool[CalibrateFramesParams]):
    """Calibrate original mono/CFA lights using explicit or dataset master selections."""

    name = "calibrate_frames"
    description = (
        "Calibrate lights before debayering; preserve negative values and mask unsafe flat pixels."
    )
    params_model = CalibrateFramesParams

    def execute(
        self, dataset: AstroDataset, params: CalibrateFramesParams, context: DatasetContext
    ) -> DatasetToolResult:
        """Execute the explicit calibration plan, optionally followed by bilinear debayering."""
        plan = CalibrationPlan.model_validate(params.model_dump(exclude={"debayer", "combine"}))
        output = calibrate_frames(
            dataset,
            context.output,
            plan,
            debayer=params.debayer,
            combine=params.combine,
            overwrite=context.overwrite,
        )
        return DatasetToolResult(output, output.reports["calibration"])


class DebayerFramesParams(SchemaModel):
    """Bilinear Bayer interpolation with an explicit optional pattern override."""

    method: Literal["bilinear"] = "bilinear"
    pattern: str | None = None

    @field_validator("pattern")
    @classmethod
    def explicit_cfa(cls, value: str | None) -> str | None:
        """Require an actual Bayer pattern whenever an override is supplied."""
        return CFAMetadata(pattern=value).pattern if value is not None else None


class DebayerFramesTool(DatasetTool[DebayerFramesParams]):
    """Convert CFA datasets to consistently sampled RGB datasets."""

    name = "debayer_frames"
    description = (
        "Demosaic raw CFA to RGB using explicit FITS patterns and offsets after calibration."
    )
    params_model = DebayerFramesParams

    def execute(
        self, dataset: AstroDataset, params: DebayerFramesParams, context: DatasetContext
    ) -> DatasetToolResult:
        """Apply the same independently usable debayer engine to every frame."""
        output = debayer_frames(
            dataset, context.output, pattern=params.pattern, overwrite=context.overwrite
        )
        return DatasetToolResult(
            output, {"frames": [str(p) for p in output.frames], "method": params.method}
        )
