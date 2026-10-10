from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator

from astroagent.models.base import SchemaModel
from astroagent.models.layout import CFAMetadata, ImageLayout


class FrameType(StrEnum):
    """Normalized astronomical frame purpose; unknown is never silently guessed."""

    LIGHT = "light"
    DARK = "dark"
    FLAT = "flat"
    BIAS = "bias"
    DARK_FLAT = "dark_flat"
    UNKNOWN = "unknown"


class FrameInfo(SchemaModel):
    """Normalized header-only frame metadata, including sampling compatibility."""

    path: Path
    frame_type: FrameType
    exposure: float | None = Field(default=None, ge=0)
    gain: float | None = None
    offset: float | None = None
    temperature: float | None = None
    filter_name: str | None = None
    cfa: CFAMetadata | None = None
    layout: ImageLayout = ImageLayout.MONO
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    binning: tuple[int, int] = (1, 1)
    bitpix: int
    master: bool = False
    input_hdu: int | None = Field(default=None, ge=0)
    purpose_source: str = "header-or-path"
    capture_time: str | None = None
    camera: str | None = None
    telescope: str | None = None
    pixel_scale_arcsec: tuple[float, float] | None = None
    metadata_sources: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class AstroSession(SchemaModel):
    """Discovered session frames and warnings; no pixel arrays enter planning."""

    frames: list[FrameInfo]
    warnings: list[str] = Field(default_factory=list)

    def of_type(self, kind: FrameType) -> list[FrameInfo]:
        """Return stable frame ordering for the requested purpose, excluding masters."""
        return [f for f in self.frames if f.frame_type == kind and not f.master]

    def counts(self) -> dict[str, int]:
        """Summarize frame purposes for CLI and planners."""
        return {kind.value: len(self.of_type(kind)) for kind in FrameType}


class MasterFrame(SchemaModel):
    """Master provenance and compatibility; dark bias content is always explicit."""

    path: Path
    info: FrameInfo
    contains_bias: bool = False
    input_paths: list[Path] = Field(default_factory=list)
    rejected: dict[str, list[str]] = Field(default_factory=dict)


class CalibrationPlan(SchemaModel):
    """Numerically explicit corrections; dark_contains_bias may override foreign headers."""

    master_bias: Path | None = None
    master_dark: Path | None = None
    master_flat: Path | None = None
    dark_contains_bias: bool | None = None
    dark_scaling: bool = False
    cosmetic_correction: bool = False
    hot_pixel_sigma: float = Field(default=8, gt=0)
    flat_min_fraction: float = Field(default=0.05, gt=0, lt=1)
    max_invalid_flat_fraction: float = Field(default=0.2, ge=0, le=1)
    temperature_threshold: float = Field(default=3, ge=0)
    cfa_pattern: str | None = None

    @field_validator("cfa_pattern")
    @classmethod
    def explicit_cfa(cls, value: str | None) -> str | None:
        """Validate optional pattern overrides before an executor creates artifacts."""
        return CFAMetadata(pattern=value).pattern if value is not None else None
