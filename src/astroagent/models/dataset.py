from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field

from astroagent.models.base import SchemaModel

if TYPE_CHECKING:
    from astroagent.analysis.frame_quality import FrameQualityMetrics
    from astroagent.calibration.models import MasterFrame
    from astroagent.registration.engine import RegistrationResult
    from astroagent.registration.stars import StarCatalog


@dataclass
class AstroDataset:
    """Disk-backed frames and typed analysis, independent of individual image metadata.

    Intermediate pixels live in files rather than a list of full images in RAM.
    Reports carry provenance across dataset operations and a saved replay pipeline.
    """

    frames: list[Path]
    source: Path | None = None
    reference: Path | None = None
    catalogs: dict[str, StarCatalog] = field(default_factory=dict)
    qualities: list[FrameQualityMetrics] = field(default_factory=list)
    registrations: list[RegistrationResult] = field(default_factory=list)
    masters: list[MasterFrame] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    purpose_overrides: dict[str, str] = field(default_factory=dict)


class DatasetMetrics(SchemaModel):
    """Pixel-free session and frame facts exposed to workflow planners."""

    number_of_frames: int
    session_counts: dict[str, int]
    frames: list[dict[str, Any]]
    selected_reference: str | None = None
    registration_statistics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
