"""Half-flux radii from aperture curves of growth, independent of FWHM proxies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, model_validator
from scipy.spatial import cKDTree

from astroagent.analysis.background import luminance
from astroagent.execution import checkpoint
from astroagent.models.base import SchemaModel
from astroagent.models.image import AstroImage

if TYPE_CHECKING:
    from astroagent.registration.stars import StarCatalog


class HFRParams(SchemaModel):
    """Aperture/annulus radii in pixels and uniform subpixel area sampling."""

    aperture_radius: float = Field(default=8, ge=2, le=64)
    background_inner: float = Field(default=10, ge=2, le=128)
    background_outer: float = Field(default=14, ge=3, le=160)
    subpixels: int = Field(default=5, ge=1, le=9)

    @model_validator(mode="after")
    def ordered_radii(self) -> HFRParams:
        """Require an external sky annulus with nonzero radial width."""
        if not self.aperture_radius < self.background_inner < self.background_outer:
            raise ValueError("Require aperture_radius < background_inner < background_outer.")
        return self


def measure_hfr(image: AstroImage, catalog: StarCatalog, params: HFRParams) -> None:
    """Attach reliable per-star HFR values without changing pixels or quality scoring.

    DAO centroids are zero-based pixel centers. Sky is the annular median;
    negative residual flux is clipped to zero. Each pixel is treated as uniform
    flux split across a regular subpixel grid. Sorted radial cumulative flux is
    linearly interpolated at half the finite aperture flux. Border, masked,
    saturated and catalog-blended apertures are unavailable. This finite-aperture
    measure is biased by noise, undetected neighbors and truncated broad wings.
    """
    data = luminance(image)
    offsets = (np.arange(params.subpixels) + 0.5) / params.subpixels - 0.5
    unavailable = 0
    radius = int(np.ceil(params.background_outer + 1))
    nearest = np.full(len(catalog.stars), np.inf)
    if len(catalog.stars) > 1:
        positions = np.array([[star.x, star.y] for star in catalog.stars])
        nearest = cKDTree(positions).query(positions, k=2)[0][:, 1]
    for index, star in enumerate(catalog.stars):
        checkpoint()
        star.hfr = None
        star.hfr_warning = None
        cx, cy = int(round(star.x)), int(round(star.y))
        if (
            cx - radius < 0
            or cy - radius < 0
            or cx + radius >= data.shape[1]
            or cy + radius >= data.shape[0]
        ):
            star.hfr_warning = "HFR sky annulus intersects the image border."
        elif nearest[index] < 2 * params.aperture_radius:
            star.hfr_warning = "HFR aperture is blended with another catalog source."
        else:
            cut = data[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
            yy, xx = np.indices(cut.shape, dtype=float)
            dx, dy = xx + cx - radius - star.x, yy + cy - radius - star.y
            radial = np.hypot(dx, dy)
            sky_mask = (radial >= params.background_inner) & (radial <= params.background_outer)
            aperture = radial <= params.aperture_radius + 1
            needed = sky_mask | aperture
            if not np.isfinite(cut[needed]).all() or not sky_mask.any():
                star.hfr_warning = "HFR aperture or sky annulus contains masked samples."
            elif (
                image.saturation_level is not None
                and (
                    image.data[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1][
                        aperture
                    ]
                    >= image.saturation_level
                ).any()
            ):
                star.hfr_warning = "HFR aperture contains saturated samples."
            else:
                sky = float(np.median(cut[sky_mask]))
                flux = np.maximum(cut - sky, 0)
                distances = np.hypot(
                    dx[..., None, None] + offsets[None, None, :, None],
                    dy[..., None, None] + offsets[None, None, None, :],
                )
                weights = np.broadcast_to(
                    flux[..., None, None] / params.subpixels**2, distances.shape
                )
                included = distances <= params.aperture_radius
                radial_samples, samples = distances[included], weights[included]
                order = np.argsort(radial_samples, kind="stable")
                cumulative = np.cumsum(samples[order], dtype=float)
                if not cumulative.size or cumulative[-1] <= 0 or not np.isfinite(cumulative[-1]):
                    star.hfr_warning = "HFR aperture has no positive background-subtracted flux."
                else:
                    value = float(np.interp(cumulative[-1] / 2, cumulative, radial_samples[order]))
                    if value > 0:
                        star.hfr = value
                    else:
                        star.hfr_warning = "HFR is unresolved at this sampling."
        if star.hfr is None:
            unavailable += 1
    if unavailable:
        catalog.warnings.append(f"HFR unavailable for {unavailable} unreliable aperture(s).")
    if not any(star.hfr is not None for star in catalog.stars):
        catalog.warnings.append("No reliable HFR measurements are available.")
