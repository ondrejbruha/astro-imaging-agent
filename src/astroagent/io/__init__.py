"""FITS input/output and safe artifact writing."""

from astroagent.io.fits import load_fits, save_fits
from astroagent.io.images import load_image, save_image

__all__ = ["load_fits", "save_fits", "load_image", "save_image"]
