import math
from typing import Any

from astropy.io.fits import Header


def header_metadata(header: Header) -> dict[str, Any]:
    """Extract JSON-safe FITS cards; the complete Header remains on AstroImage."""
    result: dict[str, Any] = {}
    for card in header.cards:
        if not card.keyword or card.keyword in {"HISTORY", "COMMENT"}:
            continue
        value = card.value
        if isinstance(value, float) and not math.isfinite(value):
            value = str(value)
        elif not isinstance(value, str | int | float | bool | type(None)):
            value = str(value)
        result[card.keyword] = value
    return result
