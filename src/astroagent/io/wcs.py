import re

from astropy.io.fits import Header

_WCS = re.compile(
    r"^(WCSAXES|WCSNAME|CTYPE\d|CUNIT\d|CRPIX\d|CRVAL\d|CDELT\d|CROTA\d|"
    r"CD\d_\d|PC\d_\d|PV\d_\d+|PS\d_\d+|LONPOLE|LATPOLE|RADESYS|EQUINOX)[A-Z]?$|"
    r"^(A|B|AP|BP)_(ORDER|\d+_\d+)$",
)


def copy_reference_wcs(target: Header, reference: Header) -> None:
    """Replace coordinate-dependent WCS/SIP cards after pixel registration."""
    for key in list(target):
        if _WCS.match(key):
            target.remove(key, ignore_missing=True, remove_all=True)
    for card in reference.cards:
        if _WCS.match(card.keyword):
            target.append(card)
