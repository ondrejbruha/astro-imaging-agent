import numpy as np
from scipy.ndimage import convolve

from astroagent.errors import PipelineError
from astroagent.execution import ExecutionContext, checkpoint, emit_progress, execution_scope
from astroagent.io.metadata import header_metadata
from astroagent.models.image import AstroImage
from astroagent.models.layout import CFAMetadata


@execution_scope
def debayer_image(
    image: AstroImage, cfa: CFAMetadata | None = None, *, context: ExecutionContext | None = None
) -> AstroImage:
    """Bilinearly reconstruct RGB from original CFA samples, respecting phase offsets.

    Interpolation uses finite-sample normalized kernels at edges and holes.
    Missing original samples remain invalid in all channels. No white balance,
    color matrix, denoising, or sharpening is applied. Calibrate before this step.
    """
    cfa = image.cfa if cfa is None else cfa
    if image.channels != 1:
        raise PipelineError("Debayer requires a single-channel raw CFA image.")
    if cfa is None:
        raise PipelineError(
            "CFA pattern is required for debayering; use --pattern or --cfa-pattern."
        )
    if min(image.data.shape) < 2:
        raise PipelineError("Debayer requires at least a 2x2 CFA image.")
    tile = cfa.tile()
    yy, xx = np.indices(image.data.shape)
    colors = tile[yy % 2, xx % 2]
    data = np.asarray(image.data, dtype=np.float32)
    valid = np.isfinite(data)
    output = np.empty((*data.shape, 3), dtype=np.float32)
    emit_progress("debayer-channels", 0, 3, "channel")
    for c, color in enumerate("RGB"):
        checkpoint()
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
        if color == "G":
            kernel = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float32)
        samples = (colors == color) & valid
        numerator = convolve(np.where(samples, data, 0), kernel, mode="constant", cval=0)
        denominator = convolve(samples.astype(np.float32), kernel, mode="constant", cval=0)
        output[..., c] = np.divide(
            numerator, denominator, out=np.full_like(data, np.nan), where=denominator > 0
        )
        output[..., c][~valid] = np.nan
        emit_progress("debayer-channels", c + 1, 3, "channel")
    result = image.with_data(output)
    result.header["DEBAYER"] = True
    result.header["CFASRC"] = cfa.pattern
    for key in ("BAYERPAT", "BAYERPATN", "XBAYROFF", "YBAYROFF"):
        result.header.remove(key, ignore_missing=True, remove_all=True)
        result.metadata.pop(key, None)
    result.header.add_history(
        f"Debayered using {cfa.pattern} / bilinear, offsets {cfa.x_offset},{cfa.y_offset}"
    )
    result.metadata.update(header_metadata(result.header))
    return result
