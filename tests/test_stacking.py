import numpy as np
import pytest
from test_registration import metric

from astroagent.errors import PipelineError
from astroagent.models.image import AstroImage
from astroagent.stacking.normalization import normalization_coefficients, normalization_statistics
from astroagent.stacking.rejection import RejectionParams, reject_frames
from astroagent.stacking.stack import CombineParams, combine_images, combine_pixels
from astroagent.stacking.weighting import QualityWeight, UniformWeight, normalized_weights


@pytest.mark.parametrize(
    "method,expected", [("mean", 4), ("median", 2), ("weighted", 6.7), ("weighted-mean", 6.7)]
)
def test_combination_methods(method, expected):
    data = np.array([1, 2, 9], dtype=np.float32)[:, None, None]
    result = combine_pixels(data, CombineParams(method=method), np.array([0.2, 0.1, 0.7]))
    assert result[0, 0] == pytest.approx(expected)


@pytest.mark.parametrize("method", ["sigma-clipped", "weighted-sigma-clipped"])
def test_cosmic_ray_rejection_in_twenty_star_frames(method):
    yy, xx = np.indices((32, 32))
    star = 10 + 100 * np.exp(-((xx - 16) ** 2 + (yy - 16) ** 2) / 4)
    frames = star + np.random.default_rng(9).normal(0, 0.2, (20, 32, 32))
    frames[0, 8, 9] += 10000
    mean = combine_pixels(frames, CombineParams(method="mean"))
    clipped = combine_pixels(frames, CombineParams(method=method), np.arange(1, 21))
    assert mean[8, 9] - star[8, 9] > 400
    assert abs(clipped[8, 9] - star[8, 9]) < 0.2
    assert abs(clipped[16, 16] - star[16, 16]) < 0.2


def test_nan_weight_renormalization_and_all_invalid():
    frames = np.array([[[2, np.nan]], [[4, np.nan]], [[np.nan, np.nan]]])
    result = combine_pixels(frames, CombineParams(method="weighted"), np.array([1, 3, 9]))
    assert result[0, 0] == pytest.approx(3.5) and np.isnan(result[0, 1])


def test_tiled_stream_matches_in_memory_and_normalizes():
    rng = np.random.default_rng(8)
    data = rng.normal(10, 1, (7, 16, 19, 3)).astype(np.float32)
    data[0, 3, 4] = 1000
    params = CombineParams(method="sigma-clipped", tile_rows=3)
    output, coefficients = combine_images((AstroImage(d) for d in data), len(data), params)
    np.testing.assert_array_equal(output.data, combine_pixels(data, params))
    assert coefficients[0]["scale"] == [1, 1, 1]
    base = data[1]
    normalized, coefficients = combine_images(
        iter([AstroImage(base), AstroImage(2 * base + 7)]),
        2,
        CombineParams(method="mean"),
        normalize=True,
    )
    np.testing.assert_allclose(normalized.data, base, atol=2e-6)
    assert coefficients[1]["scale"] == pytest.approx([0.5] * 3)


def test_frame_rejection_and_weights():
    frames = [metric(str(i), quality_score=i / 9) for i in range(10)]
    rejected = reject_frames(frames, RejectionParams(reject_worst_fraction=0.1))
    assert list(rejected) == ["0"]
    rejected = reject_frames(frames, RejectionParams(min_quality=0.5))
    assert len(rejected) == 5
    assert len(reject_frames(frames, RejectionParams(max_fwhm=2))) == 10
    assert len(reject_frames(frames, RejectionParams(max_registration_residual=1))) == 10
    assert UniformWeight().calculate(frames[0]) == 1
    weights = normalized_weights(frames, QualityWeight())
    assert weights.sum() == pytest.approx(1) and weights[9] > weights[0] > 0
    with pytest.raises(PipelineError, match="weights"):
        combine_pixels(np.ones((2, 1)), CombineParams(), np.array([-1, 2]))


def test_normalization_constant_and_invalid():
    stats = normalization_statistics(np.ones((5, 5)))
    assert normalization_coefficients(stats, ([3], [0])) == ([1], [2])
    with pytest.raises(PipelineError, match="finite"):
        normalization_statistics(np.full((5, 5), np.nan))
