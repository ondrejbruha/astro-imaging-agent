import json

import numpy as np
import pytest

from astroagent.analysis.background import analyze_background, estimate_surface
from astroagent.analysis.quality import analyze_image
from astroagent.analysis.stars import analyze_stars
from astroagent.analysis.statistics import ImageMetrics, inspect_image
from astroagent.errors import AnalysisError
from astroagent.models.image import AstroImage


def test_metrics_are_correct_and_json_serializable(image):
    result = inspect_image(image)
    assert result.dimensions == (32, 32)
    assert result.number_of_channels == 1
    assert result.datatype == "uint16"
    assert result.min == 0 and result.max == 1023
    assert result.mean == result.median == 511.5
    assert result.standard_deviation == pytest.approx(image.data.std())
    for name, percentile in [
        ("percentile_1", 1),
        ("percentile_5", 5),
        ("percentile_50", 50),
        ("percentile_95", 95),
        ("percentile_99", 99),
    ]:
        assert getattr(result, name) == pytest.approx(np.percentile(image.data, percentile))
    assert ImageMetrics.model_validate_json(result.model_dump_json()) == result
    assert json.loads(result.model_dump_json())["fits_metadata"]["OBJECT"] == "Synthetic nebula"


def test_saturation_levels_and_nonfinite_samples():
    metrics = inspect_image(AstroImage(np.array([[0, 1], [np.nan, np.inf]])))
    assert metrics.finite_samples == 2 and metrics.nonfinite_samples == 2
    assert metrics.fraction_of_saturated_pixels == 0.5
    assert metrics.warnings
    metrics = inspect_image(AstroImage(np.array([[1, 2], [3, 4.0]])))
    assert metrics.fraction_of_saturated_pixels is None
    metrics = inspect_image(AstroImage(np.array([[1, 2], [3, 4.0]]), saturation_level=3))
    assert metrics.fraction_of_saturated_pixels == 0.5
    assert (
        inspect_image(AstroImage(np.array([[0, 255]], dtype=np.uint8))).fraction_of_saturated_pixels
        == 0.5
    )


def test_rgb_channels_and_all_invalid():
    assert inspect_image(AstroImage(np.zeros((5, 6, 3)))).number_of_channels == 3
    with pytest.raises(AnalysisError, match="finite"):
        inspect_image(AstroImage(np.full((3, 4), np.nan)))
    with pytest.raises(AnalysisError, match="finite"):
        analyze_background(AstroImage(np.full((3, 4), np.nan)))


def test_robust_background_outliers_and_gradient():
    rng = np.random.default_rng(123)
    y, x = np.indices((100, 120))
    data = 30 + 0.05 * x + 0.03 * y + rng.normal(0, 0.5, x.shape)
    data[::15, ::15] += 1000
    result = analyze_background(AstroImage(data))
    assert 32 < result.median < 37
    assert result.sigma == pytest.approx(0.5, abs=0.05)
    assert result.gradient_estimate > 5


def test_tiny_background_falls_back():
    metrics = analyze_background(AstroImage(np.ones((1, 1))))
    assert metrics.gradient_estimate == 0
    assert metrics.warnings


@pytest.mark.parametrize(
    "kwargs", [{"grid_size": 1}, {"polynomial_degree": 4}, {"sigma": -1}, {"sigma": float("nan")}]
)
def test_surface_validation(kwargs):
    with pytest.raises(AnalysisError):
        estimate_surface(np.zeros((16, 16)), **kwargs)


def test_surface_rejects_invalid_data_and_rank():
    with pytest.raises(AnalysisError, match="two-dimensional"):
        estimate_surface(np.zeros((8, 8, 3)))
    with pytest.raises(AnalysisError, match="tiles|rank"):
        estimate_surface(np.zeros((1, 8)), polynomial_degree=2)
    with pytest.raises(AnalysisError, match="tiles"):
        estimate_surface(np.full((8, 8), np.nan))


def test_star_detection_count_and_shapes(star_image):
    metrics = analyze_stars(star_image)
    assert metrics.star_count == 4, metrics.warnings
    assert metrics.median_fwhm == pytest.approx(2.35482 * 1.3, abs=0.6)
    assert metrics.median_ellipticity is not None and metrics.median_ellipticity < 0.1
    assert metrics.median_eccentricity is not None and metrics.median_eccentricity < 0.45


def test_elliptical_star_shape():
    y, x = np.indices((64, 64))
    data = 50 + np.random.default_rng(33).normal(0, 0.05, x.shape)
    data += 100 * np.exp(-((x - 32) ** 2 / (2 * 2.0**2) + (y - 32) ** 2 / (2 * 1.2**2)))
    metrics = analyze_stars(AstroImage(data), fwhm=3.5)
    assert metrics.star_count == 1
    assert metrics.median_ellipticity == pytest.approx(0.4, abs=0.1)


def test_no_stars_and_failed_detection_return_warnings(monkeypatch):
    metrics = analyze_stars(AstroImage(np.zeros((32, 32))))
    assert metrics.star_count == 0 and metrics.median_fwhm is None and metrics.warnings
    metrics = analyze_stars(AstroImage(np.full((32, 32), np.nan)))
    assert metrics.star_count is None and metrics.warnings

    def fail(*args, **kwargs):
        raise RuntimeError("detection failure")

    monkeypatch.setattr("astroagent.registration.stars.DAOStarFinder", fail)
    metrics = analyze_stars(AstroImage(np.random.default_rng(4).normal(size=(32, 32))))
    assert metrics.star_count is None and "unavailable" in metrics.warnings[0]


def test_stars_validation_and_rgb(star_image):
    with pytest.raises(AnalysisError):
        analyze_stars(star_image, threshold_sigma=0)
    rgb = star_image.with_data(np.repeat(star_image.data[..., None], 3, axis=-1))
    assert analyze_stars(rgb).star_count == 4


def test_analysis_does_not_mutate_and_recognizes_stretch(star_image):
    original = star_image.data.copy()
    metrics = analyze_image(star_image)
    assert metrics.background is not None and metrics.stars is not None
    assert metrics.appears_linear
    np.testing.assert_array_equal(star_image.data, original)
    star_image.header["ASTRSTR"] = True
    assert not analyze_image(star_image).appears_linear
