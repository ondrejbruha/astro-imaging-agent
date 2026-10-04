import json

import numpy as np
import pytest
from pydantic import ValidationError
from synthetic import star_image, star_positions

from astroagent.analysis.frame_quality import FrameQualityMetrics, measure_frame, score_frames
from astroagent.errors import PipelineError
from astroagent.io.fits import load_fits, save_fits
from astroagent.models.dataset import AstroDataset
from astroagent.registration.engine import RegistrationParams, register_frames
from astroagent.registration.matching import match_stars
from astroagent.registration.reference import select_reference
from astroagent.registration.resample import resample_image
from astroagent.registration.stars import DetectedStar, StarCatalog, detect_stars
from astroagent.registration.transform import (
    RegistrationTransform,
    apply_transform,
    ransac_transform,
)


def similarity(angle=0, scale=1, shift=(0, 0)):
    a = np.deg2rad(angle)
    matrix = np.eye(3)
    matrix[:2, :2] = scale * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    center = np.array([64, 56])
    matrix[:2, 2] = center - matrix[:2, :2] @ center + shift
    return matrix


@pytest.mark.parametrize(
    "angle,scale,shift",
    [(0, 1, (6.2, -4.5)), (8, 1, (2, 1)), (0, 1.06, (-1, 2)), (-5, 0.95, (2, -1))],
)
def test_known_image_transform(angle, scale, shift):
    truth = similarity(angle, scale, shift)
    original = star_positions()
    reference = detect_stars(star_image())
    target = detect_stars(star_image(apply_transform(original, truth), seed=11))
    assert len(reference.stars) >= 20
    matches = match_stars(reference, target)
    transform, inliers = ransac_transform(matches.target, matches.reference)
    error = np.linalg.norm(
        apply_transform(apply_transform(original, truth), np.array(transform.matrix)) - original,
        axis=1,
    )
    assert error.max() < 0.12
    assert transform.inlier_count >= 20 and inliers.all()
    assert transform.residual_rms < 0.1


def test_catalog_matching_missing_stars_and_false_detections():
    points = star_positions()
    truth = similarity(12, 1.03, (20, -15))
    rng = np.random.default_rng(42)
    target_points = np.concatenate(
        [apply_transform(points[4:], truth), rng.uniform(0, 110, (15, 2))]
    )

    def catalog(p):
        return StarCatalog(
            stars=[DetectedStar(x=x, y=y, flux=100 + i) for i, (x, y) in enumerate(p)],
            image_width=160,
            image_height=160,
        )

    matches = match_stars(catalog(points), catalog(target_points), trials=1000)
    transform, _ = ransac_transform(matches.target, matches.reference)
    np.testing.assert_allclose(transform.matrix, np.linalg.inv(truth), atol=1e-8)
    assert transform.inlier_count >= len(points) - 4


@pytest.mark.parametrize("model", ["similarity", "affine"])
def test_ransac_outliers_and_reproducibility(model):
    rng = np.random.default_rng(5)
    source = rng.uniform(0, 100, (70, 2))
    truth = similarity(7, 1.04, (3, 4))
    if model == "affine":
        truth[0, 1] += 0.02
    target = apply_transform(source, truth)
    target[:20] = rng.uniform(0, 100, (20, 2))
    transform, mask = ransac_transform(source, target, model=model, threshold=0.3, random_seed=10)
    assert mask.sum() == 50
    assert not mask[:20].any()
    np.testing.assert_allclose(transform.matrix, truth, atol=1e-9)
    assert (
        transform == ransac_transform(source, target, model=model, threshold=0.3, random_seed=10)[0]
    )


@pytest.mark.parametrize("interpolation", ["nearest", "bilinear", "bicubic"])
def test_rgb_resampling_and_nan_support(interpolation):
    image = star_image(rgb=True)
    image.data[50, 50] = np.nan
    transform = RegistrationTransform(
        matrix=similarity(3, 1, (2, 1)).tolist(), inlier_count=20, residual_rms=0
    )
    result = resample_image(image, transform, image.data.shape[:2], interpolation=interpolation)
    assert result.data.shape == image.data.shape and result.data.dtype == np.float32
    assert np.isnan(result.data).any() and np.isfinite(result.data).mean() > 0.85
    finite = np.isfinite(result.data[..., 0])
    np.testing.assert_allclose(
        result.data[..., 1][finite], 0.7 * result.data[..., 0][finite], rtol=2e-5, atol=1e-4
    )


def metric(path="a", **values):
    return FrameQualityMetrics(
        path=path,
        star_count=30,
        median_fwhm=3,
        median_eccentricity=0.3,
        background_median=100,
        background_sigma=1,
        saturation_fraction=0,
        **values,
    )


def test_quality_score_and_reference_selection():
    base = metric()
    poor = base.model_copy(
        update={
            "path": "b",
            "median_fwhm": 5,
            "median_eccentricity": 0.8,
            "star_count": 10,
            "background_sigma": 3,
            "saturation_fraction": 0.1,
        }
    )
    scored = score_frames([poor, base])
    assert scored[1].quality_score == pytest.approx(1)
    assert scored[0].quality_score == pytest.approx(0)
    assert select_reference(scored).path == "a"
    tied = score_frames([base, base.model_copy(update={"path": "b"})])
    assert tied[0].quality_score == pytest.approx(0.5)
    assert select_reference(tied).path == "a"
    assert score_frames([base])[0].quality_score == pytest.approx(0.5)


def test_measurement_psf_noise_and_empty_reference():
    metrics, stars = measure_frame(star_image())
    assert metrics.star_count == len(stars.stars)
    assert 2.7 < metrics.median_fwhm < 3.5
    assert metrics.background_sigma == pytest.approx(1, rel=0.2)
    assert metrics.snr_estimate > 100
    with pytest.raises(PipelineError, match="reference"):
        select_reference([metrics.model_copy(update={"star_count": 2})])
    with pytest.raises(ValidationError, match="unstable"):
        RegistrationTransform(
            matrix=[[0, 0, 0], [0, 0, 0], [0, 0, 1]], inlier_count=0, residual_rms=0
        )


def test_registration_failure_report_and_metadata(tmp_path):
    paths = []
    for i in range(2):
        image = star_image(star_positions() + [2 * i, i], seed=i)
        image.header["CRPIX1"] = 20 + i
        image.header["CTYPE1"] = "RA---TAN"
        paths.append(save_fits(image, tmp_path / f"light_{i}.fit"))
    blank = star_image()
    blank.data[:] = 10
    paths.append(save_fits(blank, tmp_path / "bad.fit"))
    dataset = register_frames(
        AstroDataset(paths), tmp_path / "registered", RegistrationParams(reference=paths[0].name)
    )
    assert len(dataset.frames) == 2
    assert dataset.registrations[-1].success is False
    assert dataset.registrations[1].residual_rms < 0.1
    registered = load_fits(dataset.frames[1])
    assert registered.header["OBJECT"] == "Synthetic stars"
    assert registered.header["CRPIX1"] == 20
    assert registered.header["REGISTER"]
    report = json.loads((tmp_path / "registered" / "registration.json").read_text())
    assert len(report["frames"]) == 3 and report["params"]["random_seed"] == 0
