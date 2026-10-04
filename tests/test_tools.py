import numpy as np
import pytest
from pydantic import ValidationError

from astroagent.errors import PipelineError
from astroagent.models.image import AstroImage
from astroagent.tools.background import BackgroundExtractParams, BackgroundExtractTool
from astroagent.tools.denoise import DenoiseParams, DenoiseTool
from astroagent.tools.normalization import NormalizeParams, NormalizeTool
from astroagent.tools.registry import ToolRegistry, default_registry
from astroagent.tools.stretch import StretchParams, StretchTool


def test_normalize_and_rgb_shared_scale(image):
    result = NormalizeTool().execute(image, NormalizeParams())
    assert result.image.data.dtype == np.float64
    assert result.image.data.min() == 0 and result.image.data.max() == 1
    assert result.metrics_before is not None and result.metrics_after is not None
    rgb = AstroImage(np.array([[[0, 50, 100], [100, 150, 200]]]))
    output = NormalizeTool().execute(rgb, NormalizeParams(lower=0.1, upper=0.9)).image
    np.testing.assert_allclose(output.data, 0.1 + rgb.data / 200 * 0.8)


@pytest.mark.parametrize(
    "tool,params",
    [
        (NormalizeTool(), NormalizeParams()),
        (StretchTool(), StretchParams()),
        (DenoiseTool(), DenoiseParams()),
        (BackgroundExtractTool(), BackgroundExtractParams(polynomial_degree=1)),
    ],
)
def test_tools_are_deterministic_nonmutating_and_preserve_header(image, tool, params):
    original = image.data.copy()
    first, second = tool.execute(image, params), tool.execute(image, params)
    np.testing.assert_array_equal(first.image.data, second.image.data)
    np.testing.assert_array_equal(image.data, original)
    assert not np.shares_memory(image.data, first.image.data)
    assert first.image.header["OBJECT"] == image.header["OBJECT"]
    assert "Original processing history" in first.image.header["HISTORY"]
    first.image.metadata["OBJECT"] = "Changed"
    assert image.metadata["OBJECT"] == "Synthetic nebula"
    assert len(first.image.header["HISTORY"]) > len(image.header["HISTORY"])


def test_constant_normalize_and_stretch():
    image = AstroImage(np.full((8, 8), 42.0))
    for tool, params in [(NormalizeTool(), NormalizeParams()), (StretchTool(), StretchParams())]:
        result = tool.execute(image, params)
        assert np.all(result.image.data == 0)
        assert result.warnings


def test_asinh_is_monotonic_bounded_and_preserves_endpoint():
    image = AstroImage(np.linspace(10, 100, 100).reshape(10, 10))
    tool = StretchTool()
    linear = tool.execute(image, StretchParams(method="linear")).image.data
    stretched = tool.execute(image, StretchParams(strength=0.6)).image.data
    assert stretched.min() == 0 and stretched.max() == 1
    assert np.all(np.diff(stretched.ravel()) >= 0)
    assert np.all(stretched >= linear - 1e-15)
    np.testing.assert_array_equal(tool.execute(image, StretchParams(strength=0)).image.data, linear)
    expected_gain = 10**1.8 - 1
    np.testing.assert_allclose(
        stretched, np.arcsinh(expected_gain * linear) / np.arcsinh(expected_gain)
    )


def test_black_point_clipping_and_invalid_point(image):
    result = StretchTool().execute(image, StretchParams(black_point=500))
    assert np.all(result.image.data[image.data < 500] == 0)
    assert result.warnings
    with pytest.raises(ValueError, match="black_point"):
        StretchTool().execute(image, StretchParams(black_point=2000))


def test_denoise_reduces_noise_without_mixing_rgb():
    noisy = np.random.default_rng(2).normal(size=(32, 32))
    result = DenoiseTool().execute(AstroImage(noisy), DenoiseParams()).image.data
    assert result.std() < noisy.std() * 0.5
    rgb = AstroImage(np.stack([noisy, np.zeros_like(noisy), np.ones_like(noisy)], axis=-1))
    output = DenoiseTool().execute(rgb, DenoiseParams()).image.data
    assert np.all(output[..., 1] == 0)
    np.testing.assert_allclose(output[..., 2], 1)


@pytest.mark.parametrize("degree", [1, 2])
def test_background_removes_gradient_with_sources(degree):
    y, x = np.indices((96, 96))
    sky = 20 + 0.06 * x + 0.1 * y
    if degree == 2:
        sky += 0.0003 * x * y + 0.0005 * x**2
    source = 30 * np.exp(-((x - 42) ** 2 + (y - 51) ** 2) / (2 * 1.2**2))
    result = (
        BackgroundExtractTool()
        .execute(
            AstroImage(sky + source),
            BackgroundExtractParams(grid_size=12, polynomial_degree=degree),
        )
        .image.data
    )
    mask = source < 0.01
    assert result[mask].std() < 0.025
    assert abs(np.median(result[mask])) < 0.025
    assert result.max() > 29


def test_background_rgb_independent_planes():
    y, x = np.indices((32, 32))
    planes = np.stack([10 + x, 20 + y, 30 + x + y], axis=-1).astype(float)
    result = (
        BackgroundExtractTool()
        .execute(AstroImage(planes), BackgroundExtractParams(polynomial_degree=1))
        .image.data
    )
    np.testing.assert_allclose(result, 0, atol=1e-12)


@pytest.mark.parametrize(
    "model,params",
    [
        (StretchParams, {"strength": -0.1}),
        (StretchParams, {"strength": 1.1}),
        (StretchParams, {"strength": float("nan")}),
        (StretchParams, {"black_point": float("inf")}),
        (StretchParams, {"method": "unknown"}),
        (NormalizeParams, {"lower": 1, "upper": 0}),
        (NormalizeParams, {"lower": -1}),
        (DenoiseParams, {"sigma": 0}),
        (DenoiseParams, {"method": "wavelet"}),
        (BackgroundExtractParams, {"grid_size": 1}),
        (BackgroundExtractParams, {"polynomial_degree": 4}),
        (BackgroundExtractParams, {"grid_size": 3.5}),
        (StretchParams, {"strenght": 0.5}),
    ],
)
def test_invalid_parameters(model, params):
    with pytest.raises(ValidationError):
        model.model_validate(params)


def test_nonfinite_processing_rejected():
    with pytest.raises(PipelineError, match="finite"):
        NormalizeTool().execute(AstroImage(np.array([[np.inf, 1.0]])), NormalizeParams())


def test_registry_descriptions_and_custom_registry():
    first, second = default_registry(), default_registry()
    assert first.get("stretch") is not second.get("stretch")
    descriptions = first.describe()
    assert {item.name for item in descriptions} >= {
        "normalize",
        "stretch",
        "denoise",
        "background_extract",
    }
    assert all(item.description and item.parameters["type"] == "object" for item in descriptions)
    assert descriptions[1].parameters["properties"]["strength"]["maximum"] == 1
    with pytest.raises(PipelineError, match="Unknown tool"):
        first.get("missing")
    with pytest.raises(ValueError, match="Duplicate"):
        ToolRegistry([StretchTool(), StretchTool()])
