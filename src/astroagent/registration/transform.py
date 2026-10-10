from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator

from astroagent.errors import PipelineError
from astroagent.execution import checkpoint
from astroagent.models.base import SchemaModel

TransformModel = Literal["similarity", "affine"]


class RegistrationTransform(SchemaModel):
    """Forward target-to-reference homogeneous matrix in zero-based (x, y) pixels."""

    matrix: list[list[float]]
    inlier_count: int = Field(ge=0)
    residual_rms: float = Field(ge=0)

    @field_validator("matrix")
    @classmethod
    def stable_matrix(cls, value: list[list[float]]) -> list[list[float]]:
        """Reject singular, projective, nonfinite, and ill-conditioned transforms."""
        matrix = np.asarray(value, dtype=float)
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[2], [0, 0, 1])
            or np.linalg.cond(matrix[:2, :2]) > 100
            or np.linalg.det(matrix[:2, :2]) <= 1e-8
        ):
            raise ValueError("Registration transform is numerically unstable or reflected.")
        return value


def apply_transform(
    points: NDArray[np.float64], matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply a forward affine matrix to Nx2 x/y coordinates."""
    return np.asarray(points @ matrix[:2, :2].T + matrix[:2, 2], dtype=np.float64)


def fit_transform(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    model: TransformModel = "similarity",
) -> NDArray[np.float64]:
    """Fit uniform scale/rotation/translation or a full six-parameter affine model."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise PipelineError("Matching coordinates must have identical Nx2 shapes.")
    if len(source) < 3 or np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        raise PipelineError("At least three non-collinear matching stars are required.")
    result = np.eye(3)
    if model == "affine":
        design = np.column_stack([source, np.ones(len(source))])
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        result[:2] = coefficients.T
    else:
        x, y = source.T
        design = np.zeros((2 * len(source), 4))
        design[0::2] = np.column_stack([x, -y, np.ones(len(source)), np.zeros(len(source))])
        design[1::2] = np.column_stack([y, x, np.zeros(len(source)), np.ones(len(source))])
        a, b, tx, ty = np.linalg.lstsq(design, target.ravel(), rcond=None)[0]
        result[:2] = [[a, -b, tx], [b, a, ty]]
    RegistrationTransform(matrix=result.tolist(), inlier_count=len(source), residual_rms=0)
    return result


def ransac_transform(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    model: TransformModel = "similarity",
    threshold: float = 2,
    trials: int = 500,
    random_seed: int = 0,
) -> tuple[RegistrationTransform, NDArray[np.bool_]]:
    """Reject false correspondences using reproducible three-star sampling and refitting."""
    if threshold <= 0 or trials < 1 or len(source) < 3:
        raise PipelineError("RANSAC requires three matches and positive thresholds/trials.")
    rng = np.random.default_rng(random_seed)
    best = np.zeros(len(source), dtype=bool)
    best_error = np.inf
    for _ in range(trials):
        checkpoint()
        indices = rng.choice(len(source), 3, replace=False)
        try:
            matrix = fit_transform(source[indices], target[indices], model)
        except (PipelineError, ValueError, np.linalg.LinAlgError):
            continue
        residual = np.linalg.norm(apply_transform(source, matrix) - target, axis=1)
        keep = residual <= threshold
        error = float(np.mean(residual[keep] ** 2)) if keep.any() else np.inf
        if keep.sum() > best.sum() or (keep.sum() == best.sum() and error < best_error):
            best, best_error = keep, error
            if best.all():
                break
    if best.sum() < 3:
        raise PipelineError("RANSAC found fewer than three consistent matching stars.")
    for _ in range(10):
        checkpoint()
        matrix = fit_transform(source[best], target[best], model)
        residual = np.linalg.norm(apply_transform(source, matrix) - target, axis=1)
        keep = residual <= threshold
        if np.array_equal(keep, best) or keep.sum() < 3:
            break
        best = keep
    matrix = fit_transform(source[best], target[best], model)
    residual = np.linalg.norm(apply_transform(source[best], matrix) - target[best], axis=1)
    return RegistrationTransform(
        matrix=matrix.tolist(),
        inlier_count=int(best.sum()),
        residual_rms=float(np.sqrt(np.mean(residual**2))),
    ), best
