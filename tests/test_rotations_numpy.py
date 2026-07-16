from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from focodyn.rotations import quaternion_wxyz_to_matrix as quaternion_to_matrix_torch
from focodyn.rotations_numpy import (
    matrix_to_rpy,
    normalize_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
    transform_from_position_quaternion_wxyz,
)


ATOL = 1e-9


def test_numpy_quaternion_matrix_matches_torch_backend() -> None:
    quaternions = np.stack(
        (
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            np.asarray([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]),
            np.asarray([0.3, -0.4, 0.5, 0.7]),
            np.zeros(4),
        )
    )

    actual = quaternion_wxyz_to_matrix(quaternions)
    expected = (
        quaternion_to_matrix_torch(torch.as_tensor(quaternions, dtype=torch.float64))
        .detach()
        .numpy()
    )

    assert actual.shape == (4, 3, 3)
    assert np.allclose(actual, expected, atol=ATOL)


def test_numpy_rotation_helpers_for_usd_conversion() -> None:
    half_sqrt = math.sqrt(0.5)
    quaternion_wxyz = np.asarray([half_sqrt, 0.0, 0.0, half_sqrt])

    rotation = quaternion_wxyz_to_matrix(quaternion_wxyz)
    rpy = matrix_to_rpy(rotation)
    transform = transform_from_position_quaternion_wxyz(
        np.asarray([1.0, 2.0, 3.0]), quaternion_wxyz
    )

    assert np.allclose(
        rotation,
        np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=ATOL,
    )
    assert np.allclose(rpy, (0.0, 0.0, math.pi / 2.0), atol=ATOL)
    assert np.allclose(transform[:3, :3], rotation, atol=ATOL)
    assert np.array_equal(transform[:3, 3], np.asarray([1.0, 2.0, 3.0]))


def test_numpy_rotation_helpers_validate_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        normalize_quaternion_wxyz(np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        quaternion_wxyz_to_matrix(np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        matrix_to_rpy(np.eye(4))
    with pytest.raises(ValueError, match="shape"):
        transform_from_position_quaternion_wxyz(np.zeros(2), np.zeros(4))
