"""NumPy rotation helpers using the same conventions as :mod:`focodyn.rotations`.

Quaternions use scalar-first ``(w, x, y, z)`` order and represent local-to-
parent rotations. Rotation matrices map local vectors into the parent frame.
This module is intended for non-autograd tooling; differentiable code should
use the Torch implementation in :mod:`focodyn.rotations`.
"""

from __future__ import annotations

import numpy as np


def normalize_quaternion_wxyz(
    quaternion_wxyz: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Normalize scalar-first quaternions along the last axis.

    Inputs below ``eps`` are divided by ``eps``, matching the behavior of the
    Torch backend. Consequently, a zero quaternion remains zero and converts
    to the identity matrix.
    """
    quaternion = _last_axis(quaternion_wxyz, size=4, name="quaternion_wxyz")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return quaternion / np.maximum(norm, float(eps))


def quaternion_wxyz_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternions to NumPy rotation matrices."""
    quaternion = normalize_quaternion_wxyz(quaternion_wxyz)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            np.stack(
                (
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ),
                axis=-1,
            ),
            np.stack(
                (
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ),
                axis=-1,
            ),
            np.stack(
                (
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ),
                axis=-1,
            ),
        ),
        axis=-2,
    )


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to fixed-axis URDF roll-pitch-yaw angles."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must have shape (..., 3, 3). Got {matrix.shape}.")
    pitch = np.arctan2(
        -matrix[..., 2, 0],
        np.hypot(matrix[..., 0, 0], matrix[..., 1, 0]),
    )
    regular = np.abs(np.cos(pitch)) > 1e-10
    roll = np.where(
        regular,
        np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2]),
        np.arctan2(-matrix[..., 1, 2], matrix[..., 1, 1]),
    )
    yaw = np.where(
        regular,
        np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]),
        0.0,
    )
    return np.stack((roll, pitch, yaw), axis=-1)


def transform_from_position_quaternion_wxyz(
    position: np.ndarray, quaternion_wxyz: np.ndarray
) -> np.ndarray:
    """Build a homogeneous transform from a position and quaternion."""
    translation = _last_axis(position, size=3, name="position")
    rotation = quaternion_wxyz_to_matrix(quaternion_wxyz)
    batch_shape = np.broadcast_shapes(translation.shape[:-1], rotation.shape[:-2])
    transform = np.broadcast_to(
        np.eye(4, dtype=np.float64), batch_shape + (4, 4)
    ).copy()
    transform[..., :3, :3] = np.broadcast_to(rotation, batch_shape + (3, 3))
    transform[..., :3, 3] = np.broadcast_to(translation, batch_shape + (3,))
    return transform


def _last_axis(values: np.ndarray, *, size: int, name: str) -> np.ndarray:
    """Return a float64 array and validate its final coordinate dimension."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != size:
        raise ValueError(f"{name} must have shape (..., {size}). Got {array.shape}.")
    return array
