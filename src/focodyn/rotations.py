"""Rotation and angular-velocity helpers using the repository's conventions.

Unless a function says otherwise, tensors may have arbitrary leading batch
dimensions and the rotation/vector coordinate dimension is stored in the last
axis. Trajectory utilities default to the batch-first sequence layout
``(..., frames, channels)`` by treating the second-to-last axis as time; pass
``time_dim=0`` for legacy frame-first tensors such as ``(frames, batch, 4)``.
Quaternions use scalar-first ``(w, x, y, z)`` order and represent body-to-world
rotations. Rotation matrices have shape ``(..., 3, 3)`` and map local/body
vectors into the parent/world frame.
"""

from __future__ import annotations

import math

import torch


def normalize_quaternion_wxyz(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize scalar-first quaternions along the last axis.

    Args:
        q: Quaternion tensor with shape ``(..., 4)`` in ``(w, x, y, z)``
            order. Leading dimensions are treated as batch dimensions.
        eps: Minimum norm used to avoid division by zero. Inputs with norm below
            ``eps`` are divided by ``eps`` instead of their actual norm.

    Returns:
        Tensor with shape ``(..., 4)`` and the same dtype/device as ``q``.
        Nonzero inputs are unit length up to numerical precision.
    """
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=eps)


def quaternion_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply scalar-first quaternions.

    The product follows the Hamilton convention. If ``left`` and ``right`` both
    represent body-to-world rotations, the returned quaternion represents their
    composition ``left * right`` in ``(w, x, y, z)`` order.

    Args:
        left: Left quaternion tensor with shape ``(..., 4)``. Leading
            dimensions must be broadcast-compatible with ``right``.
        right: Right quaternion tensor with shape ``(..., 4)``.

    Returns:
        Quaternion product with broadcasted shape ``(..., 4)``.
    """
    lw, lx, ly, lz = torch.unbind(left, dim=-1)
    rw, rx, ry, rz = torch.unbind(right, dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_conjugate_wxyz(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Return the scalar-first quaternion conjugate.

    Args:
        quaternion_wxyz: Quaternion tensor with shape ``(..., 4)`` in
            ``(w, x, y, z)`` order.

    Returns:
        Tensor with shape ``(..., 4)`` equal to ``(w, -x, -y, -z)``.
    """
    return torch.cat((quaternion_wxyz[..., :1], -quaternion_wxyz[..., 1:]), dim=-1)


def quaternion_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first quaternions to rotation matrices.

    The input is normalized internally. The quaternion is interpreted as a
    body-to-world rotation, so the returned matrix maps body/local vectors into
    world/parent coordinates.

    Args:
        q: Quaternion tensor with shape ``(..., 4)`` in ``(w, x, y, z)`` order.
            Leading dimensions are arbitrary batch dimensions.

    Returns:
        Rotation matrix tensor with shape ``(..., 3, 3)``.
    """
    q = normalize_quaternion_wxyz(q)
    w, x, y, z = torch.unbind(q, dim=-1)

    two = torch.as_tensor(2.0, dtype=q.dtype, device=q.device)
    one = torch.as_tensor(1.0, dtype=q.dtype, device=q.device)

    r00 = one - two * (y * y + z * z)
    r01 = two * (x * y - z * w)
    r02 = two * (x * z + y * w)
    r10 = two * (x * y + z * w)
    r11 = one - two * (x * x + z * z)
    r12 = two * (y * z - x * w)
    r20 = two * (x * z - y * w)
    r21 = two * (y * z + x * w)
    r22 = one - two * (x * x + y * y)

    return torch.stack(
        (
            torch.stack((r00, r01, r02), dim=-1),
            torch.stack((r10, r11, r12), dim=-1),
            torch.stack((r20, r21, r22), dim=-1),
        ),
        dim=-2,
    )


def matrix_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to scalar-first quaternions.

    The returned quaternion uses ``(w, x, y, z)`` order and represents the same
    body-to-world rotation as ``matrix``. The implementation uses a
    candidate-based branch selection with a masked square root so gradients stay
    finite at identity and 180-degree rotations.

    Args:
        matrix: Rotation matrix tensor with shape ``(..., 3, 3)``. The final two
            axes are interpreted as the matrix axes; leading dimensions are
            arbitrary batch dimensions.

    Returns:
        Unit quaternion tensor with shape ``(..., 4)`` in ``(w, x, y, z)``
        order.
    """
    batch_shape = matrix.shape[:-2]
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    quaternion_abs = _sqrt_positive_part(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )

    quaternion_candidates = torch.stack(
        (
            torch.stack((quaternion_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, quaternion_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, quaternion_abs[..., 2] ** 2, m21 + m12), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, quaternion_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    quaternion_candidates = quaternion_candidates / (2.0 * torch.clamp(quaternion_abs[..., None], min=0.1))
    selection = quaternion_abs.argmax(dim=-1)
    gather_index = selection.reshape(*batch_shape, 1, 1).expand(*batch_shape, 1, 4)
    quaternion = torch.gather(quaternion_candidates, dim=-2, index=gather_index).squeeze(-2)
    return normalize_quaternion_wxyz(quaternion)


def rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """Convert fixed-axis URDF roll-pitch-yaw angles to rotation matrices.

    The convention matches URDF fixed-axis RPY: values are ordered as
    ``(roll, pitch, yaw)`` and the resulting matrix is
    ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``. The matrix maps local vectors into the
    parent frame.

    Args:
        rpy: Angle tensor with shape ``(..., 3)`` in radians. Leading dimensions
            are arbitrary batch dimensions.

    Returns:
        Rotation matrix tensor with shape ``(..., 3, 3)``.
    """
    roll, pitch, yaw = torch.unbind(rpy, dim=-1)
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    return torch.stack(
        (
            torch.stack((r00, r01, r02), dim=-1),
            torch.stack((r10, r11, r12), dim=-1),
            torch.stack((r20, r21, r22), dim=-1),
        ),
        dim=-2,
    )


def quaternion_derivative_from_world_angular_velocity(
    quaternion_wxyz: torch.Tensor, angular_velocity_world: torch.Tensor
) -> torch.Tensor:
    """Compute quaternion derivatives from world-frame angular velocity.

    The quaternion maps body-frame vectors to the world frame. Angular velocity
    is expressed in world coordinates. With this convention,
    ``q_dot = 0.5 * [0, omega_W] * q``.

    Args:
        quaternion_wxyz: Quaternion tensor with shape ``(..., 4)`` in
            ``(w, x, y, z)`` order. Leading dimensions must be
            broadcast-compatible with ``angular_velocity_world``.
        angular_velocity_world: World-frame angular velocity tensor with shape
            ``(..., 3)``.

    Returns:
        Quaternion derivative tensor with broadcasted shape ``(..., 4)``.
    """
    q = normalize_quaternion_wxyz(quaternion_wxyz)
    w, x, y, z = torch.unbind(q, dim=-1)
    wx, wy, wz = torch.unbind(angular_velocity_world, dim=-1)

    return 0.5 * torch.stack(
        (
            -wx * x - wy * y - wz * z,
            wx * w + wy * z - wz * y,
            -wx * z + wy * w + wz * x,
            wx * y - wy * x + wz * w,
        ),
        dim=-1,
    )


def world_angular_velocity_from_quaternion_derivative(
    quaternion_wxyz: torch.Tensor,
    quaternion_derivative_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Convert scalar-first quaternion derivatives to world angular velocity.

    This is the inverse of
    :func:`quaternion_derivative_from_world_angular_velocity` for unit
    quaternions under the convention ``q_dot = 0.5 * [0, omega_W] * q``.

    Args:
        quaternion_wxyz: Quaternion tensor with shape ``(..., 4)`` in
            ``(w, x, y, z)`` order. Inputs are normalized internally.
        quaternion_derivative_wxyz: Quaternion derivative tensor with shape
            ``(..., 4)`` and broadcast-compatible leading dimensions.

    Returns:
        World-frame angular velocity tensor with shape ``(..., 3)``.
    """
    q = normalize_quaternion_wxyz(quaternion_wxyz)
    omega_quaternion = 2.0 * quaternion_multiply_wxyz(
        quaternion_derivative_wxyz,
        quaternion_conjugate_wxyz(q),
    )
    return omega_quaternion[..., 1:]


def quaternion_second_derivative_from_world_angular_acceleration(
    quaternion_wxyz: torch.Tensor,
    quaternion_derivative_wxyz: torch.Tensor,
    angular_velocity_world: torch.Tensor,
    angular_acceleration_world: torch.Tensor,
) -> torch.Tensor:
    """Compute ``q_ddot`` from world-frame angular velocity and acceleration.

    The convention is the time derivative of
    ``q_dot = 0.5 * [0, omega_W] * q``, where the quaternion maps body vectors
    to world coordinates and ``omega_W`` / ``alpha_W`` are expressed in the
    world frame.

    Args:
        quaternion_wxyz: Quaternion tensor with shape ``(..., 4)`` in
            ``(w, x, y, z)`` order.
        quaternion_derivative_wxyz: Quaternion derivative tensor with shape
            ``(..., 4)``.
        angular_velocity_world: World-frame angular velocity tensor with shape
            ``(..., 3)``.
        angular_acceleration_world: World-frame angular acceleration tensor
            with shape ``(..., 3)``.

    Returns:
        Quaternion second derivative tensor with broadcasted shape ``(..., 4)``.
    """
    zeros = torch.zeros_like(angular_velocity_world[..., :1])
    omega_quaternion = torch.cat((zeros, angular_velocity_world), dim=-1)
    alpha_quaternion = torch.cat((zeros, angular_acceleration_world), dim=-1)
    return 0.5 * (
        quaternion_multiply_wxyz(alpha_quaternion, normalize_quaternion_wxyz(quaternion_wxyz))
        + quaternion_multiply_wxyz(omega_quaternion, quaternion_derivative_wxyz)
    )


def continuous_quaternions_wxyz(quaternions: torch.Tensor, *, time_dim: int = -2) -> torch.Tensor:
    """Flip quaternion signs to keep a trajectory on one quaternion hemisphere.

    Quaternion signs are ambiguous: ``q`` and ``-q`` represent the same
    rotation. This function chooses signs so adjacent samples along the time
    axis have nonnegative dot products. The default time convention is
    batch-first, ``(..., frames, 4)``, so the second-to-last axis is treated as
    time. Unbatched ``(frames, 4)`` inputs also use this default. Pass
    ``time_dim=0`` for frame-first tensors such as ``(frames, batch, 4)``.

    Args:
        quaternions: Quaternion trajectory with shape ``(..., frames, 4)`` by
            default. The last axis must have length ``4`` and stores
            scalar-first ``(w, x, y, z)`` quaternions.
        time_dim: Axis containing trajectory frames. Defaults to ``-2`` for the
            batch-first sequence layout ``(..., frames, 4)``.

    Returns:
        Normalized quaternion tensor with the same shape as ``quaternions``.
        Adjacent samples along ``time_dim`` are sign-flipped independently for
        every batch entry when their quaternion dot product is negative.

    Raises:
        ValueError: If ``quaternions`` does not end in a quaternion axis or if
            ``time_dim`` points at that final quaternion axis.
    """
    if quaternions.ndim < 2 or quaternions.shape[-1] != 4:
        raise ValueError("quaternions must have shape (..., frames, 4).")

    time_axis = _resolve_time_dim(quaternions, time_dim)
    if time_axis == quaternions.ndim - 1:
        raise ValueError("time_dim cannot point at the final quaternion axis.")

    q = torch.movedim(normalize_quaternion_wxyz(quaternions), time_axis, -2)
    adjacent_dot = torch.sum(q[..., 1:, :] * q[..., :-1, :], dim=-1, keepdim=True)
    relative_sign = torch.where(adjacent_dot < 0.0, -torch.ones_like(adjacent_dot), torch.ones_like(adjacent_dot))
    first_sign = torch.ones_like(q[..., :1, :1])
    signs = torch.cumprod(torch.cat((first_sign, relative_sign), dim=-2), dim=-2)
    return torch.movedim(q * signs, -2, time_axis)


def matrix_to_rotation_6d(rotations: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to a 6D continuous representation.

    The representation stores the first two columns of a body-to-world rotation
    matrix. This matches the Zhou et al. 6D rotation representation used by
    neural networks, with columns rather than rows.

    Args:
        rotations: Rotation matrix tensor with shape ``(..., 3, 3)``.

    Returns:
        Tensor with shape ``(..., 6)`` ordered as
        ``(R[:, 0], R[:, 1])`` along the last axis.
    """
    return torch.cat((rotations[..., :, 0], rotations[..., :, 1]), dim=-1)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Project a 6D rotation representation back to SO(3).

    The final axis is interpreted as two 3D column vectors. The function uses
    Gram-Schmidt orthogonalization and deterministic fallbacks for zero or
    collinear inputs, so outputs and gradients remain finite at degenerate
    inputs.

    Args:
        rotation_6d: Tensor with shape ``(..., 6)`` ordered as two candidate
            rotation columns.

    Returns:
        Rotation matrix tensor with shape ``(..., 3, 3)``.
    """
    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:6]
    first = _normalize_vector(first, fallback=_canonical_unit_vector_like(first, axis=0))
    second_orthogonal = second - torch.sum(first * second, dim=-1, keepdim=True) * first
    second_fallback = _orthogonal_unit_vector(first)
    second = _normalize_vector(second_orthogonal, fallback=second_fallback)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def rotation_6d_to_matrix_and_derivative(
    rotation_6d: torch.Tensor,
    rotation_6d_derivative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project 6D rotations to SO(3) and propagate a first derivative.

    This function differentiates the same Gram-Schmidt projection implemented
    by :func:`rotation_6d_to_matrix`. It is intended for time trajectories where
    ``rotation_6d_derivative`` is the derivative of ``rotation_6d`` with respect
    to time.

    Args:
        rotation_6d: Tensor with shape ``(..., 6)``.
        rotation_6d_derivative: Tensor with shape ``(..., 6)`` and
            broadcast-compatible leading dimensions, representing
            ``d(rotation_6d) / dt``.

    Returns:
        Tuple ``(rotation, rotation_derivative)``. Both tensors have shape
        ``(..., 3, 3)``. ``rotation_derivative`` is ``dR/dt`` for the projected
        rotation matrix.
    """
    first_raw = rotation_6d[..., :3]
    second_raw = rotation_6d[..., 3:6]
    first_raw_dot = rotation_6d_derivative[..., :3]
    second_raw_dot = rotation_6d_derivative[..., 3:6]

    first, first_dot = _normalize_vector_and_derivative(
        first_raw,
        first_raw_dot,
        fallback=_canonical_unit_vector_like(first_raw, axis=0),
    )
    projection = torch.sum(first * second_raw, dim=-1, keepdim=True)
    projection_dot = torch.sum(first_dot * second_raw + first * second_raw_dot, dim=-1, keepdim=True)
    second_orthogonal = second_raw - projection * first
    second_orthogonal_dot = second_raw_dot - projection_dot * first - projection * first_dot
    second, second_dot = _normalize_vector_and_derivative(
        second_orthogonal,
        second_orthogonal_dot,
        fallback=_orthogonal_unit_vector(first),
    )
    third = torch.cross(first, second, dim=-1)
    third_dot = torch.cross(first_dot, second, dim=-1) + torch.cross(first, second_dot, dim=-1)
    return torch.stack((first, second, third), dim=-1), torch.stack((first_dot, second_dot, third_dot), dim=-1)


def world_angular_velocity_from_rotation_derivative(
    rotation: torch.Tensor,
    rotation_derivative: torch.Tensor,
) -> torch.Tensor:
    """Convert ``R_dot`` for a body-to-world rotation to world angular velocity.

    For a body-to-world rotation matrix ``R``, this assumes
    ``R_dot = skew(omega_W) @ R``. The skew-symmetric part of
    ``R_dot @ R.T`` is extracted and converted to ``omega_W``.

    Args:
        rotation: Rotation matrix tensor with shape ``(..., 3, 3)``.
        rotation_derivative: Time derivative of ``rotation`` with shape
            ``(..., 3, 3)``.

    Returns:
        World-frame angular velocity tensor with shape ``(..., 3)``.
    """
    angular_matrix = rotation_derivative @ rotation.transpose(-1, -2)
    return torch.stack(
        (
            0.5 * (angular_matrix[..., 2, 1] - angular_matrix[..., 1, 2]),
            0.5 * (angular_matrix[..., 0, 2] - angular_matrix[..., 2, 0]),
            0.5 * (angular_matrix[..., 1, 0] - angular_matrix[..., 0, 1]),
        ),
        dim=-1,
    )


def unwrap_angles(angles: torch.Tensor, *, time_dim: int = -2) -> torch.Tensor:
    """Unwrap angular trajectories along the frame dimension.

    This is a trajectory utility for periodic scalar coordinates. The default
    time convention is batch-first, ``(..., frames, coordinates)``, so the
    second-to-last axis is unwrapped. Unbatched ``(frames, coordinates)`` and
    one-dimensional ``(frames,)`` inputs also work with the default. Jumps
    larger than ``pi`` are shifted by integer multiples of ``2*pi`` to avoid
    artificial derivative spikes at ``-pi/pi`` or ``0/2*pi`` representation
    boundaries.

    Args:
        angles: Angular trajectory in radians. The recommended shape is
            ``(..., frames, coordinates)``; ``(frames, coordinates)`` and
            ``(frames,)`` are accepted.
        time_dim: Axis containing trajectory frames. Defaults to ``-2`` for
            ``(..., frames, coordinates)``. For frame-first batched data such as
            ``(frames, batch, coordinates)``, pass ``time_dim=0``.

    Returns:
        Tensor with the same shape as ``angles`` containing an unwrapped
        trajectory along ``time_dim``.

    Raises:
        ValueError: If ``angles`` is a scalar tensor.
    """
    time_axis = _resolve_time_dim(angles, time_dim)
    if angles.shape[time_axis] <= 1:
        return angles.clone()

    period = torch.as_tensor(2.0 * math.pi, dtype=angles.dtype, device=angles.device)
    half_period = torch.as_tensor(math.pi, dtype=angles.dtype, device=angles.device)
    deltas = torch.diff(angles, dim=time_axis)
    wrapped = torch.remainder(deltas + half_period, period) - half_period
    wrapped = torch.where((wrapped == -half_period) & (deltas > 0.0), half_period, wrapped)
    corrections = wrapped - deltas
    corrections = torch.where(torch.abs(deltas) <= half_period, torch.zeros_like(corrections), corrections)
    cumulative = torch.cumsum(corrections, dim=time_axis)
    leading_zero = torch.zeros_like(angles.narrow(time_axis, 0, 1))
    return angles + torch.cat((leading_zero, cumulative), dim=time_axis)


def skew(vector: torch.Tensor) -> torch.Tensor:
    """Return skew-symmetric cross-product matrices.

    Args:
        vector: Vector tensor with shape ``(..., 3)``.

    Returns:
        Tensor with shape ``(..., 3, 3)`` such that
        ``skew(vector) @ y == torch.cross(vector, y)`` for compatible
        ``y`` tensors.
    """
    x, y, z = torch.unbind(vector, dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )


def _resolve_time_dim(tensor: torch.Tensor, time_dim: int) -> int:
    """Resolve a trajectory time axis against a concrete tensor rank.

    The public trajectory helpers default to ``time_dim=-2`` for batch-first
    data. One-dimensional angle trajectories are a special case: their only
    axis is treated as time when the default is used.

    Args:
        tensor: Trajectory tensor with at least one dimension.
        time_dim: User-provided time axis, possibly negative.

    Returns:
        Nonnegative axis index into ``tensor``.

    Raises:
        ValueError: If ``tensor`` is scalar.
    """
    if tensor.ndim == 0:
        raise ValueError("trajectory tensor must have at least one dimension.")
    if tensor.ndim == 1 and time_dim == -2:
        return 0
    return time_dim % tensor.ndim


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Return ``sqrt(max(0, x))`` with zero subgradient at ``x == 0``.

    Args:
        x: Tensor of any shape.

    Returns:
        Tensor with the same shape as ``x``.
    """
    result = torch.zeros_like(x)
    positive = x > 0.0
    result[positive] = torch.sqrt(x[positive])
    return result


def _normalize_vector(vector: torch.Tensor, *, fallback: torch.Tensor | None = None) -> torch.Tensor:
    """Normalize vectors along the last axis while avoiding division by zero.

    Args:
        vector: Tensor with shape ``(..., D)``.
        fallback: Optional tensor with shape ``(..., D)`` used wherever the
            input norm is below the internal threshold.

    Returns:
        Tensor with shape ``(..., D)``.
    """
    norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    normalized = vector / torch.clamp(norm, min=1e-12)
    if fallback is None:
        return normalized
    return torch.where(norm > 1e-12, normalized, fallback)


def _normalize_vector_and_derivative(
    vector: torch.Tensor,
    vector_derivative: torch.Tensor,
    *,
    fallback: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a vector and propagate the derivative of the normalization.

    Args:
        vector: Tensor with shape ``(..., D)``.
        vector_derivative: Derivative of ``vector`` with shape ``(..., D)``.
        fallback: Optional normalized value used for near-zero vectors. When
            fallback is used, the propagated derivative is zero.

    Returns:
        Tuple ``(normalized, normalized_derivative)``, both with shape
        ``(..., D)``.
    """
    raw_norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    norm = torch.clamp(raw_norm, min=1e-12)
    normalized = vector / norm
    tangent_derivative = vector_derivative - normalized * torch.sum(
        normalized * vector_derivative,
        dim=-1,
        keepdim=True,
    )
    normalized_derivative = tangent_derivative / norm
    if fallback is None:
        return normalized, normalized_derivative
    valid = raw_norm > 1e-12
    return torch.where(valid, normalized, fallback), torch.where(valid, normalized_derivative, torch.zeros_like(vector))


def _orthogonal_unit_vector(vector: torch.Tensor) -> torch.Tensor:
    """Return a deterministic unit vector orthogonal to ``vector``.

    Args:
        vector: Unit vector tensor with shape ``(..., 3)``.

    Returns:
        Unit vector tensor with shape ``(..., 3)`` and zero dot product with
        ``vector`` up to numerical precision.
    """
    x_axis = torch.zeros_like(vector)
    x_axis[..., 0] = 1.0
    y_axis = torch.zeros_like(vector)
    y_axis[..., 1] = 1.0
    basis = torch.where(torch.abs(vector[..., :1]) < 0.9, x_axis, y_axis)
    orthogonal = basis - torch.sum(vector * basis, dim=-1, keepdim=True) * vector
    return _normalize_vector(orthogonal)


def _canonical_unit_vector_like(vector: torch.Tensor, *, axis: int) -> torch.Tensor:
    """Return a canonical unit vector broadcast-compatible with ``vector``.

    Args:
        vector: Reference tensor with shape ``(..., D)``.
        axis: Coordinate index set to one in the returned tensor.

    Returns:
        Tensor with the same shape, dtype, and device as ``vector``.
    """
    result = torch.zeros_like(vector)
    result[..., axis] = 1.0
    return result
