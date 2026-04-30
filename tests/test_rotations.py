from __future__ import annotations

import math

import pytest
import torch

from focodyn.rotations import (
    continuous_quaternions_wxyz,
    matrix_to_quaternion_wxyz,
    matrix_to_rotation_6d,
    normalize_quaternion_wxyz,
    quaternion_conjugate_wxyz,
    quaternion_derivative_from_world_angular_velocity,
    quaternion_multiply_wxyz,
    quaternion_second_derivative_from_world_angular_acceleration,
    quaternion_wxyz_to_matrix,
    rotation_6d_to_matrix,
    rotation_6d_to_matrix_and_derivative,
    rpy_to_matrix,
    skew,
    unwrap_angles,
    world_angular_velocity_from_quaternion_derivative,
    world_angular_velocity_from_rotation_derivative,
)


DTYPE = torch.float64
ATOL = 1e-9


def _weighted_sum(tensor: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(0.37, 1.91, tensor.numel(), dtype=tensor.dtype, device=tensor.device)
    return torch.sum(tensor * weights.reshape_as(tensor))


def _assert_finite_output_and_gradients(function, *inputs: torch.Tensor) -> None:
    differentiable_inputs = tuple(input.detach().clone().requires_grad_(True) for input in inputs)
    output = function(*differentiable_inputs)
    outputs = output if isinstance(output, tuple) else (output,)
    for value in outputs:
        assert torch.isfinite(value).all()
    loss = sum(_weighted_sum(value) for value in outputs)
    gradients = torch.autograd.grad(loss, differentiable_inputs, allow_unused=False)
    for gradient in gradients:
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def _axis_angle_quaternion(axis: tuple[float, float, float], angle: float) -> torch.Tensor:
    axis_tensor = torch.tensor(axis, dtype=DTYPE)
    axis_tensor = axis_tensor / torch.linalg.norm(axis_tensor)
    half_angle = 0.5 * torch.as_tensor(angle, dtype=DTYPE)
    return torch.cat((torch.cos(half_angle).reshape(1), torch.sin(half_angle) * axis_tensor))


def _boundary_quaternions() -> torch.Tensor:
    return torch.stack(
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
            torch.tensor([-1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
            _axis_angle_quaternion((1.0, 0.0, 0.0), math.pi),
            _axis_angle_quaternion((0.0, 1.0, 0.0), math.pi),
            _axis_angle_quaternion((0.0, 0.0, 1.0), math.pi),
            _axis_angle_quaternion((1.0, 2.0, -3.0), math.pi - 1e-9),
            _axis_angle_quaternion((-2.0, 0.5, 1.0), 1e-12),
        )
    )


def _assert_same_rotation_quaternion(actual: torch.Tensor, expected: torch.Tensor, *, atol: float = ATOL) -> None:
    actual = normalize_quaternion_wxyz(actual)
    expected = normalize_quaternion_wxyz(expected)
    alignment = torch.abs(torch.sum(actual * expected, dim=-1))
    assert torch.allclose(alignment, torch.ones_like(alignment), atol=atol, rtol=0.0)


def _assert_rotation_matrix(rotation: torch.Tensor, *, atol: float = 1e-8) -> None:
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device).expand(rotation.shape[:-2] + (3, 3))
    orthogonality = rotation.transpose(-1, -2) @ rotation
    assert torch.allclose(orthogonality, identity, atol=atol, rtol=0.0)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(rotation.shape[:-2], dtype=rotation.dtype), atol=atol)


def test_quaternion_matrix_round_trip_at_branch_boundaries() -> None:
    quaternions = _boundary_quaternions()
    rotations = quaternion_wxyz_to_matrix(quaternions)

    recovered = matrix_to_quaternion_wxyz(rotations)

    _assert_same_rotation_quaternion(recovered, quaternions)
    _assert_finite_output_and_gradients(quaternion_wxyz_to_matrix, quaternions)
    _assert_finite_output_and_gradients(matrix_to_quaternion_wxyz, rotations)


def test_normalize_quaternion_outputs_unit_length_for_nonzero_inputs() -> None:
    quaternions = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, -3.0, 4.0, 0.0],
            [1e-14, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
    )

    normalized = normalize_quaternion_wxyz(quaternions)

    assert torch.allclose(torch.linalg.norm(normalized[:2], dim=-1), torch.ones(2, dtype=DTYPE), atol=ATOL)
    assert torch.allclose(normalized[2], torch.tensor([0.01, 0.0, 0.0, 0.0], dtype=DTYPE), atol=ATOL)
    assert torch.allclose(normalized[3], torch.zeros(4, dtype=DTYPE), atol=ATOL)


def test_quaternion_multiply_matches_matrix_composition_and_conjugate_inverse() -> None:
    left = _axis_angle_quaternion((1.0, -2.0, 0.5), 0.9)
    right = _axis_angle_quaternion((-0.4, 0.2, 1.0), -1.3)

    product = quaternion_multiply_wxyz(left, right)
    identity = quaternion_multiply_wxyz(left, quaternion_conjugate_wxyz(left))

    assert torch.allclose(
        quaternion_wxyz_to_matrix(product),
        quaternion_wxyz_to_matrix(left) @ quaternion_wxyz_to_matrix(right),
        atol=ATOL,
        rtol=0.0,
    )
    assert torch.allclose(identity, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE), atol=ATOL, rtol=0.0)


def test_rotation_helpers_preserve_arbitrary_leading_dimensions() -> None:
    quaternions = normalize_quaternion_wxyz(torch.randn(2, 3, 4, dtype=DTYPE))
    angular_velocity = torch.randn(2, 3, 3, dtype=DTYPE)
    angular_acceleration = torch.randn(2, 3, 3, dtype=DTYPE)
    rpy = torch.randn(2, 3, 3, dtype=DTYPE)
    rotation_6d = torch.randn(2, 3, 6, dtype=DTYPE)
    rotation_6d_derivative = torch.randn(2, 3, 6, dtype=DTYPE)

    rotations = quaternion_wxyz_to_matrix(quaternions)
    quaternion_derivative = quaternion_derivative_from_world_angular_velocity(quaternions, angular_velocity)
    rotation, rotation_derivative = rotation_6d_to_matrix_and_derivative(rotation_6d, rotation_6d_derivative)

    assert rotations.shape == (2, 3, 3, 3)
    assert matrix_to_quaternion_wxyz(rotations).shape == (2, 3, 4)
    assert rpy_to_matrix(rpy).shape == (2, 3, 3, 3)
    assert quaternion_multiply_wxyz(quaternions, quaternions).shape == (2, 3, 4)
    assert quaternion_conjugate_wxyz(quaternions).shape == (2, 3, 4)
    assert quaternion_derivative.shape == (2, 3, 4)
    assert world_angular_velocity_from_quaternion_derivative(quaternions, quaternion_derivative).shape == (2, 3, 3)
    assert quaternion_second_derivative_from_world_angular_acceleration(
        quaternions,
        quaternion_derivative,
        angular_velocity,
        angular_acceleration,
    ).shape == (2, 3, 4)
    assert matrix_to_rotation_6d(rotations).shape == (2, 3, 6)
    assert rotation_6d_to_matrix(rotation_6d).shape == (2, 3, 3, 3)
    assert rotation.shape == (2, 3, 3, 3)
    assert rotation_derivative.shape == (2, 3, 3, 3)
    assert world_angular_velocity_from_rotation_derivative(rotation, rotation_derivative).shape == (2, 3, 3)
    assert continuous_quaternions_wxyz(quaternions).shape == (2, 3, 4)
    assert skew(angular_velocity).shape == (2, 3, 3, 3)


def test_matrix_to_quaternion_has_finite_gradients_at_identity_and_pi_rotations() -> None:
    rotations = quaternion_wxyz_to_matrix(_boundary_quaternions()[:5])

    _assert_finite_output_and_gradients(matrix_to_quaternion_wxyz, rotations)


def test_quaternion_to_matrix_handles_zero_and_tiny_inputs_with_finite_gradients() -> None:
    quaternions = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1e-14, 0.0, 0.0, 0.0],
            [0.0, -1e-14, 2e-14, -3e-14],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
    )

    _assert_finite_output_and_gradients(normalize_quaternion_wxyz, quaternions)
    _assert_finite_output_and_gradients(quaternion_wxyz_to_matrix, quaternions)


def test_rpy_to_matrix_is_valid_and_differentiable_at_gimbal_lock_boundaries() -> None:
    rpy = torch.tensor(
        [
            [0.0, math.pi / 2.0, 0.0],
            [0.0, -math.pi / 2.0, 0.0],
            [math.pi, math.pi / 2.0 - 1e-12, -math.pi],
            [-math.pi, -math.pi / 2.0 + 1e-12, math.pi],
        ],
        dtype=DTYPE,
    )

    rotations = rpy_to_matrix(rpy)

    _assert_rotation_matrix(rotations)
    _assert_finite_output_and_gradients(rpy_to_matrix, rpy)


def test_rpy_to_matrix_matches_fixed_axis_reference_values() -> None:
    quarter_turns = torch.tensor(
        [
            [math.pi / 2.0, 0.0, 0.0],
            [0.0, math.pi / 2.0, 0.0],
            [0.0, 0.0, math.pi / 2.0],
        ],
        dtype=DTYPE,
    )
    expected = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=DTYPE,
    )

    assert torch.allclose(rpy_to_matrix(quarter_turns), expected, atol=ATOL, rtol=0.0)


def test_quaternion_multiply_and_conjugate_have_finite_gradients() -> None:
    left = torch.cat((_boundary_quaternions(), torch.zeros(1, 4, dtype=DTYPE)), dim=0)
    right = torch.roll(left, shifts=2, dims=0)

    _assert_finite_output_and_gradients(quaternion_multiply_wxyz, left, right)
    _assert_finite_output_and_gradients(quaternion_conjugate_wxyz, left)


def test_quaternion_angular_velocity_conversions_are_consistent_and_differentiable() -> None:
    quaternions = _boundary_quaternions()
    angular_velocity = torch.tensor(
        [
            [0.3, -0.2, 0.7],
            [-0.4, 0.1, 0.2],
            [1.0, 0.0, -0.5],
            [0.0, -1.3, 0.4],
            [0.5, 0.6, 0.7],
            [-0.8, 0.2, 1.1],
            [1e-9, -2e-9, 3e-9],
        ],
        dtype=DTYPE,
    )

    quaternion_derivative = quaternion_derivative_from_world_angular_velocity(quaternions, angular_velocity)
    recovered = world_angular_velocity_from_quaternion_derivative(quaternions, quaternion_derivative)

    assert torch.allclose(recovered, angular_velocity, atol=1e-9, rtol=0.0)
    _assert_finite_output_and_gradients(
        quaternion_derivative_from_world_angular_velocity,
        quaternions,
        angular_velocity,
    )
    _assert_finite_output_and_gradients(
        world_angular_velocity_from_quaternion_derivative,
        torch.cat((quaternions, torch.zeros(1, 4, dtype=DTYPE)), dim=0),
        torch.cat((quaternion_derivative, torch.ones(1, 4, dtype=DTYPE)), dim=0),
    )


def test_quaternion_second_derivative_has_finite_gradients() -> None:
    quaternions = torch.cat((_boundary_quaternions(), torch.zeros(1, 4, dtype=DTYPE)), dim=0)
    quaternion_derivative = torch.linspace(-0.4, 0.5, quaternions.numel(), dtype=DTYPE).reshape_as(quaternions)
    angular_velocity = torch.linspace(-1.0, 1.0, quaternions.shape[0] * 3, dtype=DTYPE).reshape(-1, 3)
    angular_acceleration = torch.linspace(0.7, -0.8, quaternions.shape[0] * 3, dtype=DTYPE).reshape(-1, 3)

    _assert_finite_output_and_gradients(
        quaternion_second_derivative_from_world_angular_acceleration,
        quaternions,
        quaternion_derivative,
        angular_velocity,
        angular_acceleration,
    )


def test_quaternion_second_derivative_matches_jvp_of_first_derivative() -> None:
    quaternion = normalize_quaternion_wxyz(torch.tensor([0.8, -0.2, 0.3, 0.4], dtype=DTYPE))
    angular_velocity = torch.tensor([0.35, -0.6, 1.2], dtype=DTYPE)
    angular_acceleration = torch.tensor([-0.7, 0.4, 0.2], dtype=DTYPE)
    quaternion_derivative = quaternion_derivative_from_world_angular_velocity(quaternion, angular_velocity)

    _, expected = torch.autograd.functional.jvp(
        quaternion_derivative_from_world_angular_velocity,
        (quaternion, angular_velocity),
        (quaternion_derivative, angular_acceleration),
    )
    actual = quaternion_second_derivative_from_world_angular_acceleration(
        quaternion,
        quaternion_derivative,
        angular_velocity,
        angular_acceleration,
    )

    assert torch.allclose(actual, expected, atol=ATOL, rtol=ATOL)


def test_continuous_quaternions_flips_antipodal_samples_and_has_finite_gradients() -> None:
    sequence = torch.stack(
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
            torch.tensor([-0.99, -0.01, 0.0, 0.0], dtype=DTYPE),
            torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=DTYPE),
            torch.tensor([0.0, -1.0, 0.0, 0.0], dtype=DTYPE),
        )
    )

    continuous = continuous_quaternions_wxyz(sequence)
    adjacent_alignment = torch.sum(continuous[:-1] * continuous[1:], dim=-1)

    assert torch.all(adjacent_alignment >= -1e-12)
    _assert_finite_output_and_gradients(continuous_quaternions_wxyz, sequence)

    batched_sequence = torch.stack(
        (
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]],
                dtype=DTYPE,
            ),
            torch.tensor(
                [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]],
                dtype=DTYPE,
            ),
        )
    )
    batched_continuous = continuous_quaternions_wxyz(batched_sequence)
    batched_alignment = torch.sum(batched_continuous[:, :-1] * batched_continuous[:, 1:], dim=-1)

    assert batched_continuous.shape == batched_sequence.shape
    assert torch.all(batched_alignment >= -1e-12)

    frame_first_continuous = continuous_quaternions_wxyz(batched_sequence.movedim(1, 0), time_dim=0)
    frame_first_alignment = torch.sum(frame_first_continuous[:-1] * frame_first_continuous[1:], dim=-1)

    assert frame_first_continuous.shape == (3, 2, 4)
    assert torch.all(frame_first_alignment >= -1e-12)


def test_continuous_quaternions_rejects_invalid_shapes_and_time_axes() -> None:
    with pytest.raises(ValueError, match="shape"):
        continuous_quaternions_wxyz(torch.ones(4, dtype=DTYPE))
    with pytest.raises(ValueError, match="shape"):
        continuous_quaternions_wxyz(torch.ones(2, 3, dtype=DTYPE))
    with pytest.raises(ValueError, match="time_dim"):
        continuous_quaternions_wxyz(torch.ones(2, 3, 4, dtype=DTYPE), time_dim=-1)


def test_rotation_6d_projection_is_valid_and_differentiable_for_degenerate_inputs() -> None:
    rotation_6d = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [1e-14, 0.0, 0.0, 0.0, 1e-14, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0 + 1e-14, 0.0],
        ],
        dtype=DTYPE,
    )

    rotations = rotation_6d_to_matrix(rotation_6d)

    _assert_rotation_matrix(rotations)
    _assert_finite_output_and_gradients(rotation_6d_to_matrix, rotation_6d)
    _assert_finite_output_and_gradients(matrix_to_rotation_6d, rotations)


def test_rotation_6d_round_trip_preserves_valid_rotation_matrices() -> None:
    rotations = quaternion_wxyz_to_matrix(_boundary_quaternions())

    recovered = rotation_6d_to_matrix(matrix_to_rotation_6d(rotations))

    assert torch.allclose(recovered, rotations, atol=ATOL, rtol=0.0)


def test_rotation_6d_derivative_matches_autograd_jvp_away_from_branches() -> None:
    rotation_6d = torch.tensor([0.8, -0.2, 0.3, 0.1, 1.2, -0.4], dtype=DTYPE)
    rotation_6d_derivative = torch.tensor([-0.3, 0.7, 0.2, 0.5, -0.1, 0.4], dtype=DTYPE)

    rotation, rotation_derivative = rotation_6d_to_matrix_and_derivative(rotation_6d, rotation_6d_derivative)
    jvp_rotation, jvp_derivative = torch.autograd.functional.jvp(
        rotation_6d_to_matrix,
        (rotation_6d,),
        (rotation_6d_derivative,),
    )

    assert torch.allclose(rotation, jvp_rotation, atol=1e-10, rtol=1e-10)
    assert torch.allclose(rotation_derivative, jvp_derivative, atol=1e-10, rtol=1e-10)


def test_rotation_6d_derivative_has_finite_gradients_for_degenerate_inputs() -> None:
    rotation_6d = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0 + 1e-14, 0.0],
        ],
        dtype=DTYPE,
    )
    rotation_6d_derivative = torch.linspace(-0.5, 0.6, rotation_6d.numel(), dtype=DTYPE).reshape_as(rotation_6d)

    _assert_finite_output_and_gradients(rotation_6d_to_matrix_and_derivative, rotation_6d, rotation_6d_derivative)


def test_world_angular_velocity_from_rotation_derivative_matches_skew_convention() -> None:
    quaternions = _boundary_quaternions()[2:5]
    rotations = quaternion_wxyz_to_matrix(quaternions)
    angular_velocity = torch.tensor(
        [
            [0.2, -0.3, 0.5],
            [-1.1, 0.4, 0.8],
            [0.7, 0.9, -0.2],
        ],
        dtype=DTYPE,
    )
    rotation_derivative = skew(angular_velocity) @ rotations

    recovered = world_angular_velocity_from_rotation_derivative(rotations, rotation_derivative)

    assert torch.allclose(recovered, angular_velocity, atol=1e-9, rtol=0.0)
    _assert_finite_output_and_gradients(world_angular_velocity_from_rotation_derivative, rotations, rotation_derivative)


def test_skew_matrix_matches_cross_product_and_has_finite_gradients() -> None:
    vectors = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 3.0], [1e-12, -1e-12, 2e-12]], dtype=DTYPE)
    operands = torch.tensor([[4.0, 5.0, 6.0], [-0.5, 0.25, 0.75], [3.0, -2.0, 1.0]], dtype=DTYPE)

    result = torch.matmul(skew(vectors), operands.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(result, torch.cross(vectors, operands, dim=-1), atol=ATOL, rtol=0.0)
    _assert_finite_output_and_gradients(skew, vectors)


@pytest.mark.parametrize(
    "angles",
    (
        torch.tensor([[3.00], [3.10], [-3.10], [-3.00]], dtype=DTYPE),
        torch.tensor([[0.0], [2.0 * math.pi - 1e-12], [1e-12], [-2.0 * math.pi + 1e-12]], dtype=DTYPE),
        torch.tensor([[-math.pi], [math.pi], [-math.pi], [math.pi]], dtype=DTYPE),
    ),
)
def test_unwrap_angles_removes_periodic_jumps_and_has_finite_gradients(angles: torch.Tensor) -> None:
    unwrapped = unwrap_angles(angles)

    assert torch.max(torch.abs(torch.diff(unwrapped, dim=0))) <= math.pi
    _assert_finite_output_and_gradients(unwrap_angles, angles)


def test_unwrap_angles_defaults_to_batch_first_time_axis() -> None:
    batch_first_angles = torch.tensor(
        [
            [[3.00, -0.1], [3.10, -0.2], [-3.10, -0.3], [-3.00, -0.4]],
            [[0.0, 1.0], [2.0 * math.pi - 1e-12, 1.1], [1e-12, 1.2], [0.1, 1.3]],
        ],
        dtype=DTYPE,
    )
    one_dimensional_angles = torch.tensor([3.00, 3.10, -3.10, -3.00], dtype=DTYPE)

    batch_first_unwrapped = unwrap_angles(batch_first_angles)
    one_dimensional_unwrapped = unwrap_angles(one_dimensional_angles)

    assert torch.max(torch.abs(torch.diff(batch_first_unwrapped, dim=1))) <= math.pi
    assert torch.max(torch.abs(torch.diff(one_dimensional_unwrapped, dim=0))) <= math.pi
    _assert_finite_output_and_gradients(unwrap_angles, batch_first_angles)
    _assert_finite_output_and_gradients(unwrap_angles, one_dimensional_angles)


def test_unwrap_angles_supports_frame_first_time_axis_and_rejects_scalars() -> None:
    frame_first_angles = torch.tensor(
        [
            [[3.00, 0.0], [0.0, -1.0]],
            [[3.10, 0.1], [2.0 * math.pi - 1e-12, -1.1]],
            [[-3.10, 0.2], [1e-12, -1.2]],
            [[-3.00, 0.3], [0.1, -1.3]],
        ],
        dtype=DTYPE,
    )
    single_frame_angles = torch.tensor([[[3.0, -2.0], [1.0, 0.0]]], dtype=DTYPE)

    frame_first_unwrapped = unwrap_angles(frame_first_angles, time_dim=0)
    single_frame_unwrapped = unwrap_angles(single_frame_angles, time_dim=0)

    assert torch.max(torch.abs(torch.diff(frame_first_unwrapped, dim=0))) <= math.pi
    assert torch.allclose(single_frame_unwrapped, single_frame_angles)
    with pytest.raises(ValueError, match="at least one dimension"):
        unwrap_angles(torch.tensor(0.0, dtype=DTYPE))
