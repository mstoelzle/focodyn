from __future__ import annotations

from dataclasses import dataclass

import torch

from .dynamics import FloatingBaseDynamics
from .rotations import (
    continuous_quaternions_wxyz,
    matrix_to_quaternion_wxyz,
    matrix_to_rotation_6d,
    quaternion_wxyz_to_matrix,
    rotation_6d_to_matrix_and_derivative,
    unwrap_angles,
    world_angular_velocity_from_rotation_derivative,
)


@dataclass(frozen=True)
class MotionDerivativeEstimate:
    """Whittaker-smoothed motion derivatives for a floating-base trajectory.

    The state convention matches :class:`focodyn.dynamics.FloatingBaseDynamics`:
    ``q = (p_WB, quat_WB, s)`` and
    ``nu = (v_WB, omega_WB, s_dot)``. The quaternion uses scalar-first
    ``(w, x, y, z)`` order and maps base-frame vectors to world coordinates.
    Angular velocity and angular acceleration are expressed in the world frame.

    Attributes:
        states: Smoothed state trajectory with shape ``(frames, state_dim)``.
            The velocity block contains generalized velocities estimated from
            the smoothed configuration.
        generalized_accelerations: Estimated ``nu_dot`` tensor with shape
            ``(frames, nv)``.
        configurations: Smoothed configuration tensor ``q`` with shape
            ``(frames, nq)``.
        configuration_velocities: Estimated tangent-space ``q_dot`` with shape
            ``(frames, nv)`` ordered as ``(p_dot, omega_WB, s_dot)``.
            Floating-base orientation uses angular velocity rather than
            quaternion component derivatives.
        configuration_accelerations: Estimated tangent-space ``q_ddot`` with
            shape ``(frames, nv)`` ordered as
            ``(p_ddot, omega_dot_WB, s_ddot)``.
        times: Time vector with shape ``(frames,)``.
        lmbda: Whittaker-Eilers smoothing parameter used for the estimate.
        d_order: Difference penalty order used by the Whittaker filter.
    """

    states: torch.Tensor
    generalized_accelerations: torch.Tensor
    configurations: torch.Tensor
    configuration_velocities: torch.Tensor
    configuration_accelerations: torch.Tensor
    times: torch.Tensor
    lmbda: float
    d_order: int


def estimate_motion_derivatives(
    model: FloatingBaseDynamics,
    states: torch.Tensor,
    times: torch.Tensor,
    *,
    lmbda: float = 100.0,
    d_order: int = 2,
) -> MotionDerivativeEstimate:
    """Estimate velocities and accelerations using torch-dxdt Whittaker filters.

    Base orientation is smoothed in a continuous 6D rotation representation and
    projected back to SO(3). Joint angles are unwrapped before filtering so
    periodic representation jumps do not create artificial derivative spikes.
    Returned velocity and acceleration fields are tangent-space generalized
    quantities, not ambient quaternion component derivatives.

    Args:
        model: Floating-base dynamics model defining state dimensions and
            dtype/device. Its convention is
            ``q = (p_WB, quat_WB, s)`` and
            ``nu = (v_WB, omega_WB, s_dot)``.
        states: Kinematic state trajectory with shape
            ``(frames, model.state_dim)``. Only the configuration block
            ``states[:, :model.nq]`` is filtered; the input velocity block is
            ignored.
        times: Time vector with shape ``(frames,)``.
        lmbda: Whittaker-Eilers smoothing parameter. Larger values produce
            smoother derivatives.
        d_order: Whittaker difference penalty order.

    Returns:
        :class:`MotionDerivativeEstimate` with ``states`` shaped
        ``(frames, model.state_dim)``, ``configurations`` shaped
        ``(frames, model.nq)``, and derivative fields shaped
        ``(frames, model.nv)``.

    Raises:
        ImportError: If ``torch-dxdt`` is unavailable.
        ValueError: If input shapes are inconsistent with ``model``.
    """
    torch_dxdt = _require_torch_dxdt()
    states = states.to(dtype=model.dtype, device=model.device)
    times = times.to(dtype=model.dtype, device=model.device)
    if states.ndim != 2 or states.shape[-1] != model.state_dim:
        raise ValueError(f"Expected states with shape (frames, {model.state_dim}).")
    if times.ndim != 1 or times.shape[0] != states.shape[0]:
        raise ValueError("times must have shape (frames,) matching states.")
    if states.shape[0] < 4:
        raise ValueError("At least four frames are required for derivative estimation.")

    q = states[:, : model.nq].clone()
    joint_positions = unwrap_angles(q[:, 7:])

    differentiator = torch_dxdt.Whittaker(lmbda=float(lmbda), d_order=int(d_order))

    euclidean_q = torch.cat((q[:, :3], joint_positions), dim=-1)
    euclidean_orders = differentiator.d_orders(euclidean_q, times, orders=[0, 1, 2], dim=0)
    euclidean_smooth = _order_value(euclidean_orders, 0)
    euclidean_dot = _order_value(euclidean_orders, 1)
    euclidean_ddot = _order_value(euclidean_orders, 2)

    orientation = _estimate_orientation_derivatives(differentiator, q[:, 3:7], times)

    q_smooth = torch.empty_like(q)
    q_smooth[:, :3] = euclidean_smooth[:, :3]
    q_smooth[:, 3:7] = orientation.quaternions
    q_smooth[:, 7:] = euclidean_smooth[:, 3:]

    base_linear_velocity = euclidean_dot[:, :3]
    base_linear_acceleration = euclidean_ddot[:, :3]
    joint_velocity = euclidean_dot[:, 3:]
    joint_acceleration = euclidean_ddot[:, 3:]
    base_angular_velocity = orientation.angular_velocities
    base_angular_acceleration = orientation.angular_accelerations
    generalized_velocity = torch.cat((base_linear_velocity, base_angular_velocity, joint_velocity), dim=-1)
    generalized_acceleration = torch.cat(
        (base_linear_acceleration, base_angular_acceleration, joint_acceleration),
        dim=-1,
    )

    estimated_states = torch.zeros_like(states)
    estimated_states[:, : model.nq] = q_smooth
    estimated_states[:, model.nq :] = generalized_velocity

    return MotionDerivativeEstimate(
        states=estimated_states,
        generalized_accelerations=generalized_acceleration,
        configurations=q_smooth,
        configuration_velocities=generalized_velocity,
        configuration_accelerations=generalized_acceleration,
        times=times,
        lmbda=float(lmbda),
        d_order=int(d_order),
    )


@dataclass(frozen=True)
class _OrientationDerivativeEstimate:
    """Internal orientation derivative bundle.

    Attributes:
        quaternions: Smoothed unit quaternions with shape ``(frames, 4)`` in
            ``(w, x, y, z)`` order.
        angular_velocities: World-frame angular velocities with shape
            ``(frames, 3)``.
        angular_accelerations: World-frame angular accelerations with shape
            ``(frames, 3)``.
    """

    quaternions: torch.Tensor
    angular_velocities: torch.Tensor
    angular_accelerations: torch.Tensor


def _require_torch_dxdt():
    """Import ``torch_dxdt`` with a package-specific error message.

    Args:
        None.

    Returns:
        Imported ``torch_dxdt`` module.

    Raises:
        ImportError: If ``torch-dxdt`` is not installed.
    """
    try:
        import torch_dxdt
    except ImportError as exc:
        raise ImportError(
            "torch-dxdt is required for Whittaker derivative estimation. "
            "Install it with `uv sync` after updating dependencies or "
            "`pip install torch-dxdt`."
        ) from exc
    return torch_dxdt


def _estimate_orientation_derivatives(
    differentiator,
    quaternions: torch.Tensor,
    times: torch.Tensor,
) -> _OrientationDerivativeEstimate:
    """Smooth base orientation in 6D rotation space and estimate derivatives.

    The input quaternions are first sign-continuized along the frame dimension,
    converted to rotation matrices, then represented by the first two matrix
    columns. The Whittaker filter smooths and differentiates this 6D path. The
    smoothed path is projected back to SO(3), converted to scalar-first
    quaternions, and differentiated into world-frame angular velocity.

    Args:
        differentiator: torch-dxdt differentiator instance with a
            ``d_orders(values, times, orders, dim)`` method.
        quaternions: Base-to-world quaternion trajectory with shape
            ``(frames, 4)`` in ``(w, x, y, z)`` order.
        times: Time vector with shape ``(frames,)``.

    Returns:
        :class:`_OrientationDerivativeEstimate` containing smoothed
        quaternions, world-frame angular velocities, and world-frame angular
        accelerations.
    """
    rotations = quaternion_wxyz_to_matrix(continuous_quaternions_wxyz(quaternions))
    rotation_6d = matrix_to_rotation_6d(rotations)

    rotation_6d_orders = differentiator.d_orders(rotation_6d, times, orders=[0, 1], dim=0)
    rotation_6d_smooth = _order_value(rotation_6d_orders, 0)
    rotation_6d_dot = _order_value(rotation_6d_orders, 1)
    rotations_smooth, rotation_dot = rotation_6d_to_matrix_and_derivative(rotation_6d_smooth, rotation_6d_dot)
    quaternions_smooth = continuous_quaternions_wxyz(matrix_to_quaternion_wxyz(rotations_smooth))

    angular_velocity = world_angular_velocity_from_rotation_derivative(rotations_smooth, rotation_dot)
    angular_acceleration_orders = differentiator.d_orders(angular_velocity, times, orders=[1], dim=0)
    angular_acceleration = _order_value(angular_acceleration_orders, 1)

    return _OrientationDerivativeEstimate(
        quaternions=quaternions_smooth,
        angular_velocities=angular_velocity,
        angular_accelerations=angular_acceleration,
    )


def _order_value(orders, order: int) -> torch.Tensor:
    """Read a derivative order from torch-dxdt's order container.

    Args:
        orders: Container returned by ``torch_dxdt``. Supported forms are a
            mapping from integer order to tensor, or an indexable sequence.
        order: Requested derivative order.

    Returns:
        Tensor stored for ``order``. Shape depends on the differentiated input.
    """
    if isinstance(orders, dict):
        return orders[order]
    return orders[order]
