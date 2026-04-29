from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from ._torch import normalize_quaternion_wxyz
from .dynamics import FloatingBaseDynamics
from .motion import (
    KinematicMotionReference,
    default_g1_motion_reference,
    load_kinematic_motion_reference,
)


@dataclass(frozen=True)
class MotionDerivativeEstimate:
    """Whittaker-smoothed motion derivatives for a floating-base trajectory.

    Attributes:
        states: Smoothed state trajectory with shape ``(frames, state_dim)``.
            The velocity block contains generalized velocities estimated from
            the smoothed configuration.
        generalized_accelerations: Estimated ``nu_dot`` tensor with shape
            ``(frames, nv)``.
        configurations: Smoothed configuration tensor ``q`` with shape
            ``(frames, nq)``.
        configuration_velocities: Estimated ``q_dot`` with shape
            ``(frames, nq)``. The quaternion derivative entries are the direct
            Whittaker derivatives before conversion to angular velocity.
        configuration_accelerations: Estimated ``q_ddot`` with shape
            ``(frames, nq)``.
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

    Args:
        model: Floating-base dynamics model defining state dimensions and
            dtype/device.
        states: Kinematic state trajectory with shape
            ``(frames, model.state_dim)``. Only the configuration block is used
            for filtering.
        times: Time vector with shape ``(frames,)``.
        lmbda: Whittaker-Eilers smoothing parameter. Larger values produce
            smoother derivatives.
        d_order: Whittaker difference penalty order.

    Returns:
        :class:`MotionDerivativeEstimate`.

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
    q[:, 3:7] = _continuous_quaternions(q[:, 3:7])

    differentiator = torch_dxdt.Whittaker(lmbda=float(lmbda), d_order=int(d_order))
    q_orders = differentiator.d_orders(q, times, orders=[0, 1, 2], dim=0)
    q_smooth = _order_value(q_orders, 0)
    q_dot = _order_value(q_orders, 1)
    q_ddot = _order_value(q_orders, 2)
    q_smooth = q_smooth.clone()
    q_smooth[:, 3:7] = normalize_quaternion_wxyz(q_smooth[:, 3:7])

    base_linear_velocity = q_dot[:, :3]
    base_linear_acceleration = q_ddot[:, :3]
    joint_velocity = q_dot[:, 7:]
    joint_acceleration = q_ddot[:, 7:]

    base_angular_velocity = world_angular_velocity_from_quaternion_derivative(
        q_smooth[:, 3:7],
        q_dot[:, 3:7],
    )
    omega_orders = differentiator.d_orders(base_angular_velocity, times, orders=[1], dim=0)
    base_angular_acceleration = _order_value(omega_orders, 1)

    estimated_states = torch.zeros_like(states)
    estimated_states[:, : model.nq] = q_smooth
    estimated_states[:, model.nq : model.nq + 3] = base_linear_velocity
    estimated_states[:, model.nq + 3 : model.nq + 6] = base_angular_velocity
    estimated_states[:, model.nq + 6 :] = joint_velocity
    generalized_acceleration = torch.cat(
        (base_linear_acceleration, base_angular_acceleration, joint_acceleration),
        dim=-1,
    )

    return MotionDerivativeEstimate(
        states=estimated_states,
        generalized_accelerations=generalized_acceleration,
        configurations=q_smooth,
        configuration_velocities=q_dot,
        configuration_accelerations=q_ddot,
        times=times,
        lmbda=float(lmbda),
        d_order=int(d_order),
    )


def world_angular_velocity_from_quaternion_derivative(
    quaternion_wxyz: torch.Tensor,
    quaternion_derivative_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Convert scalar-first quaternion derivatives to world angular velocity.

    The convention matches
    :func:`focodyn._torch.quaternion_derivative_from_world_angular_velocity`,
    i.e. ``q_dot = 0.5 * [0, omega_W] * q``.

    Args:
        quaternion_wxyz: Unit or near-unit quaternions with shape ``(..., 4)``.
        quaternion_derivative_wxyz: Quaternion derivatives with shape
            ``(..., 4)``.

    Returns:
        World-frame angular velocities with shape ``(..., 3)``.
    """
    q = normalize_quaternion_wxyz(quaternion_wxyz)
    q_conjugate = torch.cat((q[..., :1], -q[..., 1:]), dim=-1)
    omega_quaternion = 2.0 * _quaternion_multiply(quaternion_derivative_wxyz, q_conjugate)
    return omega_quaternion[..., 1:]


def plot_motion_derivative_lambdas(
    model: FloatingBaseDynamics,
    motion: KinematicMotionReference,
    lambdas: Iterable[float],
    *,
    output_path: str | Path | None = None,
    joint_names: Iterable[str] | None = None,
    d_order: int = 2,
):
    """Plot smoothed motion, velocity, and acceleration for lambda selection.

    Args:
        model: Floating-base dynamics model defining state dimensions and
            joint order.
        motion: Loaded kinematic motion reference.
        lambdas: Whittaker ``lmbda`` values to compare.
        output_path: Optional path where the figure should be saved.
        joint_names: Optional joint names to include. Defaults to a small set
            of leg joints when available.
        d_order: Whittaker difference penalty order.

    Returns:
        The Matplotlib figure.

    Raises:
        ImportError: If Matplotlib or ``torch-dxdt`` is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install plotting dependencies with `uv sync --extra viz`.") from exc

    selected = _plot_coordinate_indices(model, joint_names=joint_names)
    lambda_values = tuple(float(value) for value in lambdas)
    if not lambda_values:
        raise ValueError("At least one lambda value is required.")

    estimates = [
        estimate_motion_derivatives(model, motion.states, motion.times, lmbda=value, d_order=d_order)
        for value in lambda_values
    ]

    rows = ("q", "q_dot", "q_ddot")
    fig, axes = plt.subplots(
        len(rows),
        len(selected),
        figsize=(4.2 * len(selected), 8.0),
        squeeze=False,
        sharex=True,
    )
    t = motion.times.detach().cpu().numpy()
    original_q = motion.states[:, : model.nq].detach().cpu().numpy()

    for col, (label, index) in enumerate(selected):
        axes[0, col].plot(t, original_q[:, index], color="black", linewidth=1.1, linestyle="--", label="raw")
        for estimate in estimates:
            color_label = f"lambda={estimate.lmbda:g}"
            axes[0, col].plot(
                t,
                estimate.configurations[:, index].detach().cpu().numpy(),
                linewidth=1.0,
                label=color_label,
            )
            axes[1, col].plot(
                t,
                estimate.configuration_velocities[:, index].detach().cpu().numpy(),
                linewidth=1.0,
                label=color_label,
            )
            axes[2, col].plot(
                t,
                estimate.configuration_accelerations[:, index].detach().cpu().numpy(),
                linewidth=1.0,
                label=color_label,
            )
        axes[0, col].set_title(label)
        for row, row_label in enumerate(rows):
            axes[row, col].set_ylabel(row_label)
            axes[row, col].grid(True, alpha=0.3)
        axes[-1, col].set_xlabel("time [s]")

    axes[0, 0].legend(loc="best", fontsize="small")
    fig.suptitle(f"Whittaker-Eilers derivative estimates: {motion.source_name}")
    fig.tight_layout()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    return fig


def main() -> None:
    """Parse CLI arguments and generate Whittaker derivative plots."""
    parser = argparse.ArgumentParser(description="Plot Whittaker-smoothed motion derivatives.")
    parser.add_argument("--asset", default="unitree_g1")
    parser.add_argument("--motion-reference", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("motion_derivative_lambdas.png"))
    parser.add_argument("--lambdas", type=float, nargs="+", default=(10.0, 100.0, 1000.0, 10000.0))
    parser.add_argument("--joints", nargs="*", default=None)
    parser.add_argument("--d-order", type=int, default=2)
    args = parser.parse_args()

    model = FloatingBaseDynamics(args.asset, include_contact_forces=True, dtype=torch.float64)
    motion = (
        load_kinematic_motion_reference(args.motion_reference, model)
        if args.motion_reference is not None
        else default_g1_motion_reference(model)
    )
    plot_motion_derivative_lambdas(
        model,
        motion,
        args.lambdas,
        output_path=args.output,
        joint_names=args.joints,
        d_order=args.d_order,
    )
    print(f"Saved derivative plot to {args.output}")


def _require_torch_dxdt():
    """Import torch-dxdt with a package-specific error message."""
    try:
        import torch_dxdt
    except ImportError as exc:
        raise ImportError(
            "torch-dxdt is required for Whittaker derivative estimation. "
            "Install it with `uv sync` after updating dependencies or "
            "`pip install torch-dxdt`."
        ) from exc
    return torch_dxdt


def _continuous_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    """Flip quaternion signs to keep adjacent samples on the same hemisphere."""
    result = normalize_quaternion_wxyz(quaternions).clone()
    for index in range(1, result.shape[0]):
        if torch.sum(result[index - 1] * result[index]) < 0:
            result[index] = -result[index]
    return result


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply scalar-first quaternions."""
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


def _order_value(orders, order: int) -> torch.Tensor:
    """Read a derivative order from torch-dxdt's order container."""
    if isinstance(orders, dict):
        return orders[order]
    return orders[order]


def _plot_coordinate_indices(
    model: FloatingBaseDynamics,
    *,
    joint_names: Iterable[str] | None,
) -> tuple[tuple[str, int], ...]:
    """Choose configuration coordinates shown by derivative plots."""
    selected: list[tuple[str, int]] = [("root z", 2)]
    if joint_names is None:
        candidates = (
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
        )
    else:
        candidates = tuple(joint_names)

    for name in candidates:
        if name in model.joint_names:
            selected.append((name, 7 + model.joint_names.index(name)))
    return tuple(selected[:7])


if __name__ == "__main__":
    main()
