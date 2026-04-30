from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch

from focodyn import (
    FloatingBaseDynamics,
    KinematicMotionReference,
    default_g1_motion_reference,
    estimate_motion_derivatives,
    load_kinematic_motion_reference,
)


def plot_motion_derivative_lambdas(
    model: FloatingBaseDynamics,
    motion: KinematicMotionReference,
    lambdas: Iterable[float],
    *,
    output_path: str | Path | None = None,
    joint_names: Iterable[str] | None = None,
    d_order: int = 2,
):
    """Plot smoothed motion derivatives for comparing Whittaker lambdas.

    Args:
        model: Floating-base dynamics model defining state dimensions and
            joint order.
        motion: Kinematic motion reference with ``motion.states`` shaped
            ``(frames, model.state_dim)`` and ``motion.times`` shaped
            ``(frames,)``.
        lambdas: Whittaker ``lmbda`` values to compare.
        output_path: Optional path where the figure is saved.
        joint_names: Optional joint names to include. Defaults to a small set
            of leg joints when those names exist in the model.
        d_order: Whittaker difference penalty order.

    Returns:
        Matplotlib figure containing ``q``, tangent-space ``q_dot``, and
        tangent-space ``q_ddot`` rows.
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

    for col, (label, q_index, derivative_index) in enumerate(selected):
        axes[0, col].plot(t, original_q[:, q_index], color="black", linewidth=1.1, linestyle="--", label="raw")
        for estimate in estimates:
            color_label = f"lambda={estimate.lmbda:g}"
            axes[0, col].plot(
                t,
                estimate.configurations[:, q_index].detach().cpu().numpy(),
                linewidth=1.0,
                label=color_label,
            )
            axes[1, col].plot(
                t,
                estimate.configuration_velocities[:, derivative_index].detach().cpu().numpy(),
                linewidth=1.0,
                label=color_label,
            )
            axes[2, col].plot(
                t,
                estimate.configuration_accelerations[:, derivative_index].detach().cpu().numpy(),
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


def _plot_coordinate_indices(
    model: FloatingBaseDynamics,
    *,
    joint_names: Iterable[str] | None,
) -> tuple[tuple[str, int, int], ...]:
    selected: list[tuple[str, int, int]] = [("root z", 2, 2)]
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
            joint_index = model.joint_names.index(name)
            selected.append((name, 7 + joint_index, 6 + joint_index))
    return tuple(selected[:7])


if __name__ == "__main__":
    main()
