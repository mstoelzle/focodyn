from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from .contacts import FlatTerrainContactDetector
from .dynamics import FloatingBaseDynamics
from .motion import (
    KinematicMotionReference,
    default_g1_motion_reference,
    load_kinematic_motion_reference,
)
from .motion_derivatives import estimate_motion_derivatives


@dataclass(frozen=True)
class FixedContactForceTrajectoryAnalysis:
    """Fixed-contact force diagnostics over a kinematic trajectory."""

    times: torch.Tensor
    active_contacts: torch.Tensor
    contact_heights: torch.Tensor
    world_forces: torch.Tensor
    normal_forces: torch.Tensor
    friction_ratios: torch.Tensor
    total_normal_force: torch.Tensor
    support_force_demand: torch.Tensor
    contact_space_rank: torch.Tensor
    contact_space_condition: torch.Tensor
    equation_residual_norm: torch.Tensor
    contact_velocity_residual_norm: torch.Tensor
    contact_acceleration_residual_norm: torch.Tensor
    contact_names: tuple[str, ...]
    source_name: str
    contact_threshold: float


def analyze_fixed_contact_forces(
    model: FloatingBaseDynamics,
    motion: KinematicMotionReference,
    *,
    whittaker_lmbda: float = 100.0,
    whittaker_d_order: int = 2,
    contact_threshold: float = 0.025,
    friction_coefficient: float = 0.8,
    pinv_rcond: float = 1e-10,
    max_frames: int | None = None,
) -> FixedContactForceTrajectoryAnalysis:
    """Estimate fixed-contact forces along a kinematic trajectory.

    The force estimate assumes that contacts selected by the flat-terrain
    detector are sticking contacts. The returned residuals make violations of
    ``J_c nu = 0`` and ``J_c nu_dot + Jdot_c nu = 0`` visible in plots.

    Args:
        model: Floating-base dynamics model with contact forces enabled.
        motion: Kinematic trajectory to analyze.
        whittaker_lmbda: Whittaker-Eilers smoothing parameter used to estimate
            velocities and accelerations from the kinematic states.
        whittaker_d_order: Whittaker difference penalty order.
        contact_threshold: Flat-ground signed-distance threshold for active
            contacts in meters.
        friction_coefficient: Coulomb coefficient used only to report
            tangential-to-normal ratios relative to a nominal friction limit.
        pinv_rcond: Relative cutoff for contact-space pseudoinverses.
        max_frames: Optional frame limit for quick analysis and tests.

    Returns:
        :class:`FixedContactForceTrajectoryAnalysis`.

    Raises:
        RuntimeError: If ``model`` does not have a contact model.
    """
    if model.contact_model is None:
        raise RuntimeError("analyze_fixed_contact_forces requires include_contact_forces=True.")

    frame_count = motion.states.shape[0] if max_frames is None else min(max_frames, motion.states.shape[0])
    if frame_count <= 0:
        raise ValueError("At least one frame is required.")
    states = motion.states[:frame_count]
    times = motion.times[:frame_count]
    estimate = estimate_motion_derivatives(
        model,
        states,
        times,
        lmbda=whittaker_lmbda,
        d_order=whittaker_d_order,
    )
    detector = FlatTerrainContactDetector(
        contact_threshold=contact_threshold,
        dtype=model.dtype,
        device=model.device,
    )

    active_contacts = []
    contact_heights = []
    world_forces = []
    normal_forces = []
    friction_ratios = []
    total_normal_force = []
    support_force_demand = []
    contact_space_rank = []
    contact_space_condition = []
    equation_residual_norm = []
    contact_velocity_residual_norm = []
    contact_acceleration_residual_norm = []

    terrain_normal = detector.normal.to(dtype=model.dtype, device=model.device)
    for frame, state in enumerate(estimate.states):
        split = model.split_state(state)
        contact_poses = model.contact_model.contact_poses(
            model.base_transform(state),
            split.joint_positions.squeeze(0),
        )
        terrain_state = detector.detect(contact_poses.positions)
        generalized_demand = model.generalized_forces_from_acceleration(
            state,
            estimate.generalized_accelerations[frame],
        )
        fixed_forces = model.fixed_contact_forces_from_joint_torques(
            state,
            generalized_demand[6:],
            active_contacts=terrain_state.in_contact,
            force_frame="world",
            force_direction="environment_on_robot",
            pinv_rcond=pinv_rcond,
        )
        terms = fixed_forces.contact_space_dynamics
        acceleration_residual = (
            torch.matmul(terms.contact_jacobian, estimate.generalized_accelerations[frame].unsqueeze(-1))
            .squeeze(-1)
            + terms.jacobian_dot_velocity
        )
        normals = contact_poses.transforms[..., :3, 2].to(dtype=model.dtype, device=model.device)
        tangential_forces = fixed_forces.world_forces - fixed_forces.normal_forces.unsqueeze(-1) * normals
        tangential_norm = torch.linalg.norm(tangential_forces, dim=-1)
        friction_ratio = tangential_norm / torch.clamp(
            friction_coefficient * fixed_forces.normal_forces,
            min=1e-9,
        )
        singular_values = terms.singular_values
        if singular_values.numel() == 0 or int(terms.rank.item()) == 0:
            condition = torch.as_tensor(float("inf"), dtype=model.dtype, device=model.device)
        else:
            positive = singular_values[singular_values > 0.0]
            condition = positive[0] / torch.clamp(positive[-1], min=1e-12)

        active_contacts.append(terrain_state.in_contact)
        contact_heights.append(terrain_state.signed_distances)
        world_forces.append(fixed_forces.world_forces)
        normal_forces.append(fixed_forces.normal_forces)
        friction_ratios.append(friction_ratio)
        total_normal_force.append(torch.sum(fixed_forces.normal_forces))
        support_force_demand.append(torch.clamp(torch.dot(generalized_demand[:3], terrain_normal), min=0.0))
        contact_space_rank.append(terms.rank)
        contact_space_condition.append(condition)
        equation_residual_norm.append(torch.linalg.norm(fixed_forces.equation_residual))
        contact_velocity_residual_norm.append(torch.linalg.norm(terms.contact_velocity))
        contact_acceleration_residual_norm.append(torch.linalg.norm(acceleration_residual))

    return FixedContactForceTrajectoryAnalysis(
        times=estimate.times,
        active_contacts=torch.stack(active_contacts),
        contact_heights=torch.stack(contact_heights),
        world_forces=torch.stack(world_forces),
        normal_forces=torch.stack(normal_forces),
        friction_ratios=torch.stack(friction_ratios),
        total_normal_force=torch.stack(total_normal_force),
        support_force_demand=torch.stack(support_force_demand),
        contact_space_rank=torch.stack(contact_space_rank),
        contact_space_condition=torch.stack(contact_space_condition),
        equation_residual_norm=torch.stack(equation_residual_norm),
        contact_velocity_residual_norm=torch.stack(contact_velocity_residual_norm),
        contact_acceleration_residual_norm=torch.stack(contact_acceleration_residual_norm),
        contact_names=model.contact_model.contact_names,
        source_name=motion.source_name,
        contact_threshold=float(contact_threshold),
    )


def plot_fixed_contact_force_analysis(
    analysis: FixedContactForceTrajectoryAnalysis,
    output_dir: str | Path = "outputs/contact_forces",
    *,
    line_width: float = 1.6,
    friction_ratio_plot_cap: float = 3.0,
) -> Path:
    """Save a multi-panel PDF summary of fixed-contact force diagnostics.

    Args:
        analysis: Precomputed fixed-contact force diagnostics.
        output_dir: Directory receiving ``fixed_contact_forces_summary.pdf``.
        line_width: Matplotlib line width used for plotted trajectories.
        friction_ratio_plot_cap: Maximum displayed friction ratio. Larger
            values are clipped visually to keep the Coulomb-limit region
            readable.

    Returns:
        Path to the saved PDF.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install plotting dependencies with `uv sync --extra viz`.") from exc

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "fixed_contact_forces_summary.pdf"
    time = analysis.times.detach().cpu().numpy()
    names = _pretty_contact_labels(analysis.contact_names)

    fig, axes = plt.subplots(4, 2, figsize=(14.0, 12.5), sharex=True)
    flat_axes = axes.reshape(-1)

    active_count = torch.sum(analysis.active_contacts.to(dtype=torch.float64), dim=-1)
    flat_axes[0].plot(time, active_count.detach().cpu().numpy(), color="black", linewidth=line_width)
    flat_axes[0].set_ylabel("active contacts")
    flat_axes[0].set_title(r"Detected fixed-contact set")

    heights = analysis.contact_heights.detach().cpu().numpy()
    threshold = float(analysis.contact_threshold)
    flat_axes[1].axhspan(
        0.0,
        threshold,
        color="tab:orange",
        alpha=0.10,
        label=rf"active band: $0 \leq d_c \leq {threshold:g}$ m",
    )
    for index, name in enumerate(names):
        flat_axes[1].plot(time, heights[:, index], linewidth=line_width, label=name)
    flat_axes[1].axhline(0.0, color="black", linewidth=1.0, label=r"ground: $d_c=0$")
    flat_axes[1].axhline(
        threshold,
        color="tab:orange",
        linewidth=1.1,
        linestyle="--",
        label=rf"activation threshold: $d_c={threshold:g}$ m",
    )
    flat_axes[1].set_ylabel(r"signed distance $d_c$ [m]")
    flat_axes[1].set_title(r"Contact heights and activation threshold")

    normal_forces = analysis.normal_forces.detach().cpu().numpy()
    normal_min = min(float(analysis.normal_forces.min().detach().cpu()), 0.0)
    normal_max = max(
        float(analysis.normal_forces.max().detach().cpu()),
        1.0,
    )
    normal_margin = max(1.0, 0.08 * (normal_max - normal_min))
    normal_ymin = normal_min - normal_margin
    normal_ymax = normal_max + normal_margin
    flat_axes[2].fill_between(
        time,
        normal_ymin,
        0.0,
        color="tab:red",
        alpha=0.10,
        label=r"unilateral violation: $f_n < 0$",
    )
    for index, name in enumerate(names):
        flat_axes[2].plot(time, normal_forces[:, index], linewidth=line_width, label=name)
    flat_axes[2].set_ylim(normal_ymin, normal_ymax)
    flat_axes[2].set_ylabel(r"normal force $f_n$ [N]")
    flat_axes[2].set_title(r"Per-contact normal forces")

    flat_axes[3].plot(
        time,
        analysis.total_normal_force.detach().cpu().numpy(),
        label="estimated",
        linewidth=line_width,
    )
    flat_axes[3].plot(
        time,
        analysis.support_force_demand.detach().cpu().numpy(),
        linestyle="--",
        label="inverse-dynamics demand",
        linewidth=line_width,
    )
    flat_axes[3].set_ylabel("force [N]")
    flat_axes[3].set_title(r"Total vertical support $\sum_i f_{n,i}$")
    flat_axes[3].legend(loc="best", fontsize="small")

    ratios_tensor = torch.clamp(analysis.friction_ratios, min=0.0, max=friction_ratio_plot_cap)
    ratios = ratios_tensor.detach().cpu().numpy()
    ratio_ymax = max(1.25, float(torch.max(ratios_tensor).detach().cpu()) + 0.10)
    flat_axes[4].axhspan(1.0, ratio_ymax, color="tab:red", alpha=0.10, label=r"Coulomb violation: $\rho > 1$")
    for index, name in enumerate(names):
        flat_axes[4].plot(time, ratios[:, index], linewidth=line_width, label=name)
    flat_axes[4].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    flat_axes[4].set_ylim(0.0, ratio_ymax)
    flat_axes[4].set_ylabel(r"$\rho=\|f_t\|/(\mu f_n)$")
    flat_axes[4].set_title(
        rf"Friction ratio; shaded when $\rho>1$ "
        rf"(values clipped at {friction_ratio_plot_cap:g})"
    )

    flat_axes[5].plot(
        time,
        analysis.contact_space_rank.detach().cpu().numpy(),
        label="rank",
        linewidth=line_width,
    )
    flat_axes[5].set_ylabel("rank")
    twin = flat_axes[5].twinx()
    twin.semilogy(
        time,
        analysis.contact_space_condition.detach().cpu().numpy(),
        color="tab:red",
        label="condition",
        linewidth=line_width,
    )
    twin.set_ylabel("condition")
    flat_axes[5].set_title(r"Rank and conditioning of $J_cM^{-1}J_c^\top$")

    flat_axes[6].semilogy(
        time,
        torch.clamp(analysis.equation_residual_norm, min=1e-16).detach().cpu().numpy(),
        label=r"Eq. residual",
        linewidth=line_width,
    )
    flat_axes[6].semilogy(
        time,
        torch.clamp(analysis.contact_velocity_residual_norm, min=1e-16).detach().cpu().numpy(),
        label=r"$J_c\nu$",
        linewidth=line_width,
    )
    flat_axes[6].semilogy(
        time,
        torch.clamp(analysis.contact_acceleration_residual_norm, min=1e-16).detach().cpu().numpy(),
        label=r"$J_c\dot{\nu}+\dot{J}_c\nu$",
        linewidth=line_width,
    )
    flat_axes[6].set_ylabel("norm")
    flat_axes[6].set_title(r"Fixed-contact assumption residuals")
    flat_axes[6].legend(loc="best", fontsize="small")

    vertical_forces = analysis.world_forces[..., 2].detach().cpu().numpy()
    for index, name in enumerate(names):
        flat_axes[7].plot(time, vertical_forces[:, index], linewidth=line_width, label=name)
    flat_axes[7].set_ylabel(r"world $z$ force [N]")
    flat_axes[7].set_title(r"World-frame vertical force component $f_z$")

    for axis in flat_axes:
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("time [s]")
    flat_axes[1].legend(loc="best", fontsize="x-small", ncols=2)
    flat_axes[2].legend(loc="best", fontsize="x-small")
    flat_axes[4].legend(loc="best", fontsize="x-small", ncols=2)
    fig.suptitle(rf"Fixed-contact force analysis: {analysis.source_name}")
    fig.text(
        0.5,
        0.012,
        r"Friction ratio: $\rho=\|f_t\|/(\mu f_n)$. "
        r"The red shaded band marks $\rho>1$, where the tangential force exceeds "
        r"the Coulomb limit $\|f_t\|\leq\mu f_n$.",
        ha="center",
        va="bottom",
        fontsize="small",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _pretty_contact_labels(contact_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return compact, human-readable labels for configured contacts."""
    labels = []
    g1_corner_labels = {
        "0": "heel y-",
        "1": "heel y+",
        "2": "toe y-",
        "3": "toe y+",
    }
    for name in contact_names:
        if name == "left_ankle_roll_link:center":
            labels.append("L foot center")
        elif name == "right_ankle_roll_link:center":
            labels.append("R foot center")
        elif name.startswith("left_ankle_roll_link:"):
            suffix = name.split(":", maxsplit=1)[1]
            labels.append(f"L {g1_corner_labels.get(suffix, suffix)}")
        elif name.startswith("right_ankle_roll_link:"):
            suffix = name.split(":", maxsplit=1)[1]
            labels.append(f"R {g1_corner_labels.get(suffix, suffix)}")
        else:
            labels.append(name.replace("_ankle_roll_link", " foot").replace("_", " "))
    return tuple(labels)


def main() -> None:
    """Run fixed-contact force analysis from the command line."""
    parser = argparse.ArgumentParser(description="Analyze fixed-contact forces on a kinematic trajectory.")
    parser.add_argument("--asset", default="unitree_g1")
    parser.add_argument("--contact-mode", default="feet_corners", choices=("feet_corners", "feet_centers"))
    parser.add_argument("--motion-reference", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/contact_forces"))
    parser.add_argument("--whittaker-lambda", type=float, default=100.0)
    parser.add_argument("--whittaker-d-order", type=int, default=2)
    parser.add_argument("--contact-threshold", type=float, default=0.025)
    parser.add_argument("--friction-coefficient", type=float, default=0.8)
    parser.add_argument("--pinv-rcond", type=float, default=1e-10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--line-width", type=float, default=1.6)
    parser.add_argument("--friction-ratio-plot-cap", type=float, default=3.0)
    parser.add_argument("--export-video", type=Path, default=None)
    parser.add_argument("--export-width", type=int, default=1280)
    parser.add_argument("--export-height", type=int, default=720)
    parser.add_argument("--export-frames", type=int, default=None)
    parser.add_argument("--export-browser", type=Path, default=None)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-meshes", action="store_true")
    parser.add_argument("--robot-opacity", type=float, default=0.35)
    parser.add_argument("--video-force-scale", type=float, default=0.01)
    args = parser.parse_args()

    model = FloatingBaseDynamics(
        args.asset,
        include_contact_forces=True,
        contact_mode=args.contact_mode,
        dtype=torch.float64,
    )
    motion = (
        load_kinematic_motion_reference(args.motion_reference, model)
        if args.motion_reference is not None
        else default_g1_motion_reference(model)
    )
    analysis = analyze_fixed_contact_forces(
        model,
        motion,
        whittaker_lmbda=args.whittaker_lambda,
        whittaker_d_order=args.whittaker_d_order,
        contact_threshold=args.contact_threshold,
        friction_coefficient=args.friction_coefficient,
        pinv_rcond=args.pinv_rcond,
        max_frames=args.max_frames,
    )
    output = plot_fixed_contact_force_analysis(
        analysis,
        args.output_dir,
        line_width=args.line_width,
        friction_ratio_plot_cap=args.friction_ratio_plot_cap,
    )
    print(f"Saved fixed-contact force analysis to {output}")

    if args.export_video is not None:
        from .visualization import DynamicsVerificationViewer

        viewer = DynamicsVerificationViewer(
            asset_name=args.asset,
            contact_mode=args.contact_mode,
            contact_force_frame="world",
            port=args.port,
            load_meshes=not args.no_meshes,
            robot_opacity=args.robot_opacity,
            motion_reference=args.motion_reference,
            whittaker_lmbda=args.whittaker_lambda,
            whittaker_d_order=args.whittaker_d_order,
            contact_threshold=args.contact_threshold,
        )
        viewer.dynamics_mode = DynamicsVerificationViewer.FIXED_CONTACT_MODE
        viewer.force_scale = float(args.video_force_scale)
        export_frame_count = args.export_frames if args.export_frames is not None else args.max_frames
        video_output = viewer.export_video(
            args.export_video,
            width=args.export_width,
            height=args.export_height,
            frame_count=export_frame_count,
            browser_executable=args.export_browser,
        )
        print(f"Saved fixed-contact force rendering to {video_output}")


if __name__ == "__main__":
    main()
