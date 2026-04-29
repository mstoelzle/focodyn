from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .dynamics import HumanoidDynamics
from .walking import simple_walking_sequence


def run_contact_viewer(
    *,
    asset_name: str = "unitree_g1",
    contact_mode: str = "feet_corners",
    fps: float = 30.0,
    port: int = 8080,
    load_meshes: bool = True,
    max_frames: int | None = None,
) -> None:
    """Run a Viser URDF viewer with Adam FK contact points overlaid."""
    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise ImportError("Install visualization extras with `uv sync --extra viz`.") from exc

    model = HumanoidDynamics(
        asset_name,
        include_contact_forces=True,
        contact_mode=contact_mode,
        dtype=torch.float64,
    )
    if model.contact_model is None:
        raise RuntimeError("Contact model was not initialized.")

    states, _ = simple_walking_sequence(model, frames=180, dt=1.0 / fps)
    server = viser.ViserServer(port=port)
    server.scene.add_grid("/ground", width=2.0, height=2.0)
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    urdf_vis = ViserUrdf(
        server,
        urdf_or_path=Path(model.asset.urdf_path),
        root_node_name="/robot",
        load_meshes=load_meshes,
        load_collision_meshes=False,
    )

    contact_handles = [
        server.scene.add_icosphere(
            f"/contacts/{name.replace(':', '_')}",
            radius=0.018,
            color=(255, 80, 20),
        )
        for name in model.contact_model.contact_names
    ]
    contact_frame_handles = [
        server.scene.add_frame(
            f"/contact_frames/{name.replace(':', '_')}",
            axes_length=0.06,
            axes_radius=0.004,
        )
        for name in model.contact_model.contact_names
    ]

    frame = 0
    frames_rendered = 0
    period = 1.0 / fps
    while True:
        state = states[frame]
        split = model.split_state(state)
        joint_positions = split.joint_positions.squeeze(0).detach().cpu().numpy()
        base_position = split.base_position.squeeze(0).detach().cpu().numpy()
        base_quaternion = split.base_quaternion_wxyz.squeeze(0).detach().cpu().numpy()
        urdf_vis.update_cfg(joint_positions)
        robot_root.position = tuple(float(value) for value in base_position)
        robot_root.wxyz = tuple(float(value) for value in base_quaternion)

        base_transform = model.base_transform(state)
        contact_poses = model.contact_model.contact_poses(
            base_transform,
            split.joint_positions.squeeze(0),
        )
        contact_positions = (
            contact_poses.positions
            .detach()
            .cpu()
            .numpy()
        )
        contact_quaternions = (
            contact_poses.quaternions_wxyz
            .detach()
            .cpu()
            .numpy()
        )
        for handle, frame_handle, point, quat in zip(
            contact_handles,
            contact_frame_handles,
            np.asarray(contact_positions),
            np.asarray(contact_quaternions),
        ):
            handle.position = tuple(float(value) for value in point)
            frame_handle.position = tuple(float(value) for value in point)
            frame_handle.wxyz = tuple(float(value) for value in quat)

        frame = (frame + 1) % len(states)
        frames_rendered += 1
        if max_frames is not None and frames_rendered >= max_frames:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        time.sleep(period)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Unitree G1 contact-point FK.")
    parser.add_argument("--asset", default="unitree_g1")
    parser.add_argument("--contact-mode", default="feet_corners", choices=("feet_corners", "feet_centers"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-meshes", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    run_contact_viewer(
        asset_name=args.asset,
        contact_mode=args.contact_mode,
        fps=args.fps,
        port=args.port,
        load_meshes=not args.no_meshes,
        max_frames=args.max_frames,
    )
