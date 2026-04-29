from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch

from .dynamics import HumanoidDynamics
from .motion import (
    EMBER_G1_MOTION_REFERENCE,
    bundled_motion_reference_path,
    default_g1_motion_reference,
    load_kinematic_motion_reference,
)
from .walking import simple_walking_sequence


@dataclass(frozen=True)
class _ViewerMotion:
    """Kinematic motion sequence available in the viewer.

    Attributes:
        label: GUI-facing motion name.
        states: State trajectory with shape ``(frames, state_dim)``.
    """

    label: str
    states: torch.Tensor


def run_contact_viewer(
    *,
    asset_name: str = "unitree_g1",
    contact_mode: str = "feet_corners",
    fps: float = 30.0,
    port: int = 8080,
    load_meshes: bool = True,
    max_frames: int | None = None,
    motion_reference: str | Path | None = None,
    synthetic_motion: bool = False,
) -> None:
    """Run a Viser URDF viewer with Adam FK contact frames overlaid.

    Args:
        asset_name: Asset alias or URDF path passed to
            :class:`HumanoidDynamics`.
        contact_mode: Contact extraction mode passed to
            :class:`HumanoidDynamics`, such as ``"feet_corners"`` or
            ``"feet_centers"``.
        fps: Playback frequency for the synthetic walking sequence.
        port: TCP port used by the local Viser server.
        load_meshes: Whether to load visual meshes from the URDF.
        max_frames: Optional number of frames to render before exiting. This
            is intended for smoke tests; ``None`` runs the viewer indefinitely.
        motion_reference: Optional path to a retargeted G1 motion ``.npz`` or
            raw ``.npy`` file. If ``None``, the bundled retargeted AMASS/G1
            sample is used unless ``synthetic_motion`` is ``True``.
        synthetic_motion: Whether to use the old deterministic walking-like
            fallback instead of a retargeted motion reference.

    Returns:
        None. The function starts a local Viser server and updates scene nodes
        until interrupted or until ``max_frames`` is reached.
    """
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

    motion_options, initial_motion_label = _load_viewer_motion_options(
        model,
        fps=fps,
        motion_reference=motion_reference,
        synthetic_motion=synthetic_motion,
    )
    states = motion_options[initial_motion_label].states

    floor_states = torch.cat([motion.states for motion in motion_options.values()], dim=0)
    floor_width, floor_height, floor_position = _floor_geometry_from_states(floor_states)

    server = viser.ViserServer(port=port)
    floor_thickness = 0.018
    server.scene.add_box(
        "/floor_plane",
        dimensions=(floor_width, floor_height, floor_thickness),
        color=(238, 241, 245),
        material="standard",
        position=(floor_position[0], floor_position[1], -0.5 * floor_thickness),
        cast_shadow=False,
        receive_shadow=True,
    )
    server.scene.add_grid(
        "/ground_grid",
        width=floor_width,
        height=floor_height,
        plane="xy",
        cell_size=0.25,
        section_size=1.0,
        cell_color=(198, 206, 216),
        section_color=(126, 140, 156),
        cell_thickness=0.8,
        section_thickness=1.2,
        shadow_opacity=0.25,
        fade_distance=max(floor_width, floor_height) * 1.5,
        fade_strength=0.35,
        position=(floor_position[0], floor_position[1], 0.002),
    )
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

    playback_lock = threading.RLock()
    playback = {
        "motion_label": initial_motion_label,
        "states": states,
        "frame": 0,
        "playing": True,
        "suppress_slider_callback": False,
    }
    max_frame_index = max(motion.states.shape[0] for motion in motion_options.values()) - 1

    with server.gui.add_folder("Animation", expand_by_default=True):
        motion_dropdown = server.gui.add_dropdown(
            "Motion",
            tuple(motion_options),
            initial_value=initial_motion_label,
            hint="Kinematic sequence to visualize.",
        )
        play_button = server.gui.add_button(
            "Play",
            hint="Start playback from the current frame.",
        )
        pause_button = server.gui.add_button(
            "Pause",
            hint="Pause playback at the current frame.",
        )
        reset_button = server.gui.add_button(
            "Reset",
            hint="Return to frame zero and pause playback.",
        )
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max_frame_index,
            step=1,
            initial_value=0,
            hint="Scrub the sequence. Scrubbing pauses playback.",
        )
        status_text = server.gui.add_text(
            "Status",
            _animation_status(
                motion_label=initial_motion_label,
                frame=0,
                num_frames=states.shape[0],
                playing=True,
            ),
            disabled=True,
        )

    def render_frame(sequence: torch.Tensor, frame_index: int) -> None:
        """Render one motion frame and its contact frames.

        Args:
            sequence: State trajectory with shape ``(frames, state_dim)``.
            frame_index: Frame index to render.

        Returns:
            None.
        """
        state = sequence[frame_index]
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
        contact_positions = contact_poses.positions.detach().cpu().numpy()
        contact_quaternions = contact_poses.quaternions_wxyz.detach().cpu().numpy()
        for handle, frame_handle, point, quat in zip(
            contact_handles,
            contact_frame_handles,
            np.asarray(contact_positions),
            np.asarray(contact_quaternions),
        ):
            handle.position = tuple(float(value) for value in point)
            frame_handle.position = tuple(float(value) for value in point)
            frame_handle.wxyz = tuple(float(value) for value in quat)

    def sync_controls() -> None:
        """Synchronize GUI widgets from the current playback state.

        Args:
            None.

        Returns:
            None.
        """
        playback["frame"] = _clamp_frame(int(playback["frame"]), playback["states"].shape[0])
        playback["suppress_slider_callback"] = True
        frame_slider.value = int(playback["frame"])
        playback["suppress_slider_callback"] = False
        status_text.value = _animation_status(
            motion_label=str(playback["motion_label"]),
            frame=int(playback["frame"]),
            num_frames=playback["states"].shape[0],
            playing=bool(playback["playing"]),
        )

    @play_button.on_click
    def _(_) -> None:
        """Start animation playback from the GUI.

        Args:
            _: Viser button event.

        Returns:
            None.
        """
        with playback_lock:
            playback["playing"] = True
            sync_controls()

    @pause_button.on_click
    def _(_) -> None:
        """Pause animation playback from the GUI.

        Args:
            _: Viser button event.

        Returns:
            None.
        """
        with playback_lock:
            playback["playing"] = False
            sync_controls()

    @reset_button.on_click
    def _(_) -> None:
        """Reset the sequence to its first frame from the GUI.

        Args:
            _: Viser button event.

        Returns:
            None.
        """
        with playback_lock:
            playback["frame"] = 0
            playback["playing"] = False
            sync_controls()
            render_frame(playback["states"], int(playback["frame"]))

    @frame_slider.on_update
    def _(_) -> None:
        """Scrub to a frame selected by the GUI slider.

        Args:
            _: Viser slider event.

        Returns:
            None.
        """
        with playback_lock:
            if playback["suppress_slider_callback"]:
                return
            playback["frame"] = _clamp_frame(int(frame_slider.value), playback["states"].shape[0])
            playback["playing"] = False
            sync_controls()
            render_frame(playback["states"], int(playback["frame"]))

    @motion_dropdown.on_update
    def _(_) -> None:
        """Switch the displayed kinematic motion from the GUI dropdown.

        Args:
            _: Viser dropdown event.

        Returns:
            None.
        """
        with playback_lock:
            label = str(motion_dropdown.value)
            playback["motion_label"] = label
            playback["states"] = motion_options[label].states
            playback["frame"] = 0
            playback["playing"] = False
            sync_controls()
            render_frame(playback["states"], int(playback["frame"]))

    frames_rendered = 0
    period = 1.0 / fps
    while True:
        with playback_lock:
            frame = _clamp_frame(int(playback["frame"]), playback["states"].shape[0])
            sequence = playback["states"]
            playing = bool(playback["playing"])
            render_frame(sequence, frame)
            playback["frame"] = (frame + 1) % sequence.shape[0] if playing else frame
            sync_controls()

        frames_rendered += 1
        if max_frames is not None and frames_rendered >= max_frames:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        time.sleep(period)


def main() -> None:
    """Parse CLI arguments and launch the contact viewer.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Visualize Unitree G1 contact-point FK.")
    parser.add_argument("--asset", default="unitree_g1")
    parser.add_argument("--contact-mode", default="feet_corners", choices=("feet_corners", "feet_centers"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-meshes", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--motion-reference", type=Path, default=None)
    parser.add_argument("--synthetic-motion", action="store_true")
    args = parser.parse_args()
    run_contact_viewer(
        asset_name=args.asset,
        contact_mode=args.contact_mode,
        fps=args.fps,
        port=args.port,
        load_meshes=not args.no_meshes,
        max_frames=args.max_frames,
        motion_reference=args.motion_reference,
        synthetic_motion=args.synthetic_motion,
    )


def _floor_geometry_from_states(
    states: torch.Tensor,
    *,
    margin: float = 1.5,
    min_width: float = 8.0,
    min_height: float = 4.0,
) -> tuple[float, float, tuple[float, float, float]]:
    """Choose a ground-plane footprint that covers a walking sequence.

    Args:
        states: Motion state tensor with shape ``(num_frames, state_dim)``.
            The first two state entries are interpreted as the floating-base
            ``x`` and ``y`` world positions.
        margin: Extra meters added to both sides of the motion extent.
        min_width: Minimum ground size in meters along world ``x``.
        min_height: Minimum ground size in meters along world ``y``.

    Returns:
        Tuple ``(width, height, position)`` where ``width`` and ``height`` are
        meter dimensions for an ``xy`` Viser grid, and ``position`` is the
        grid center with shape ``(3,)``.

    Raises:
        ValueError: If ``states`` does not have shape ``(num_frames, state_dim)``
            with at least two state entries.
    """
    if states.ndim != 2 or states.shape[-1] < 2:
        raise ValueError("Expected states with shape (num_frames, state_dim).")

    xy = states[..., :2].detach().cpu()
    lower = torch.amin(xy, dim=0)
    upper = torch.amax(xy, dim=0)
    center = 0.5 * (lower + upper)
    span = upper - lower
    width = max(min_width, float(span[0]) + 2.0 * margin)
    height = max(min_height, float(span[1]) + 2.0 * margin)
    position = (float(center[0]), float(center[1]), 0.0)
    return width, height, position


def _load_viewer_motion_options(
    model: HumanoidDynamics,
    *,
    fps: float,
    motion_reference: str | Path | None,
    synthetic_motion: bool,
) -> tuple[dict[str, _ViewerMotion], str]:
    """Load kinematic motion choices shown in the Viser GUI.

    Args:
        model: Humanoid dynamics model used to map motions into state tensors.
        fps: Frame rate used for the deterministic synthetic fallback.
        motion_reference: Optional user-specified motion path.
        synthetic_motion: Whether the synthetic fallback should be selected
            initially.

    Returns:
        Tuple ``(motion_options, initial_label)``. ``motion_options`` maps GUI
        labels to loaded motion sequences, and ``initial_label`` is one of its
        keys.
    """
    options: dict[str, _ViewerMotion] = {}

    default_motion = default_g1_motion_reference(model)
    default_label = "Fleaven JOOF walk (default)"
    options[default_label] = _ViewerMotion(default_label, default_motion.states)

    ember_motion = load_kinematic_motion_reference(
        bundled_motion_reference_path(EMBER_G1_MOTION_REFERENCE),
        model,
        source_name="CMU 06_01 retargeted AMASS for Unitree G1",
    )
    ember_label = "Ember CMU 06_01"
    options[ember_label] = _ViewerMotion(ember_label, ember_motion.states)

    synthetic_states, _ = simple_walking_sequence(model, frames=180, dt=1.0 / fps)
    synthetic_label = "Synthetic fallback"
    options[synthetic_label] = _ViewerMotion(synthetic_label, synthetic_states)

    custom_label = None
    if motion_reference is not None:
        custom_motion = load_kinematic_motion_reference(motion_reference, model)
        custom_label = f"Custom: {Path(motion_reference).name}"
        options[custom_label] = _ViewerMotion(custom_label, custom_motion.states)

    if synthetic_motion:
        initial_label = synthetic_label
    elif custom_label is not None:
        initial_label = custom_label
    else:
        initial_label = default_label
    return options, initial_label


def _clamp_frame(frame: int, num_frames: int) -> int:
    """Clamp a requested frame index to a valid sequence index.

    Args:
        frame: Requested frame index.
        num_frames: Number of frames in the active sequence.

    Returns:
        Clamped frame index in ``[0, num_frames - 1]``.

    Raises:
        ValueError: If ``num_frames`` is not positive.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    return min(max(frame, 0), num_frames - 1)


def _animation_status(
    *,
    motion_label: str,
    frame: int,
    num_frames: int,
    playing: bool,
) -> str:
    """Format playback state for the Viser GUI.

    Args:
        motion_label: Name of the active motion sequence.
        frame: Current zero-based frame index.
        num_frames: Number of frames in the active sequence.
        playing: Whether playback is currently active.

    Returns:
        Human-readable status string.
    """
    state = "Playing" if playing else "Paused"
    return f"{state} | {motion_label} | frame {frame + 1}/{num_frames}"
