from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
from typing import Literal

import numpy as np
import torch

from .contacts import BasicContactForceResolver, ContactPoses, FlatTerrainContactDetector
from .dynamics import FloatingBaseDynamics, SplitState
from .motion import (
    EMBER_G1_MOTION_REFERENCE,
    bundled_motion_reference_path,
    default_g1_motion_reference,
    load_kinematic_motion_reference,
)
from .motion_derivatives import MotionDerivativeEstimate, estimate_motion_derivatives
from .walking import simple_walking_sequence


@dataclass(frozen=True)
class _ViewerMotion:
    """Kinematic motion sequence available in the viewer.

    Attributes:
        label: GUI-facing motion name.
        states: State trajectory with shape ``(frames, state_dim)``.
        times: Time vector with shape ``(frames,)``.
    """

    label: str
    states: torch.Tensor
    times: torch.Tensor


class KinematicTrajectoryViewer:
    """Viser viewer for a floating-base kinematic trajectory."""

    def __init__(
        self,
        *,
        asset_name: str = "unitree_g1",
        contact_mode: str = "feet_corners",
        contact_force_frame: Literal["world", "contact"] = "world",
        fps: float = 30.0,
        port: int = 8080,
        load_meshes: bool = True,
        robot_opacity: float = 1.0,
        max_frames: int | None = None,
        motion_reference: str | Path | None = None,
        synthetic_motion: bool = False,
    ) -> None:
        """Initialize the trajectory viewer.

        Args:
            asset_name: Asset alias or URDF path passed to
                :class:`FloatingBaseDynamics`.
            contact_mode: Contact extraction mode such as ``"feet_corners"``
                or ``"feet_centers"``.
            contact_force_frame: Contact-force coordinate frame used by the
                underlying dynamics model.
            fps: Playback frequency.
            port: TCP port used by the Viser server.
            load_meshes: Whether to load visual meshes from the URDF.
            robot_opacity: Visual mesh opacity in ``[0, 1]``. Values below
                one render the robot with a neutral transparent material.
            max_frames: Optional frame limit for smoke tests. ``None`` runs
                indefinitely.
            motion_reference: Optional custom motion reference.
            synthetic_motion: Whether to select the deterministic synthetic
                fallback initially.

        Returns:
            None.
        """
        self.asset_name = asset_name
        self.contact_mode = contact_mode
        self.contact_force_frame = contact_force_frame
        self.fps = float(fps)
        self.port = int(port)
        self.load_meshes = bool(load_meshes)
        self.robot_opacity = float(np.clip(robot_opacity, 0.0, 1.0))
        self.max_frames = max_frames
        self.motion_reference = motion_reference
        self.synthetic_motion = synthetic_motion

        self.model = FloatingBaseDynamics(
            asset_name,
            include_contact_forces=True,
            contact_mode=contact_mode,
            contact_force_frame=contact_force_frame,
            dtype=torch.float64,
        )
        if self.model.contact_model is None:
            raise RuntimeError("Contact model was not initialized.")

        self.motion_options, self.initial_motion_label = _load_viewer_motion_options(
            self.model,
            fps=fps,
            motion_reference=motion_reference,
            synthetic_motion=synthetic_motion,
        )

        self.server: Any | None = None
        self.robot_root: Any | None = None
        self.urdf_vis: Any | None = None
        self.contact_handles: list[Any] = []
        self.contact_frame_handles: list[Any] = []
        self.playback_lock = threading.RLock()
        self.playback: dict[str, Any] = {
            "motion_label": self.initial_motion_label,
            "motion": self.motion_options[self.initial_motion_label],
            "frame": 0,
            "playing": True,
            "suppress_slider_callback": False,
        }
        self.frame_slider: Any | None = None
        self.status_text: Any | None = None

    def run(self) -> None:
        """Run the Viser viewer until interrupted or ``max_frames`` is reached."""
        try:
            import viser
            from viser.extras import ViserUrdf
        except ImportError as exc:
            raise ImportError("Install visualization extras with `uv sync --extra viz`.") from exc

        self.server = viser.ViserServer(port=self.port)
        self._setup_scene(ViserUrdf)
        self._setup_gui()
        self._render_frame(self.active_motion, int(self.playback["frame"]))

        frames_rendered = 0
        period = 1.0 / self.fps
        while True:
            with self.playback_lock:
                motion = self.active_motion
                frame = _clamp_frame(int(self.playback["frame"]), motion.states.shape[0])
                playing = bool(self.playback["playing"])
                self._render_frame(motion, frame)
                self.playback["frame"] = (frame + 1) % motion.states.shape[0] if playing else frame
                self._sync_controls()

            frames_rendered += 1
            if self.max_frames is not None and frames_rendered >= self.max_frames:
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)
            time.sleep(period)

    @property
    def active_motion(self) -> _ViewerMotion:
        """Return the currently selected motion."""
        return self.playback["motion"]

    def _setup_scene(self, ViserUrdf) -> None:
        """Create static scene objects and robot/contact handles."""
        assert self.server is not None
        floor_states = torch.cat([motion.states for motion in self.motion_options.values()], dim=0)
        floor_width, floor_height, floor_position = _floor_geometry_from_states(floor_states)
        floor_thickness = 0.018
        self.server.scene.add_box(
            "/floor_plane",
            dimensions=(floor_width, floor_height, floor_thickness),
            color=(238, 241, 245),
            material="standard",
            position=(floor_position[0], floor_position[1], -0.5 * floor_thickness),
            cast_shadow=False,
            receive_shadow=True,
        )
        self.server.scene.add_grid(
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
        self.robot_root = self.server.scene.add_frame("/robot", show_axes=False)
        self.urdf_vis = ViserUrdf(
            self.server,
            urdf_or_path=Path(self.model.asset.urdf_path),
            root_node_name="/robot",
            mesh_color_override=self._mesh_color_override(),
            load_meshes=self.load_meshes,
            load_collision_meshes=False,
        )
        self.contact_handles = [
            self.server.scene.add_icosphere(
                f"/contacts/{name.replace(':', '_')}",
                radius=0.018,
                color=(255, 80, 20),
            )
            for name in self.model.contact_model.contact_names
        ]
        self.contact_frame_handles = [
            self.server.scene.add_frame(
                f"/contact_frames/{name.replace(':', '_')}",
                axes_length=0.06,
                axes_radius=0.004,
            )
            for name in self.model.contact_model.contact_names
        ]
        self._setup_extra_scene()

    def _setup_extra_scene(self) -> None:
        """Subclass hook for additional scene handles."""

    def _mesh_color_override(self) -> tuple[int, int, int, float] | None:
        """Return the optional Viser URDF mesh color override."""
        if self.robot_opacity >= 0.999:
            return None
        return (190, 198, 208, self.robot_opacity)

    def _setup_gui(self) -> None:
        """Create animation controls and subclass GUI controls."""
        assert self.server is not None
        max_frame_index = max(motion.states.shape[0] for motion in self.motion_options.values()) - 1
        with self.server.gui.add_folder("Animation", expand_by_default=True):
            motion_dropdown = self.server.gui.add_dropdown(
                "Motion",
                tuple(self.motion_options),
                initial_value=self.initial_motion_label,
                hint="Kinematic sequence to visualize.",
            )
            play_button = self.server.gui.add_button("Play", hint="Start playback from the current frame.")
            pause_button = self.server.gui.add_button("Pause", hint="Pause playback at the current frame.")
            reset_button = self.server.gui.add_button(
                "Reset",
                hint="Return to frame zero and pause playback.",
            )
            self.frame_slider = self.server.gui.add_slider(
                "Frame",
                min=0,
                max=max_frame_index,
                step=1,
                initial_value=0,
                hint="Scrub the sequence. Scrubbing pauses playback.",
            )
            self.status_text = self.server.gui.add_text(
                "Status",
                _animation_status(
                    motion_label=self.initial_motion_label,
                    frame=0,
                    num_frames=self.active_motion.states.shape[0],
                    playing=True,
                ),
                disabled=True,
            )

        @play_button.on_click
        def _(_) -> None:
            with self.playback_lock:
                self.playback["playing"] = True
                self._sync_controls()

        @pause_button.on_click
        def _(_) -> None:
            with self.playback_lock:
                self.playback["playing"] = False
                self._sync_controls()

        @reset_button.on_click
        def _(_) -> None:
            with self.playback_lock:
                self.playback["frame"] = 0
                self.playback["playing"] = False
                self._sync_controls()
                self._render_frame(self.active_motion, int(self.playback["frame"]))

        @self.frame_slider.on_update
        def _(_) -> None:
            with self.playback_lock:
                if self.playback["suppress_slider_callback"]:
                    return
                frame = _clamp_frame(int(self.frame_slider.value), self.active_motion.states.shape[0])
                self.playback["frame"] = frame
                self.playback["playing"] = False
                self._sync_controls()
                self._render_frame(self.active_motion, frame)

        @motion_dropdown.on_update
        def _(_) -> None:
            with self.playback_lock:
                label = str(motion_dropdown.value)
                self.playback["motion_label"] = label
                self.playback["motion"] = self.motion_options[label]
                self.playback["frame"] = 0
                self.playback["playing"] = False
                self._on_motion_changed(self.active_motion)
                self._sync_controls()
                self._render_frame(self.active_motion, int(self.playback["frame"]))

        self._setup_extra_gui()

    def _setup_extra_gui(self) -> None:
        """Subclass hook for additional GUI controls."""

    def _on_motion_changed(self, motion: _ViewerMotion) -> None:
        """Subclass hook called after the active motion changes."""

    def _sync_controls(self) -> None:
        """Synchronize GUI widgets from the current playback state."""
        assert self.frame_slider is not None
        assert self.status_text is not None
        motion = self.active_motion
        self.playback["frame"] = _clamp_frame(int(self.playback["frame"]), motion.states.shape[0])
        self.playback["suppress_slider_callback"] = True
        self.frame_slider.value = int(self.playback["frame"])
        self.playback["suppress_slider_callback"] = False
        self.status_text.value = _animation_status(
            motion_label=str(self.playback["motion_label"]),
            frame=int(self.playback["frame"]),
            num_frames=motion.states.shape[0],
            playing=bool(self.playback["playing"]),
        )

    def _state_for_render(self, motion: _ViewerMotion, frame_index: int) -> torch.Tensor:
        """Return the state rendered for a motion/frame pair."""
        return motion.states[frame_index]

    def _render_frame(self, motion: _ViewerMotion, frame_index: int) -> None:
        """Render one motion frame and its overlays."""
        assert self.robot_root is not None
        assert self.urdf_vis is not None
        state = self._state_for_render(motion, frame_index)
        split = self.model.split_state(state)
        joint_positions = split.joint_positions.squeeze(0).detach().cpu().numpy()
        base_position = split.base_position.squeeze(0).detach().cpu().numpy()
        base_quaternion = split.base_quaternion_wxyz.squeeze(0).detach().cpu().numpy()
        self.urdf_vis.update_cfg(joint_positions)
        self.robot_root.position = tuple(float(value) for value in base_position)
        self.robot_root.wxyz = tuple(float(value) for value in base_quaternion)

        base_transform = self.model.base_transform(state)
        contact_poses = self.model.contact_model.contact_poses(
            base_transform,
            split.joint_positions.squeeze(0),
        )
        contact_positions = contact_poses.positions.detach().cpu().numpy()
        contact_quaternions = contact_poses.quaternions_wxyz.detach().cpu().numpy()
        for handle, frame_handle, point, quat in zip(
            self.contact_handles,
            self.contact_frame_handles,
            np.asarray(contact_positions),
            np.asarray(contact_quaternions),
        ):
            handle.position = tuple(float(value) for value in point)
            frame_handle.position = tuple(float(value) for value in point)
            frame_handle.wxyz = tuple(float(value) for value in quat)

        self._render_extra_frame(motion, frame_index, state, split, contact_poses)

    def _render_extra_frame(
        self,
        motion: _ViewerMotion,
        frame_index: int,
        state: torch.Tensor,
        split: SplitState,
        contact_poses: ContactPoses,
    ) -> None:
        """Subclass hook for per-frame overlays."""


class DynamicsVerificationViewer(KinematicTrajectoryViewer):
    """Viser viewer for intuitive ``f(x)`` and ``g(x)`` verification."""

    F_MODE = "f(x): inverse dynamics"
    G_MODE = "g(x): contact projection"
    JOINT_TORQUE_ARROWS = "arrows"
    JOINT_TORQUE_LABELS = "labels"
    JOINT_TORQUE_ARROWS_AND_LABELS = "arrows + labels"

    def __init__(
        self,
        *,
        whittaker_lmbda: float = 100.0,
        whittaker_d_order: int = 2,
        contact_threshold: float = 0.025,
        top_joint_count: int = 8,
        robot_opacity: float = 0.35,
        **kwargs,
    ) -> None:
        """Initialize the dynamics verification viewer.

        Args:
            whittaker_lmbda: Initial Whittaker-Eilers smoothing parameter.
            whittaker_d_order: Whittaker difference penalty order.
            contact_threshold: Flat-ground contact distance threshold in
                meters.
            top_joint_count: Number of largest joint torques shown.
            robot_opacity: Visual mesh opacity. Dynamics verification defaults
                to a transparent robot so force and torque vectors are easier
                to inspect.
            **kwargs: Arguments forwarded to
                :class:`KinematicTrajectoryViewer`.

        Returns:
            None.
        """
        super().__init__(robot_opacity=robot_opacity, **kwargs)
        self.whittaker_lmbda = float(whittaker_lmbda)
        self.whittaker_d_order = int(whittaker_d_order)
        self.top_joint_count = int(top_joint_count)
        self.dynamics_mode = self.F_MODE
        self.force_scale = 0.001
        self.torque_scale = 0.01
        self.joint_torque_scale = 0.01
        self.joint_torque_viz = self.JOINT_TORQUE_ARROWS
        self.contact_detector = FlatTerrainContactDetector(
            contact_threshold=contact_threshold,
            dtype=self.model.dtype,
            device=self.model.device,
        )
        self.contact_resolver = BasicContactForceResolver(force_frame=self.contact_force_frame)
        self._derivative_cache: dict[tuple[str, float, int], MotionDerivativeEstimate] = {}
        self.root_force_handle: Any | None = None
        self.root_torque_handle: Any | None = None
        self.contact_force_handles: list[Any] = []
        self.joint_torque_positive_handles: list[Any] = []
        self.joint_torque_negative_handles: list[Any] = []
        self.joint_torque_positive_tip_handles: list[Any] = []
        self.joint_torque_negative_tip_handles: list[Any] = []
        self.joint_torque_labels: list[Any | None] = []
        self.dynamics_status_text: Any | None = None

    def _setup_extra_scene(self) -> None:
        """Create dynamics vector and torque handles."""
        assert self.server is not None
        self.root_force_handle = self.server.scene.add_cylinder(
            "/dynamics/root_force",
            radius=0.015,
            height=1.0,
            color=(30, 120, 255),
            visible=False,
        )
        self.root_torque_handle = self.server.scene.add_cylinder(
            "/dynamics/root_torque",
            radius=0.012,
            height=1.0,
            color=(180, 80, 210),
            visible=False,
        )
        self.contact_force_handles = [
            self.server.scene.add_cylinder(
                f"/dynamics/contact_force/{name.replace(':', '_')}",
                radius=0.01,
                height=1.0,
                color=(20, 170, 90),
                visible=False,
            )
            for name in self.model.contact_model.contact_names
        ]
        for index in range(self.top_joint_count):
            self.joint_torque_positive_handles.append(
                self.server.scene.add_cylinder(
                    f"/dynamics/joint_torque/positive_{index}",
                    radius=0.011,
                    height=1.0,
                    color=(220, 70, 60),
                    visible=False,
                )
            )
            self.joint_torque_negative_handles.append(
                self.server.scene.add_cylinder(
                    f"/dynamics/joint_torque/negative_{index}",
                    radius=0.011,
                    height=1.0,
                    color=(55, 105, 215),
                    visible=False,
                )
            )
            self.joint_torque_positive_tip_handles.append(
                self.server.scene.add_icosphere(
                    f"/dynamics/joint_torque/positive_tip_{index}",
                    radius=0.022,
                    color=(220, 70, 60),
                    visible=False,
                )
            )
            self.joint_torque_negative_tip_handles.append(
                self.server.scene.add_icosphere(
                    f"/dynamics/joint_torque/negative_tip_{index}",
                    radius=0.022,
                    color=(55, 105, 215),
                    visible=False,
                )
            )
            self.joint_torque_labels.append(None)

    def _setup_extra_gui(self) -> None:
        """Create dynamics verification GUI controls."""
        assert self.server is not None
        with self.server.gui.add_folder("Dynamics Verification", expand_by_default=True):
            mode_dropdown = self.server.gui.add_dropdown(
                "Mode",
                (self.F_MODE, self.G_MODE),
                initial_value=self.dynamics_mode,
                hint="Choose whether to visualize inverse dynamics or contact-force projection.",
            )
            joint_viz_dropdown = self.server.gui.add_dropdown(
                "Joint torque viz",
                (
                    self.JOINT_TORQUE_ARROWS,
                    self.JOINT_TORQUE_LABELS,
                    self.JOINT_TORQUE_ARROWS_AND_LABELS,
                ),
                initial_value=self.joint_torque_viz,
                hint="Choose how the largest joint torques are drawn.",
            )
            lambda_number = self.server.gui.add_number(
                "Whittaker lambda",
                self.whittaker_lmbda,
                min=0.0,
                step=10.0,
                hint="Higher values smooth the motion derivatives more strongly.",
            )
            force_scale_number = self.server.gui.add_number(
                "Force scale",
                self.force_scale,
                min=0.0,
                step=0.0005,
                hint="Meters of vector length per Newton.",
            )
            torque_scale_number = self.server.gui.add_number(
                "Torque scale",
                self.torque_scale,
                min=0.0,
                step=0.002,
                hint="Meters of vector length per Newton-meter.",
            )
            joint_scale_number = self.server.gui.add_number(
                "Joint scale",
                self.joint_torque_scale,
                min=0.0,
                step=0.002,
                hint="Meters of joint torque arrow length per Newton-meter.",
            )
            self.dynamics_status_text = self.server.gui.add_text(
                "Dynamics",
                "",
                disabled=True,
            )

        @mode_dropdown.on_update
        def _(_) -> None:
            with self.playback_lock:
                self.dynamics_mode = str(mode_dropdown.value)
                self._render_frame(self.active_motion, int(self.playback["frame"]))

        @joint_viz_dropdown.on_update
        def _(_) -> None:
            with self.playback_lock:
                self.joint_torque_viz = str(joint_viz_dropdown.value)
                self._render_frame(self.active_motion, int(self.playback["frame"]))

        @lambda_number.on_update
        def _(_) -> None:
            with self.playback_lock:
                self.whittaker_lmbda = float(lambda_number.value)
                self._render_frame(self.active_motion, int(self.playback["frame"]))

        @force_scale_number.on_update
        def _(_) -> None:
            with self.playback_lock:
                self.force_scale = float(force_scale_number.value)
                self._render_frame(self.active_motion, int(self.playback["frame"]))

        @torque_scale_number.on_update
        def _(_) -> None:
            with self.playback_lock:
                self.torque_scale = float(torque_scale_number.value)
                self._render_frame(self.active_motion, int(self.playback["frame"]))

        @joint_scale_number.on_update
        def _(_) -> None:
            with self.playback_lock:
                self.joint_torque_scale = float(joint_scale_number.value)
                self._render_frame(self.active_motion, int(self.playback["frame"]))

    def _state_for_render(self, motion: _ViewerMotion, frame_index: int) -> torch.Tensor:
        """Render the Whittaker-smoothed state in dynamics verification mode."""
        estimate = self._estimate_for_motion(motion)
        return estimate.states[frame_index]

    def _render_extra_frame(
        self,
        motion: _ViewerMotion,
        frame_index: int,
        state: torch.Tensor,
        split: SplitState,
        contact_poses: ContactPoses,
    ) -> None:
        """Render inverse-dynamics or contact-projection overlays."""
        estimate = self._estimate_for_motion(motion)
        generalized_acceleration = estimate.generalized_accelerations[frame_index]
        if self.dynamics_mode == self.F_MODE:
            generalized_force = self.model.generalized_forces_from_acceleration(
                state,
                generalized_acceleration,
            )
            self._hide_contact_forces()
            contact_summary = "contacts ignored"
        else:
            generalized_force, contact_summary = self._contact_projected_generalized_force(
                state,
                split,
                contact_poses,
                generalized_acceleration,
            )

        self._update_root_wrench(state, generalized_force[:6])
        self._update_joint_torques(state, generalized_force[6:])
        self._update_status(generalized_force, contact_summary)

    def _estimate_for_motion(self, motion: _ViewerMotion) -> MotionDerivativeEstimate:
        """Return cached Whittaker derivatives for a motion."""
        key = (motion.label, float(self.whittaker_lmbda), int(self.whittaker_d_order))
        if key not in self._derivative_cache:
            self._derivative_cache[key] = estimate_motion_derivatives(
                self.model,
                motion.states,
                motion.times,
                lmbda=self.whittaker_lmbda,
                d_order=self.whittaker_d_order,
            )
        return self._derivative_cache[key]

    def _contact_projected_generalized_force(
        self,
        state: torch.Tensor,
        split: SplitState,
        contact_poses: ContactPoses,
        generalized_acceleration: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        """Resolve contact forces and project them through ``B(q)``."""
        assert self.model.contact_model is not None
        contact_state = self.contact_detector.detect(contact_poses.positions)
        contact_velocities = self._contact_velocities(state, split)
        desired_generalized_force = self.model.generalized_forces_from_acceleration(
            state,
            generalized_acceleration,
        )
        support_force = torch.clamp(
            torch.dot(
                desired_generalized_force[:3],
                self.contact_detector.normal.to(
                    dtype=desired_generalized_force.dtype,
                    device=desired_generalized_force.device,
                ),
            ),
            min=0.0,
        )
        resolved = self.contact_resolver.resolve(
            contact_state,
            contact_velocities=contact_velocities,
            total_normal_force=support_force,
            contact_poses=contact_poses,
        )
        control_input = torch.zeros(self.model.input_dim, dtype=self.model.dtype, device=self.model.device)
        control_input[self.model.n_joints :] = resolved.forces.reshape(-1)
        contact_state_derivative = self.model.g(state).matmul(control_input)
        contact_acceleration = contact_state_derivative[self.model.nq :]
        mass = self.model.dynamics_terms(state).mass_matrix
        generalized_force = torch.matmul(mass, contact_acceleration.unsqueeze(-1)).squeeze(-1)
        self._update_contact_forces(contact_poses.positions, resolved.world_forces)
        active_count = int(torch.count_nonzero(resolved.active).item())
        total_normal = float(torch.sum(resolved.normal_forces).detach().cpu())
        return generalized_force, f"{active_count} contacts | normal {total_normal:.1f} N"

    def _contact_velocities(self, state: torch.Tensor, split: SplitState) -> torch.Tensor:
        """Compute world-frame contact point velocities."""
        assert self.model.contact_model is not None
        jacobian = self.model.contact_model.contact_jacobian(
            self.model.base_transform(state),
            split.joint_positions.squeeze(0),
        )
        velocity = state[self.model.nq :]
        return torch.matmul(jacobian, velocity).reshape(self.model.contact_model.num_contacts, 3)

    def _update_root_wrench(self, state: torch.Tensor, wrench: torch.Tensor) -> None:
        """Update root force and moment vectors."""
        root = state[:3].detach().cpu().numpy()
        force_vector = wrench[:3].detach().cpu().numpy() * self.force_scale
        torque_vector = wrench[3:6].detach().cpu().numpy() * self.torque_scale
        _update_vector_cylinder(self.root_force_handle, root, force_vector)
        _update_vector_cylinder(self.root_torque_handle, root + np.array([0.0, 0.0, 0.08]), torque_vector)

    def _update_contact_forces(self, positions: torch.Tensor, world_forces: torch.Tensor) -> None:
        """Update contact force vectors."""
        points = positions.detach().cpu().numpy()
        forces = world_forces.detach().cpu().numpy()
        for handle, point, force in zip(self.contact_force_handles, points, forces):
            _update_vector_cylinder(handle, point, force * self.force_scale)

    def _hide_contact_forces(self) -> None:
        """Hide contact force vectors."""
        for handle in self.contact_force_handles:
            handle.visible = False

    def _update_joint_torques(self, state: torch.Tensor, joint_torques: torch.Tensor) -> None:
        """Update top joint torque arrows and labels."""
        count = min(self.top_joint_count, joint_torques.numel())
        if count <= 0:
            return
        magnitudes, indices = torch.topk(torch.abs(joint_torques), k=count)
        positions, axes = self._joint_child_positions_and_axes(state, indices.tolist())
        show_arrows = self.joint_torque_viz in {
            self.JOINT_TORQUE_ARROWS,
            self.JOINT_TORQUE_ARROWS_AND_LABELS,
        }
        show_labels = self.joint_torque_viz in {
            self.JOINT_TORQUE_LABELS,
            self.JOINT_TORQUE_ARROWS_AND_LABELS,
        }
        for rank in range(self.top_joint_count):
            positive = self.joint_torque_positive_handles[rank]
            negative = self.joint_torque_negative_handles[rank]
            positive_tip = self.joint_torque_positive_tip_handles[rank]
            negative_tip = self.joint_torque_negative_tip_handles[rank]
            if rank >= count or float(magnitudes[rank]) <= 1e-9:
                self._hide_joint_torque_overlay(rank)
                continue
            joint_index = int(indices[rank])
            torque = float(joint_torques[joint_index].detach().cpu())
            origin = positions[rank]
            vector = axes[rank] * torque * self.joint_torque_scale
            if show_arrows:
                _update_vector_cylinder(positive if torque >= 0.0 else negative, origin, vector)
                active_tip = positive_tip if torque >= 0.0 else negative_tip
                inactive_tip = negative_tip if torque >= 0.0 else positive_tip
                if np.linalg.norm(vector) > 1e-9:
                    active_tip.position = tuple(float(value) for value in origin + vector)
                    active_tip.visible = True
                else:
                    active_tip.visible = False
                inactive_tip.visible = False
            else:
                positive.visible = False
                negative.visible = False
                positive_tip.visible = False
                negative_tip.visible = False
            (negative if torque >= 0.0 else positive).visible = False

            if show_labels:
                self._replace_joint_torque_label(
                    rank,
                    f"{self.model.joint_names[joint_index]} {torque:.1f} Nm",
                    origin + np.array([0.0, 0.0, 0.08]),
                )
            else:
                self._remove_joint_torque_label(rank)

        for rank in range(count, self.top_joint_count):
            self._hide_joint_torque_overlay(rank)

    def _hide_joint_torque_overlay(self, rank: int) -> None:
        """Hide one ranked joint torque overlay."""
        self.joint_torque_positive_handles[rank].visible = False
        self.joint_torque_negative_handles[rank].visible = False
        self.joint_torque_positive_tip_handles[rank].visible = False
        self.joint_torque_negative_tip_handles[rank].visible = False
        self._remove_joint_torque_label(rank)

    def _replace_joint_torque_label(self, rank: int, text: str, position: np.ndarray) -> None:
        """Replace a Viser label because label text is immutable after creation."""
        assert self.server is not None
        self._remove_joint_torque_label(rank)
        self.joint_torque_labels[rank] = self.server.scene.add_label(
            f"/dynamics/joint_torque/label_{rank}",
            text,
            position=tuple(float(value) for value in position),
            font_screen_scale=0.75,
            anchor="bottom-center",
        )

    def _remove_joint_torque_label(self, rank: int) -> None:
        """Remove an existing joint torque label handle."""
        label = self.joint_torque_labels[rank]
        if label is not None:
            label.remove()
            self.joint_torque_labels[rank] = None

    def _joint_child_positions_and_axes(
        self,
        state: torch.Tensor,
        joint_indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return child-link positions and world joint axes for selected joints."""
        split = self.model.split_state(state)
        base_transform = self.model.base_transform(state)
        positions = []
        axes = []
        for joint_index in joint_indices:
            child_link = self.model.asset.urdf.joints[joint_index].child
            transform = self.model.kindyn.forward_kinematics(
                child_link,
                base_transform,
                split.joint_positions.squeeze(0),
            )
            positions.append(transform[:3, 3].detach().cpu().numpy())
            jacobian = self.model.kindyn.jacobian(
                child_link,
                base_transform,
                split.joint_positions.squeeze(0),
            )
            axis = jacobian[3:6, 6 + joint_index]
            if torch.linalg.norm(axis) <= 1e-9:
                axis = jacobian[:3, 6 + joint_index]
            axis = axis / torch.clamp(torch.linalg.norm(axis), min=1e-9)
            axes.append(axis.detach().cpu().numpy())
        return np.asarray(positions), np.asarray(axes)

    def _update_status(self, generalized_force: torch.Tensor, contact_summary: str) -> None:
        """Update dynamics status text."""
        if self.dynamics_status_text is None:
            return
        root_force_norm = float(torch.linalg.norm(generalized_force[:3]).detach().cpu())
        root_torque_norm = float(torch.linalg.norm(generalized_force[3:6]).detach().cpu())
        max_joint = float(torch.max(torch.abs(generalized_force[6:])).detach().cpu())
        self.dynamics_status_text.value = (
            f"{self.dynamics_mode} | root F {root_force_norm:.1f} N | "
            f"root M {root_torque_norm:.1f} Nm | max joint {max_joint:.1f} Nm | {contact_summary}"
        )


def run_contact_viewer(
    *,
    asset_name: str = "unitree_g1",
    contact_mode: str = "feet_corners",
    fps: float = 30.0,
    port: int = 8080,
    load_meshes: bool = True,
    robot_opacity: float = 1.0,
    max_frames: int | None = None,
    motion_reference: str | Path | None = None,
    synthetic_motion: bool = False,
) -> None:
    """Run a Viser URDF viewer with Adam FK contact frames overlaid.

    Args:
        asset_name: Asset alias or URDF path passed to
            :class:`FloatingBaseDynamics`.
        contact_mode: Contact extraction mode passed to
            :class:`FloatingBaseDynamics`, such as ``"feet_corners"`` or
            ``"feet_centers"``.
        fps: Playback frequency.
        port: TCP port used by the local Viser server.
        load_meshes: Whether to load visual meshes from the URDF.
        robot_opacity: Visual mesh opacity in ``[0, 1]``.
        max_frames: Optional number of frames to render before exiting.
        motion_reference: Optional path to a retargeted G1 motion.
        synthetic_motion: Whether to use the deterministic walking fallback.

    Returns:
        None.
    """
    KinematicTrajectoryViewer(
        asset_name=asset_name,
        contact_mode=contact_mode,
        fps=fps,
        port=port,
        load_meshes=load_meshes,
        robot_opacity=robot_opacity,
        max_frames=max_frames,
        motion_reference=motion_reference,
        synthetic_motion=synthetic_motion,
    ).run()


def run_dynamics_verification_viewer(
    *,
    asset_name: str = "unitree_g1",
    contact_mode: str = "feet_corners",
    contact_force_frame: Literal["world", "contact"] = "world",
    fps: float = 30.0,
    port: int = 8080,
    load_meshes: bool = True,
    robot_opacity: float = 0.35,
    max_frames: int | None = None,
    motion_reference: str | Path | None = None,
    synthetic_motion: bool = False,
    whittaker_lmbda: float = 100.0,
    whittaker_d_order: int = 2,
    contact_threshold: float = 0.025,
    top_joint_count: int = 8,
) -> None:
    """Run the specialized Viser viewer for ``f(x)`` and ``g(x)`` checks."""
    DynamicsVerificationViewer(
        asset_name=asset_name,
        contact_mode=contact_mode,
        contact_force_frame=contact_force_frame,
        fps=fps,
        port=port,
        load_meshes=load_meshes,
        robot_opacity=robot_opacity,
        max_frames=max_frames,
        motion_reference=motion_reference,
        synthetic_motion=synthetic_motion,
        whittaker_lmbda=whittaker_lmbda,
        whittaker_d_order=whittaker_d_order,
        contact_threshold=contact_threshold,
        top_joint_count=top_joint_count,
    ).run()


def main() -> None:
    """Parse CLI arguments and launch a Viser viewer."""
    parser = argparse.ArgumentParser(description="Visualize Unitree G1 trajectories and dynamics.")
    parser.add_argument("--asset", default="unitree_g1")
    parser.add_argument("--contact-mode", default="feet_corners", choices=("feet_corners", "feet_centers"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-meshes", action="store_true")
    parser.add_argument("--robot-opacity", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--motion-reference", type=Path, default=None)
    parser.add_argument("--synthetic-motion", action="store_true")
    parser.add_argument("--dynamics-verification", action="store_true")
    parser.add_argument("--contact-force-frame", choices=("world", "contact"), default="world")
    parser.add_argument("--whittaker-lambda", type=float, default=100.0)
    parser.add_argument("--whittaker-d-order", type=int, default=2)
    parser.add_argument("--contact-threshold", type=float, default=0.025)
    parser.add_argument("--top-joints", type=int, default=8)
    args = parser.parse_args()

    robot_opacity = args.robot_opacity
    if robot_opacity is None:
        robot_opacity = 0.35 if args.dynamics_verification else 1.0
    kwargs = dict(
        asset_name=args.asset,
        contact_mode=args.contact_mode,
        fps=args.fps,
        port=args.port,
        load_meshes=not args.no_meshes,
        robot_opacity=robot_opacity,
        max_frames=args.max_frames,
        motion_reference=args.motion_reference,
        synthetic_motion=args.synthetic_motion,
    )
    if args.dynamics_verification:
        run_dynamics_verification_viewer(
            **kwargs,
            contact_force_frame=args.contact_force_frame,
            whittaker_lmbda=args.whittaker_lambda,
            whittaker_d_order=args.whittaker_d_order,
            contact_threshold=args.contact_threshold,
            top_joint_count=args.top_joints,
        )
    else:
        run_contact_viewer(**kwargs)


def dynamics_main() -> None:
    """Launch the dynamics verification viewer from its dedicated CLI."""
    sys.argv.insert(1, "--dynamics-verification")
    main()


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
    model: FloatingBaseDynamics,
    *,
    fps: float,
    motion_reference: str | Path | None,
    synthetic_motion: bool,
) -> tuple[dict[str, _ViewerMotion], str]:
    """Load kinematic motion choices shown in the Viser GUI."""
    options: dict[str, _ViewerMotion] = {}

    default_motion = default_g1_motion_reference(model)
    default_label = "Fleaven JOOF walk (default)"
    options[default_label] = _ViewerMotion(default_label, default_motion.states, default_motion.times)

    ember_motion = load_kinematic_motion_reference(
        bundled_motion_reference_path(EMBER_G1_MOTION_REFERENCE),
        model,
        source_name="CMU 06_01 retargeted AMASS for Unitree G1",
    )
    ember_label = "Ember CMU 06_01"
    options[ember_label] = _ViewerMotion(ember_label, ember_motion.states, ember_motion.times)

    synthetic_states, synthetic_times = simple_walking_sequence(model, frames=180, dt=1.0 / fps)
    synthetic_label = "Synthetic fallback"
    options[synthetic_label] = _ViewerMotion(synthetic_label, synthetic_states, synthetic_times)

    custom_label = None
    if motion_reference is not None:
        custom_motion = load_kinematic_motion_reference(motion_reference, model)
        custom_label = f"Custom: {Path(motion_reference).name}"
        options[custom_label] = _ViewerMotion(custom_label, custom_motion.states, custom_motion.times)

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
    """Format playback state for the Viser GUI."""
    state = "Playing" if playing else "Paused"
    return f"{state} | {motion_label} | frame {frame + 1}/{num_frames}"


def _update_vector_cylinder(handle: Any | None, origin: np.ndarray, vector: np.ndarray) -> None:
    """Update a Viser cylinder so it represents a vector."""
    if handle is None:
        return
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 1e-9:
        handle.visible = False
        return
    midpoint = np.asarray(origin, dtype=np.float64) + 0.5 * np.asarray(vector, dtype=np.float64)
    handle.position = tuple(float(value) for value in midpoint)
    handle.wxyz = _wxyz_from_z_axis(vector)
    handle.scale = (1.0, 1.0, length)
    handle.visible = True


def _wxyz_from_z_axis(vector: np.ndarray) -> tuple[float, float, float, float]:
    """Return a quaternion rotating the local z-axis onto ``vector``."""
    direction = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    direction = direction / norm
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if dot > 0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    if dot < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)
    axis = np.cross(z_axis, direction)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    half = 0.5 * angle
    sin_half = np.sin(half)
    return (
        float(np.cos(half)),
        float(axis[0] * sin_half),
        float(axis[1] * sin_half),
        float(axis[2] * sin_half),
    )
