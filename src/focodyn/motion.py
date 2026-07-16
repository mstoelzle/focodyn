from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import torch

from .dynamics import FloatingBaseDynamics
from .joint_conventions import (
    UNITREE_G1_29DOF_JOINT_NAMES,
    is_g1_37dof_minimal_joint_set,
    map_g1_29dof_to_37dof,
)


DEFAULT_G1_MOTION_REFERENCE = "g1_fleaven_retargeted/JOOF_walk_poses_120_jpos.npy"
EMBER_G1_MOTION_REFERENCE = "g1_amass_retargeted/cmu_06_01_poses_120_jpos.npz"

@dataclass(frozen=True)
class KinematicMotionReference:
    """Kinematic state trajectory for a floating-base robot model.

    Attributes:
        states: State tensor with shape ``(frames, model.state_dim)`` using
            the model convention ``(p_WB, quat_WB, s, v_WB, omega_WB, s_dot)``.
        times: Time vector with shape ``(frames,)`` in seconds.
        fps: Frame rate in Hz.
        source_path: File path from which the motion was loaded.
        source_name: Human-readable source label.
    """

    states: torch.Tensor
    times: torch.Tensor
    fps: float
    source_path: Path
    source_name: str


def bundled_motion_reference_path(name: str = DEFAULT_G1_MOTION_REFERENCE) -> Path:
    """Resolve a bundled motion reference path.

    Args:
        name: Motion path relative to
            ``focodyn/assets/motions``.

    Returns:
        Filesystem path to the bundled motion reference.
    """
    path = resources.files("focodyn") / "assets" / "motions" / name
    return Path(str(path))


def load_kinematic_motion_reference(
    path: str | Path,
    model: FloatingBaseDynamics,
    *,
    root_body_name: str = "pelvis",
    source_name: str | None = None,
) -> KinematicMotionReference:
    """Load a retargeted G1 kinematic motion reference into model states.

    The preferred format is the NPZ layout used by
    ``ember-lab-berkeley/AMASS_Retargeted_for_G1``:
    ``dof_names`` ``(n_dofs,)``, ``dof_positions`` ``(frames, n_dofs)``,
    ``dof_velocities`` ``(frames, n_dofs)``, ``body_names`` ``(n_bodies,)``,
    ``body_positions`` ``(frames, n_bodies, 3)``, ``body_rotations``
    ``(frames, n_bodies, 4)``, ``body_linear_velocities``
    ``(frames, n_bodies, 3)``, and ``body_angular_velocities``
    ``(frames, n_bodies, 3)``.

    A simpler 29-DOF NPZ format with ``dof``, ``root_trans``, and
    ``root_rot_quat`` is also accepted. In that case, velocities are computed
    by finite differences.

    Raw 36-column ``.npy`` files from ``fleaven/Retargeted_AMASS_for_robotics``
    are also accepted. These files store ``(root_position, root_quat_xyzw,
    joint_positions)``; the loader converts the root quaternion to this
    package's scalar-first ``(w, x, y, z)`` convention.

    Args:
        path: Path to a retargeted G1 motion ``.npz`` or raw ``.npy`` file.
        model: Floating-base dynamics model whose state shape, dtype, device, and
            joint order should be used.
        root_body_name: Body/link name used as the floating-base pose source.
        source_name: Optional human-readable source label. If ``None``, the
            filename is used.

    Returns:
        :class:`KinematicMotionReference` with states shaped
        ``(frames, model.state_dim)`` and times shaped ``(frames,)``.

    Raises:
        ValueError: If the file layout is unsupported, if no matching root
            body exists, or if a required model joint is absent from named DOF
            data.
    """
    motion_path = Path(path)
    ground_to_contacts = False
    if motion_path.suffix == ".npy":
        raw_motion = np.asarray(
            np.load(motion_path, allow_pickle=False), dtype=np.float64
        )
        if raw_motion.ndim != 2 or raw_motion.shape[1] != 36:
            raise ValueError(
                "Raw G1 motion .npy files must have shape (frames, 36): "
                "(root_position, root_quat_xyzw, joint_positions)."
            )
        fps = _fps_from_path(motion_path, default=120.0)
        root_positions = raw_motion[:, :3]
        root_quaternions = _xyzw_to_wxyz(raw_motion[:, 3:7])
        joint_positions = _remap_unnamed_dofs(raw_motion[:, 7:36], model)
        joint_velocities = _finite_difference(joint_positions, fps)
        root_linear_velocities = _finite_difference(root_positions, fps)
        root_angular_velocities = np.zeros_like(root_linear_velocities)
        ground_to_contacts = True
    else:
        data = np.load(motion_path, allow_pickle=False)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 30.0

        if "dof_positions" in data:
            root_index = _name_index(data["body_names"], root_body_name, what="body")
            joint_positions = _remap_named_dofs(
                data["dof_names"], data["dof_positions"], model
            )
            joint_velocities = _remap_named_dofs(
                data["dof_names"], data["dof_velocities"], model
            )
            root_positions = np.asarray(
                data["body_positions"][:, root_index, :], dtype=np.float64
            )
            root_quaternions = np.asarray(
                data["body_rotations"][:, root_index, :], dtype=np.float64
            )
            root_linear_velocities = np.asarray(
                data["body_linear_velocities"][:, root_index, :], dtype=np.float64
            )
            root_angular_velocities = np.asarray(
                data["body_angular_velocities"][:, root_index, :], dtype=np.float64
            )
        elif {"dof", "root_trans", "root_rot_quat"}.issubset(data.files):
            joint_positions = _remap_unnamed_dofs(data["dof"], model)
            root_positions = np.asarray(data["root_trans"], dtype=np.float64)
            root_quaternions = np.asarray(data["root_rot_quat"], dtype=np.float64)
            joint_velocities = _finite_difference(joint_positions, fps)
            root_linear_velocities = _finite_difference(root_positions, fps)
            root_angular_velocities = np.zeros_like(root_linear_velocities)
        else:
            raise ValueError(f"Unsupported motion reference format: {motion_path}")

    frames = joint_positions.shape[0]
    states = torch.zeros(
        frames, model.state_dim, dtype=model.dtype, device=model.device
    )
    states[:, :3] = _as_tensor(root_positions, model)
    states[:, 3:7] = _as_tensor(_normalize_quaternions(root_quaternions), model)
    states[:, 7 : 7 + model.n_joints] = _as_tensor(joint_positions, model)
    if ground_to_contacts:
        _shift_lowest_contact_to_ground(states, model)
    velocity_start = model.nq
    states[:, velocity_start : velocity_start + 3] = _as_tensor(
        root_linear_velocities, model
    )
    states[:, velocity_start + 3 : velocity_start + 6] = _as_tensor(
        root_angular_velocities, model
    )
    states[:, velocity_start + 6 :] = _as_tensor(joint_velocities, model)
    times = torch.arange(frames, dtype=model.dtype, device=model.device) / fps

    return KinematicMotionReference(
        states=states,
        times=times,
        fps=fps,
        source_path=motion_path,
        source_name=source_name or motion_path.name,
    )


def default_g1_motion_reference(
    model: FloatingBaseDynamics,
) -> KinematicMotionReference:
    """Load the bundled G1 retargeted AMASS walking reference.

    Args:
        model: Floating-base dynamics model whose state shape, dtype, device, and
            joint order should be used.

    Returns:
        :class:`KinematicMotionReference` backed by the bundled sample motion.
    """
    return load_kinematic_motion_reference(
        bundled_motion_reference_path(),
        model,
        source_name="Transitions JOOF walk retargeted AMASS for Unitree G1",
    )


def _name_index(names: np.ndarray, name: str, *, what: str) -> int:
    """Find a name in a numpy string array.

    Args:
        names: Array with shape ``(n_names,)``.
        name: Name to search for.
        what: Label used in error messages.

    Returns:
        Integer index of ``name``.

    Raises:
        ValueError: If ``name`` is absent.
    """
    names_list = [str(item) for item in names.tolist()]
    if name not in names_list:
        raise ValueError(
            f"Motion reference is missing {what} {name!r}. Available: {names_list}"
        )
    return names_list.index(name)


def _remap_named_dofs(
    names: np.ndarray, values: np.ndarray, model: FloatingBaseDynamics
) -> np.ndarray:
    """Remap named DOF values into the model's joint order.

    Args:
        names: DOF name array with shape ``(n_dofs,)``.
        values: DOF value matrix with shape ``(frames, n_dofs)``.
        model: Floating-base dynamics model defining the target joint order.

    Returns:
        Array with shape ``(frames, model.n_joints)``.

    Raises:
        ValueError: If any model joint is missing from ``names``.
    """
    names_list = [str(item) for item in names.tolist()]
    if is_g1_37dof_minimal_joint_set(model.joint_names):
        return map_g1_29dof_to_37dof(values, names_list, model.joint_names)
    missing = [name for name in model.joint_names if name not in names_list]
    if missing:
        raise ValueError(f"Motion reference is missing model joints: {missing}")
    indices = [names_list.index(name) for name in model.joint_names]
    return np.asarray(values[:, indices], dtype=np.float64)


def _remap_unnamed_dofs(values: np.ndarray, model: FloatingBaseDynamics) -> np.ndarray:
    """Validate unnamed DOF values against the model joint count.

    Args:
        values: DOF value matrix with shape ``(frames, n_dofs)``.
        model: Floating-base dynamics model defining ``n_joints``.

    Returns:
        Array with shape ``(frames, model.n_joints)``.

    Raises:
        ValueError: If ``n_dofs`` does not match ``model.n_joints``.
    """
    values = np.asarray(values, dtype=np.float64)
    if is_g1_37dof_minimal_joint_set(model.joint_names) and values.shape[1] == len(
        UNITREE_G1_29DOF_JOINT_NAMES
    ):
        return map_g1_29dof_to_37dof(
            values, UNITREE_G1_29DOF_JOINT_NAMES, model.joint_names
        )
    if values.shape[1] != model.n_joints:
        raise ValueError(
            "Unnamed motion reference DOFs must match model.n_joints. "
            f"Got {values.shape[1]}, expected {model.n_joints}."
        )
    return values


def _fps_from_path(path: Path, *, default: float) -> float:
    """Infer a fixed frame rate from a motion filename.

    Args:
        path: Motion file path. Filenames containing ``"_120_"`` or
            ``"_60_"`` are interpreted as 120 Hz or 60 Hz respectively.
        default: Frame rate returned when no supported token is present.

    Returns:
        Inferred frame rate in Hz.
    """
    name = path.name
    if "_120_" in name:
        return 120.0
    if "_60_" in name:
        return 60.0
    return float(default)


def _xyzw_to_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """Convert vector-first quaternions to scalar-first order.

    Args:
        quaternions: Quaternion array with shape ``(..., 4)`` in
            ``(x, y, z, w)`` order.

    Returns:
        Quaternion array with shape ``(..., 4)`` in ``(w, x, y, z)`` order.

    Raises:
        ValueError: If the last dimension is not 4.
    """
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.shape[-1] != 4:
        raise ValueError("Expected xyzw quaternions with last dimension 4.")
    return np.concatenate((quaternions[..., 3:4], quaternions[..., :3]), axis=-1)


def _shift_lowest_contact_to_ground(
    states: torch.Tensor, model: FloatingBaseDynamics
) -> None:
    """Translate root height so the lowest contact candidate is on ``z = 0``.

    Args:
        states: State tensor with shape ``(frames, model.state_dim)``. The
            tensor is modified in place by adding a constant to the root
            ``z`` coordinate.
        model: Floating-base dynamics model used to evaluate contact FK. If the
            model does not already own a contact model, a temporary one is
            created using the same Adam kinematics object.

    Returns:
        None.
    """
    from .contacts import FloatingBaseContactModel

    contact_model = model.contact_model
    if contact_model is None:
        contact_model = FloatingBaseContactModel(
            model.asset,
            kin_dyn=model.kindyn,
            dtype=model.dtype,
            device=model.device,
        )
    base_transform = model.base_transform(states)
    joint_positions = states[:, 7 : 7 + model.n_joints]
    contact_positions = contact_model.contact_positions(base_transform, joint_positions)
    min_height = torch.amin(contact_positions[..., 2])
    states[:, 2] = states[:, 2] - min_height


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    """Compute first derivatives for a fixed-rate sequence.

    Args:
        values: Sequence array with shape ``(frames, dim)``.
        fps: Frame rate in Hz.

    Returns:
        Derivative array with shape ``(frames, dim)``.
    """
    return np.gradient(np.asarray(values, dtype=np.float64), 1.0 / fps, axis=0)


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Normalize scalar-first quaternions.

    Args:
        quaternions: Quaternion array with shape ``(frames, 4)`` in
            ``(w, x, y, z)`` order.

    Returns:
        Unit quaternion array with shape ``(frames, 4)``.
    """
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    return quaternions / np.clip(norms, 1e-12, None)


def _as_tensor(values: np.ndarray, model: FloatingBaseDynamics) -> torch.Tensor:
    """Convert numpy values to a tensor using model dtype and device.

    Args:
        values: Numpy array with arbitrary shape.
        model: Floating-base dynamics model defining target dtype and device.

    Returns:
        Tensor with the same shape as ``values``.
    """
    return torch.as_tensor(values, dtype=model.dtype, device=model.device)
