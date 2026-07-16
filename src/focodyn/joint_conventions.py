"""Named joint-order conventions and conversions between G1 variants."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np


JointOrder = Literal["source", "unitree_g1_29dof"]

UNITREE_G1_29DOF_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_UNITREE_G1_29DOF_NAME_BY_VARIANT_NAME = {
    "torso_joint": "waist_yaw_joint",
    "left_elbow_pitch_joint": "left_elbow_joint",
    "left_elbow_roll_joint": "left_wrist_roll_joint",
    "right_elbow_pitch_joint": "right_elbow_joint",
    "right_elbow_roll_joint": "right_wrist_roll_joint",
}
_VARIANT_NAME_BY_UNITREE_G1_29DOF_NAME = {
    reference_name: variant_name
    for variant_name, reference_name in _UNITREE_G1_29DOF_NAME_BY_VARIANT_NAME.items()
}
_UNITREE_G1_CORE_JOINT_NAMES = frozenset(UNITREE_G1_29DOF_JOINT_NAMES[:12])
_G1_37DOF_MINIMAL_FINGER_JOINTS = frozenset(
    f"{side}_{index_name}_joint"
    for side in ("left", "right")
    for index_name in ("zero", "one", "two", "three", "four", "five", "six")
)
_G1_29DOF_JOINTS_ABSENT_FROM_37DOF = frozenset(
    {
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    }
)


def joint_names_for_order(
    source_joint_names: Sequence[str], joint_order: JointOrder
) -> tuple[str, ...]:
    """Resolve an active joint order without changing source joint names."""
    source_names = tuple(source_joint_names)
    if joint_order == "source":
        return source_names
    if joint_order != "unitree_g1_29dof":
        raise ValueError(
            f"joint_order must be 'source' or 'unitree_g1_29dof'. Got {joint_order!r}."
        )

    source_name_set = set(source_names)
    if not _UNITREE_G1_CORE_JOINT_NAMES.issubset(source_name_set):
        raise ValueError(
            "joint_order='unitree_g1_29dof' requires a Unitree G1-compatible "
            "asset containing the 12 leg joints."
        )

    rank = {name: index for index, name in enumerate(UNITREE_G1_29DOF_JOINT_NAMES)}
    matched: list[tuple[int, str]] = []
    source_only: list[str] = []
    used_reference_names: set[str] = set()
    for source_name in source_names:
        reference_name = _UNITREE_G1_29DOF_NAME_BY_VARIANT_NAME.get(
            source_name, source_name
        )
        if reference_name not in rank:
            source_only.append(source_name)
            continue
        if reference_name in used_reference_names:
            raise ValueError(
                "Multiple source joints map to Unitree G1 29-DoF joint "
                f"{reference_name!r}."
            )
        used_reference_names.add(reference_name)
        matched.append((rank[reference_name], source_name))

    matched.sort(key=lambda item: item[0])
    return tuple(name for _, name in matched) + tuple(source_only)


def map_g1_29dof_to_37dof(
    values: np.ndarray,
    source_joint_names: Sequence[str],
    target_joint_names: Sequence[str],
) -> np.ndarray:
    """Map named G1 29-DoF values to a named 37-DoF minimal-model order.

    Shared joints are copied by name or by the torso/elbow/wrist aliases. The
    14 finger joints are initialized to zero. Joint coordinates occupy the
    final array dimension; any leading dimensions are preserved.
    """
    source_names = _validated_joint_axis(values, source_joint_names)
    target_names = _validated_unique_names(target_joint_names, what="target")
    array = np.asarray(values, dtype=np.float64)
    remapped = np.zeros(array.shape[:-1] + (len(target_names),), dtype=np.float64)
    source_indices = {name: index for index, name in enumerate(source_names)}
    missing: list[str] = []
    for target_index, target_name in enumerate(target_names):
        if target_name in _G1_37DOF_MINIMAL_FINGER_JOINTS:
            continue
        source_name = _UNITREE_G1_29DOF_NAME_BY_VARIANT_NAME.get(
            target_name, target_name
        )
        source_index = source_indices.get(source_name)
        if source_index is None:
            missing.append(f"{target_name} (source {source_name})")
            continue
        remapped[..., target_index] = array[..., source_index]
    if missing:
        raise ValueError(f"Motion reference is missing model joints: {missing}")
    return remapped


def map_g1_37dof_to_29dof(
    values: np.ndarray,
    source_joint_names: Sequence[str],
    target_joint_names: Sequence[str] = UNITREE_G1_29DOF_JOINT_NAMES,
) -> np.ndarray:
    """Map named 37-DoF minimal-model values to a named G1 29-DoF order.

    Finger values are dropped. The waist roll/pitch and wrist pitch/yaw joints
    do not exist in the 37-DoF model and are initialized to zero, making this
    conversion intentionally lossy. Joint coordinates occupy the final array
    dimension; any leading dimensions are preserved.
    """
    source_names = _validated_joint_axis(values, source_joint_names)
    target_names = _validated_unique_names(target_joint_names, what="target")
    array = np.asarray(values, dtype=np.float64)
    remapped = np.zeros(array.shape[:-1] + (len(target_names),), dtype=np.float64)
    source_indices = {name: index for index, name in enumerate(source_names)}
    missing: list[str] = []
    for target_index, target_name in enumerate(target_names):
        source_name = _VARIANT_NAME_BY_UNITREE_G1_29DOF_NAME.get(
            target_name, target_name
        )
        source_index = source_indices.get(source_name)
        if source_index is None:
            if target_name in _G1_29DOF_JOINTS_ABSENT_FROM_37DOF:
                continue
            missing.append(f"{target_name} (source {source_name})")
            continue
        remapped[..., target_index] = array[..., source_index]
    if missing:
        raise ValueError(f"37-DoF reference is missing mapped joints: {missing}")
    return remapped


def is_g1_37dof_minimal_joint_set(joint_names: Sequence[str]) -> bool:
    """Return whether names contain the complete minimal-model finger set."""
    return _G1_37DOF_MINIMAL_FINGER_JOINTS.issubset(joint_names)


def _validated_joint_axis(
    values: np.ndarray, joint_names: Sequence[str]
) -> tuple[str, ...]:
    """Validate a named final joint axis and return normalized names."""
    array = np.asarray(values)
    names = _validated_unique_names(joint_names, what="source")
    if array.ndim == 0 or array.shape[-1] != len(names):
        actual = None if array.ndim == 0 else array.shape[-1]
        raise ValueError(
            "Joint value final dimension must match source_joint_names. "
            f"Got {actual}, expected {len(names)}."
        )
    return names


def _validated_unique_names(
    joint_names: Sequence[str], *, what: str
) -> tuple[str, ...]:
    """Normalize a joint-name sequence and reject ambiguous duplicates."""
    names = tuple(str(name) for name in joint_names)
    if len(set(names)) != len(names):
        raise ValueError(f"{what}_joint_names must not contain duplicates.")
    return names
