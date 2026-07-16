from __future__ import annotations

import numpy as np
import pytest
import torch

from focodyn import FloatingBaseDynamics
from focodyn.joint_conventions import (
    map_g1_29dof_to_37dof,
    map_g1_37dof_to_29dof,
)


def test_g1_joint_maps_are_bidirectional_with_documented_zero_fill() -> None:
    model_29 = FloatingBaseDynamics("unitree_g1", dtype=torch.float64)
    model_37 = FloatingBaseDynamics("g1_37dof_minimal", dtype=torch.float64)
    values_29 = np.arange(2 * model_29.n_joints, dtype=np.float64).reshape(2, -1)
    unavailable = {
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    }
    for name in unavailable:
        values_29[:, model_29.joint_names.index(name)] = 0.0

    values_37 = map_g1_29dof_to_37dof(
        values_29, model_29.joint_names, model_37.joint_names
    )
    round_trip = map_g1_37dof_to_29dof(
        values_37, model_37.joint_names, model_29.joint_names
    )

    assert values_37.shape == (2, 37)
    assert np.array_equal(round_trip, values_29)
    for name in model_37.joint_names:
        if any(
            token in name
            for token in ("zero", "one", "two", "three", "four", "five", "six")
        ):
            assert np.count_nonzero(values_37[:, model_37.joint_names.index(name)]) == 0


def test_g1_37dof_inverse_drops_fingers_and_zero_fills_unavailable_joints() -> None:
    model_29 = FloatingBaseDynamics("unitree_g1", dtype=torch.float64)
    model_37 = FloatingBaseDynamics(
        "g1_37dof_minimal",
        joint_order="unitree_g1_29dof",
        dtype=torch.float64,
    )
    values_37 = np.arange(model_37.n_joints, dtype=np.float64) + 1.0

    values_29 = map_g1_37dof_to_29dof(
        values_37, model_37.joint_names, model_29.joint_names
    )

    assert values_29.shape == (29,)
    assert (
        values_29[model_29.joint_names.index("waist_yaw_joint")]
        == values_37[model_37.joint_names.index("torso_joint")]
    )
    for name in (
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ):
        assert values_29[model_29.joint_names.index(name)] == 0.0


def test_g1_joint_maps_validate_named_joint_axis() -> None:
    with pytest.raises(ValueError, match="final dimension"):
        map_g1_29dof_to_37dof(
            np.zeros((2, 3)),
            ("one", "two"),
            ("left_zero_joint",),
        )
    with pytest.raises(ValueError, match="duplicates"):
        map_g1_37dof_to_29dof(
            np.zeros(2),
            ("torso_joint", "torso_joint"),
            ("waist_yaw_joint",),
        )
