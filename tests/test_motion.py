from __future__ import annotations

import math
import builtins

import pytest
import torch

from focodyn import (
    FloatingBaseDynamics,
    bundled_motion_reference_path,
    default_g1_motion_reference,
    estimate_motion_derivatives,
    load_kinematic_motion_reference,
    simple_walking_sequence,
)
from focodyn.motion import EMBER_G1_MOTION_REFERENCE
from focodyn.rotations import matrix_to_quaternion_wxyz, unwrap_angles
from focodyn.motion_derivatives import _order_value, _require_torch_dxdt


def test_bundled_g1_motion_reference_loads() -> None:
    model = FloatingBaseDynamics(
        "unitree_g1", include_contact_forces=True, dtype=torch.float64
    )
    motion = default_g1_motion_reference(model)
    assert motion.states.ndim == 2
    assert motion.states.shape[1] == model.state_dim
    assert motion.times.shape == (motion.states.shape[0],)
    assert motion.fps == 120.0
    assert motion.source_path.suffix == ".npy"
    assert torch.isfinite(motion.states).all()
    assert torch.allclose(
        torch.linalg.norm(motion.states[:, 3:7], dim=-1), torch.ones_like(motion.times)
    )


def test_ember_g1_motion_reference_still_loads() -> None:
    model = FloatingBaseDynamics(
        "unitree_g1", include_contact_forces=True, dtype=torch.float64
    )
    motion = load_kinematic_motion_reference(
        bundled_motion_reference_path(EMBER_G1_MOTION_REFERENCE),
        model,
    )
    assert motion.states.shape[1] == model.state_dim
    assert motion.fps == 30.0
    assert torch.isfinite(motion.states).all()


@pytest.mark.parametrize(
    "motion_name",
    [None, EMBER_G1_MOTION_REFERENCE],
    ids=["raw-npy", "named-npz"],
)
@pytest.mark.parametrize(
    "joint_order", ["source", "unitree_g1_29dof"], ids=["source", "compatible"]
)
def test_bundled_g1_motions_map_to_37dof_minimal(
    motion_name: str | None, joint_order: str
) -> None:
    source_model = FloatingBaseDynamics("unitree_g1", dtype=torch.float64)
    target_model = FloatingBaseDynamics(
        "g1_37dof_minimal", joint_order=joint_order, dtype=torch.float64
    )
    path = (
        bundled_motion_reference_path(motion_name)
        if motion_name
        else bundled_motion_reference_path()
    )
    source = load_kinematic_motion_reference(path, source_model)
    target = load_kinematic_motion_reference(path, target_model)

    assert target.states.shape[1] == 87
    assert torch.isfinite(target.states).all()
    aliases = {
        "torso_joint": "waist_yaw_joint",
        "left_elbow_pitch_joint": "left_elbow_joint",
        "left_elbow_roll_joint": "left_wrist_roll_joint",
        "right_elbow_pitch_joint": "right_elbow_joint",
        "right_elbow_roll_joint": "right_wrist_roll_joint",
    }
    for target_name in target_model.joint_names:
        target_index = target_model.joint_names.index(target_name)
        target_positions = target.states[:, 7 + target_index]
        target_velocities = target.states[:, target_model.nq + 6 + target_index]
        if any(
            number in target_name
            for number in ("zero", "one", "two", "three", "four", "five", "six")
        ):
            assert torch.count_nonzero(target_positions) == 0
            assert torch.count_nonzero(target_velocities) == 0
            continue
        source_name = aliases.get(target_name, target_name)
        source_index = source_model.joint_names.index(source_name)
        assert torch.allclose(target_positions, source.states[:, 7 + source_index])
        assert torch.allclose(
            target_velocities,
            source.states[:, source_model.nq + 6 + source_index],
        )


def test_bundled_g1_motion_has_knee_bend() -> None:
    model = FloatingBaseDynamics("unitree_g1", dtype=torch.float64)
    motion = load_kinematic_motion_reference(bundled_motion_reference_path(), model)
    q = motion.states[:, 7 : 7 + model.n_joints]
    left_knee = q[:, model.joint_names.index("left_knee_joint")]
    right_knee = q[:, model.joint_names.index("right_knee_joint")]
    assert torch.max(left_knee) - torch.min(left_knee) > 0.2
    assert torch.max(right_knee) - torch.min(right_knee) > 0.2


def test_bundled_g1_motion_contact_heights_are_grounded() -> None:
    model = FloatingBaseDynamics(
        "unitree_g1", include_contact_forces=True, dtype=torch.float64
    )
    assert model.contact_model is not None
    motion = default_g1_motion_reference(model)
    contact_heights = []
    contact_normals = []
    for state in motion.states[:: max(1, motion.states.shape[0] // 10)]:
        split = model.split_state(state)
        poses = model.contact_model.contact_poses(
            model.base_transform(state),
            split.joint_positions.squeeze(0),
        )
        contact_heights.append(poses.positions[:, 2])
        contact_normals.append(poses.transforms[..., :3, 2])
    heights = torch.cat(contact_heights)
    normals = torch.cat(contact_normals)
    stance = heights < 0.035
    assert torch.min(heights) < 0.03
    assert torch.max(heights) > 0.08
    assert torch.mean(normals[stance, 2]) > 0.9


def test_joint_angle_unwrap_removes_periodic_discontinuity() -> None:
    angles = torch.tensor(
        [
            [3.00, -0.1],
            [3.10, -0.2],
            [-3.10, -0.3],
            [-3.00, -0.4],
        ],
        dtype=torch.float64,
    )

    unwrapped = unwrap_angles(angles)

    assert torch.max(torch.abs(torch.diff(unwrapped[:, 0]))) < 0.2
    assert torch.allclose(unwrapped[:, 1], angles[:, 1])
    assert unwrapped[-1, 0] > math.pi


def test_whittaker_orientation_derivative_estimates_world_angular_velocity() -> None:
    import pytest

    pytest.importorskip("torch_dxdt")
    model = FloatingBaseDynamics("unitree_g1", dtype=torch.float64)
    frames = 60
    dt = 1.0 / 60.0
    yaw_rate = 1.25
    times = torch.arange(frames, dtype=model.dtype) * dt
    yaw = yaw_rate * times
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rotations = torch.zeros(frames, 3, 3, dtype=model.dtype)
    rotations[:, 0, 0] = cos_yaw
    rotations[:, 0, 1] = -sin_yaw
    rotations[:, 1, 0] = sin_yaw
    rotations[:, 1, 1] = cos_yaw
    rotations[:, 2, 2] = 1.0

    states = torch.zeros(frames, model.state_dim, dtype=model.dtype)
    states[:, 3:7] = matrix_to_quaternion_wxyz(rotations)

    estimate = estimate_motion_derivatives(model, states, times, lmbda=1.0)
    angular_velocity = estimate.configuration_velocities[10:-10, 3:6]

    assert torch.allclose(
        angular_velocity.mean(dim=0),
        torch.tensor([0.0, 0.0, yaw_rate], dtype=model.dtype),
        atol=1e-3,
    )


def test_whittaker_motion_derivative_estimate_shapes() -> None:
    pytest.importorskip("torch_dxdt")
    model = FloatingBaseDynamics("unitree_g1", dtype=torch.float64)
    states, times = simple_walking_sequence(model, frames=24, dt=1.0 / 60.0)

    estimate = estimate_motion_derivatives(model, states, times, lmbda=10.0)

    assert estimate.states.shape == states.shape
    assert estimate.generalized_accelerations.shape == (states.shape[0], model.nv)
    assert estimate.configurations.shape == (states.shape[0], model.nq)
    assert estimate.configuration_velocities.shape == (states.shape[0], model.nv)
    assert estimate.configuration_accelerations.shape == (states.shape[0], model.nv)
    assert torch.allclose(
        estimate.configuration_velocities, estimate.states[:, model.nq :]
    )
    assert torch.allclose(
        estimate.configuration_accelerations, estimate.generalized_accelerations
    )
    assert torch.isfinite(estimate.states).all()
    assert torch.isfinite(estimate.generalized_accelerations).all()
    assert torch.allclose(
        torch.linalg.norm(estimate.states[:, 3:7], dim=-1),
        torch.ones(states.shape[0], dtype=model.dtype),
        atol=1e-7,
    )


def test_motion_derivative_estimator_rejects_invalid_shapes() -> None:
    pytest.importorskip("torch_dxdt")
    model = FloatingBaseDynamics("unitree_g1", dtype=torch.float64)
    states, times = simple_walking_sequence(model, frames=5, dt=1.0 / 60.0)

    with pytest.raises(ValueError, match="Expected states"):
        estimate_motion_derivatives(model, states[:, :-1], times)
    with pytest.raises(ValueError, match="times"):
        estimate_motion_derivatives(model, states, times[:-1])
    with pytest.raises(ValueError, match="At least four"):
        estimate_motion_derivatives(model, states[:3], times[:3])


def test_motion_derivative_private_dependency_and_order_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch_dxdt":
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    values = [torch.tensor([0.0]), torch.tensor([1.0])]

    assert torch.equal(_order_value(values, 1), torch.tensor([1.0]))
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="torch-dxdt"):
        _require_torch_dxdt()
