from __future__ import annotations

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


def test_bundled_g1_motion_reference_loads() -> None:
    model = FloatingBaseDynamics("unitree_g1", include_contact_forces=True, dtype=torch.float64)
    motion = default_g1_motion_reference(model)
    assert motion.states.ndim == 2
    assert motion.states.shape[1] == model.state_dim
    assert motion.times.shape == (motion.states.shape[0],)
    assert motion.fps == 120.0
    assert motion.source_path.suffix == ".npy"
    assert torch.isfinite(motion.states).all()
    assert torch.allclose(torch.linalg.norm(motion.states[:, 3:7], dim=-1), torch.ones_like(motion.times))


def test_ember_g1_motion_reference_still_loads() -> None:
    model = FloatingBaseDynamics("unitree_g1", include_contact_forces=True, dtype=torch.float64)
    motion = load_kinematic_motion_reference(
        bundled_motion_reference_path(EMBER_G1_MOTION_REFERENCE),
        model,
    )
    assert motion.states.shape[1] == model.state_dim
    assert motion.fps == 30.0
    assert torch.isfinite(motion.states).all()


def test_bundled_g1_motion_has_knee_bend() -> None:
    model = FloatingBaseDynamics("unitree_g1", dtype=torch.float64)
    motion = load_kinematic_motion_reference(bundled_motion_reference_path(), model)
    q = motion.states[:, 7 : 7 + model.n_joints]
    left_knee = q[:, model.joint_names.index("left_knee_joint")]
    right_knee = q[:, model.joint_names.index("right_knee_joint")]
    assert torch.max(left_knee) - torch.min(left_knee) > 0.2
    assert torch.max(right_knee) - torch.min(right_knee) > 0.2


def test_bundled_g1_motion_contact_heights_are_grounded() -> None:
    model = FloatingBaseDynamics("unitree_g1", include_contact_forces=True, dtype=torch.float64)
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


def test_whittaker_motion_derivative_estimate_shapes() -> None:
    import pytest

    pytest.importorskip("torch_dxdt")
    model = FloatingBaseDynamics("unitree_g1", dtype=torch.float64)
    states, times = simple_walking_sequence(model, frames=24, dt=1.0 / 60.0)

    estimate = estimate_motion_derivatives(model, states, times, lmbda=10.0)

    assert estimate.states.shape == states.shape
    assert estimate.generalized_accelerations.shape == (states.shape[0], model.nv)
    assert estimate.configurations.shape == (states.shape[0], model.nq)
    assert torch.isfinite(estimate.states).all()
    assert torch.isfinite(estimate.generalized_accelerations).all()
    assert torch.allclose(
        torch.linalg.norm(estimate.states[:, 3:7], dim=-1),
        torch.ones(states.shape[0], dtype=model.dtype),
        atol=1e-7,
    )
