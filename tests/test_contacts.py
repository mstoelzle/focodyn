from __future__ import annotations

from pathlib import Path

import pytest
import torch

from focodyn import FloatingBaseDynamics
from focodyn.assets import RobotAsset
from focodyn.contacts import (
    BasicContactForceResolver,
    ContactPoses,
    FlatTerrainContactDetector,
    FloatingBaseContactModel,
)
from focodyn.urdf import CollisionInfo, UrdfInfo


@pytest.fixture(scope="module")
def model() -> FloatingBaseDynamics:
    pytest.importorskip("adam")
    return FloatingBaseDynamics("unitree_g1", include_contact_forces=True, dtype=torch.float64)


def test_contact_fk_and_jacobian_shapes(model: FloatingBaseDynamics) -> None:
    assert model.contact_model is not None
    x = model.neutral_state()
    split = model.split_state(x)
    base_transform = model.base_transform(x)
    positions = model.contact_model.contact_positions(base_transform, split.joint_positions.squeeze(0))
    poses = model.contact_model.contact_poses(base_transform, split.joint_positions.squeeze(0))
    quaternions = model.contact_model.contact_quaternions(base_transform, split.joint_positions.squeeze(0))
    normals = model.contact_model.contact_normals(base_transform, split.joint_positions.squeeze(0))
    jacobian = model.contact_model.contact_jacobian(base_transform, split.joint_positions.squeeze(0))
    spatial_jacobian = model.contact_model.contact_spatial_jacobian(
        base_transform, split.joint_positions.squeeze(0)
    )
    assert positions.shape == (8, 3)
    assert poses.positions.shape == (8, 3)
    assert poses.quaternions_wxyz.shape == (8, 4)
    assert quaternions.shape == (8, 4)
    assert poses.transforms.shape == (8, 4, 4)
    assert normals.shape == (8, 3)
    assert jacobian.shape == (24, model.nv)
    assert spatial_jacobian.shape == (48, model.nv)
    assert torch.isfinite(positions).all()
    assert torch.isfinite(poses.quaternions_wxyz).all()
    assert torch.isfinite(quaternions).all()
    assert torch.isfinite(normals).all()
    assert torch.isfinite(jacobian).all()
    unit = torch.ones(8, dtype=model.dtype)
    assert torch.allclose(torch.linalg.norm(poses.quaternions_wxyz, dim=-1), unit)
    assert torch.allclose(torch.linalg.norm(quaternions, dim=-1), unit)
    assert torch.allclose(torch.linalg.norm(normals, dim=-1), unit)


def test_batched_contact_fk_and_jacobian_shapes(model: FloatingBaseDynamics) -> None:
    assert model.contact_model is not None
    x0 = model.neutral_state()
    x1 = x0.clone()
    x1[0] = 0.1
    x1[7:] = 0.02
    x = torch.stack((x0, x1))
    split = model.split_state(x)
    base_transform = model.base_transform(x)
    poses = model.contact_model.contact_poses(base_transform, split.joint_positions)
    normals = model.contact_model.contact_normals(base_transform, split.joint_positions)
    jacobian = model.contact_model.contact_jacobian(base_transform, split.joint_positions)
    spatial_jacobian = model.contact_model.contact_spatial_jacobian(
        base_transform, split.joint_positions
    )
    assert poses.positions.shape == (2, 8, 3)
    assert poses.quaternions_wxyz.shape == (2, 8, 4)
    assert poses.transforms.shape == (2, 8, 4, 4)
    assert normals.shape == (2, 8, 3)
    assert jacobian.shape == (2, 24, model.nv)
    assert spatial_jacobian.shape == (2, 48, model.nv)
    assert torch.isfinite(poses.positions).all()
    assert torch.isfinite(poses.quaternions_wxyz).all()
    assert torch.isfinite(normals).all()
    assert torch.isfinite(jacobian).all()
    assert torch.isfinite(spatial_jacobian).all()


def test_contact_jacobian_calls_adam_once_per_unique_link(
    model: FloatingBaseDynamics, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert model.contact_model is not None
    x = model.neutral_state()
    split = model.split_state(x)
    base_transform = model.base_transform(x)
    q = split.joint_positions.squeeze(0)
    fk_calls: list[str] = []
    jacobian_calls: list[str] = []
    original_fk = model.contact_model._fk
    original_jacobian = model.contact_model._jacobian

    def counting_fk(
        link_name: str, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        fk_calls.append(link_name)
        return original_fk(link_name, base_transform, joint_positions)

    def counting_jacobian(
        link_name: str, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        jacobian_calls.append(link_name)
        return original_jacobian(link_name, base_transform, joint_positions)

    monkeypatch.setattr(model.contact_model, "_fk", counting_fk)
    monkeypatch.setattr(model.contact_model, "_jacobian", counting_jacobian)
    model.contact_model.contact_jacobian(base_transform, q)

    unique_links = list(model.contact_model.unique_contact_link_names)
    assert fk_calls == unique_links
    assert jacobian_calls == unique_links


def test_contact_jacobian_matches_autograd_contact_fk(model: FloatingBaseDynamics) -> None:
    assert model.contact_model is not None
    x = model.neutral_state()
    split = model.split_state(x)
    q = split.joint_positions.squeeze(0).detach()
    q = q + 0.03 * torch.sin(torch.arange(model.n_joints, dtype=model.dtype))
    q = q.requires_grad_(True)
    base_transform = model.base_transform(x)

    def contact_positions_from_joints(joint_positions: torch.Tensor) -> torch.Tensor:
        return model.contact_model.contact_positions(base_transform, joint_positions).reshape(-1)

    autodiff_jacobian = torch.autograd.functional.jacobian(contact_positions_from_joints, q)
    analytical_jacobian = model.contact_model.contact_jacobian(base_transform, q.detach())[:, 6:]
    assert torch.allclose(analytical_jacobian, autodiff_jacobian, atol=1e-8, rtol=1e-7)


def test_contact_frame_force_mapping_is_supported() -> None:
    pytest.importorskip("adam")
    model = FloatingBaseDynamics(
        "unitree_g1",
        include_contact_forces=True,
        contact_force_frame="contact",
        dtype=torch.float64,
    )
    x = model.neutral_state()
    control = model.g(x)
    assert control.shape == (model.state_dim, model.input_dim)
    assert torch.isfinite(control).all()


def test_flat_terrain_contact_detector_reports_distances() -> None:
    detector = FlatTerrainContactDetector(contact_threshold=0.025, dtype=torch.float64)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.03],
            [1.0, 0.0, -0.02],
        ],
        dtype=torch.float64,
    )

    state = detector.detect(positions)

    assert torch.allclose(state.signed_distances, torch.tensor([0.03, -0.02], dtype=torch.float64))
    assert torch.allclose(state.penetration_depths, torch.tensor([0.0, 0.02], dtype=torch.float64))
    assert torch.allclose(state.nearest_points[:, 2], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(state.in_contact, torch.tensor([False, True]))
    assert torch.allclose(state.normals, torch.tensor([[0.0, 0.0, 1.0]] * 2, dtype=torch.float64))


def test_flat_terrain_contact_detector_validates_normal_and_position_shape() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        FlatTerrainContactDetector(normal=(0.0, 0.0, 0.0))

    detector = FlatTerrainContactDetector(dtype=torch.float64)
    with pytest.raises(ValueError, match="last dimension 3"):
        detector(torch.zeros(2, 2, dtype=torch.float64))


def test_basic_contact_force_resolver_support_force_and_frames() -> None:
    detector = FlatTerrainContactDetector(contact_threshold=0.025, dtype=torch.float64)
    positions = torch.tensor(
        [
            [0.0, 0.0, -0.01],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    contact_state = detector.detect(positions)
    resolver = BasicContactForceResolver(
        force_frame="world",
        normal_stiffness=1000.0,
        normal_damping=0.0,
    )

    resolved = resolver.resolve(contact_state, total_normal_force=torch.tensor(20.0))

    assert resolved.force_frame == "world"
    assert torch.allclose(resolved.normal_forces, torch.tensor([20.0, 10.0], dtype=torch.float64))
    assert torch.allclose(resolved.world_forces[:, :2], torch.zeros(2, 2, dtype=torch.float64))
    assert torch.allclose(resolved.world_forces[:, 2], resolved.normal_forces)

    transforms = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    poses = ContactPoses(
        positions=positions,
        quaternions_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64),
        transforms=transforms,
    )
    contact_resolver = BasicContactForceResolver(force_frame="contact", normal_stiffness=1000.0)
    contact_forces = contact_resolver.resolve(contact_state, total_normal_force=20.0, contact_poses=poses)
    assert contact_forces.force_frame == "contact"
    assert torch.allclose(contact_forces.forces, contact_forces.world_forces)


def test_basic_contact_force_resolver_applies_velocity_damping_and_friction_limits() -> None:
    detector = FlatTerrainContactDetector(contact_threshold=0.05, dtype=torch.float64)
    positions = torch.tensor(
        [
            [[0.0, 0.0, -0.01], [1.0, 0.0, 0.03]],
            [[0.0, 0.0, 0.10], [1.0, 0.0, -0.02]],
        ],
        dtype=torch.float64,
    )
    velocities = torch.tensor(
        [
            [[1.0, 0.0, -0.2], [0.0, 0.5, 0.1]],
            [[1.0, 1.0, -0.3], [0.5, 0.0, -0.4]],
        ],
        dtype=torch.float64,
    )
    contact_state = detector(positions)
    resolver = BasicContactForceResolver(
        normal_stiffness=1000.0,
        normal_damping=100.0,
        tangential_damping=50.0,
        friction_coefficient=0.5,
    )

    resolved = resolver(contact_state, contact_velocities=velocities, total_normal_force=torch.tensor(20.0))
    tangential_norm = torch.linalg.norm(resolved.world_forces[..., :2], dim=-1)

    assert resolved.forces.shape == positions.shape
    assert torch.allclose(resolved.world_forces[1, 0], torch.zeros(3, dtype=torch.float64))
    assert torch.all(resolved.normal_forces[contact_state.in_contact] > 0.0)
    assert torch.all(tangential_norm <= 0.5 * resolved.normal_forces + 1e-12)


def test_contact_frame_resolver_rotates_world_forces_into_contact_frame() -> None:
    detector = FlatTerrainContactDetector(dtype=torch.float64)
    positions = torch.tensor([[0.0, 0.0, -0.01]], dtype=torch.float64)
    contact_state = detector(positions)
    transform = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    transform[0, :3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    poses = ContactPoses(
        positions=positions,
        quaternions_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        transforms=transform,
    )
    resolver = BasicContactForceResolver(force_frame="contact", normal_stiffness=1000.0)

    resolved = resolver(contact_state, contact_poses=poses)

    assert torch.allclose(
        resolved.forces,
        torch.matmul(transform[..., :3, :3].transpose(-1, -2), resolved.world_forces.unsqueeze(-1)).squeeze(-1),
    )


def test_contact_frame_force_resolver_requires_contact_poses() -> None:
    detector = FlatTerrainContactDetector(dtype=torch.float64)
    contact_state = detector.detect(torch.zeros(1, 3, dtype=torch.float64))
    resolver = BasicContactForceResolver(force_frame="contact")
    with pytest.raises(ValueError, match="contact_poses"):
        resolver.resolve(contact_state)


def test_contact_force_resolver_rejects_unknown_force_frame() -> None:
    with pytest.raises(ValueError, match="force_frame"):
        BasicContactForceResolver(force_frame="local")


def test_contact_model_without_kindyn_exposes_static_specs_and_errors() -> None:
    asset = _contact_asset(
        default_contact_links=("left_foot",),
        collisions=(
            CollisionInfo("left_foot", (0.2, -0.1, 0.0), (0.0, 0.0, 0.0), "sphere", (0.02,)),
            CollisionInfo("left_foot", (-0.2, 0.1, 0.0), (0.0, 0.0, 0.0), "sphere", (0.02,)),
        ),
    )
    model = FloatingBaseContactModel(asset, mode="feet_centers", dtype=torch.float64)
    base_transform = torch.eye(4, dtype=torch.float64)
    joint_positions = torch.empty(0, dtype=torch.float64)

    assert model.num_contacts == 1
    assert model.force_dim == 3
    assert model.contact_names == ("left_foot:center",)
    assert torch.allclose(model.local_offsets.squeeze(0), torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(
        model.contact_force_transform(base_transform, joint_positions, force_frame="world"),
        torch.eye(3, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="force_frame"):
        model.contact_force_transform(base_transform, joint_positions, force_frame="bad")
    with pytest.raises(RuntimeError, match="KinDynComputations"):
        model._fk("left_foot", base_transform, joint_positions)
    with pytest.raises(RuntimeError, match="KinDynComputations"):
        model._jacobian("left_foot", base_transform, joint_positions)


def test_contact_model_reports_missing_or_unknown_static_contact_specs() -> None:
    with pytest.raises(ValueError, match="no default"):
        FloatingBaseContactModel(_contact_asset(default_contact_links=(), collisions=()))

    with pytest.raises(ValueError, match="No sphere"):
        FloatingBaseContactModel(
            _contact_asset(
                default_contact_links=("left_foot",),
                collisions=(CollisionInfo("left_foot", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "box", (1.0, 1.0, 1.0)),),
            )
        )

    with pytest.raises(ValueError, match="Unknown contact mode"):
        FloatingBaseContactModel(
            _contact_asset(
                default_contact_links=("left_foot",),
                collisions=(CollisionInfo("left_foot", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "sphere", (0.02,)),),
            ),
            mode="toes",
        )


def _contact_asset(
    *,
    default_contact_links: tuple[str, ...],
    collisions: tuple[CollisionInfo, ...],
) -> RobotAsset:
    info = UrdfInfo(
        name="test_robot",
        root_link="base",
        joints=(),
        collisions=collisions,
    )
    return RobotAsset(
        name="test_robot",
        urdf_path=Path("test_robot.urdf"),
        adam_urdf_path=Path("test_robot.urdf"),
        root_link="base",
        joint_names=(),
        default_contact_links=default_contact_links,
        urdf=info,
    )
