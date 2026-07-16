from __future__ import annotations

import hashlib

import pytest

from focodyn import FloatingBaseContactModel, available_assets, load_asset


G1_37DOF_MINIMAL_JOINTS = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "torso_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
    "left_five_joint",
    "left_three_joint",
    "left_zero_joint",
    "left_six_joint",
    "left_four_joint",
    "left_one_joint",
    "left_two_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
    "right_five_joint",
    "right_three_joint",
    "right_zero_joint",
    "right_six_joint",
    "right_four_joint",
    "right_one_joint",
    "right_two_joint",
)

G1_37DOF_UNITREE_29DOF_ORDER = (
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
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
    "left_five_joint",
    "left_three_joint",
    "left_zero_joint",
    "left_six_joint",
    "left_four_joint",
    "left_one_joint",
    "left_two_joint",
    "right_five_joint",
    "right_three_joint",
    "right_zero_joint",
    "right_six_joint",
    "right_four_joint",
    "right_one_joint",
    "right_two_joint",
)


def test_unitree_g1_asset_metadata() -> None:
    assert "unitree_g1" in available_assets()
    asset = load_asset("unitree_g1")
    assert asset.urdf_path.name == "g1_29dof_rev_1_0.urdf"
    assert asset.adam_urdf_path.name == "g1_29dof_rev_1_0.adam.urdf"
    assert asset.root_link == "pelvis"
    assert len(asset.joint_names) == 29
    assert "left_ankle_roll_link" in asset.default_contact_links
    assert "right_ankle_roll_link" in asset.default_contact_links


def test_deprecated_unitree_g1_files_are_not_vendored() -> None:
    asset = load_asset("unitree_g1")
    folder = asset.urdf_path.parent
    deprecated = {
        "g1_23dof.urdf",
        "g1_23dof.xml",
        "g1_29dof.urdf",
        "g1_29dof.xml",
        "g1_29dof_with_hand.urdf",
        "g1_29dof_with_hand.xml",
        "g1_29dof_lock_waist.urdf",
        "g1_29dof_lock_waist.xml",
    }
    assert not deprecated.intersection(path.name for path in folder.iterdir())


def test_contact_modes_from_collision_geometry() -> None:
    corners = FloatingBaseContactModel("unitree_g1", mode="feet_corners")
    centers = FloatingBaseContactModel("unitree_g1", mode="feet_centers")
    assert corners.num_contacts == 8
    assert centers.num_contacts == 2
    assert corners.local_offsets.shape == (8, 3)
    assert centers.local_offsets.shape == (2, 3)


def test_g1_37dof_minimal_asset_metadata_and_source() -> None:
    assert "g1_37dof_minimal" in available_assets()
    asset = load_asset("g1_37dof_minimal")
    assert asset.urdf_path.name == "g1_37dof_minimal.urdf"
    assert asset.adam_urdf_path.name == "g1_37dof_minimal.adam.urdf"
    assert asset.usd_path is not None
    assert asset.usd_path.name == "g1_37dof_minimal.usd"
    assert hashlib.sha256(asset.usd_path.read_bytes()).hexdigest() == (
        "4f5e0600f24bed04d4c45b3921b6f3d7b6205463b0b1ac051cf2883dd3aaee67"
    )
    assert asset.root_link == "pelvis"
    assert asset.joint_names == G1_37DOF_MINIMAL_JOINTS
    assert asset.source_joint_names == G1_37DOF_MINIMAL_JOINTS
    assert asset.joint_order == "source"
    assert asset.default_contact_links == (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )


def test_g1_37dof_minimal_can_match_unitree_29dof_joint_order() -> None:
    source = load_asset("g1_37dof_minimal")
    compatible = load_asset("g1_37dof_minimal", joint_order="unitree_g1_29dof")

    assert compatible.joint_names == G1_37DOF_UNITREE_29DOF_ORDER
    assert compatible.source_joint_names == source.joint_names
    assert compatible.joint_order == "unitree_g1_29dof"
    assert set(compatible.joint_names) == set(source.joint_names)


def test_existing_g1_source_and_unitree_29dof_orders_are_equal() -> None:
    source = load_asset("unitree_g1")
    compatible = load_asset("unitree_g1", joint_order="unitree_g1_29dof")
    assert compatible.joint_names == source.joint_names
    assert compatible.source_joint_names == source.source_joint_names


def test_load_asset_rejects_invalid_or_incompatible_joint_order(tmp_path) -> None:
    with pytest.raises(ValueError, match="joint_order must be"):
        load_asset("unitree_g1", joint_order="g1")  # type: ignore[arg-type]

    urdf_path = tmp_path / "one_joint.urdf"
    urdf_path.write_text(
        """<robot name="one_joint">
  <link name="base"/>
  <link name="tip"/>
  <joint name="joint" type="revolute">
    <parent link="base"/><child link="tip"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
"""
    )
    with pytest.raises(ValueError, match="requires a Unitree G1-compatible"):
        load_asset(str(urdf_path), joint_order="unitree_g1_29dof")


def test_g1_37dof_minimal_usd_path_uses_companion_urdf() -> None:
    asset = load_asset("g1_37dof_minimal")
    assert asset.usd_path is not None
    direct = load_asset(str(asset.usd_path))
    compatible = load_asset(str(asset.usd_path), joint_order="unitree_g1_29dof")
    assert direct.urdf_path == asset.urdf_path
    assert direct.adam_urdf_path == asset.adam_urdf_path
    assert direct.usd_path == asset.usd_path
    assert direct.joint_names == G1_37DOF_MINIMAL_JOINTS
    assert compatible.joint_names == G1_37DOF_UNITREE_29DOF_ORDER


def test_direct_usd_requires_same_stem_urdf(tmp_path) -> None:
    usd_path = tmp_path / "robot.usd"
    usd_path.write_bytes(b"PXR-USDC")
    with pytest.raises(ValueError, match="same-stem URDF companion"):
        load_asset(str(usd_path))


def test_g1_37dof_minimal_contact_modes_from_box_geometry() -> None:
    corners = FloatingBaseContactModel("g1_37dof_minimal", mode="feet_corners")
    centers = FloatingBaseContactModel("g1_37dof_minimal", mode="feet_centers")
    assert corners.num_contacts == 8
    assert centers.num_contacts == 2
    assert corners.local_offsets.shape == (8, 3)
    assert centers.local_offsets.shape == (2, 3)
