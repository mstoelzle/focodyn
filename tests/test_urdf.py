from __future__ import annotations

import numpy as np
import pytest

from focodyn.urdf import foot_collision_points, parse_urdf


def test_parse_urdf_reads_joints_collisions_and_geometry_variants(tmp_path) -> None:
    urdf_path = tmp_path / "toy.urdf"
    urdf_path.write_text(
        """\
<robot name="toy">
  <link name="base">
    <collision>
      <geometry><sphere radius="0.2"/></geometry>
    </collision>
    <collision>
      <origin xyz="1 2 3" rpy="0.1 0.2 0.3"/>
      <geometry><box size="0.4 0.5 0.6"/></geometry>
    </collision>
    <collision>
      <origin xyz="-1 0 0"/>
      <geometry><cylinder radius="0.7" length="0.8"/></geometry>
    </collision>
    <collision>
      <geometry><mesh filename="package://mesh.stl"/></geometry>
    </collision>
    <collision>
      <geometry><capsule radius="1.0"/></geometry>
    </collision>
    <collision/>
  </link>
  <link name="hip"/>
  <link name="knee"/>
  <link name="foot"/>
  <joint name="fixed_joint" type="fixed">
    <parent link="base"/>
    <child link="hip"/>
  </joint>
  <joint name="hip_joint" type="revolute">
    <parent link="hip"/>
    <child link="knee"/>
    <limit lower="-1.0" upper="2.0" effort="3.0" velocity="4.0"/>
  </joint>
  <joint name="knee_joint" type="continuous">
    <parent link="knee"/>
    <child link="foot"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )

    info = parse_urdf(urdf_path)

    assert info.name == "toy"
    assert info.root_link == "base"
    assert info.joint_names == ("hip_joint", "knee_joint")
    assert info.joints[0].lower == -1.0
    assert info.joints[0].upper == 2.0
    assert info.joints[0].effort == 3.0
    assert info.joints[0].velocity == 4.0
    assert info.joints[1].lower is None
    assert [collision.geometry_type for collision in info.collisions] == [
        "sphere",
        "box",
        "cylinder",
        "mesh",
        "unknown",
    ]
    assert info.collisions[0].xyz == (0.0, 0.0, 0.0)
    assert info.collisions[1].xyz == (1.0, 2.0, 3.0)
    assert info.collisions[1].rpy == (0.1, 0.2, 0.3)
    assert info.collisions[1].size == (0.4, 0.5, 0.6)
    assert info.collisions[2].size == (0.7, 0.8)
    assert info.collisions[3].size == (len("package://mesh.stl"),)

    points = foot_collision_points(info, ("base",), geometry_type="sphere")
    assert np.allclose(points["base"], np.array([[0.0, 0.0, 0.0]]))


def test_foot_collision_points_reports_missing_requested_links(tmp_path) -> None:
    urdf_path = tmp_path / "missing_contacts.urdf"
    urdf_path.write_text(
        """\
<robot>
  <link name="base">
    <collision><geometry><box size="1 1 1"/></geometry></collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    info = parse_urdf(urdf_path)

    with pytest.raises(ValueError, match="No collision-origin contact points"):
        foot_collision_points(info, ("base",), geometry_type="sphere")
    with pytest.raises(ValueError, match="No collision-origin contact points"):
        foot_collision_points(info, ("missing_link",), geometry_type="box")


def test_parse_urdf_rejects_missing_root_missing_link_and_bad_vectors(tmp_path) -> None:
    no_root = tmp_path / "cycle.urdf"
    no_root.write_text(
        """\
<robot>
  <link name="a"/>
  <link name="b"/>
  <joint name="a_to_b" type="fixed"><parent link="a"/><child link="b"/></joint>
  <joint name="b_to_a" type="fixed"><parent link="b"/><child link="a"/></joint>
</robot>
""",
        encoding="utf-8",
    )
    missing_child = tmp_path / "missing_child.urdf"
    missing_child.write_text(
        """\
<robot>
  <link name="base"/>
  <joint name="bad" type="fixed"><parent link="base"/></joint>
</robot>
""",
        encoding="utf-8",
    )
    bad_origin = tmp_path / "bad_origin.urdf"
    bad_origin.write_text(
        """\
<robot>
  <link name="base">
    <collision>
      <origin xyz="1 2"/>
      <geometry><sphere radius="0.1"/></geometry>
    </collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no root link"):
        parse_urdf(no_root)
    with pytest.raises(ValueError, match="missing <child"):
        parse_urdf(missing_child)
    with pytest.raises(ValueError, match="Expected three values"):
        parse_urdf(bad_origin)
