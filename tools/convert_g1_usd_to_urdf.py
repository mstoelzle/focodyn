#!/usr/bin/env python3
"""Generate FoCoDyn's URDF companions from the supplied minimal G1 USD.

This is an offline asset-maintenance tool, not a general USD-to-URDF importer.
It uses the standalone OpenUSD Python bindings and does not require Isaac Sim.
Movable joints are emitted in USD stage traversal order; fixed joints follow,
also retaining their relative stage traversal order.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import struct
import xml.etree.ElementTree as ET

import numpy as np

from focodyn.rotations import (
    matrix_to_rpy_numpy,
    quaternion_xyzw_to_matrix_numpy,
    transform_from_position_quaternion_xyzw_numpy,
)

try:
    from pxr import Usd, UsdGeom, UsdPhysics
except ImportError as error:  # pragma: no cover - exercised without tool dependency
    raise SystemExit(
        "OpenUSD Python bindings are required. Install FoCoDyn's optional USD "
        "dependencies and run the converter with:\n"
        "  uv run --python 3.12 --extra usd python "
        "tools/convert_g1_usd_to_urdf.py --help"
    ) from error


EXPECTED_MOVABLE_JOINT_NAMES = frozenset(
    {
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
        "left_zero_joint",
        "left_one_joint",
        "left_two_joint",
        "left_three_joint",
        "left_four_joint",
        "left_five_joint",
        "left_six_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
        "right_zero_joint",
        "right_one_joint",
        "right_two_joint",
        "right_three_joint",
        "right_four_joint",
        "right_five_joint",
        "right_six_joint",
    }
)


def _fmt(values) -> str:
    if isinstance(values, (float, int, np.floating)):
        return f"{float(values):.12g}"
    return " ".join(f"{float(value):.12g}" for value in values)


def _quat_xyzw(quaternion) -> np.ndarray:
    imaginary = quaternion.GetImaginary()
    return np.asarray(
        [
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
            float(quaternion.GetReal()),
        ],
        dtype=np.float64,
    )


def _rpy_from_quat(quaternion) -> tuple[float, float, float]:
    return matrix_to_rpy_numpy(quaternion_xyzw_to_matrix_numpy(_quat_xyzw(quaternion)))


def _visual_triangles(
    stage, link_prim
) -> tuple[list[np.ndarray], tuple[float, float, float, float]]:
    visuals = stage.GetPrimAtPath(link_prim.GetPath().AppendChild("visuals"))
    if not visuals:
        return [], (0.7, 0.7, 0.7, 1.0)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    triangles: list[np.ndarray] = []
    color = (0.7, 0.7, 0.7, 1.0)
    for prim in Usd.PrimRange(visuals, Usd.TraverseInstanceProxies()):
        color_attr = prim.GetAttribute("inputs:diffuse_color_constant")
        if color_attr:
            value = color_attr.Get()
            if value is not None:
                color = (float(value[0]), float(value[1]), float(value[2]), 1.0)
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        relative, _ = xform_cache.ComputeRelativeTransform(prim, link_prim)
        transformed = np.asarray(
            [tuple(relative.Transform(point)) for point in points], dtype=np.float64
        )
        cursor = 0
        for count in counts:
            face = [int(index) for index in indices[cursor : cursor + count]]
            cursor += count
            for offset in range(1, len(face) - 1):
                triangles.append(transformed[[face[0], face[offset], face[offset + 1]]])
    return triangles, color


def _write_binary_stl(path: Path, triangles: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"FoCoDyn g1_37dof_minimal".ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
            stream.write(struct.pack("<3f", *normal.astype(np.float32)))
            for point in triangle:
                stream.write(struct.pack("<3f", *point.astype(np.float32)))
            stream.write(struct.pack("<H", 0))


def _add_origin(parent: ET.Element, xyz, rpy) -> None:
    ET.SubElement(parent, "origin", {"xyz": _fmt(xyz), "rpy": _fmt(rpy)})


def _add_link(
    robot: ET.Element,
    stage,
    link_prim,
    mesh_dir: Path,
    mesh_reference_dir: str,
) -> None:
    name = link_prim.GetName()
    link = ET.SubElement(robot, "link", {"name": name})
    mass_api = UsdPhysics.MassAPI(link_prim)
    mass = mass_api.GetMassAttr().Get()
    center = mass_api.GetCenterOfMassAttr().Get()
    diagonal = mass_api.GetDiagonalInertiaAttr().Get()
    principal = mass_api.GetPrincipalAxesAttr().Get()
    if mass is not None and diagonal is not None:
        center_values = (
            np.asarray(center, dtype=np.float64) if center is not None else np.zeros(3)
        )
        if not np.all(np.isfinite(center_values)):
            center_values = np.zeros(3)
        inertial = ET.SubElement(link, "inertial")
        _add_origin(
            inertial,
            center_values,
            _rpy_from_quat(principal) if principal is not None else (0.0, 0.0, 0.0),
        )
        ET.SubElement(inertial, "mass", {"value": _fmt(mass)})
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": _fmt(diagonal[0]),
                "ixy": "0",
                "ixz": "0",
                "iyy": _fmt(diagonal[1]),
                "iyz": "0",
                "izz": _fmt(diagonal[2]),
            },
        )

    triangles, color = _visual_triangles(stage, link_prim)
    if triangles:
        mesh_name = f"{name}.stl"
        _write_binary_stl(mesh_dir / mesh_name, triangles)
        visual = ET.SubElement(link, "visual")
        _add_origin(visual, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(
            geometry, "mesh", {"filename": f"{mesh_reference_dir}/{mesh_name}"}
        )
        material = ET.SubElement(visual, "material", {"name": f"{name}_material"})
        ET.SubElement(material, "color", {"rgba": _fmt(color)})

    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetParent() != link_prim:
            continue
        translation = prim.GetAttribute("xformOp:translate").Get() or (0.0, 0.0, 0.0)
        rotation = prim.GetAttribute("xformOp:orient").Get()
        scale = prim.GetAttribute("xformOp:scale").Get() or (1.0, 1.0, 1.0)
        if prim.IsA(UsdGeom.Cube):
            local_size = np.full(3, UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0)
        elif prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            counts = tuple(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or ())
            if points.shape != (8, 3) or counts != (4, 4, 4, 4, 4, 4):
                raise RuntimeError(
                    f"Collision mesh {prim.GetPath()} is not the expected box."
                )
            local_size = np.max(points, axis=0) - np.min(points, axis=0)
        else:
            raise RuntimeError(
                f"Unsupported collision geometry {prim.GetTypeName()!r} at "
                f"{prim.GetPath()}; this converter supports box collisions."
            )
        size = np.abs(np.asarray(scale, dtype=np.float64)) * local_size
        collision = ET.SubElement(link, "collision")
        _add_origin(
            collision,
            translation,
            _rpy_from_quat(rotation) if rotation is not None else (0.0, 0.0, 0.0),
        )
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "box", {"size": _fmt(size)})


def _joint_record(prim) -> dict:
    body0_targets = prim.GetRelationship("physics:body0").GetTargets()
    body1_targets = prim.GetRelationship("physics:body1").GetTargets()
    if len(body0_targets) != 1 or len(body1_targets) != 1:
        raise RuntimeError(f"Joint {prim.GetPath()} must connect exactly two bodies.")
    pos0 = prim.GetAttribute("physics:localPos0").Get()
    pos1 = prim.GetAttribute("physics:localPos1").Get()
    rot0 = prim.GetAttribute("physics:localRot0").Get()
    rot1 = prim.GetAttribute("physics:localRot1").Get()
    origin = transform_from_position_quaternion_xyzw_numpy(
        pos0, _quat_xyzw(rot0)
    ) @ np.linalg.inv(
        transform_from_position_quaternion_xyzw_numpy(pos1, _quat_xyzw(rot1))
    )
    return {
        "prim": prim,
        "name": prim.GetName(),
        "parent": body0_targets[0].name,
        "child": body1_targets[0].name,
        "xyz": origin[:3, 3],
        "rpy": matrix_to_rpy_numpy(origin[:3, :3]),
    }


def _add_joint(robot: ET.Element, record: dict, *, movable: bool) -> None:
    prim = record["prim"]
    joint = ET.SubElement(
        robot,
        "joint",
        {"name": record["name"], "type": "revolute" if movable else "fixed"},
    )
    _add_origin(joint, record["xyz"], record["rpy"])
    ET.SubElement(joint, "parent", {"link": record["parent"]})
    ET.SubElement(joint, "child", {"link": record["child"]})
    if not movable:
        return
    axis_name = str(prim.GetAttribute("physics:axis").Get())
    try:
        axis = {
            "X": (1.0, 0.0, 0.0),
            "Y": (0.0, 1.0, 0.0),
            "Z": (0.0, 0.0, 1.0),
        }[axis_name]
    except KeyError as error:
        raise RuntimeError(f"Unsupported joint axis {axis_name!r}.") from error
    ET.SubElement(joint, "axis", {"xyz": _fmt(axis)})
    lower = float(prim.GetAttribute("physics:lowerLimit").Get()) * math.pi / 180.0
    upper = float(prim.GetAttribute("physics:upperLimit").Get()) * math.pi / 180.0
    effort = float(prim.GetAttribute("drive:angular:physics:maxForce").Get())
    velocity = (
        float(prim.GetAttribute("physxJoint:maxJointVelocity").Get()) * math.pi / 180.0
    )
    ET.SubElement(
        joint,
        "limit",
        {
            "lower": _fmt(lower),
            "upper": _fmt(upper),
            "effort": _fmt(effort),
            "velocity": _fmt(velocity),
        },
    )


def convert(
    source: Path, output_dir: Path, *, force: bool = False
) -> tuple[Path, Path]:
    """Convert the supplied minimal G1 USD to URDF, STL, and Adam companions."""
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    urdf_path = output_dir / "g1_37dof_minimal.urdf"
    adam_path = output_dir / "g1_37dof_minimal.adam.urdf"
    existing = [path for path in (urdf_path, adam_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"Refusing to overwrite {existing}. Pass --force after reviewing the paths."
        )

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"Could not open {source}")
    robot_prim = stage.GetPrimAtPath("/g1")
    if not robot_prim:
        raise RuntimeError("Expected the supplied robot root prim at /g1.")
    rigid_bodies = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and prim.GetPath().HasPrefix(robot_prim.GetPath())
    ]
    joints = [
        _joint_record(prim)
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_prim.GetPath())
        and (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.FixedJoint))
    ]
    movable_records = [
        record for record in joints if record["prim"].IsA(UsdPhysics.RevoluteJoint)
    ]
    movable_names = tuple(record["name"] for record in movable_records)
    if set(movable_names) != EXPECTED_MOVABLE_JOINT_NAMES or len(movable_names) != 37:
        raise RuntimeError(
            "Unexpected movable joints: "
            f"missing={sorted(EXPECTED_MOVABLE_JOINT_NAMES - set(movable_names))}, "
            f"extra={sorted(set(movable_names) - EXPECTED_MOVABLE_JOINT_NAMES)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_reference_dir = "meshes/g1_37dof_minimal"
    mesh_dir = output_dir / mesh_reference_dir
    robot = ET.Element("robot", {"name": "g1_37dof_minimal"})
    for link_prim in rigid_bodies:
        _add_link(robot, stage, link_prim, mesh_dir, mesh_reference_dir)
    for record in movable_records:
        _add_joint(robot, record, movable=True)
    for record in joints:
        if not record["prim"].IsA(UsdPhysics.RevoluteJoint):
            _add_joint(robot, record, movable=False)

    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    shutil.copyfile(urdf_path, adam_path)
    return urdf_path, adam_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to g1_37dof_minimal.usd")
    parser.add_argument("output_dir", type=Path, help="Directory for companions")
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing URDF companions"
    )
    args = parser.parse_args()
    urdf_path, adam_path = convert(args.source, args.output_dir, force=args.force)
    print(f"Wrote {urdf_path}")
    print(f"Wrote {adam_path}")
    print("Movable joints preserve USD stage traversal order.")


if __name__ == "__main__":
    main()
