from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import NamedTuple

import numpy as np
import torch

from .assets import RobotAsset, load_asset
from .rotations import matrix_to_quaternion_wxyz, rpy_to_matrix, skew


@dataclass(frozen=True)
class ContactPointSpec:
    """Static contact-frame specification in a link frame.

    Attributes:
        name: Human-readable contact name.
        link_name: Link whose frame owns this contact transform.
        offset: Contact origin translation in ``link_name`` coordinates with
            shape ``(3,)``.
        rpy: Contact-frame roll-pitch-yaw orientation in radians relative to
            ``link_name`` with shape ``(3,)``.
    """

    name: str
    link_name: str
    offset: tuple[float, float, float]
    rpy: tuple[float, float, float]


class ContactPoses(NamedTuple):
    """Batched contact pose outputs.

    Attributes:
        positions: World-frame contact positions with shape
            ``(..., num_contacts, 3)``.
        quaternions_wxyz: Contact-to-world quaternions with shape
            ``(..., num_contacts, 4)`` in ``(w, x, y, z)`` order.
        transforms: Homogeneous contact-to-world transforms ``W_H_C`` with
            shape ``(..., num_contacts, 4, 4)``.
    """

    positions: torch.Tensor
    quaternions_wxyz: torch.Tensor
    transforms: torch.Tensor


class TerrainContactState(NamedTuple):
    """Contact candidate state relative to a terrain model.

    Attributes:
        positions: World-frame contact candidate positions with shape
            ``(..., num_contacts, 3)``.
        nearest_points: World-frame nearest terrain points with shape
            ``(..., num_contacts, 3)``.
        signed_distances: Signed distances along the terrain normal with shape
            ``(..., num_contacts)``. Positive values are above the terrain and
            negative values indicate penetration.
        penetration_depths: Non-negative penetration depths with shape
            ``(..., num_contacts)``.
        normals: World-frame terrain normals at the nearest points with shape
            ``(..., num_contacts, 3)``.
        in_contact: Boolean contact indicators with shape
            ``(..., num_contacts)``.
    """

    positions: torch.Tensor
    nearest_points: torch.Tensor
    signed_distances: torch.Tensor
    penetration_depths: torch.Tensor
    normals: torch.Tensor
    in_contact: torch.Tensor


class ResolvedContactForces(NamedTuple):
    """Contact force estimate returned by a contact resolver.

    Attributes:
        forces: Stacked 3D contact forces in ``force_frame`` with shape
            ``(..., num_contacts, 3)``.
        world_forces: The same forces expressed in the world frame with shape
            ``(..., num_contacts, 3)``.
        normal_forces: Scalar normal force magnitudes with shape
            ``(..., num_contacts)``.
        force_frame: Coordinate frame used by ``forces``.
        active: Boolean active-contact mask with shape ``(..., num_contacts)``.
    """

    forces: torch.Tensor
    world_forces: torch.Tensor
    normal_forces: torch.Tensor
    force_frame: Literal["world", "contact"]
    active: torch.Tensor


class FlatTerrainContactDetector(torch.nn.Module):
    """Detect contact candidates against a flat terrain plane.

    The current terrain model is the plane ``normal.dot(x) = height``. The
    default is the common ``z = 0`` ground plane with upward normal
    ``(0, 0, 1)``. The public output is intentionally independent of this
    implementation detail so a mesh-backed detector can provide the same
    quantities later.
    """

    def __init__(
        self,
        *,
        height: float = 0.0,
        normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
        contact_threshold: float = 0.02,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the flat-terrain contact detector.

        Args:
            height: Plane offset measured along ``normal`` in meters.
            normal: Upward terrain normal in world coordinates.
            contact_threshold: Maximum positive signed distance considered
                contact, in meters.
            dtype: Torch dtype used for detector buffers.
            device: Torch device used for detector buffers. ``None`` selects
                CPU.

        Returns:
            None.
        """
        super().__init__()
        self.height = float(height)
        self.contact_threshold = float(contact_threshold)
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        normal_tensor = torch.as_tensor(normal, dtype=dtype, device=self.device)
        normal_norm = torch.linalg.norm(normal_tensor)
        if normal_norm <= 0:
            raise ValueError("Terrain normal must be non-zero.")
        self.register_buffer("normal", normal_tensor / normal_norm, persistent=False)

    def forward(self, contact_positions: torch.Tensor) -> TerrainContactState:
        """Evaluate terrain contact state for contact candidate positions.

        Args:
            contact_positions: World-frame contact positions with shape
                ``(..., num_contacts, 3)``.

        Returns:
            :class:`TerrainContactState` with one entry per contact candidate.

        Raises:
            ValueError: If the last dimension of ``contact_positions`` is not
                three.
        """
        return self.detect(contact_positions)

    def detect(self, contact_positions: torch.Tensor) -> TerrainContactState:
        """Evaluate terrain contact state for contact candidate positions.

        Args:
            contact_positions: World-frame contact positions with shape
                ``(..., num_contacts, 3)``.

        Returns:
            :class:`TerrainContactState`.

        Raises:
            ValueError: If the last dimension of ``contact_positions`` is not
                three.
        """
        positions = contact_positions.to(dtype=self.dtype, device=self.device)
        if positions.shape[-1] != 3:
            raise ValueError("Contact positions must have last dimension 3.")

        normal = self.normal.to(dtype=positions.dtype, device=positions.device)
        signed_distances = torch.sum(positions * normal, dim=-1) - self.height
        nearest_points = positions - signed_distances.unsqueeze(-1) * normal
        penetration_depths = torch.clamp(-signed_distances, min=0.0)
        normals = normal.expand_as(positions)
        in_contact = signed_distances <= self.contact_threshold
        return TerrainContactState(
            positions=positions,
            nearest_points=nearest_points,
            signed_distances=signed_distances,
            penetration_depths=penetration_depths,
            normals=normals,
            in_contact=in_contact,
        )


class BasicContactForceResolver(torch.nn.Module):
    """Estimate normal contact forces for detected flat-terrain contacts.

    The resolver combines a simple penalty normal force with an optional total
    normal support force distributed over active contacts. It is deliberately
    basic: it provides deterministic, inspectable contact forces for
    visualization and debugging, not a full complementarity solver.
    """

    def __init__(
        self,
        *,
        force_frame: Literal["world", "contact"] = "world",
        normal_stiffness: float = 15_000.0,
        normal_damping: float = 150.0,
        tangential_damping: float = 0.0,
        friction_coefficient: float = 0.8,
    ) -> None:
        """Initialize the contact-force resolver.

        Args:
            force_frame: Coordinate frame for the returned ``forces`` tensor.
                The default is ``"world"``.
            normal_stiffness: Penalty stiffness in N/m applied to penetration.
            normal_damping: Damping in N/(m/s) applied to closing normal
                velocity.
            tangential_damping: Optional viscous tangential damping in
                N/(m/s), friction-limited by ``friction_coefficient``.
            friction_coefficient: Coulomb limit used for tangential damping.

        Returns:
            None.

        Raises:
            ValueError: If ``force_frame`` is not ``"world"`` or
                ``"contact"``.
        """
        super().__init__()
        if force_frame not in {"world", "contact"}:
            raise ValueError("force_frame must be 'world' or 'contact'.")
        self.force_frame = force_frame
        self.normal_stiffness = float(normal_stiffness)
        self.normal_damping = float(normal_damping)
        self.tangential_damping = float(tangential_damping)
        self.friction_coefficient = float(friction_coefficient)

    def forward(
        self,
        contact_state: TerrainContactState,
        *,
        contact_velocities: torch.Tensor | None = None,
        total_normal_force: torch.Tensor | float | None = None,
        contact_poses: ContactPoses | None = None,
    ) -> ResolvedContactForces:
        """Resolve contact forces for a terrain contact state.

        Args:
            contact_state: Terrain contact state returned by a detector.
            contact_velocities: Optional world-frame contact velocities with
                shape ``(..., num_contacts, 3)``.
            total_normal_force: Optional non-negative normal force budget
                distributed over active contacts. Shape must broadcast to
                ``contact_state.in_contact.shape[:-1]``.
            contact_poses: Contact poses required when ``force_frame`` is
                ``"contact"``.

        Returns:
            :class:`ResolvedContactForces`.

        Raises:
            ValueError: If contact-frame output is requested without
                ``contact_poses``.
        """
        return self.resolve(
            contact_state,
            contact_velocities=contact_velocities,
            total_normal_force=total_normal_force,
            contact_poses=contact_poses,
        )

    def resolve(
        self,
        contact_state: TerrainContactState,
        *,
        contact_velocities: torch.Tensor | None = None,
        total_normal_force: torch.Tensor | float | None = None,
        contact_poses: ContactPoses | None = None,
    ) -> ResolvedContactForces:
        """Resolve contact forces for a terrain contact state.

        Args:
            contact_state: Terrain contact state returned by a detector.
            contact_velocities: Optional world-frame contact velocities with
                shape ``(..., num_contacts, 3)``.
            total_normal_force: Optional non-negative normal force budget
                distributed over active contacts.
            contact_poses: Contact poses required for contact-frame output.

        Returns:
            :class:`ResolvedContactForces`.
        """
        positions = contact_state.positions
        normals = contact_state.normals.to(dtype=positions.dtype, device=positions.device)
        active = contact_state.in_contact
        active_float = active.to(dtype=positions.dtype)

        normal_force = self.normal_stiffness * contact_state.penetration_depths
        if contact_velocities is not None:
            velocities = contact_velocities.to(dtype=positions.dtype, device=positions.device)
            normal_velocity = torch.sum(velocities * normals, dim=-1)
            normal_force = normal_force + self.normal_damping * torch.clamp(-normal_velocity, min=0.0)

        normal_force = torch.clamp(normal_force, min=0.0) * active_float
        if total_normal_force is not None:
            total = torch.as_tensor(total_normal_force, dtype=positions.dtype, device=positions.device)
            while total.ndim < active.ndim - 1:
                total = total.unsqueeze(-1)
            active_count = torch.clamp(active_float.sum(dim=-1), min=1.0)
            support_force = torch.clamp(total, min=0.0) / active_count
            normal_force = normal_force + support_force.unsqueeze(-1) * active_float

        world_forces = normal_force.unsqueeze(-1) * normals
        if contact_velocities is not None and self.tangential_damping > 0.0:
            velocities = contact_velocities.to(dtype=positions.dtype, device=positions.device)
            normal_velocity = torch.sum(velocities * normals, dim=-1, keepdim=True) * normals
            tangential_velocity = velocities - normal_velocity
            tangential_force = -self.tangential_damping * tangential_velocity
            tangential_norm = torch.linalg.norm(tangential_force, dim=-1, keepdim=True)
            max_tangential = self.friction_coefficient * normal_force.unsqueeze(-1)
            scale = torch.minimum(
                torch.ones_like(tangential_norm),
                max_tangential / torch.clamp(tangential_norm, min=1e-12),
            )
            world_forces = world_forces + tangential_force * scale * active_float.unsqueeze(-1)

        if self.force_frame == "world":
            forces = world_forces
        else:
            if contact_poses is None:
                raise ValueError("contact_poses are required for contact-frame force output.")
            rotations = contact_poses.transforms[..., :3, :3].to(
                dtype=positions.dtype, device=positions.device
            )
            forces = torch.matmul(rotations.transpose(-1, -2), world_forces.unsqueeze(-1)).squeeze(-1)

        return ResolvedContactForces(
            forces=forces,
            world_forces=world_forces,
            normal_forces=normal_force,
            force_frame=self.force_frame,
            active=active,
        )


class FloatingBaseContactModel(torch.nn.Module):
    """Differentiable contact-pose kinematics for floating-base legged robots.

    Contact frames are initialized from collision geometry in the URDF. For the
    Unitree G1, the ankle-roll links contain four small sphere collision origins
    per foot, which are used as foot-corner contact candidates.

    The returned pose convention follows the Adam homogeneous-transform
    convention: ``W_H_C`` maps contact-frame coordinates into world coordinates.
    Quaternions are stored as ``(w, x, y, z)`` and represent the same
    contact-to-world orientation. The contact normal is the contact frame's
    positive z-axis expressed in world coordinates.
    """

    def __init__(
        self,
        asset: str | RobotAsset = "unitree_g1",
        *,
        kin_dyn=None,
        mode: str = "feet_corners",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize contact frames from a floating-base robot asset.

        Args:
            asset: Built-in asset alias, direct URDF path, or pre-resolved
                :class:`RobotAsset`.
            kin_dyn: Adam ``KinDynComputations`` instance used for link
                forward kinematics and Jacobians. Public kinematic methods
                require this object.
            mode: Contact extraction mode. ``"feet_corners"`` creates one
                contact frame per foot-corner collision sphere; ``"feet_centers"``
                creates one averaged frame per foot.
            dtype: Torch dtype used for contact buffers.
            device: Torch device used for contact buffers. ``None`` selects CPU.

        Returns:
            None.
        """
        super().__init__()
        self.asset = load_asset(asset) if isinstance(asset, str) else asset
        self.mode = mode
        self.kindyn = kin_dyn
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        specs = _contact_specs_from_asset(self.asset, mode)
        self.contact_specs = tuple(specs)
        self.contact_names = tuple(spec.name for spec in specs)
        self.contact_link_names = tuple(spec.link_name for spec in specs)
        unique_link_names: list[str] = []
        contact_link_indices: list[int] = []
        for spec in specs:
            if spec.link_name not in unique_link_names:
                unique_link_names.append(spec.link_name)
            contact_link_indices.append(unique_link_names.index(spec.link_name))
        self.unique_contact_link_names = tuple(unique_link_names)
        offsets = torch.as_tensor([spec.offset for spec in specs], dtype=dtype, device=self.device)
        local_rpy = torch.as_tensor([spec.rpy for spec in specs], dtype=dtype, device=self.device)
        link_indices = torch.as_tensor(contact_link_indices, dtype=torch.long, device=self.device)
        self.register_buffer("contact_link_indices", link_indices, persistent=False)
        self.register_buffer("local_offsets", offsets, persistent=False)
        self.register_buffer("local_rotations", rpy_to_matrix(local_rpy), persistent=False)

    @property
    def num_contacts(self) -> int:
        """Return the number of configured contact frames.

        Args:
            None.

        Returns:
            Number of contacts. Stacked contact position tensors use this as
            ``num_contacts``.
        """
        return len(self.contact_specs)

    @property
    def force_dim(self) -> int:
        """Return the stacked 3D contact force dimension.

        Args:
            None.

        Returns:
            Integer ``3 * num_contacts``.
        """
        return 3 * self.num_contacts

    def contact_positions(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Return world-frame contact point positions.

        Args:
            base_transform: Base-to-world transform ``W_H_B`` with shape
                ``(4, 4)`` or ``(batch, 4, 4)``.
            joint_positions: Joint position tensor with shape ``(n_joints,)``
                or ``(batch, n_joints)`` in ``asset.joint_names`` order.

        Returns:
            World-frame contact positions with shape ``(num_contacts, 3)`` for
            a single state or ``(batch, num_contacts, 3)`` for batched inputs.
        """
        return self.contact_transforms(base_transform, joint_positions)[..., :3, 3]

    def contact_quaternions(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Return contact-to-world quaternions.

        Args:
            base_transform: Base-to-world transform ``W_H_B`` with shape
                ``(4, 4)`` or ``(batch, 4, 4)``.
            joint_positions: Joint position tensor with shape ``(n_joints,)``
                or ``(batch, n_joints)``.

        Returns:
            Quaternion tensor with shape ``(num_contacts, 4)`` or
            ``(batch, num_contacts, 4)`` in Adam/scalar-first ``(w, x, y, z)``
            order.
        """
        rotations = self.contact_transforms(base_transform, joint_positions)[..., :3, :3]
        return matrix_to_quaternion_wxyz(rotations)

    def contact_poses(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> ContactPoses:
        """Return world-frame SE(3) contact poses.

        Args:
            base_transform: Base-to-world transform ``W_H_B`` with shape
                ``(4, 4)`` or ``(batch, 4, 4)``.
            joint_positions: Joint position tensor with shape ``(n_joints,)``
                or ``(batch, n_joints)``.

        Returns:
            :class:`ContactPoses` containing positions with shape
            ``(..., num_contacts, 3)``, quaternions with shape
            ``(..., num_contacts, 4)``, and transforms with shape
            ``(..., num_contacts, 4, 4)``.
        """
        transforms = self.contact_transforms(base_transform, joint_positions)
        return ContactPoses(
            positions=transforms[..., :3, 3],
            quaternions_wxyz=matrix_to_quaternion_wxyz(transforms[..., :3, :3]),
            transforms=transforms,
        )

    def contact_normals(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Return each contact frame's positive z-axis in world coordinates.

        Args:
            base_transform: Base-to-world transform ``W_H_B`` with shape
                ``(4, 4)`` or ``(batch, 4, 4)``.
            joint_positions: Joint position tensor with shape ``(n_joints,)``
                or ``(batch, n_joints)``.

        Returns:
            Unit normal tensor with shape ``(num_contacts, 3)`` or
            ``(batch, num_contacts, 3)``.
        """
        transforms = self.contact_transforms(base_transform, joint_positions)
        return transforms[..., :3, 2]

    def contact_transforms(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Return ``W_H_C`` contact transforms.

        Args:
            base_transform: Base-to-world transform ``W_H_B`` with shape
                ``(4, 4)`` or ``(batch, 4, 4)``.
            joint_positions: Joint position tensor with shape ``(n_joints,)``
                or ``(batch, n_joints)``.

        Returns:
            Contact-to-world transforms with shape ``(num_contacts, 4, 4)`` for
            a single state or ``(batch, num_contacts, 4, 4)`` for batched
            inputs.
        """
        link_transforms = self._contact_link_tensors(
            self._stack_link_transforms(base_transform, joint_positions)
        )
        rotation = link_transforms[..., :3, :3]
        translation = link_transforms[..., :3, 3]
        offsets = self.local_offsets.to(dtype=rotation.dtype, device=rotation.device)
        local_rotations = self.local_rotations.to(dtype=rotation.dtype, device=rotation.device)

        contact_rotation = torch.matmul(rotation, local_rotations)
        contact_translation = translation + torch.matmul(
            rotation, offsets.unsqueeze(-1)
        ).squeeze(-1)

        contact_transform = torch.zeros(
            *contact_rotation.shape[:-2],
            4,
            4,
            dtype=rotation.dtype,
            device=rotation.device,
        )
        contact_transform[..., :3, :3] = contact_rotation
        contact_transform[..., :3, 3] = contact_translation
        contact_transform[..., 3, 3] = 1.0
        return contact_transform

    def contact_jacobian(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Return the translational contact Jacobian in world coordinates.

        The output maps Adam mixed generalized velocity
        ``nu = (v_WB, omega_WB, s_dot)`` to stacked contact point velocities.

        Args:
            base_transform: Base-to-world transform ``W_H_B`` with shape
                ``(4, 4)`` or ``(batch, 4, 4)``.
            joint_positions: Joint position tensor with shape ``(n_joints,)``
                or ``(batch, n_joints)``.

        Returns:
            Translational Jacobian with shape
            ``(3 * num_contacts, 6 + n_joints)`` or batched shape
            ``(batch, 3 * num_contacts, 6 + n_joints)``.
        """
        link_transforms = self._contact_link_tensors(
            self._stack_link_transforms(base_transform, joint_positions)
        )
        link_jacobians = self._contact_link_tensors(
            self._stack_link_jacobians(base_transform, joint_positions)
        )
        rotation = link_transforms[..., :3, :3]
        offsets = self.local_offsets.to(dtype=rotation.dtype, device=rotation.device)
        r_world = torch.matmul(rotation, offsets.unsqueeze(-1)).squeeze(-1)
        linear = link_jacobians[..., :3, :]
        angular = link_jacobians[..., 3:6, :]
        point_jacobian = linear - torch.matmul(skew(r_world), angular)
        return point_jacobian.reshape(
            *point_jacobian.shape[:-3],
            3 * self.num_contacts,
            point_jacobian.shape[-1],
        )

    def contact_spatial_jacobian(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Return stacked Adam-order spatial Jacobians at the contact frames.

        Each 6-row block is ``(linear_velocity, angular_velocity)`` in world
        coordinates, matching the mixed-representation convention used for
        contact poses and contact force mapping.

        Args:
            base_transform: Base-to-world transform ``W_H_B`` with shape
                ``(4, 4)`` or ``(batch, 4, 4)``.
            joint_positions: Joint position tensor with shape ``(n_joints,)``
                or ``(batch, n_joints)``.

        Returns:
            Spatial Jacobian with shape
            ``(6 * num_contacts, 6 + n_joints)`` or batched shape
            ``(batch, 6 * num_contacts, 6 + n_joints)``.
        """
        link_transforms = self._contact_link_tensors(
            self._stack_link_transforms(base_transform, joint_positions)
        )
        link_jacobians = self._contact_link_tensors(
            self._stack_link_jacobians(base_transform, joint_positions)
        )
        rotation = link_transforms[..., :3, :3]
        offsets = self.local_offsets.to(dtype=rotation.dtype, device=rotation.device)
        r_world = torch.matmul(rotation, offsets.unsqueeze(-1)).squeeze(-1)
        linear = link_jacobians[..., :3, :] - torch.matmul(
            skew(r_world), link_jacobians[..., 3:6, :]
        )
        spatial_blocks = torch.cat((linear, link_jacobians[..., 3:6, :]), dim=-2)
        return spatial_blocks.reshape(
            *spatial_blocks.shape[:-3],
            6 * self.num_contacts,
            spatial_blocks.shape[-1],
        )

    def contact_force_transform(
        self,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        *,
        force_frame: str,
    ) -> torch.Tensor:
        """Return the block map from contact-force coordinates to world forces.

        Args:
            base_transform: Base-to-world transform ``W_H_B`` with shape
                ``(4, 4)`` or ``(batch, 4, 4)``.
            joint_positions: Joint position tensor with shape ``(n_joints,)``
                or ``(batch, n_joints)``.
            force_frame: ``"world"`` to leave stacked contact forces unchanged
                or ``"contact"`` to rotate each contact-frame force into the
                world frame.

        Returns:
            Block-diagonal transform with shape
            ``(3 * num_contacts, 3 * num_contacts)`` or batched shape
            ``(batch, 3 * num_contacts, 3 * num_contacts)``.

        Raises:
            ValueError: If ``force_frame`` is not ``"world"`` or ``"contact"``.
        """
        if force_frame == "world":
            dim = self.force_dim
            batch_shape = base_transform.shape[:-2]
            eye = torch.eye(dim, dtype=base_transform.dtype, device=base_transform.device)
            return eye.expand(*batch_shape, dim, dim)
        if force_frame != "contact":
            raise ValueError("force_frame must be 'world' or 'contact'.")

        rotations = self.contact_transforms(base_transform, joint_positions)[..., :3, :3]
        return _block_diag_rotations(rotations)

    def _fk(
        self, link_name: str, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate Adam forward kinematics for a link.

        Args:
            link_name: Link/frame name known to Adam.
            base_transform: Base-to-world transform ``W_H_B`` with shape
                ``(4, 4)`` or ``(batch, 4, 4)``.
            joint_positions: Joint position tensor with shape ``(n_joints,)``
                or ``(batch, n_joints)``.

        Returns:
            Link-to-world transform with shape ``(4, 4)`` or
            ``(batch, 4, 4)``.

        Raises:
            RuntimeError: If no Adam kinematics object was provided.
        """
        if self.kindyn is None:
            raise RuntimeError("FloatingBaseContactModel requires an Adam KinDynComputations instance.")
        return self.kindyn.forward_kinematics(link_name, base_transform, joint_positions)

    def _stack_link_transforms(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate FK once per unique contact link and stack the results.

        Args:
            base_transform: Base-to-world transform ``W_H_B`` with shape
                ``(4, 4)`` or ``(batch, 4, 4)``.
            joint_positions: Joint position tensor with shape ``(n_joints,)``
                or ``(batch, n_joints)``.

        Returns:
            Link-to-world transforms with shape
            ``(num_unique_contact_links, 4, 4)`` or batched shape
            ``(batch, num_unique_contact_links, 4, 4)``.
        """
        transforms = [
            self._fk(link_name, base_transform, joint_positions)
            for link_name in self.unique_contact_link_names
        ]
        return torch.stack(transforms, dim=-3)

    def _jacobian(
        self, link_name: str, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate Adam mixed-representation link Jacobian.

        Args:
            link_name: Link/frame name known to Adam.
            base_transform: Base-to-world transform ``W_H_B`` with shape
                ``(4, 4)`` or ``(batch, 4, 4)``.
            joint_positions: Joint position tensor with shape ``(n_joints,)``
                or ``(batch, n_joints)``.

        Returns:
            Spatial Jacobian with shape ``(6, 6 + n_joints)`` or
            ``(batch, 6, 6 + n_joints)``. Rows are Adam-order
            ``(linear_velocity, angular_velocity)``.

        Raises:
            RuntimeError: If no Adam kinematics object was provided.
        """
        if self.kindyn is None:
            raise RuntimeError("FloatingBaseContactModel requires an Adam KinDynComputations instance.")
        return self.kindyn.jacobian(link_name, base_transform, joint_positions)

    def _stack_link_jacobians(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate Jacobians once per unique contact link and stack them.

        Args:
            base_transform: Base-to-world transform ``W_H_B`` with shape
                ``(4, 4)`` or ``(batch, 4, 4)``.
            joint_positions: Joint position tensor with shape ``(n_joints,)``
                or ``(batch, n_joints)``.

        Returns:
            Adam-order spatial Jacobians with shape
            ``(num_unique_contact_links, 6, 6 + n_joints)`` or batched shape
            ``(batch, num_unique_contact_links, 6, 6 + n_joints)``.
        """
        jacobians = [
            self._jacobian(link_name, base_transform, joint_positions)
            for link_name in self.unique_contact_link_names
        ]
        return torch.stack(jacobians, dim=-3)

    def _contact_link_tensors(self, unique_link_tensors: torch.Tensor) -> torch.Tensor:
        """Gather unique-link tensors into per-contact tensors.

        Args:
            unique_link_tensors: Tensor whose third-from-last dimension indexes
                ``num_unique_contact_links``. Supported shapes include
                ``(num_unique_contact_links, 4, 4)``,
                ``(batch, num_unique_contact_links, 4, 4)``,
                ``(num_unique_contact_links, 6, nv)``, and
                ``(batch, num_unique_contact_links, 6, nv)``.

        Returns:
            Tensor with the same shape as ``unique_link_tensors`` except the
            third-from-last dimension is expanded to ``num_contacts`` according
            to ``self.contact_link_indices``.
        """
        indices = self.contact_link_indices.to(device=unique_link_tensors.device)
        return torch.index_select(unique_link_tensors, dim=-3, index=indices)


def _contact_specs_from_asset(asset: RobotAsset, mode: str) -> list[ContactPointSpec]:
    """Build contact specifications from parsed asset collision geometry.

    Args:
        asset: Resolved robot asset with parsed URDF collision metadata.
        mode: Contact extraction mode, ``"feet_corners"`` or
            ``"feet_centers"``.

    Returns:
        List of :class:`ContactPointSpec` entries. Each entry stores one
        contact origin with shape ``(3,)`` and one contact orientation with
        shape ``(3,)`` in the owning link frame.

    Raises:
        ValueError: If the asset has no default contact links, if required foot
            collision spheres are missing, or if ``mode`` is unknown.
    """
    if not asset.default_contact_links:
        raise ValueError(f"Asset {asset.name!r} has no default foot contact links.")

    foot_collisions = {
        link: [
            collision
            for collision in asset.urdf.collisions
            if collision.link_name == link and collision.geometry_type == "sphere"
        ]
        for link in asset.default_contact_links
    }
    missing = [link for link, collisions in foot_collisions.items() if not collisions]
    if missing:
        raise ValueError(f"No sphere collision contact candidates found for {missing}.")

    specs: list[ContactPointSpec] = []
    if mode == "feet_corners":
        for link_name, collisions in foot_collisions.items():
            for index, collision in enumerate(_sort_foot_collisions(collisions)):
                specs.append(
                    ContactPointSpec(
                        name=f"{link_name}:{index}",
                        link_name=link_name,
                        offset=tuple(float(value) for value in collision.xyz),
                        rpy=tuple(float(value) for value in collision.rpy),
                    )
                )
        return specs

    if mode == "feet_centers":
        for link_name, collisions in foot_collisions.items():
            points = np.asarray([collision.xyz for collision in collisions], dtype=np.float64)
            center = np.mean(points, axis=0)
            specs.append(
                ContactPointSpec(
                    name=f"{link_name}:center",
                    link_name=link_name,
                    offset=tuple(float(value) for value in center),
                    rpy=(0.0, 0.0, 0.0),
                )
            )
        return specs

    raise ValueError("Unknown contact mode. Expected 'feet_corners' or 'feet_centers'.")


def _sort_foot_collisions(collisions):
    """Sort foot collision candidates in a stable heel/toe, lateral order.

    Args:
        collisions: Sequence of collision metadata objects with ``xyz`` fields
            of shape ``(3,)``.

    Returns:
        List of collisions sorted by x coordinate and then y coordinate.
    """
    # Stable order: heel/toe by x, then left/right by y.
    points = np.asarray([collision.xyz for collision in collisions], dtype=np.float64)
    order = np.lexsort((points[:, 1], points[:, 0]))
    return [collisions[index] for index in order]


def _block_diag_rotations(rotations: torch.Tensor) -> torch.Tensor:
    """Create a block-diagonal matrix from contact rotations.

    Args:
        rotations: Rotation matrix tensor with shape
            ``(..., num_contacts, 3, 3)``.

    Returns:
        Block-diagonal tensor with shape
        ``(..., 3 * num_contacts, 3 * num_contacts)``.
    """
    num_contacts = rotations.shape[-3]
    eye = torch.eye(num_contacts, dtype=rotations.dtype, device=rotations.device)
    blocks = torch.einsum("...cij,cd->...cidj", rotations, eye)
    return blocks.reshape(
        *rotations.shape[:-3],
        3 * num_contacts,
        3 * num_contacts,
    )


HumanoidContactModel = FloatingBaseContactModel
