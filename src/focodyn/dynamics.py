from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import NamedTuple

import torch

from ._torch import (
    ensure_batch,
    make_transform,
)
from .assets import RobotAsset, load_asset
from .contacts import FloatingBaseContactModel
from .rotations import (
    normalize_quaternion_wxyz,
    quaternion_derivative_from_world_angular_velocity,
)


class SplitState(NamedTuple):
    """Structured view of a floating-base robot state tensor.

    Attributes:
        base_position: World-frame base position with shape ``(batch, 3)``.
        base_quaternion_wxyz: Base-to-world quaternion with shape
            ``(batch, 4)`` in ``(w, x, y, z)`` order.
        joint_positions: Joint position tensor with shape
            ``(batch, n_joints)``.
        base_velocity: Mixed-representation base velocity with shape
            ``(batch, 6)`` storing ``(v_WB, omega_WB)``.
        joint_velocities: Joint velocity tensor with shape
            ``(batch, n_joints)``.
        was_single: Whether the original input state had shape
            ``(state_dim,)`` and was internally promoted to a batch of one.
    """

    base_position: torch.Tensor
    base_quaternion_wxyz: torch.Tensor
    joint_positions: torch.Tensor
    base_velocity: torch.Tensor
    joint_velocities: torch.Tensor
    was_single: bool


@dataclass(frozen=True)
class DynamicsTerms:
    """Rigid-body dynamics terms returned by Adam.

    Attributes:
        mass_matrix: Generalized mass matrix with shape ``(..., nv, nv)``.
        coriolis: Coriolis/centrifugal generalized force vector with shape
            ``(..., nv)``.
        gravity: Gravity generalized force vector with shape ``(..., nv)``.
        bias: Full bias force vector ``coriolis + gravity`` with shape
            ``(..., nv)``.
    """

    mass_matrix: torch.Tensor
    coriolis: torch.Tensor
    gravity: torch.Tensor
    bias: torch.Tensor


class FloatingBaseDynamics(torch.nn.Module):
    r"""Control-affine differentiable floating-base dynamics.

    State convention
    ----------------
    The state is ``x = (q, nu)`` with
    ``q = (p_WB, quat_WB, s)`` and
    ``nu = (v_WB, omega_WB, s_dot)``.

    ``p_WB`` is the world-frame position of the floating-base/root link
    origin. ``quat_WB`` is a unit quaternion in ``(w, x, y, z)`` order and maps
    vectors from the base/root frame ``B`` to the world frame ``W``. The joint
    vector ``s`` is ordered exactly as ``self.joint_names``; this list is parsed
    from the URDF and passed to Adam.

    The floating-base velocity uses Adam's mixed representation:
    ``v_WB`` and ``omega_WB`` are both expressed in the inertial/world frame.
    Therefore the configuration derivative is
    ``p_dot = v_WB`` and ``quat_dot = 0.5 * [0, omega_WB] * quat_WB``.

    Generalized acceleration is
    ``nu_dot = (v_dot_WB, omega_dot_WB, s_ddot)``. Adam provides terms in
    ``M(q) nu_dot + h(q, nu) = S^T tau + J_c(q)^T lambda``. The drift ``f(x)``
    uses zero joint torques and zero contact forces. The input matrix ``g(x)``
    maps joint torques through ``S^T`` and, when enabled, stacked contact
    forces through translational contact Jacobians. Contact forces can be
    represented in world coordinates or in each contact frame, controlled by
    ``contact_force_frame``.
    """

    def __init__(
        self,
        asset_name: str = "unitree_g1",
        *,
        include_contact_forces: bool = False,
        contact_mode: str = "feet_corners",
        contact_force_frame: Literal["world", "contact"] = "world",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize an Adam-backed floating-base dynamics module.

        Args:
            asset_name: Built-in asset alias or direct URDF path. The asset
                determines ``n_joints``, joint ordering, root link, and default
                contact links.
            include_contact_forces: Whether :meth:`g` includes contact-force
                input columns after the joint-torque columns.
            contact_mode: Contact extraction mode passed to
                :class:`FloatingBaseContactModel`.
            contact_force_frame: Frame for optional contact-force inputs.
                ``"world"`` means contact forces are already world-frame
                vectors. ``"contact"`` means each force is expressed in its
                contact frame and is rotated to the world frame before
                applying ``J_c(q)^T``.
            dtype: Torch dtype used for dynamics computations and generated
                tensors.
            device: Torch device used for dynamics computations. ``None``
                selects CPU.

        Returns:
            None.

        Raises:
            ValueError: If ``contact_force_frame`` is not ``"world"`` or
                ``"contact"``.
            ImportError: If Adam's PyTorch backend is unavailable.
        """
        super().__init__()
        self.asset: RobotAsset = load_asset(asset_name)
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.include_contact_forces = include_contact_forces
        self.contact_force_frame = contact_force_frame
        if contact_force_frame not in {"world", "contact"}:
            raise ValueError("contact_force_frame must be 'world' or 'contact'.")

        self.joint_names = self.asset.joint_names
        self.root_link = self.asset.root_link
        self.n_joints = len(self.joint_names)
        self.nv = 6 + self.n_joints
        self.nq = 7 + self.n_joints
        self.state_dim = self.nq + self.nv

        self.kindyn = _build_adam_kindyn(self.asset, self.dtype, self.device)
        self.contact_model: FloatingBaseContactModel | None = None
        if include_contact_forces:
            self.contact_model = FloatingBaseContactModel(
                self.asset,
                kin_dyn=self.kindyn,
                mode=contact_mode,
                dtype=dtype,
                device=self.device,
            )
        self.input_dim = self.n_joints + (
            self.contact_model.force_dim if self.contact_model is not None else 0
        )

    def neutral_state(self, *, base_height: float = 0.78) -> torch.Tensor:
        """Return a neutral floating-base state for smoke tests and examples.

        Args:
            base_height: Initial world-frame base height in meters.

        Returns:
            State tensor with shape ``(state_dim,)``. The quaternion is
            identity ``(1, 0, 0, 0)``, joint positions and velocities are zero,
            and base position is ``(0, 0, base_height)``.
        """
        state = torch.zeros(self.state_dim, dtype=self.dtype, device=self.device)
        state[2] = base_height
        state[3] = 1.0
        return state

    def split_state(self, x: torch.Tensor) -> SplitState:
        """Split a state tensor into named position and velocity blocks.

        Args:
            x: State tensor with shape ``(state_dim,)`` or
                ``(batch, state_dim)``. The state convention is
                ``(p_WB, quat_WB, s, v_WB, omega_WB, s_dot)``.

        Returns:
            :class:`SplitState` whose tensor fields always include a leading
            batch dimension.

        Raises:
            ValueError: If the last dimension of ``x`` is not ``state_dim``.
        """
        x, was_single = ensure_batch(x.to(dtype=self.dtype, device=self.device))
        if x.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, got {x.shape[-1]}")

        joint_start = 7
        velocity_start = 7 + self.n_joints
        base_position = x[..., :3]
        base_quaternion = normalize_quaternion_wxyz(x[..., 3:7])
        joint_positions = x[..., joint_start:velocity_start]
        base_velocity = x[..., velocity_start : velocity_start + 6]
        joint_velocities = x[..., velocity_start + 6 :]
        return SplitState(
            base_position=base_position,
            base_quaternion_wxyz=base_quaternion,
            joint_positions=joint_positions,
            base_velocity=base_velocity,
            joint_velocities=joint_velocities,
            was_single=was_single,
        )

    def base_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Return the base-to-world homogeneous transform from state.

        Args:
            x: State tensor with shape ``(state_dim,)`` or
                ``(batch, state_dim)``.

        Returns:
            Transform ``W_H_B`` with shape ``(4, 4)`` for a single state or
            ``(batch, 4, 4)`` for batched states.
        """
        split = self.split_state(x)
        transform = make_transform(split.base_position, split.base_quaternion_wxyz)
        return transform.squeeze(0) if split.was_single else transform

    def dynamics_terms(self, x: torch.Tensor) -> DynamicsTerms:
        """Compute mass, Coriolis, gravity, and bias terms using Adam.

        Args:
            x: State tensor with shape ``(state_dim,)`` or
                ``(batch, state_dim)``.

        Returns:
            :class:`DynamicsTerms`. For an unbatched input, each tensor is
            squeezed to shapes ``(nv, nv)`` and ``(nv,)``. For batched input,
            shapes are ``(batch, nv, nv)`` and ``(batch, nv)``.
        """
        split = self.split_state(x)
        base_transform = make_transform(split.base_position, split.base_quaternion_wxyz)
        mass = self.kindyn.mass_matrix(base_transform, split.joint_positions)
        coriolis = self.kindyn.coriolis_term(
            base_transform,
            split.joint_positions,
            split.base_velocity,
            split.joint_velocities,
        )
        gravity = self.kindyn.gravity_term(base_transform, split.joint_positions)
        bias = coriolis + gravity
        if split.was_single:
            return DynamicsTerms(
                mass_matrix=mass.squeeze(0),
                coriolis=coriolis.squeeze(0),
                gravity=gravity.squeeze(0),
                bias=bias.squeeze(0),
            )
        return DynamicsTerms(mass_matrix=mass, coriolis=coriolis, gravity=gravity, bias=bias)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        """Return the autonomous drift dynamics ``x_dot = f(x)``.

        The drift uses zero joint torque and zero contact force. It solves
        ``M(q) nu_dot = -h(q, nu)`` and combines ``nu_dot`` with the
        configuration derivative.

        Args:
            x: State tensor with shape ``(state_dim,)`` or
                ``(batch, state_dim)``.

        Returns:
            State derivative with shape ``(state_dim,)`` or
            ``(batch, state_dim)`` matching the input batch convention.
        """
        split = self.split_state(x)
        base_transform = make_transform(split.base_position, split.base_quaternion_wxyz)
        mass = self.kindyn.mass_matrix(base_transform, split.joint_positions)
        bias = self._bias_force(base_transform, split)

        acceleration = torch.linalg.solve(mass, -bias.unsqueeze(-1)).squeeze(-1)
        q_dot = self._configuration_derivative(split)
        x_dot = torch.cat((q_dot, acceleration), dim=-1)
        return x_dot.squeeze(0) if split.was_single else x_dot

    def g(self, x: torch.Tensor) -> torch.Tensor:
        """Return the control matrix in ``x_dot = f(x) + g(x) u``.

        Joint-torque columns are mapped through ``M(q)^{-1} S^T``. When contact
        forces are enabled, the remaining columns are mapped through
        ``M(q)^{-1} J_c(q)^T`` for world-frame forces or
        ``M(q)^{-1} J_c(q)^T R_WC`` for contact-frame forces.

        Args:
            x: State tensor with shape ``(state_dim,)`` or
                ``(batch, state_dim)``.

        Returns:
            Control matrix with shape ``(state_dim, input_dim)`` for a single
            state or ``(batch, state_dim, input_dim)`` for batched states. The
            first ``nq`` rows are zero because controls affect generalized
            acceleration directly.
        """
        split = self.split_state(x)
        base_transform = make_transform(split.base_position, split.base_quaternion_wxyz)
        mass = self.kindyn.mass_matrix(base_transform, split.joint_positions)
        generalized_input = self._generalized_input_matrix(base_transform, split.joint_positions)
        acceleration_map = torch.linalg.solve(mass, generalized_input)

        control_matrix = self._control_matrix_from_acceleration_map(
            acceleration_map,
            batch_size=split.base_position.shape[0],
        )
        return control_matrix.squeeze(0) if split.was_single else control_matrix

    def f_and_g(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return drift and control matrices with shared dynamics work.

        This fused path is useful when both :meth:`f` and :meth:`g` are needed
        for the same state, for example during CBF/CLF construction. It
        evaluates the mass matrix once and solves one block linear system with
        right-hand side ``[-h(q, nu), B(q)]``, where ``B(q)`` maps inputs to
        generalized forces.

        Args:
            x: State tensor with shape ``(state_dim,)`` or
                ``(batch, state_dim)``.

        Returns:
            Tuple ``(drift, control_matrix)``. For a single state, ``drift``
            has shape ``(state_dim,)`` and ``control_matrix`` has shape
            ``(state_dim, input_dim)``. For batched states, shapes are
            ``(batch, state_dim)`` and ``(batch, state_dim, input_dim)``.
        """
        split = self.split_state(x)
        base_transform = make_transform(split.base_position, split.base_quaternion_wxyz)
        mass = self.kindyn.mass_matrix(base_transform, split.joint_positions)
        bias = self._bias_force(base_transform, split)
        if bias.ndim == 1:
            bias = bias.unsqueeze(0)
        generalized_input = self._generalized_input_matrix(base_transform, split.joint_positions)
        rhs = torch.cat((-bias.unsqueeze(-1), generalized_input), dim=-1)
        solution = torch.linalg.solve(mass, rhs)

        acceleration = solution[..., 0]
        acceleration_map = solution[..., 1:]
        q_dot = self._configuration_derivative(split)
        drift = torch.cat((q_dot, acceleration), dim=-1)
        control_matrix = self._control_matrix_from_acceleration_map(
            acceleration_map,
            batch_size=split.base_position.shape[0],
        )
        if split.was_single:
            return drift.squeeze(0), control_matrix.squeeze(0)
        return drift, control_matrix

    def generalized_forces_from_acceleration(
        self,
        x: torch.Tensor,
        generalized_acceleration: torch.Tensor,
    ) -> torch.Tensor:
        """Return generalized forces for a desired generalized acceleration.

        This evaluates ``M(q) nu_dot + h(q, nu)`` using the same Adam terms as
        :meth:`f`. The first six entries are the floating-base generalized
        wrench in Adam's mixed representation. The remaining entries are
        generalized joint forces in model joint order.

        Args:
            x: State tensor with shape ``(state_dim,)`` or
                ``(batch, state_dim)``.
            generalized_acceleration: Desired generalized acceleration
                ``nu_dot`` with shape ``(nv,)`` or ``(batch, nv)``.

        Returns:
            Generalized force tensor with shape ``(nv,)`` or ``(batch, nv)``
            matching the input batch convention.

        Raises:
            ValueError: If ``generalized_acceleration`` does not have trailing
                dimension ``nv``.
        """
        split = self.split_state(x)
        acceleration = generalized_acceleration.to(dtype=self.dtype, device=self.device)
        if acceleration.shape[-1] != self.nv:
            raise ValueError(f"Expected generalized acceleration dimension {self.nv}.")
        if split.was_single and acceleration.ndim == 1:
            acceleration = acceleration.unsqueeze(0)

        base_transform = make_transform(split.base_position, split.base_quaternion_wxyz)
        mass = self.kindyn.mass_matrix(base_transform, split.joint_positions)
        bias = self._bias_force(base_transform, split)
        if bias.ndim == 1:
            bias = bias.unsqueeze(0)
        generalized_force = torch.matmul(mass, acceleration.unsqueeze(-1)).squeeze(-1) + bias
        return generalized_force.squeeze(0) if split.was_single else generalized_force

    def generalized_input_matrix(
        self,
        x: torch.Tensor,
        *,
        contact_force_frame: Literal["world", "contact"] | None = None,
    ) -> torch.Tensor:
        """Return the generalized-force input matrix ``B(q)``.

        ``B(q)`` is the matrix used internally by :meth:`g` before multiplying
        by ``M(q)^{-1}``. Its first columns map joint torques through
        ``S^T``. When contact forces are enabled, the remaining columns map
        stacked contact forces through ``J_c(q)^T``.

        Args:
            x: State tensor with shape ``(state_dim,)`` or
                ``(batch, state_dim)``.
            contact_force_frame: Optional override for contact-force input
                coordinates. If ``None``, ``self.contact_force_frame`` is used.

        Returns:
            Generalized-force input matrix with shape ``(nv, input_dim)`` or
            ``(batch, nv, input_dim)``.
        """
        split = self.split_state(x)
        base_transform = make_transform(split.base_position, split.base_quaternion_wxyz)
        generalized_input = self._generalized_input_matrix(
            base_transform,
            split.joint_positions,
            contact_force_frame=contact_force_frame,
        )
        return generalized_input.squeeze(0) if split.was_single else generalized_input

    def generalized_forces_from_input(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        *,
        contact_force_frame: Literal["world", "contact"] | None = None,
    ) -> torch.Tensor:
        """Project inputs into generalized forces.

        Args:
            x: State tensor with shape ``(state_dim,)`` or
                ``(batch, state_dim)``.
            u: Input tensor with shape ``(input_dim,)`` or
                ``(batch, input_dim)``.
            contact_force_frame: Optional override for contact-force input
                coordinates.

        Returns:
            Generalized forces with shape ``(nv,)`` or ``(batch, nv)``.

        Raises:
            ValueError: If ``u`` does not have trailing dimension
                ``input_dim``.
        """
        split = self.split_state(x)
        inputs = u.to(dtype=self.dtype, device=self.device)
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}.")
        if split.was_single and inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        matrix = self.generalized_input_matrix(x, contact_force_frame=contact_force_frame)
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        generalized_force = torch.matmul(matrix, inputs.unsqueeze(-1)).squeeze(-1)
        return generalized_force.squeeze(0) if split.was_single else generalized_force

    def forward(self, x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate the control-affine dynamics.

        Args:
            x: State tensor with shape ``(state_dim,)`` or
                ``(batch, state_dim)``.
            u: Optional input tensor with shape ``(input_dim,)`` or
                ``(batch, input_dim)``. If ``None``, only :meth:`f` is
                evaluated.

        Returns:
            State derivative with shape ``(state_dim,)`` or
            ``(batch, state_dim)``.
        """
        if u is None:
            return self.f(x)
        drift, control = self.f_and_g(x)
        return drift + torch.matmul(control, u.unsqueeze(-1)).squeeze(-1)

    def selection_matrix_transpose(self, *, batch_size: int | None = None) -> torch.Tensor:
        """Return ``S^T`` mapping joint torques to generalized forces.

        Args:
            batch_size: Optional batch size. If provided, the returned tensor is
                expanded across this batch dimension.

        Returns:
            Selection matrix with shape ``(nv, n_joints)`` when
            ``batch_size`` is ``None`` or ``(batch_size, nv, n_joints)``
            otherwise.
        """
        selection = torch.zeros(self.nv, self.n_joints, dtype=self.dtype, device=self.device)
        selection[6:, :] = torch.eye(self.n_joints, dtype=self.dtype, device=self.device)
        if batch_size is None:
            return selection
        return selection.expand(batch_size, self.nv, self.n_joints)

    def _configuration_derivative(self, split: SplitState) -> torch.Tensor:
        """Compute ``q_dot`` from a split state.

        Args:
            split: Split state with batched fields. ``base_velocity`` has shape
                ``(batch, 6)`` and stores world-frame linear and angular base
                velocity.

        Returns:
            Configuration derivative with shape ``(batch, nq)`` ordered as
            ``(p_dot, quat_dot, s_dot)``.
        """
        p_dot = split.base_velocity[..., :3]
        omega_world = split.base_velocity[..., 3:6]
        quat_dot = quaternion_derivative_from_world_angular_velocity(
            split.base_quaternion_wxyz, omega_world
        )
        return torch.cat((p_dot, quat_dot, split.joint_velocities), dim=-1)

    def _control_matrix_from_acceleration_map(
        self,
        acceleration_map: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        """Embed a generalized-acceleration map in the full state derivative.

        Args:
            acceleration_map: Tensor with shape ``(batch, nv, input_dim)``
                mapping inputs to generalized accelerations.
            batch_size: Leading batch dimension used for the returned tensor.

        Returns:
            Control matrix with shape ``(batch, state_dim, input_dim)``. The
            configuration-derivative rows are zero and the last ``nv`` rows
            contain ``acceleration_map``.
        """
        control_matrix = torch.zeros(
            batch_size,
            self.state_dim,
            self.input_dim,
            dtype=self.dtype,
            device=self.device,
        )
        control_matrix[..., self.nq :, :] = acceleration_map
        return control_matrix

    def _bias_force(self, base_transform: torch.Tensor, split: SplitState) -> torch.Tensor:
        """Return the combined Coriolis/centrifugal and gravity force.

        Adam's ``bias_force`` is the reduced RNEA call with current generalized
        velocity and the configured gravity vector, i.e. the same ``h(q, nu)``
        as ``coriolis_term + gravity_term``. Use it when available because
        drift dynamics only need the combined vector, not the separated terms.

        Args:
            base_transform: Base-to-world transform ``W_H_B`` with shape
                ``(batch, 4, 4)``.
            split: Split state with batched joint positions and velocities.

        Returns:
            Bias force tensor with shape ``(batch, nv)`` or ``(nv,)``,
            matching Adam's batch squeezing behavior.
        """
        if hasattr(self.kindyn, "bias_force"):
            return self.kindyn.bias_force(
                base_transform,
                split.joint_positions,
                split.base_velocity,
                split.joint_velocities,
            )
        return self.kindyn.coriolis_term(
            base_transform,
            split.joint_positions,
            split.base_velocity,
            split.joint_velocities,
        ) + self.kindyn.gravity_term(base_transform, split.joint_positions)

    def _generalized_input_matrix(
        self,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        *,
        contact_force_frame: Literal["world", "contact"] | None = None,
    ) -> torch.Tensor:
        """Build the generalized-force input matrix before mass inversion.

        Args:
            base_transform: Base-to-world transform ``W_H_B`` with shape
                ``(batch, 4, 4)``.
            joint_positions: Joint position tensor with shape
                ``(batch, n_joints)``.

        Returns:
            Generalized input map with shape ``(batch, nv, input_dim)``. The
            first ``n_joints`` columns are ``S^T``; optional remaining columns
            map contact-force inputs to generalized forces.
        """
        batch = base_transform.shape[0]
        torque_map = self.selection_matrix_transpose(batch_size=batch)
        if self.contact_model is None:
            return torque_map

        contact_jacobian = self.contact_model.contact_jacobian(base_transform, joint_positions)
        force_transform = self.contact_model.contact_force_transform(
            base_transform,
            joint_positions,
            force_frame=contact_force_frame or self.contact_force_frame,
        )
        contact_map = torch.matmul(contact_jacobian.transpose(-1, -2), force_transform)
        return torch.cat((torque_map, contact_map), dim=-1)


def _build_adam_kindyn(asset: RobotAsset, dtype: torch.dtype, device: torch.device):
    """Construct Adam's PyTorch kinematics/dynamics object for an asset.

    Args:
        asset: Resolved robot asset containing the Adam-compatible URDF path
            and ordered joint names.
        dtype: Torch dtype requested for Adam computations.
        device: Torch device requested for Adam computations.

    Returns:
        Adam ``KinDynComputations`` object configured with mixed velocity
        representation and gravity ``[0, 0, -9.80665, 0, 0, 0]``.

    Raises:
        ImportError: If Adam's PyTorch backend is unavailable.
    """
    try:
        import adam
        from adam.pytorch import KinDynComputations
    except ImportError as exc:
        raise ImportError(
            "Adam is required for FloatingBaseDynamics. Install with "
            "`uv sync` or `pip install 'adam-robotics[pytorch]'`."
        ) from exc

    gravity = torch.as_tensor([0.0, 0.0, -9.80665, 0.0, 0.0, 0.0], dtype=dtype, device=device)
    try:
        kindyn = KinDynComputations(
            str(asset.adam_urdf_path),
            list(asset.joint_names),
            device=device,
            dtype=dtype,
            gravity=gravity,
        )
    except TypeError:
        kindyn = KinDynComputations(
            str(asset.adam_urdf_path),
            list(asset.joint_names),
            gravity=gravity,
        )
    kindyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
    return kindyn


HumanoidDynamics = FloatingBaseDynamics
