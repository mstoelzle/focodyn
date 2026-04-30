from __future__ import annotations

import pytest
import torch

from focodyn import (
    AffineInputConstraint,
    InputConstraintSet,
    JointTorqueLimits,
    LinearizedFrictionCone,
    PositiveNormalContactForces,
    StaticAffineInputConstraint,
)


def test_joint_torque_limits_build_affine_residuals() -> None:
    constraint = JointTorqueLimits(
        lower=torch.tensor([-1.0, -2.0]),
        upper=torch.tensor([3.0, 4.0]),
        input_dim=5,
        dtype=torch.float64,
    )
    u = torch.tensor([3.5, -3.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    terms = constraint.affine_terms()
    residual = constraint(u)

    assert terms.matrix.shape == (4, 5)
    assert terms.upper_bound.shape == (4,)
    assert torch.allclose(residual, torch.tensor([0.5, -7.0, -4.5, 1.0], dtype=torch.float64))
    assert torch.allclose(
        constraint.violation(u),
        torch.tensor([0.5, 0.0, 0.0, 1.0], dtype=torch.float64),
    )
    assert not bool(constraint.is_satisfied(u))
    assert bool(constraint.is_satisfied(torch.zeros(5, dtype=torch.float64)))


def test_joint_torque_limits_accept_scalar_bounds_with_joint_count() -> None:
    constraint = JointTorqueLimits(
        -2.0,
        2.0,
        n_joints=3,
        input_dim=5,
        torque_start=1,
        dtype=torch.float64,
    )
    u = torch.tensor([10.0, -2.0, 0.0, 2.0, 10.0], dtype=torch.float64)

    assert constraint.input_dim == 5
    assert constraint.n_joints == 3
    assert bool(constraint.is_satisfied(u))


def test_joint_torque_limits_infer_default_input_dimension_from_vector_bounds() -> None:
    constraint = JointTorqueLimits(
        torch.tensor([-1.0, -2.0], dtype=torch.float64),
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        dtype=torch.float64,
    )

    assert constraint.input_dim == 2
    assert bool(constraint.is_satisfied(torch.zeros(2, dtype=torch.float64)))


def test_positive_normal_contact_forces_use_local_normal_components() -> None:
    constraint = PositiveNormalContactForces(
        input_dim=8,
        num_contacts=2,
        contact_force_start=2,
        minimum_normal_force=torch.tensor([1.0, 0.0]),
        dtype=torch.float64,
    )
    u = torch.tensor(
        [
            0.0,
            0.0,
            0.5,
            -0.2,
            1.5,
            0.1,
            0.2,
            -0.1,
        ],
        dtype=torch.float64,
    )

    assert constraint.affine_terms().matrix.shape == (2, 8)
    assert torch.allclose(constraint(u), torch.tensor([-0.5, 0.1], dtype=torch.float64))
    assert torch.allclose(constraint.violation(u), torch.tensor([0.0, 0.1], dtype=torch.float64))


def test_linearized_friction_cone_uses_local_contact_force_blocks() -> None:
    constraint = LinearizedFrictionCone(
        0.5,
        input_dim=8,
        num_contacts=2,
        contact_force_start=2,
        num_facets=4,
        dtype=torch.float64,
    )
    u = torch.tensor(
        [
            0.0,
            0.0,
            0.2,
            -0.1,
            1.0,
            0.4,
            0.0,
            1.0,
        ],
        dtype=torch.float64,
    )

    residual = constraint(u)

    assert constraint.num_constraints == 8
    assert constraint.affine_terms().matrix.shape == (8, 8)
    limit = 0.5 * torch.cos(torch.tensor(torch.pi / 4.0, dtype=torch.float64))
    assert torch.allclose(
        residual,
        torch.tensor(
            [
                0.2 - limit,
                -0.1 - limit,
                -0.2 - limit,
                0.1 - limit,
                0.4 - limit,
                -limit,
                -0.4 - limit,
                -limit,
            ],
            dtype=torch.float64,
        ),
    )
    assert torch.allclose(
        constraint.violation(u),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.4 - limit, 0.0, 0.0, 0.0], dtype=torch.float64),
    )


def test_linearized_friction_cone_generates_regular_polygon_facets() -> None:
    phase = float(torch.pi / 8.0)
    constraint = LinearizedFrictionCone(
        0.8,
        input_dim=5,
        num_contacts=1,
        contact_force_start=2,
        num_facets=8,
        facet_phase=phase,
        conservative=False,
        dtype=torch.float64,
    )

    terms = constraint.affine_terms()
    first_direction = torch.tensor([torch.cos(torch.tensor(phase)), torch.sin(torch.tensor(phase))])

    assert constraint.num_facets == 8
    assert constraint.num_constraints == 8
    assert terms.matrix.shape == (8, 5)
    assert torch.allclose(terms.matrix[0, 2:4], first_direction.to(dtype=torch.float64))
    assert torch.allclose(terms.matrix[0, 4], torch.tensor(-0.8, dtype=torch.float64))


def test_input_constraint_set_concatenates_child_constraints() -> None:
    torque = JointTorqueLimits(-1.0, 1.0, n_joints=2, input_dim=8, dtype=torch.float64)
    normal = PositiveNormalContactForces(
        input_dim=8,
        num_contacts=2,
        contact_force_start=2,
        dtype=torch.float64,
    )
    friction = LinearizedFrictionCone(
        0.5,
        input_dim=8,
        num_contacts=2,
        contact_force_start=2,
        dtype=torch.float64,
    )
    constraint_set = InputConstraintSet(torque, normal, friction)
    u = torch.tensor([0.0, 0.5, 0.1, 0.2, 1.0, 0.0, -0.2, 1.0], dtype=torch.float64)

    terms = constraint_set.affine_terms()
    expected_residual = torch.cat((torque(u), normal(u), friction(u)))

    assert constraint_set.num_constraints == 14
    assert terms.matrix.shape == (14, 8)
    assert terms.upper_bound.shape == (14,)
    assert torch.allclose(constraint_set(u), expected_residual)
    assert bool(constraint_set.is_satisfied(u))


def test_input_constraint_set_rejects_mismatched_input_dimensions() -> None:
    first = JointTorqueLimits(-1.0, 1.0, n_joints=2, input_dim=2)
    second = PositiveNormalContactForces(
        input_dim=5,
        num_contacts=1,
        contact_force_start=2,
    )

    with pytest.raises(ValueError, match="same input_dim"):
        InputConstraintSet(first, second)


def test_affine_constraint_base_and_static_validation_branches() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        AffineInputConstraint(0)
    with pytest.raises(NotImplementedError):
        AffineInputConstraint(1).affine_terms()
    with pytest.raises(ValueError, match="matrix"):
        StaticAffineInputConstraint(torch.zeros(2), torch.zeros(2))
    with pytest.raises(ValueError, match="upper_bound"):
        StaticAffineInputConstraint(torch.zeros(2, 2), torch.zeros(2, 1))
    with pytest.raises(ValueError, match="same number"):
        StaticAffineInputConstraint(torch.zeros(2, 2), torch.zeros(1))

    constraint = StaticAffineInputConstraint(torch.eye(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))
    composed = constraint.compose(constraint)

    assert isinstance(composed, InputConstraintSet)
    with pytest.raises(ValueError, match="Expected input dimension"):
        constraint(torch.zeros(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="At least one"):
        InputConstraintSet()


def test_input_constraint_constructors_reject_invalid_limits_and_blocks() -> None:
    with pytest.raises(ValueError, match="less than or equal"):
        JointTorqueLimits(2.0, 1.0, n_joints=1)
    with pytest.raises(ValueError, match="length must be positive"):
        JointTorqueLimits(-1.0, 1.0, n_joints=0)
    with pytest.raises(ValueError, match="scalar or one-dimensional"):
        JointTorqueLimits(torch.ones(1, 1), torch.ones(1, 1))
    with pytest.raises(ValueError, match="matching lengths"):
        JointTorqueLimits(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="length must be positive"):
        JointTorqueLimits(torch.empty(0), torch.empty(0))
    with pytest.raises(ValueError, match="n_values is required"):
        JointTorqueLimits(-1.0, 1.0)
    with pytest.raises(ValueError, match="at least"):
        JointTorqueLimits(-1.0, 1.0, n_joints=2, input_dim=1)
    with pytest.raises(ValueError, match="start"):
        JointTorqueLimits(-1.0, 1.0, n_joints=1, input_dim=2, torque_start=-1)

    with pytest.raises(ValueError, match="input_dim"):
        PositiveNormalContactForces(input_dim=0, num_contacts=0, contact_force_start=0)
    with pytest.raises(ValueError, match="width"):
        PositiveNormalContactForces(input_dim=3, num_contacts=0, contact_force_start=0)
    with pytest.raises(ValueError, match="exceeds"):
        PositiveNormalContactForces(input_dim=2, num_contacts=1, contact_force_start=0)
    with pytest.raises(ValueError, match="normal_axis"):
        PositiveNormalContactForces(input_dim=3, num_contacts=1, contact_force_start=0, normal_axis=3)
    with pytest.raises(ValueError, match="length 1"):
        PositiveNormalContactForces(
            input_dim=3,
            num_contacts=1,
            contact_force_start=0,
            minimum_normal_force=[1.0, 2.0],
        )
    with pytest.raises(ValueError, match="scalar or one-dimensional"):
        PositiveNormalContactForces(
            input_dim=3,
            num_contacts=1,
            contact_force_start=0,
            minimum_normal_force=[[1.0]],
        )


def test_friction_cone_rejects_invalid_coefficients_axes_and_facets() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LinearizedFrictionCone(-0.1, input_dim=3, num_contacts=1, contact_force_start=0)
    with pytest.raises(ValueError, match="at least 3"):
        LinearizedFrictionCone(0.5, input_dim=3, num_contacts=1, contact_force_start=0, num_facets=2)
    with pytest.raises(ValueError, match="two axes"):
        LinearizedFrictionCone(0.5, input_dim=3, num_contacts=1, contact_force_start=0, tangent_axes=(0,))
    with pytest.raises(ValueError, match="permutation"):
        LinearizedFrictionCone(0.5, input_dim=3, num_contacts=1, contact_force_start=0, tangent_axes=(0, 2))
