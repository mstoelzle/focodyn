from __future__ import annotations

import builtins

import pytest
import torch

from focodyn import FloatingBaseDynamics
from focodyn.dynamics import _build_adam_kindyn, _contact_space_dynamics_from_matrices


@pytest.fixture(scope="module")
def model() -> FloatingBaseDynamics:
    pytest.importorskip("adam")
    return FloatingBaseDynamics(
        "unitree_g1", include_contact_forces=True, dtype=torch.float64
    )


def test_adam_computes_g1_dynamics_terms(model: FloatingBaseDynamics) -> None:
    x = model.neutral_state()
    terms = model.dynamics_terms(x)
    nv = model.nv
    assert terms.mass_matrix.shape == (nv, nv)
    assert terms.coriolis.shape == (nv,)
    assert terms.gravity.shape == (nv,)
    assert terms.bias.shape == (nv,)
    assert torch.isfinite(terms.mass_matrix).all()
    assert torch.isfinite(terms.coriolis).all()
    assert torch.isfinite(terms.gravity).all()
    assert torch.allclose(terms.mass_matrix, terms.mass_matrix.T, atol=1e-7, rtol=1e-7)
    assert torch.allclose(
        terms.bias, terms.coriolis + terms.gravity, atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize("joint_order", ["source", "unitree_g1_29dof"])
def test_adam_computes_g1_37dof_minimal_dynamics_and_contacts(
    joint_order: str,
) -> None:
    pytest.importorskip("adam")
    minimal = FloatingBaseDynamics(
        "g1_37dof_minimal",
        joint_order=joint_order,
        include_contact_forces=True,
        dtype=torch.float64,
    )
    assert minimal.n_joints == 37
    assert minimal.nq == 44
    assert minimal.nv == 43
    assert minimal.state_dim == 87
    assert minimal.input_dim == 61
    assert minimal.contact_model is not None
    assert minimal.contact_model.num_contacts == 8

    x = minimal.neutral_state()
    terms = minimal.dynamics_terms(x)
    drift, control = minimal.f_and_g(x)
    assert terms.mass_matrix.shape == (43, 43)
    assert terms.bias.shape == (43,)
    assert drift.shape == (87,)
    assert control.shape == (87, 61)
    assert torch.isfinite(terms.mass_matrix).all()
    assert torch.isfinite(terms.bias).all()
    assert torch.isfinite(drift).all()
    assert torch.isfinite(control).all()
    assert torch.isfinite(terms.coriolis).all()
    assert torch.isfinite(terms.gravity).all()
    assert torch.allclose(terms.mass_matrix, terms.mass_matrix.T, atol=1e-7, rtol=1e-7)
    assert torch.allclose(
        terms.bias, terms.coriolis + terms.gravity, atol=1e-10, rtol=1e-10
    )


def test_batched_dynamics_terms_and_selection_matrix_shapes(
    model: FloatingBaseDynamics,
) -> None:
    x0 = model.neutral_state()
    x1 = x0.clone()
    x1[0] = 0.1
    x1[model.nq :] = 0.02
    x = torch.stack((x0, x1))

    terms = model.dynamics_terms(x)
    selection = model.selection_matrix_transpose()

    assert terms.mass_matrix.shape == (2, model.nv, model.nv)
    assert terms.coriolis.shape == (2, model.nv)
    assert terms.gravity.shape == (2, model.nv)
    assert terms.bias.shape == (2, model.nv)
    assert selection.shape == (model.nv, model.n_joints)
    assert torch.allclose(
        selection[:6], torch.zeros(6, model.n_joints, dtype=model.dtype)
    )
    assert torch.allclose(selection[6:], torch.eye(model.n_joints, dtype=model.dtype))


def test_f_and_g_shapes(model: FloatingBaseDynamics) -> None:
    x = model.neutral_state()
    drift = model.f(x)
    control = model.g(x)
    fused_drift, fused_control = model.f_and_g(x)
    assert drift.shape == (model.state_dim,)
    assert control.shape == (model.state_dim, model.input_dim)
    assert fused_drift.shape == (model.state_dim,)
    assert fused_control.shape == (model.state_dim, model.input_dim)
    assert torch.isfinite(drift).all()
    assert torch.isfinite(control).all()
    assert torch.allclose(fused_drift, drift)
    assert torch.allclose(fused_control, control)
    assert torch.count_nonzero(control[: model.nq]) == 0
    assert model.input_dim == model.n_joints + 24


def test_f_and_g_batched_shapes(model: FloatingBaseDynamics) -> None:
    x0 = model.neutral_state()
    x1 = x0.clone()
    x1[0] = 0.1
    x1[model.nq :] = 0.02
    x = torch.stack((x0, x1))
    drift, control = model.f_and_g(x)
    assert drift.shape == (2, model.state_dim)
    assert control.shape == (2, model.state_dim, model.input_dim)
    assert torch.isfinite(drift).all()
    assert torch.isfinite(control).all()


def test_forward_matches_control_affine_formula(model: FloatingBaseDynamics) -> None:
    x = model.neutral_state()
    u = torch.linspace(-0.05, 0.05, model.input_dim, dtype=model.dtype)
    expected = model.f(x) + model.g(x).matmul(u)
    assert torch.allclose(model(x, u), expected)
    assert torch.allclose(model(x), model.f(x))


def test_generalized_forces_from_acceleration_matches_drift_equation(
    model: FloatingBaseDynamics,
) -> None:
    x = model.neutral_state()
    drift = model.f(x)
    generalized_force = model.generalized_forces_from_acceleration(x, drift[model.nq :])
    assert generalized_force.shape == (model.nv,)
    assert torch.allclose(
        generalized_force, torch.zeros_like(generalized_force), atol=1e-9
    )

    batched_acceleration = drift[model.nq :].unsqueeze(0)
    assert torch.allclose(
        model.generalized_forces_from_acceleration(x, batched_acceleration),
        generalized_force,
    )


def test_generalized_forces_from_acceleration_rejects_wrong_dimension(
    model: FloatingBaseDynamics,
) -> None:
    x = model.neutral_state()

    with pytest.raises(ValueError, match="generalized acceleration"):
        model.generalized_forces_from_acceleration(
            x, torch.zeros(model.nv + 1, dtype=model.dtype)
        )


def test_generalized_forces_from_input_projects_joint_torques(
    model: FloatingBaseDynamics,
) -> None:
    x = model.neutral_state()
    u = torch.zeros(model.input_dim, dtype=model.dtype)
    joint_torques = torch.linspace(-1.0, 1.0, model.n_joints, dtype=model.dtype)
    u[: model.n_joints] = joint_torques

    generalized_force = model.generalized_forces_from_input(x, u)

    assert generalized_force.shape == (model.nv,)
    assert torch.allclose(generalized_force[:6], torch.zeros(6, dtype=model.dtype))
    assert torch.allclose(generalized_force[6:], joint_torques)


def test_generalized_forces_from_input_supports_batched_inputs_and_rejects_wrong_dimension(
    model: FloatingBaseDynamics,
) -> None:
    x0 = model.neutral_state()
    x1 = x0.clone()
    x1[model.nq :] = 0.02
    x = torch.stack((x0, x1))
    u = torch.zeros(2, model.input_dim, dtype=model.dtype)
    u[:, : model.n_joints] = torch.linspace(
        -1.0, 1.0, model.n_joints, dtype=model.dtype
    )

    generalized_force = model.generalized_forces_from_input(x, u)

    assert generalized_force.shape == (2, model.nv)
    assert torch.allclose(
        generalized_force[:, :6], torch.zeros(2, 6, dtype=model.dtype)
    )
    assert torch.allclose(generalized_force[:, 6:], u[:, : model.n_joints])
    with pytest.raises(ValueError, match="input dimension"):
        model.generalized_forces_from_input(
            x0, torch.zeros(model.input_dim + 1, dtype=model.dtype)
        )


def test_contact_space_dynamics_from_matrices_matches_contact_space_equations() -> None:
    mass = torch.diag(torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)).unsqueeze(0)
    coriolis = torch.tensor([[0.3, -0.2, 0.4]], dtype=torch.float64)
    gravity = torch.tensor([[0.0, 1.5, -2.0]], dtype=torch.float64)
    jacobian = torch.tensor([[[1.0, 0.2, -0.1], [0.0, 1.0, 0.3]]], dtype=torch.float64)
    jacobian_dot = torch.tensor(
        [[[0.1, -0.2, 0.0], [0.0, 0.05, 0.2]]], dtype=torch.float64
    )
    velocity = torch.tensor([[0.4, -0.1, 0.2]], dtype=torch.float64)
    active_contacts = torch.tensor([[True]], dtype=torch.bool)

    terms = _contact_space_dynamics_from_matrices(
        mass_matrix=mass,
        coriolis=coriolis,
        gravity=gravity,
        contact_jacobian=jacobian,
        contact_jacobian_dot=jacobian_dot,
        generalized_velocity=velocity,
        active_contacts=active_contacts,
        pinv_rcond=1e-12,
    )

    mass_inverse_jacobian_transpose = torch.linalg.solve(
        mass, jacobian.transpose(-1, -2)
    )
    expected_inverse_contact_inertia = torch.matmul(
        jacobian, mass_inverse_jacobian_transpose
    )
    expected_contact_inertia = torch.linalg.pinv(
        expected_inverse_contact_inertia, rtol=1e-12
    )
    expected_dynamic_inverse = torch.matmul(
        mass_inverse_jacobian_transpose, expected_contact_inertia
    )
    expected_jdot_velocity = torch.matmul(jacobian_dot, velocity.unsqueeze(-1)).squeeze(
        -1
    )
    expected_bias = torch.matmul(
        expected_dynamic_inverse.transpose(-1, -2), coriolis.unsqueeze(-1)
    ).squeeze(-1) - torch.matmul(
        expected_contact_inertia, expected_jdot_velocity.unsqueeze(-1)
    ).squeeze(-1)
    expected_gravity = torch.matmul(
        expected_dynamic_inverse.transpose(-1, -2),
        gravity.unsqueeze(-1),
    ).squeeze(-1)
    gamma = torch.tensor([[0.7, -0.4, 0.2]], dtype=torch.float64)
    projected_gamma = torch.matmul(
        expected_dynamic_inverse.transpose(-1, -2), gamma.unsqueeze(-1)
    ).squeeze(-1)
    paper_contact_force = projected_gamma - expected_bias - expected_gravity

    assert torch.allclose(terms.contact_inertia, expected_contact_inertia)
    assert torch.allclose(terms.dynamic_inverse, expected_dynamic_inverse)
    assert torch.allclose(terms.jacobian_dot_velocity, expected_jdot_velocity)
    assert torch.allclose(terms.contact_bias, expected_bias)
    assert torch.allclose(terms.contact_gravity, expected_gravity)
    assert torch.allclose(
        terms.contact_bias + terms.contact_gravity + paper_contact_force,
        projected_gamma,
    )


def test_contact_space_dynamics_and_fixed_contact_force_shapes(
    model: FloatingBaseDynamics,
) -> None:
    assert model.contact_model is not None
    x = model.neutral_state()
    active_contacts = torch.zeros(model.contact_model.num_contacts, dtype=torch.bool)
    active_contacts[:2] = True

    terms = model.contact_space_dynamics(x, active_contacts=active_contacts)
    result = model.fixed_contact_forces_from_joint_torques(
        x,
        torch.zeros(model.n_joints, dtype=model.dtype),
        active_contacts=active_contacts,
    )

    assert terms.contact_jacobian.shape == (6, model.nv)
    assert terms.contact_jacobian_dot.shape == (6, model.nv)
    assert terms.contact_inertia.shape == (6, 6)
    assert terms.dynamic_inverse.shape == (model.nv, 6)
    assert terms.contact_bias.shape == (6,)
    assert terms.contact_gravity.shape == (6,)
    assert terms.jacobian_dot_velocity.shape == (6,)
    assert terms.contact_velocity.shape == (6,)
    assert terms.active_contacts.shape == (model.contact_model.num_contacts,)
    assert terms.singular_values.shape == (6,)
    assert int(terms.rank.item()) <= 6
    assert result.forces.shape == (model.contact_model.num_contacts, 3)
    assert result.world_forces.shape == (model.contact_model.num_contacts, 3)
    assert result.normal_forces.shape == (model.contact_model.num_contacts,)
    assert torch.allclose(
        result.world_forces[~active_contacts],
        torch.zeros_like(result.world_forces[~active_contacts]),
    )
    assert torch.allclose(
        result.equation_residual, torch.zeros_like(result.equation_residual), atol=1e-9
    )
    assert torch.isfinite(result.world_forces).all()


def test_fixed_contact_forces_support_batches_frames_and_signs(
    model: FloatingBaseDynamics,
) -> None:
    assert model.contact_model is not None
    x0 = model.neutral_state()
    x1 = x0.clone()
    x1[0] = 0.05
    x = torch.stack((x0, x1))
    active_contacts = torch.zeros(model.contact_model.num_contacts, dtype=torch.bool)
    active_contacts[::2] = True
    torques = torch.zeros(2, model.n_joints, dtype=model.dtype)

    env = model.fixed_contact_forces_from_joint_torques(
        x, torques, active_contacts=active_contacts
    )
    robot = model.fixed_contact_forces_from_joint_torques(
        x,
        torques,
        active_contacts=active_contacts,
        force_direction="robot_on_environment",
    )
    contact_frame = model.fixed_contact_forces_from_joint_torques(
        x0,
        torques[0],
        active_contacts=active_contacts,
        force_frame="contact",
    )

    assert env.world_forces.shape == (2, model.contact_model.num_contacts, 3)
    assert env.contact_space_dynamics.contact_jacobian.shape == (2, 12, model.nv)
    assert torch.allclose(env.world_forces, -robot.world_forces, atol=1e-9)
    assert contact_frame.force_frame == "contact"
    with pytest.raises(ValueError, match="same contacts"):
        changing_mask = torch.stack((active_contacts, ~active_contacts))
        model.contact_space_dynamics(x, active_contacts=changing_mask)
    with pytest.raises(ValueError, match="joint torque dimension"):
        model.fixed_contact_forces_from_joint_torques(
            x0, torch.zeros(model.n_joints + 1, dtype=model.dtype)
        )


def test_fixed_contact_assumption_residuals_are_reported(
    model: FloatingBaseDynamics,
) -> None:
    assert model.contact_model is not None
    x = model.neutral_state()
    x[model.nq + 6 :] = 0.02 * torch.sin(
        torch.arange(model.n_joints, dtype=model.dtype)
    )
    terms = model.contact_space_dynamics(x)
    manual_velocity = torch.matmul(
        terms.contact_jacobian, x[model.nq :].unsqueeze(-1)
    ).squeeze(-1)
    manual_acceleration_residual = terms.jacobian_dot_velocity

    assert torch.allclose(terms.contact_velocity, manual_velocity)
    assert torch.allclose(terms.jacobian_dot_velocity, manual_acceleration_residual)
    assert torch.linalg.norm(terms.contact_velocity) > 0.0


def test_split_state_and_constructor_validation() -> None:
    pytest.importorskip("adam")
    with pytest.raises(ValueError, match="contact_force_frame"):
        FloatingBaseDynamics("unitree_g1", contact_force_frame="local")

    model = FloatingBaseDynamics("unitree_g1", dtype=torch.float64)
    with pytest.raises(ValueError, match="Expected state dimension"):
        model.split_state(torch.zeros(model.state_dim + 1, dtype=model.dtype))


def test_build_adam_kindyn_reports_missing_adam(
    model: FloatingBaseDynamics,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "adam" or name.startswith("adam."):
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="Adam is required"):
        _build_adam_kindyn(model.asset, model.dtype, model.device)


def test_generalized_input_matrix_without_contacts_is_selection_map() -> None:
    pytest.importorskip("adam")
    model = FloatingBaseDynamics(
        "unitree_g1", include_contact_forces=False, dtype=torch.float64
    )
    x = model.neutral_state()
    matrix = model.generalized_input_matrix(x)

    assert matrix.shape == (model.nv, model.n_joints)
    assert torch.allclose(matrix, model.selection_matrix_transpose())


def test_bias_force_falls_back_to_coriolis_plus_gravity(
    model: FloatingBaseDynamics,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoBiasKindyn:
        def __init__(self, original) -> None:
            self._original = original

        def coriolis_term(self, *args, **kwargs):
            return self._original.coriolis_term(*args, **kwargs)

        def gravity_term(self, *args, **kwargs):
            return self._original.gravity_term(*args, **kwargs)

    x = model.neutral_state()
    split = model.split_state(x)
    base_transform = model.base_transform(x).unsqueeze(0)
    monkeypatch.setattr(model, "kindyn", NoBiasKindyn(model.kindyn))

    fallback_bias = model._bias_force(base_transform, split)

    assert torch.allclose(
        fallback_bias,
        model.kindyn.coriolis_term(
            base_transform,
            split.joint_positions,
            split.base_velocity,
            split.joint_velocities,
        )
        + model.kindyn.gravity_term(base_transform, split.joint_positions),
    )


def test_forward_with_control_uses_fused_mass_matrix(
    model: FloatingBaseDynamics, monkeypatch: pytest.MonkeyPatch
) -> None:
    x = model.neutral_state()
    u = torch.linspace(-0.05, 0.05, model.input_dim, dtype=model.dtype)
    calls = 0
    original_mass_matrix = model.kindyn.mass_matrix

    def counting_mass_matrix(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_mass_matrix(*args, **kwargs)

    monkeypatch.setattr(model.kindyn, "mass_matrix", counting_mass_matrix)
    model(x, u)
    assert calls == 1


def test_drift_is_differentiable(model: FloatingBaseDynamics) -> None:
    x = model.neutral_state().requires_grad_(True)
    y = model.f(x)[model.nq :].sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
