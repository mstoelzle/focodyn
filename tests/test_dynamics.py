from __future__ import annotations

import pytest
import torch

from focodyn import FloatingBaseDynamics


@pytest.fixture(scope="module")
def model() -> FloatingBaseDynamics:
    pytest.importorskip("adam")
    return FloatingBaseDynamics("unitree_g1", include_contact_forces=True, dtype=torch.float64)


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
    assert torch.allclose(terms.bias, terms.coriolis + terms.gravity, atol=1e-10, rtol=1e-10)


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


def test_generalized_forces_from_acceleration_matches_drift_equation(
    model: FloatingBaseDynamics,
) -> None:
    x = model.neutral_state()
    drift = model.f(x)
    generalized_force = model.generalized_forces_from_acceleration(x, drift[model.nq :])
    assert generalized_force.shape == (model.nv,)
    assert torch.allclose(generalized_force, torch.zeros_like(generalized_force), atol=1e-9)


def test_generalized_forces_from_input_projects_joint_torques(model: FloatingBaseDynamics) -> None:
    x = model.neutral_state()
    u = torch.zeros(model.input_dim, dtype=model.dtype)
    joint_torques = torch.linspace(-1.0, 1.0, model.n_joints, dtype=model.dtype)
    u[: model.n_joints] = joint_torques

    generalized_force = model.generalized_forces_from_input(x, u)

    assert generalized_force.shape == (model.nv,)
    assert torch.allclose(generalized_force[:6], torch.zeros(6, dtype=model.dtype))
    assert torch.allclose(generalized_force[6:], joint_torques)


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
