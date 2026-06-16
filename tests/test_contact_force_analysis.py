from __future__ import annotations

from pathlib import Path

import pytest
import torch

from focodyn import FloatingBaseDynamics, KinematicMotionReference, simple_walking_sequence
from focodyn.contact_force_analysis import (
    _pretty_contact_labels,
    analyze_fixed_contact_forces,
    plot_fixed_contact_force_analysis,
)


def test_fixed_contact_force_analysis_smoke(tmp_path: Path) -> None:
    pytest.importorskip("adam")
    pytest.importorskip("torch_dxdt")
    pytest.importorskip("matplotlib")
    model = FloatingBaseDynamics(
        "unitree_g1",
        include_contact_forces=True,
        contact_mode="feet_centers",
        dtype=torch.float64,
    )
    states, times = simple_walking_sequence(model, frames=8, dt=1.0 / 60.0)
    motion = KinematicMotionReference(
        states=states,
        times=times,
        fps=60.0,
        source_path=Path("synthetic"),
        source_name="synthetic test",
    )

    analysis = analyze_fixed_contact_forces(
        model,
        motion,
        whittaker_lmbda=10.0,
        contact_threshold=0.05,
        max_frames=8,
    )
    output = plot_fixed_contact_force_analysis(analysis, tmp_path)

    assert analysis.world_forces.shape == (8, model.contact_model.num_contacts, 3)
    assert analysis.normal_forces.shape == (8, model.contact_model.num_contacts)
    assert analysis.active_contacts.shape == (8, model.contact_model.num_contacts)
    assert torch.isfinite(analysis.equation_residual_norm).all()
    assert analysis.contact_threshold == pytest.approx(0.05)
    assert output.exists()


def test_pretty_contact_labels_for_g1_feet() -> None:
    assert _pretty_contact_labels(
        (
            "left_ankle_roll_link:0",
            "left_ankle_roll_link:3",
            "right_ankle_roll_link:2",
            "left_ankle_roll_link:center",
        )
    ) == ("L heel y-", "L toe y+", "R toe y-", "L foot center")
