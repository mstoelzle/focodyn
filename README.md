# FoCoDyn

FoCoDyn, short for Floating Contact Dynamics, provides differentiable
floating-base dynamics and contact-pose Jacobians for legged robots. It is
aimed at CBF/CLF, trajectory optimization, and learning workflows that need
gradients through rigid-body dynamics and contact kinematics.

The package wraps [Adam](https://github.com/gbionics/adam) through its PyTorch
backend. Adam provides the floating-base mass matrix, Coriolis/centrifugal
terms, gravity terms, Jacobians, and forward kinematics. This repository adds a
legged-robot layer around those quantities:

- an installable Python package with `uv` and Python >=3.11,
- vendored Unitree G1 URDF/MJCF assets to start,
- a shared `torch.nn.Module` exposing control-affine `f(x)` and `g(x)`,
- differentiable foot contact-pose forward kinematics and Jacobians, and
- a Viser-based viewer that overlays contact frames on a retargeted G1 motion
  reference.

## Install

The distribution and import package name is `focodyn`.

```bash
uv sync --extra dev
```

For the optional browser visualization:

```bash
uv sync --extra dev --extra viz
```

Run the Adam/G1 smoke check:

```bash
uv run focodyn-check-adam-g1
```

Run the root-level dynamics inspection example:

```bash
uv run python examples/inspect_g1_dynamics.py
```

Run the test suite:

```bash
uv run pytest
```

## Dynamics API

```python
import torch
from focodyn import FloatingBaseDynamics

model = FloatingBaseDynamics(
    "unitree_g1",
    include_contact_forces=True,
    contact_mode="feet_corners",
    dtype=torch.float64,
)

x = model.neutral_state()
xdot_drift = model.f(x)
control_map = model.g(x)
xdot_drift, control_map = model.f_and_g(x)  # shared mass matrix + solve
```

The dynamics are represented as:

```text
x_dot = f(x) + g(x) u
M(q) nu_dot + h(q, nu) = S^T tau + J_c(q)^T lambda
```

`u` is stacked as joint torques first, then optional contact forces
`lambda = (f_c0, f_c1, ...)` with three force components per contact. By
default contact-force inputs are world-frame vectors. Pass
`contact_force_frame="contact"` to express each force in its contact frame; the
module rotates those forces into world coordinates before applying `J_c^T`.

## State Convention

The state is `x = (q, nu)`.

`q = (p_WB, quat_WB, s)`:

- `p_WB` is the world-frame position of the floating-base/root link origin.
- `quat_WB` is a unit quaternion in `(w, x, y, z)` order and maps base-frame
  vectors into the world frame.
- `s` is ordered exactly as `model.joint_names`, which is parsed from the URDF
  and passed to Adam.

`nu = (v_WB, omega_WB, s_dot)` uses Adam's mixed representation:

- `v_WB` is the base linear velocity in the world frame.
- `omega_WB` is the base angular velocity in the world frame.
- `s_dot` follows the same joint order as `s`.

The generalized acceleration is `nu_dot = (v_dot_WB, omega_dot_WB, s_ddot)`.

## Contacts

`FloatingBaseContactModel` initializes contact candidates from URDF collision
geometry. For the Unitree G1, each ankle-roll link has four small sphere
collisions; `feet_corners` uses those eight sphere origins, and `feet_centers`
uses one averaged point per foot. The same interface is intended to support
other floating-base legged robots as assets and contact extraction modes are
added.

Contacts are full `SE(3)` frames, not only points. `contact_poses(...)` returns
world positions, `(w, x, y, z)` contact-to-world quaternions, and homogeneous
`W_H_C` matrices. This matches the Adam transform convention used elsewhere in
the package. The contact normal is the contact frame's positive z-axis
expressed in world coordinates and is available through `contact_normals(...)`.

The translational contact Jacobian maps Adam mixed generalized velocity to
stacked world-frame contact point velocities. The contact force contribution to
the equations of motion is `J_c(q)^T lambda` for world-frame forces, or
`J_c(q)^T R_WC lambda_C` for contact-frame forces.

`FlatTerrainContactDetector` evaluates each configured contact candidate
against a flat terrain plane and returns nearest terrain points, signed
distances, penetration depths, world normals, and binary contact flags.
`BasicContactForceResolver` provides a deterministic normal-force estimate for
debugging and visualization. It returns forces in the world frame by default,
or in each local contact frame with `force_frame="contact"`.

## Motion Derivatives

Kinematic references can be differentiated with `torch-dxdt` Whittaker-Eilers
filters to estimate `nu` and `nu_dot` for inverse-dynamics inspection. To
compare smoothing parameters and save a plot of raw/smoothed motion,
velocities, and accelerations:

```bash
uv run --extra viz python examples/plot_g1_motion_derivatives.py --lambdas 10 100 1000 10000 --output outputs/motion_derivative_lambdas.pdf
```

## Visualization

The `viz` extra provides three Viser-based viewers.

### Kinematic Trajectory Viewer

Start the standard kinematic trajectory viewer with:

```bash
uv run --extra viz focodyn-visualize-g1
```

Or run the root-level visualization example:

```bash
uv run --extra viz python examples/visualize_g1.py
```

The viewer loads the local G1 URDF, plays the bundled retargeted AMASS/G1
kinematic reference, and overlays the differentiable contact-frame FK estimates.
The Viser side panel includes playback buttons, a frame scrubber, and a motion
dropdown for switching between bundled kinematic references.
Pass `--synthetic-motion` to recover the older deterministic fallback sequence,
or `--motion-reference /path/to/file.npz` / `--motion-reference /path/to/file.npy`
to visualize another supported retargeted G1 motion.

To export the standard kinematic visualization as an MP4, use the Viser render
export path rather than a manual start/stop recording button:

```bash
uv run --extra viz focodyn-visualize-g1 \
  --export-video outputs/videos/kinematic_trajectory.mp4 \
  --export-width 1280 \
  --export-height 720 \
  --port 0
```

### Dynamics Verification Viewer

For inspecting the dynamics maps, use the specialized viewer:

```bash
uv run --extra viz focodyn-visualize-dynamics --whittaker-lambda 100
```

The dynamics viewer inherits the kinematic trajectory viewer and adds two
modes. `f(x): inverse dynamics` ignores contacts and visualizes the virtual
root wrench plus the largest joint torques from `M(q) nu_dot + h(q,nu)`.
`g(x): contact projection` resolves flat-ground contact forces, projects them
through the contact-force input block, and visualizes the resulting root wrench,
joint torques, and contact force vectors. The dynamics viewer renders the robot
semi-transparently by default and includes a `Joint torque viz` selector for
showing torque arrows, text labels, or both. Override the transparency with
`--robot-opacity 0.5` when needed.

Export both dynamics verification videos with:

```bash
uv run --extra viz focodyn-visualize-dynamics \
  --export-dynamics-videos outputs/videos \
  --export-width 1280 \
  --export-height 720 \
  --port 0
```

Video export requires `ffmpeg` and a Chromium-compatible browser. The exporter
launches a temporary headless browser client, captures each frame with Viser,
and writes H.264 MP4 files. Pass `--export-frames N` for a shorter clip during
tests, or `--export-browser /path/to/browser` if Chrome/Chromium is not on the
standard path.

### Input Constraint Verification Viewer

For checking the affine input constraints along a kinematic reference, use:

```bash
uv run --extra viz focodyn-visualize-input-constraints --whittaker-lambda 100
```

This viewer follows the same derivative-estimation path as the dynamics
verification viewer. For each frame, it estimates `nu_dot`, computes
`M(q) nu_dot + h(q, nu)`, uses the actuated block as the joint torque estimate,
and resolves rough flat-ground contact forces in the local contact frames. It
then builds `u = [tau, lambda_C]` and evaluates the joint torque, positive
normal force, and linearized friction cone constraints.

Red arrows indicate constraint violations. Joint arrows are drawn at the
violating joints and point along the joint axis, while contact arrows mark
negative normal-force or friction-cone violations at the contact frames. The
side panel exposes the URDF effort-limit scale, friction coefficient, number of
friction facets (eight by default), conservative/outer cone mode, tangential
damping used by the rough contact resolver, and separate scales for force and
torque violation arrows.

The same mode is available from the main viewer:

```bash
uv run --extra viz focodyn-visualize-g1 --input-constraint-verification
```

## Sources And Attribution

Rigid-body dynamics are computed by
[Adam](https://github.com/gbionics/adam), distributed on PyPI as
`adam-robotics`. This package depends on `adam-robotics[pytorch]` and does not
reimplement CRBA, RNEA, ABA, frame Jacobians, or forward kinematics.

Robot assets under
`src/focodyn/assets/robots/unitree_g1` come from
[unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description).
The upstream BSD 3-Clause license is included at
`src/focodyn/assets/robots/unitree_g1/LICENSE.unitree_ros`.

The default bundled sample kinematic motion under
`src/focodyn/assets/motions/g1_fleaven_retargeted` is
`g1/Transitions_mocap/mazen_c3d/JOOF_walk_poses_120_jpos.npy` from
[fleaven/Retargeted_AMASS_for_robotics](https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics).
That dataset is distributed under CC-BY-4.0 and contains AMASS motions
retargeted to Unitree G1. The raw root quaternion is documented by the source
dataset in `(x, y, z, w)` order; this package converts it to the `(w, x, y, z)`
convention used by Adam-facing APIs.

An additional bundled reference under
`src/focodyn/assets/motions/g1_amass_retargeted` is
`g1/CMU/06/06_01_poses_120_jpos.npz` from
[ember-lab-berkeley/AMASS_Retargeted_for_G1](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1).
That dataset is distributed under CC-BY-4.0 and contains AMASS motions
retargeted to Unitree G1. The original CMU source motion is from the
[CMU Graphics Lab Motion Capture Database](https://mocap.cs.cmu.edu/).

Files ending in `.adam.urdf` are generated compatibility copies of the upstream
URDFs. They add identity joint origins where the upstream URDF omits them and
remove Unitree/MuJoCo-specific XML that `urdf_parser_py` warns about. The
original upstream URDFs remain vendored unchanged and are used for source
inspection and visualization.

The visualization uses [Viser](https://viser.studio/) when the optional
`viz` extra is installed. Whittaker-Eilers derivatives use
[torch-dxdt](https://github.com/mstoelzle/torch-dxdt).
