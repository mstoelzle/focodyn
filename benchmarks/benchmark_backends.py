# Example:
# uv run --extra benchmark python benchmarks/benchmark_backends.py --devices cpu
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable

import numpy as np

from focodyn.assets import RobotAsset
from focodyn.assets import load_asset


IMPLEMENTATIONS = ("adam-torch", "adam-jax", "frax")
QUERIES = ("jacobian", "forward-dynamics", "forward-dynamics-no-coriolis")
DEVICES = ("cpu", "gpu")
FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
FRAX_FOOT_JOINTS = ("left_ankle_roll_joint", "right_ankle_roll_joint")
GRAVITY = np.array([0.0, 0.0, -9.80665, 0.0, 0.0, 0.0])
BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "outputs"
DEFAULT_PDF_FILENAME = "benchmark-results.pdf"
DEFAULT_PDF_PATH = DEFAULT_OUTPUT_DIR / DEFAULT_PDF_FILENAME


@dataclass(frozen=True)
class BenchmarkPayload:
    asset: RobotAsset
    batch_size: int
    base_transform: np.ndarray
    joint_positions: np.ndarray
    base_velocity: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    batched_base_transform: np.ndarray
    batched_joint_positions: np.ndarray
    batched_base_velocity: np.ndarray
    batched_joint_velocities: np.ndarray
    batched_joint_torques: np.ndarray
    frax_configuration: np.ndarray
    frax_velocity: np.ndarray
    frax_torque: np.ndarray
    batched_frax_configuration: np.ndarray
    batched_frax_velocity: np.ndarray
    batched_frax_torque: np.ndarray


@dataclass(frozen=True)
class BenchmarkRow:
    implementation: str
    query: str
    mode: str
    device: str
    batch_size: int
    dtype: str
    warmups: int
    iters: int
    status: str
    reason: str = ""
    mean_ms: float | None = None
    median_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    std_ms: float | None = None
    output_shape: str = ""


@dataclass(frozen=True)
class BenchmarkRunner:
    run: Callable[[], Any]
    sync: Callable[[Any], None]


class BackendUnavailable(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Adam PyTorch, Adam JAX, and Frax on Unitree G1 dynamics queries."
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["all"],
        help=f"Implementations to run: comma-separated/list values from {IMPLEMENTATIONS}, or all.",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["all"],
        help=f"Queries to run: comma-separated/list values from {QUERIES}, or all.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["all"],
        help=f"Devices to run: comma-separated/list values from {DEVICES}, or all.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--asset", default="unitree_g1")
    args = parser.parse_args(argv)
    args.implementations = _expand_selection(args.implementations, IMPLEMENTATIONS, "--implementations")
    args.queries = _expand_selection(args.queries, QUERIES, "--queries")
    args.devices = _expand_selection(args.devices, DEVICES, "--devices")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.iters < 1:
        parser.error("--iters must be at least 1")
    return args


def run_benchmarks(args: argparse.Namespace) -> list[BenchmarkRow]:
    payload = make_payload(load_asset(args.asset), args.batch_size)
    rows: list[BenchmarkRow] = []
    for implementation in args.implementations:
        for query in args.queries:
            for mode, device in _selected_modes(args.devices):
                rows.append(
                    run_one(
                        implementation=implementation,
                        query=query,
                        mode=mode,
                        device=device,
                        dtype=args.dtype,
                        warmups=args.warmups,
                        iters=args.iters,
                        payload=payload,
                    )
                )
    return rows


def run_one(
    *,
    implementation: str,
    query: str,
    mode: str,
    device: str,
    dtype: str,
    warmups: int,
    iters: int,
    payload: BenchmarkPayload,
) -> BenchmarkRow:
    try:
        runner = _make_runner(implementation, query, mode, device, dtype, payload)
    except BackendUnavailable as exc:
        return _skipped_row(implementation, query, mode, device, payload, dtype, warmups, iters, str(exc))
    except Exception as exc:
        return BenchmarkRow(
            implementation=implementation,
            query=query,
            mode=mode,
            device=device,
            batch_size=_row_batch_size(mode, payload.batch_size),
            dtype=dtype,
            warmups=warmups,
            iters=iters,
            status="error",
            reason=f"{type(exc).__name__}: {exc}",
        )

    try:
        for _ in range(warmups):
            runner.sync(runner.run())

        timings_ms: list[float] = []
        output: Any = None
        for _ in range(iters):
            start = time.perf_counter()
            output = runner.run()
            runner.sync(output)
            timings_ms.append((time.perf_counter() - start) * 1000.0)
    except Exception as exc:
        return BenchmarkRow(
            implementation=implementation,
            query=query,
            mode=mode,
            device=device,
            batch_size=_row_batch_size(mode, payload.batch_size),
            dtype=dtype,
            warmups=warmups,
            iters=iters,
            status="error",
            reason=f"{type(exc).__name__}: {exc}",
        )

    return BenchmarkRow(
        implementation=implementation,
        query=query,
        mode=mode,
        device=device,
        batch_size=_row_batch_size(mode, payload.batch_size),
        dtype=dtype,
        warmups=warmups,
        iters=iters,
        status="ok",
        mean_ms=statistics.fmean(timings_ms),
        median_ms=statistics.median(timings_ms),
        min_ms=min(timings_ms),
        max_ms=max(timings_ms),
        std_ms=statistics.pstdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        output_shape=_shape_string(output),
    )


def make_payload(asset: RobotAsset, batch_size: int) -> BenchmarkPayload:
    n_joints = len(asset.joint_names)
    base_transform = np.eye(4, dtype=np.float64)
    base_transform[:3, 3] = np.array([0.05, -0.03, 0.78])
    joint_positions = np.linspace(-0.18, 0.18, n_joints, dtype=np.float64)
    base_velocity = np.linspace(-0.25, 0.25, 6, dtype=np.float64)
    joint_velocities = np.linspace(0.35, -0.35, n_joints, dtype=np.float64)
    joint_torques = np.linspace(-6.0, 6.0, n_joints, dtype=np.float64)

    batch_offsets = np.arange(batch_size, dtype=np.float64)
    batched_base_transform = np.broadcast_to(base_transform, (batch_size, 4, 4)).copy()
    batched_base_transform[:, 0, 3] += 0.002 * batch_offsets
    batched_base_transform[:, 1, 3] -= 0.001 * batch_offsets

    joint_pattern = np.sin(np.arange(n_joints, dtype=np.float64))
    batched_joint_positions = joint_positions + 0.001 * batch_offsets[:, None] * joint_pattern
    batched_base_velocity = base_velocity + 0.0005 * batch_offsets[:, None]
    batched_joint_velocities = joint_velocities - 0.0007 * batch_offsets[:, None] * joint_pattern
    batched_joint_torques = joint_torques + 0.002 * batch_offsets[:, None] * np.cos(
        np.arange(n_joints, dtype=np.float64)
    )

    frax_configuration = np.concatenate((base_transform[:3, 3], np.zeros(3), joint_positions))
    frax_velocity = np.concatenate((base_velocity, joint_velocities))
    frax_torque = np.concatenate((np.zeros(6), joint_torques))
    batched_frax_configuration = np.concatenate(
        (batched_base_transform[:, :3, 3], np.zeros((batch_size, 3)), batched_joint_positions),
        axis=-1,
    )
    batched_frax_velocity = np.concatenate((batched_base_velocity, batched_joint_velocities), axis=-1)
    batched_frax_torque = np.concatenate(
        (np.zeros((batch_size, 6)), batched_joint_torques),
        axis=-1,
    )

    return BenchmarkPayload(
        asset=asset,
        batch_size=batch_size,
        base_transform=base_transform,
        joint_positions=joint_positions,
        base_velocity=base_velocity,
        joint_velocities=joint_velocities,
        joint_torques=joint_torques,
        batched_base_transform=batched_base_transform,
        batched_joint_positions=batched_joint_positions,
        batched_base_velocity=batched_base_velocity,
        batched_joint_velocities=batched_joint_velocities,
        batched_joint_torques=batched_joint_torques,
        frax_configuration=frax_configuration,
        frax_velocity=frax_velocity,
        frax_torque=frax_torque,
        batched_frax_configuration=batched_frax_configuration,
        batched_frax_velocity=batched_frax_velocity,
        batched_frax_torque=batched_frax_torque,
    )


def write_outputs(
    rows: list[BenchmarkRow],
    *,
    json_path: Path | None,
    csv_path: Path | None,
    pdf_path: Path | None,
) -> None:
    dictionaries = [asdict(row) for row in rows]
    if json_path is not None:
        json_path.write_text(json.dumps(dictionaries, indent=2) + "\n")
    if csv_path is not None:
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(dictionaries[0]) if dictionaries else [])
            writer.writeheader()
            writer.writerows(dictionaries)
    if pdf_path is not None:
        write_pdf(rows, pdf_path)


def write_pdf(rows: list[BenchmarkRow], pdf_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise RuntimeError(
            "PDF export requires matplotlib. Install it with `uv sync --extra benchmark`."
        ) from exc

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        ok_rows = [row for row in rows if row.status == "ok" and row.mean_ms is not None]
        _write_pdf_summary_page(pdf, plt, rows, ok_rows)
        _write_pdf_table_pages(pdf, plt, rows)


def _write_pdf_summary_page(pdf: Any, plt: Any, rows: list[BenchmarkRow], ok_rows: list[BenchmarkRow]) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    fig.suptitle("focodyn Backend Benchmark", fontsize=16, fontweight="bold", y=0.96)

    if ok_rows:
        labels = [f"{row.implementation}\n{row.query}\n{row.mode}" for row in ok_rows]
        values = [float(row.mean_ms) for row in ok_rows]
        colors = [_implementation_color(row.implementation) for row in ok_rows]
        ax.bar(range(len(ok_rows)), values, color=colors)
        ax.set_ylabel("Mean latency (ms)")
        ax.set_xticks(range(len(ok_rows)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    else:
        ax.text(0.5, 0.5, "No successful benchmark rows to plot.", ha="center", va="center", fontsize=14)
        ax.set_axis_off()

    status_counts = {status: sum(row.status == status for row in rows) for status in ("ok", "skipped", "error")}
    fig.text(
        0.02,
        0.02,
        (
            f"Rows: {len(rows)}   ok: {status_counts['ok']}   "
            f"skipped: {status_counts['skipped']}   errors: {status_counts['error']}"
        ),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def _write_pdf_table_pages(pdf: Any, plt: Any, rows: list[BenchmarkRow]) -> None:
    headers = ["Implementation", "Query", "Mode", "Status", "Mean ms", "Median ms", "Output", "Reason"]
    records = [
        [
            row.implementation,
            row.query,
            row.mode,
            row.status,
            _format_number(row.mean_ms),
            _format_number(row.median_ms),
            row.output_shape,
            _shorten(row.reason, 54),
        ]
        for row in rows
    ]
    if not records:
        records = [["", "", "", "", "", "", "", "No benchmark rows."]]

    rows_per_page = 24
    for page_start in range(0, len(records), rows_per_page):
        page_records = records[page_start : page_start + rows_per_page]
        fig, ax = plt.subplots(figsize=(11.0, 8.5))
        ax.set_title("Benchmark Results", fontsize=14, fontweight="bold", pad=14)
        ax.axis("off")
        table = ax.table(
            cellText=page_records,
            colLabels=headers,
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=[0.12, 0.19, 0.12, 0.08, 0.08, 0.08, 0.11, 0.22],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.35)
        for (row_index, _col_index), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_facecolor("#d9e2ef")
                cell.set_text_props(weight="bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#f6f8fb")
        page_number = page_start // rows_per_page + 1
        page_count = (len(records) + rows_per_page - 1) // rows_per_page
        fig.text(0.5, 0.03, f"Page {page_number} of {page_count}", ha="center", fontsize=9)
        fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.94))
        pdf.savefig(fig)
        plt.close(fig)


def print_table(rows: list[BenchmarkRow]) -> None:
    headers = ("implementation", "query", "mode", "status", "mean_ms", "median_ms", "output", "reason")
    table = []
    for row in rows:
        table.append(
            (
                row.implementation,
                row.query,
                row.mode,
                row.status,
                _format_number(row.mean_ms),
                _format_number(row.median_ms),
                row.output_shape,
                row.reason,
            )
        )
    widths = [len(header) for header in headers]
    for record in table:
        widths = [max(width, len(str(value))) for width, value in zip(widths, record)]

    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for record in table:
        print("  ".join(str(value).ljust(width) for value, width in zip(record, widths)))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = run_benchmarks(args)
    print_table(rows)
    write_outputs(rows, json_path=args.output_json, csv_path=args.output_csv, pdf_path=DEFAULT_PDF_PATH)
    return 1 if any(row.status == "error" for row in rows) else 0


def _make_runner(
    implementation: str,
    query: str,
    mode: str,
    device: str,
    dtype: str,
    payload: BenchmarkPayload,
) -> BenchmarkRunner:
    if implementation == "adam-torch":
        return _make_adam_torch_runner(query, mode, device, dtype, payload)
    if implementation == "adam-jax":
        return _make_adam_jax_runner(query, mode, device, dtype, payload)
    if implementation == "frax":
        return _make_frax_runner(query, mode, device, dtype, payload)
    raise ValueError(f"Unknown implementation {implementation!r}")


def _make_adam_torch_runner(
    query: str,
    mode: str,
    device_name: str,
    dtype_name: str,
    payload: BenchmarkPayload,
) -> BenchmarkRunner:
    try:
        import adam
        import torch
        from adam.pytorch import KinDynComputations
    except ImportError as exc:
        raise BackendUnavailable(f"Adam PyTorch unavailable: {exc}") from exc

    if device_name == "gpu":
        if not torch.cuda.is_available():
            raise BackendUnavailable("PyTorch CUDA is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    gravity = torch.as_tensor(GRAVITY, dtype=dtype, device=device)
    try:
        kindyn = KinDynComputations(
            str(payload.asset.adam_urdf_path),
            list(payload.asset.joint_names),
            device=device,
            dtype=dtype,
            gravity=gravity,
        )
    except TypeError:
        kindyn = KinDynComputations(
            str(payload.asset.adam_urdf_path),
            list(payload.asset.joint_names),
            gravity=gravity,
        )
    kindyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)

    if mode == "single-cpu":
        base_transform = torch.as_tensor(payload.base_transform, dtype=dtype, device=device)
        joint_positions = torch.as_tensor(payload.joint_positions, dtype=dtype, device=device)
        base_velocity = torch.as_tensor(payload.base_velocity, dtype=dtype, device=device)
        joint_velocities = torch.as_tensor(payload.joint_velocities, dtype=dtype, device=device)
        joint_torques = torch.as_tensor(payload.joint_torques, dtype=dtype, device=device)
    else:
        base_transform = torch.as_tensor(payload.batched_base_transform, dtype=dtype, device=device)
        joint_positions = torch.as_tensor(payload.batched_joint_positions, dtype=dtype, device=device)
        base_velocity = torch.as_tensor(payload.batched_base_velocity, dtype=dtype, device=device)
        joint_velocities = torch.as_tensor(payload.batched_joint_velocities, dtype=dtype, device=device)
        joint_torques = torch.as_tensor(payload.batched_joint_torques, dtype=dtype, device=device)

    def generalized_torque() -> Any:
        zeros_shape = (*joint_torques.shape[:-1], 6)
        zeros = torch.zeros(zeros_shape, dtype=dtype, device=device)
        return torch.cat((zeros, joint_torques), dim=-1)

    def jacobian() -> Any:
        return torch.cat(
            [
                kindyn.jacobian(link, base_transform, joint_positions)[..., :3, :]
                for link in FOOT_LINKS
            ],
            dim=-2,
        )

    def forward_dynamics(*, include_coriolis: bool) -> Any:
        mass = kindyn.mass_matrix(base_transform, joint_positions)
        gravity_term = kindyn.gravity_term(base_transform, joint_positions)
        rhs = generalized_torque() - gravity_term
        if include_coriolis:
            rhs = rhs - kindyn.coriolis_term(
                base_transform,
                joint_positions,
                base_velocity,
                joint_velocities,
            )
        return torch.linalg.solve(mass, rhs.unsqueeze(-1)).squeeze(-1)

    if query == "jacobian":
        run = jacobian
    elif query == "forward-dynamics":
        run = lambda: forward_dynamics(include_coriolis=True)
    elif query == "forward-dynamics-no-coriolis":
        run = lambda: forward_dynamics(include_coriolis=False)
    else:
        raise ValueError(f"Unknown query {query!r}")

    def sync(value: Any) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if hasattr(value, "detach"):
            value.detach()

    return BenchmarkRunner(run=run, sync=sync)


def _make_adam_jax_runner(
    query: str,
    mode: str,
    device_name: str,
    dtype_name: str,
    payload: BenchmarkPayload,
) -> BenchmarkRunner:
    try:
        import adam
        import jax

        jax.config.update("jax_enable_x64", dtype_name == "float64")
        import jax.numpy as jnp
        from adam.jax import KinDynComputations
    except ImportError as exc:
        raise BackendUnavailable(f"Adam JAX unavailable: {exc}") from exc

    device = _jax_device(jax, device_name, "Adam JAX")
    dtype = jnp.float64 if dtype_name == "float64" else jnp.float32
    gravity = jnp.asarray(GRAVITY, dtype=dtype)
    kindyn = KinDynComputations(
        str(payload.asset.adam_urdf_path),
        list(payload.asset.joint_names),
        dtype=dtype,
        gravity=gravity,
    )
    kindyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)

    if mode == "single-cpu":
        base_transform = _put_jax(jax, jnp, payload.base_transform, dtype, device)
        joint_positions = _put_jax(jax, jnp, payload.joint_positions, dtype, device)
        base_velocity = _put_jax(jax, jnp, payload.base_velocity, dtype, device)
        joint_velocities = _put_jax(jax, jnp, payload.joint_velocities, dtype, device)
        joint_torques = _put_jax(jax, jnp, payload.joint_torques, dtype, device)
    else:
        base_transform = _put_jax(jax, jnp, payload.batched_base_transform, dtype, device)
        joint_positions = _put_jax(jax, jnp, payload.batched_joint_positions, dtype, device)
        base_velocity = _put_jax(jax, jnp, payload.batched_base_velocity, dtype, device)
        joint_velocities = _put_jax(jax, jnp, payload.batched_joint_velocities, dtype, device)
        joint_torques = _put_jax(jax, jnp, payload.batched_joint_torques, dtype, device)

    def generalized_torque(tau: Any) -> Any:
        return jnp.concatenate((jnp.zeros((*tau.shape[:-1], 6), dtype=dtype), tau), axis=-1)

    def jacobian_one(transform: Any, positions: Any, _: Any, __: Any, ___: Any) -> Any:
        return jnp.concatenate(
            [kindyn.jacobian(link, transform, positions)[:3, :] for link in FOOT_LINKS],
            axis=-2,
        )

    def dynamics_one(
        transform: Any,
        positions: Any,
        base_vel: Any,
        joint_vel: Any,
        tau: Any,
        *,
        include_coriolis: bool,
    ) -> Any:
        mass = kindyn.mass_matrix(transform, positions)
        rhs = generalized_torque(tau) - kindyn.gravity_term(transform, positions)
        if include_coriolis:
            rhs = rhs - kindyn.coriolis_term(transform, positions, base_vel, joint_vel)
        return jnp.linalg.solve(mass, rhs)

    if query == "jacobian":
        single_kernel = jacobian_one
    elif query == "forward-dynamics":
        single_kernel = lambda transform, positions, base_vel, joint_vel, tau: dynamics_one(
            transform,
            positions,
            base_vel,
            joint_vel,
            tau,
            include_coriolis=True,
        )
    elif query == "forward-dynamics-no-coriolis":
        single_kernel = lambda transform, positions, base_vel, joint_vel, tau: dynamics_one(
            transform,
            positions,
            base_vel,
            joint_vel,
            tau,
            include_coriolis=False,
        )
    else:
        raise ValueError(f"Unknown query {query!r}")

    if mode == "single-cpu":
        kernel = jax.jit(single_kernel)
    else:
        kernel = jax.jit(
            lambda transforms, positions, base_velocities, joint_velocities_, torques: jax.lax.map(
                lambda sample: single_kernel(*sample),
                (transforms, positions, base_velocities, joint_velocities_, torques),
            )
        )

    def run() -> Any:
        return kernel(base_transform, joint_positions, base_velocity, joint_velocities, joint_torques)

    return BenchmarkRunner(run=run, sync=lambda value: _sync_jax(jax, value))


def _make_frax_runner(
    query: str,
    mode: str,
    device_name: str,
    dtype_name: str,
    payload: BenchmarkPayload,
) -> BenchmarkRunner:
    try:
        import jax

        jax.config.update("jax_enable_x64", dtype_name == "float64")
        import jax.numpy as jnp
        from frax.core.humanoid import Humanoid
    except ImportError as exc:
        raise BackendUnavailable(f"Frax unavailable: {exc}") from exc

    device = _jax_device(jax, device_name, "Frax")
    dtype = jnp.float64 if dtype_name == "float64" else jnp.float32
    try:
        robot = Humanoid(
            str(payload.asset.adam_urdf_path),
            left_hand_parent_joint_name="left_wrist_yaw_joint",
            right_hand_parent_joint_name="right_wrist_yaw_joint",
            left_foot_parent_joint_name=FRAX_FOOT_JOINTS[0],
            right_foot_parent_joint_name=FRAX_FOOT_JOINTS[1],
            joint_ordering=list(payload.asset.joint_names),
            add_floating_base=True,
        )
    except Exception as exc:
        raise BackendUnavailable(f"Frax could not load Unitree G1: {type(exc).__name__}: {exc}") from exc

    if getattr(robot, "num_joints", payload.frax_configuration.shape[-1]) != payload.frax_configuration.shape[-1]:
        raise BackendUnavailable(
            f"Frax model has {getattr(robot, 'num_joints', 'unknown')} generalized coordinates; "
            f"expected {payload.frax_configuration.shape[-1]}"
        )

    if mode == "single-cpu":
        q = _put_jax(jax, jnp, payload.frax_configuration, dtype, device)
        qd = _put_jax(jax, jnp, payload.frax_velocity, dtype, device)
        tau = _put_jax(jax, jnp, payload.frax_torque, dtype, device)
    else:
        q = _put_jax(jax, jnp, payload.batched_frax_configuration, dtype, device)
        qd = _put_jax(jax, jnp, payload.batched_frax_velocity, dtype, device)
        tau = _put_jax(jax, jnp, payload.batched_frax_torque, dtype, device)

    def jacobian_one(configuration: Any, _: Any, __: Any) -> Any:
        return jnp.concatenate(
            (robot.left_foot_jacobian(configuration)[:3, :], robot.right_foot_jacobian(configuration)[:3, :]),
            axis=-2,
        )

    def dynamics_one(configuration: Any, velocity: Any, generalized_force: Any, *, include_coriolis: bool) -> Any:
        mass = robot.mass_matrix(configuration)
        rhs = generalized_force - robot.gravity_vector(configuration)
        if include_coriolis:
            rhs = rhs - robot.centrifugal_coriolis_vector(configuration, velocity)
        return jnp.linalg.solve(mass, rhs)

    if query == "jacobian":
        single_kernel = jacobian_one
    elif query == "forward-dynamics":
        single_kernel = lambda configuration, velocity, generalized_force: dynamics_one(
            configuration,
            velocity,
            generalized_force,
            include_coriolis=True,
        )
    elif query == "forward-dynamics-no-coriolis":
        single_kernel = lambda configuration, velocity, generalized_force: dynamics_one(
            configuration,
            velocity,
            generalized_force,
            include_coriolis=False,
        )
    else:
        raise ValueError(f"Unknown query {query!r}")

    if mode == "single-cpu":
        kernel = jax.jit(single_kernel)
    else:
        kernel = jax.jit(
            lambda configurations, velocities, generalized_forces: jax.lax.map(
                lambda sample: single_kernel(*sample),
                (configurations, velocities, generalized_forces),
            )
        )

    def run() -> Any:
        return kernel(q, qd, tau)

    return BenchmarkRunner(run=run, sync=lambda value: _sync_jax(jax, value))


def _expand_selection(values: list[str], choices: tuple[str, ...], flag_name: str) -> list[str]:
    expanded: list[str] = []
    for value in values:
        expanded.extend(part.strip() for part in value.split(",") if part.strip())
    if not expanded or "all" in expanded:
        return list(choices)
    invalid = [value for value in expanded if value not in choices]
    if invalid:
        raise SystemExit(f"{flag_name}: invalid values {invalid}; expected one of {choices} or all")
    return expanded


def _selected_modes(devices: list[str]) -> list[tuple[str, str]]:
    modes: list[tuple[str, str]] = []
    if "cpu" in devices:
        modes.extend((("single-cpu", "cpu"), ("batched-cpu", "cpu")))
    if "gpu" in devices:
        modes.append(("batched-gpu", "gpu"))
    return modes


def _row_batch_size(mode: str, configured_batch_size: int) -> int:
    return 1 if mode == "single-cpu" else configured_batch_size


def _skipped_row(
    implementation: str,
    query: str,
    mode: str,
    device: str,
    payload: BenchmarkPayload,
    dtype: str,
    warmups: int,
    iters: int,
    reason: str,
) -> BenchmarkRow:
    return BenchmarkRow(
        implementation=implementation,
        query=query,
        mode=mode,
        device=device,
        batch_size=_row_batch_size(mode, payload.batch_size),
        dtype=dtype,
        warmups=warmups,
        iters=iters,
        status="skipped",
        reason=reason,
    )


def _jax_device(jax: Any, device_name: str, backend_name: str) -> Any:
    if device_name == "cpu":
        devices = jax.devices("cpu")
    else:
        devices = [
            device
            for device in jax.devices()
            if device.platform in {"gpu", "cuda", "rocm"}
        ]
    if not devices:
        raise BackendUnavailable(f"{backend_name} {device_name.upper()} device is not available")
    return devices[0]


def _put_jax(jax: Any, jnp: Any, value: np.ndarray, dtype: Any, device: Any) -> Any:
    return jax.device_put(jnp.asarray(value, dtype=dtype), device)


def _sync_jax(jax: Any, value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _shape_string(value: Any) -> str:
    leaves = _flatten_outputs(value)
    if not leaves:
        return ""
    return ",".join(str(tuple(getattr(leaf, "shape", ()))) for leaf in leaves)


def _flatten_outputs(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        leaves: list[Any] = []
        for item in value:
            leaves.extend(_flatten_outputs(item))
        return leaves
    return [value]


def _format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _implementation_color(implementation: str) -> str:
    return {
        "adam-torch": "#4c78a8",
        "adam-jax": "#f58518",
        "frax": "#54a24b",
    }.get(implementation, "#7f7f7f")


def _shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


if __name__ == "__main__":
    sys.exit(main())
