from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_backends.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_backends", _BENCHMARK_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
benchmark_backends = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = benchmark_backends
_SPEC.loader.exec_module(benchmark_backends)


def test_benchmark_cli_parses_comma_separated_selections() -> None:
    args = benchmark_backends.parse_args(
        [
            "--implementations",
            "adam-torch,adam-jax",
            "--queries",
            "jacobian",
            "--devices",
            "cpu",
            "--batch-size",
            "2",
            "--warmups",
            "1",
            "--iters",
            "1",
        ]
    )

    assert args.implementations == ["adam-torch", "adam-jax"]
    assert args.queries == ["jacobian"]
    assert args.devices == ["cpu"]
    assert args.batch_size == 2


def test_adam_torch_cpu_benchmark_smoke() -> None:
    args = benchmark_backends.parse_args(
        [
            "--implementations",
            "adam-torch",
            "--queries",
            "jacobian",
            "--devices",
            "cpu",
            "--batch-size",
            "2",
            "--warmups",
            "1",
            "--iters",
            "1",
        ]
    )

    rows = benchmark_backends.run_benchmarks(args)

    assert [row.mode for row in rows] == ["single-cpu", "batched-cpu"]
    assert all(row.status == "ok" for row in rows)
    assert all(row.mean_ms is not None for row in rows)


def test_optional_backend_benchmarks_report_skips_or_success() -> None:
    args = benchmark_backends.parse_args(
        [
            "--implementations",
            "adam-jax,frax",
            "--queries",
            "jacobian",
            "--devices",
            "cpu",
            "--batch-size",
            "1",
            "--warmups",
            "1",
            "--iters",
            "1",
        ]
    )

    rows = benchmark_backends.run_benchmarks(args)

    assert len(rows) == 4
    assert all(row.status in {"ok", "skipped"} for row in rows)
    frax_rows = [row for row in rows if row.implementation == "frax"]
    assert frax_rows
    assert all(row.status in {"ok", "skipped"} for row in frax_rows)


def test_pdf_export_smoke(tmp_path: Path) -> None:
    rows = [
        benchmark_backends.BenchmarkRow(
            implementation="adam-torch",
            query="jacobian",
            mode="single-cpu",
            device="cpu",
            batch_size=1,
            dtype="float64",
            warmups=1,
            iters=2,
            status="ok",
            mean_ms=1.2,
            median_ms=1.1,
            min_ms=1.0,
            max_ms=1.4,
            std_ms=0.2,
            output_shape="(6, 35)",
        ),
        benchmark_backends.BenchmarkRow(
            implementation="frax",
            query="forward-dynamics",
            mode="batched-cpu",
            device="cpu",
            batch_size=2,
            dtype="float64",
            warmups=1,
            iters=2,
            status="skipped",
            reason="backend unavailable",
        ),
    ]
    pdf_path = tmp_path / "benchmark.pdf"

    benchmark_backends.write_pdf(rows, pdf_path)

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_write_outputs_always_accepts_default_pdf_path(tmp_path: Path) -> None:
    rows = [
        benchmark_backends.BenchmarkRow(
            implementation="adam-torch",
            query="jacobian",
            mode="single-cpu",
            device="cpu",
            batch_size=1,
            dtype="float64",
            warmups=0,
            iters=1,
            status="ok",
            mean_ms=1.0,
            median_ms=1.0,
            min_ms=1.0,
            max_ms=1.0,
            std_ms=0.0,
            output_shape="(6, 35)",
        )
    ]
    pdf_path = tmp_path / benchmark_backends.DEFAULT_PDF_FILENAME

    benchmark_backends.write_outputs(rows, json_path=None, csv_path=None, pdf_path=pdf_path)

    assert pdf_path.exists()
