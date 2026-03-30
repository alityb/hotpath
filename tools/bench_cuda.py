#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from dataclasses import dataclass

import torch

from bench_cuda_kernels import KernelSpec, build_registry, clone_state


@dataclass
class BenchResult:
    kernel: str
    implementation: str
    shape: tuple[int, int]
    dtype: str
    avg_ms: float
    stddev_ms: float
    repeat_cv_pct: float
    min_ms: float
    p50_ms: float
    p99_ms: float
    bandwidth_gb_s: float
    repeats: int
    validation_passed: bool
    validation_max_abs_error: float
    unstable: bool
    warnings: list[str]


@dataclass
class GpuSnapshot:
    name: str
    driver_version: str
    pstate: str
    sm_clock_mhz: float
    mem_clock_mhz: float
    temp_c: float
    power_draw_w: float
    power_limit_w: float
    throttle_active: str
    sw_power_cap: str
    sw_thermal_slowdown: str
    hw_thermal_slowdown: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--shapes", default="1x4096,64x4096,256x4096")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def parse_shapes(spec: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for token in spec.split(","):
        if not token:
            continue
        left, right = token.split("x", 1)
        batch = int(left)
        hidden = int(right)
        if batch <= 0 or hidden <= 0:
            raise ValueError(f"invalid shape: {token}")
        shapes.append((batch, hidden))
    if not shapes:
        raise ValueError("at least one shape is required")
    return shapes


def percentile(values: list[float], quantile: float) -> float:
    values = sorted(values)
    index = int(((len(values) - 1) * quantile) + 0.999999999)
    return values[index]


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def dtype_size(dtype: torch.dtype) -> int:
    if dtype in (torch.bfloat16, torch.float16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"unsupported dtype: {dtype}")


def query_gpu_snapshot() -> GpuSnapshot | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,pstate,clocks.current.sm,clocks.current.memory,"
        "temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active,"
        "clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.sw_thermal_slowdown,"
        "clocks_throttle_reasons.hw_thermal_slowdown",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True).splitlines()[0]
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
        return None

    fields = [part.strip() for part in output.split(",")]
    if len(fields) < 12:
        return None

    return GpuSnapshot(
        name=fields[0],
        driver_version=fields[1],
        pstate=fields[2],
        sm_clock_mhz=float(fields[3]),
        mem_clock_mhz=float(fields[4]),
        temp_c=float(fields[5]),
        power_draw_w=float(fields[6]),
        power_limit_w=float(fields[7]),
        throttle_active=fields[8],
        sw_power_cap=fields[9],
        sw_thermal_slowdown=fields[10],
        hw_thermal_slowdown=fields[11],
    )


def measure_once(fn, warmup: int, n_iter: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def max_abs_error(
    reference_outputs: tuple[torch.Tensor, ...],
    candidate_outputs: tuple[torch.Tensor, ...],
) -> float:
    max_error = 0.0
    for reference, candidate in zip(reference_outputs, candidate_outputs):
        error = (reference.to(torch.float32) - candidate.to(torch.float32)).abs().max().item()
        max_error = max(max_error, float(error))
    return max_error


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 2e-2, 2e-2


def validate_implementation(spec: KernelSpec, implementation, shape, dtype_name: str) -> tuple[bool, float]:
    dtype = dtype_from_name(dtype_name)
    reference_state = clone_state(spec.setup(shape, dtype))
    candidate_state = clone_state(reference_state)
    with torch.inference_mode():
        reference_outputs = spec.reference(reference_state)
        candidate_outputs = implementation(candidate_state)
    atol, rtol = tolerances(dtype)
    max_error = max_abs_error(reference_outputs, candidate_outputs)
    passed = all(
        torch.allclose(ref, cand, atol=atol, rtol=rtol)
        for ref, cand in zip(reference_outputs, candidate_outputs)
    )
    return passed, max_error


def benchmark(
    kernel: str,
    spec: KernelSpec,
    implementation_name: str,
    implementation,
    shape: tuple[int, int],
    dtype_name: str,
    warmup: int,
    n_iter: int,
    repeats: int,
) -> BenchResult:
    dtype = dtype_from_name(dtype_name)
    validation_passed, validation_max_abs_error = validate_implementation(
        spec, implementation, shape, dtype_name
    )

    all_times: list[float] = []
    repeat_means: list[float] = []
    snapshots: list[GpuSnapshot] = []

    for _ in range(repeats):
        state = spec.setup(shape, dtype)
        times = measure_once(lambda: implementation(state), warmup, n_iter)
        all_times.extend(times)
        repeat_means.append(sum(times) / len(times))
        snapshot = query_gpu_snapshot()
        if snapshot is not None:
            snapshots.append(snapshot)

    avg_ms = sum(repeat_means) / len(repeat_means)
    stddev_ms = statistics.stdev(repeat_means) if len(repeat_means) > 1 else 0.0
    repeat_cv_pct = 0.0 if avg_ms == 0 else (stddev_ms / avg_ms) * 100.0
    warnings: list[str] = []

    if repeat_cv_pct > 5.0:
        warnings.append(f"repeat cv exceeded threshold ({repeat_cv_pct:.1f}%)")

    if snapshots:
        sm_clocks = [snapshot.sm_clock_mhz for snapshot in snapshots]
        sm_avg = sum(sm_clocks) / len(sm_clocks)
        sm_variation_pct = 0.0 if sm_avg == 0 else ((max(sm_clocks) - min(sm_clocks)) / sm_avg) * 100.0
        if sm_variation_pct > 10.0:
          warnings.append(f"sm clock varied materially ({sm_variation_pct:.1f}%)")
        if any(snapshot.sw_power_cap == "Active" for snapshot in snapshots):
          warnings.append("power cap throttling observed")
        if any(
            snapshot.sw_thermal_slowdown == "Active" or snapshot.hw_thermal_slowdown == "Active"
            for snapshot in snapshots
        ):
          warnings.append("thermal throttling observed")
        if any(snapshot.throttle_active != "Not Active" for snapshot in snapshots):
          warnings.append("clock throttle reasons were active")
        if max(snapshot.temp_c for snapshot in snapshots) >= 80.0:
          warnings.append("gpu temperature reached high operating range")

    if not validation_passed:
        warnings.append(f"validation failed (max abs error {validation_max_abs_error:.4g})")

    return BenchResult(
        kernel=kernel,
        implementation=implementation_name,
        shape=shape,
        dtype=dtype_name,
        avg_ms=avg_ms,
        stddev_ms=stddev_ms,
        repeat_cv_pct=repeat_cv_pct,
        min_ms=min(all_times),
        p50_ms=percentile(all_times, 0.50),
        p99_ms=percentile(all_times, 0.99),
        bandwidth_gb_s=spec.bytes_processed(shape) / (avg_ms / 1000.0) / 1e9,
        repeats=repeats,
        validation_passed=validation_passed,
        validation_max_abs_error=validation_max_abs_error,
        unstable=bool(warnings),
        warnings=warnings,
    )


def render_json(results: list[BenchResult], gpu_snapshot: GpuSnapshot | None) -> str:
    grouped_warnings = [
        f"{result.kernel} {result.implementation} {result.shape[0]}x{result.shape[1]}: {warning}"
        for result in results
        for warning in result.warnings
    ]
    payload = {
        "gpu": None
        if gpu_snapshot is None
        else {
            "name": gpu_snapshot.name,
            "driver_version": gpu_snapshot.driver_version,
            "sm_clock_mhz": gpu_snapshot.sm_clock_mhz,
            "mem_clock_mhz": gpu_snapshot.mem_clock_mhz,
            "temp_c": gpu_snapshot.temp_c,
            "power_draw_w": gpu_snapshot.power_draw_w,
            "power_limit_w": gpu_snapshot.power_limit_w,
        },
        "results": [
            {
                "kernel": result.kernel,
                "implementation": result.implementation,
                "shape": f"{result.shape[0]}x{result.shape[1]}",
                "dtype": result.dtype,
                "avg_us": result.avg_ms * 1000.0,
                "stddev_us": result.stddev_ms * 1000.0,
                "cv_pct": result.repeat_cv_pct,
                "min_us": result.min_ms * 1000.0,
                "p50_us": result.p50_ms * 1000.0,
                "p99_us": result.p99_ms * 1000.0,
                "bandwidth_gb_s": result.bandwidth_gb_s,
                "valid": result.validation_passed,
                "validation_max_abs_error": result.validation_max_abs_error,
                "unstable": result.unstable,
            }
            for result in results
        ],
        "warnings": grouped_warnings,
    }
    return json.dumps(payload)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for tools/bench_cuda.py")

    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be > 0")

    shapes = parse_shapes(args.shapes)
    dtype = dtype_from_name(args.dtype)
    registry = build_registry(dtype_size(dtype))

    if args.kernel not in registry:
        raise SystemExit(f"unsupported kernel: {args.kernel}")

    spec = registry[args.kernel]
    results: list[BenchResult] = []
    torch.set_grad_enabled(False)

    with torch.inference_mode():
        for implementation_name, implementation in spec.implementations:
            for shape in shapes:
                results.append(
                    benchmark(
                        kernel=args.kernel,
                        spec=spec,
                        implementation_name=implementation_name,
                        implementation=implementation,
                        shape=shape,
                        dtype_name=args.dtype,
                        warmup=args.warmup,
                        n_iter=args.n_iter,
                        repeats=args.repeats,
                    )
                )

    print(render_json(results, query_gpu_snapshot()))


if __name__ == "__main__":
    main()
