# rlprof

`rlprof` profiles vLLM inference under RL-shaped rollout traffic. It records raw CUDA kernel timing from `nsys`, vLLM `/metrics` samples, and traffic shape stats, then stores everything in SQLite for reporting, diffing, and export.

The profiler core is native C++. The benchmark command uses a Python CUDA helper on GPU machines so timing comes from `torch.cuda.Event`, not CPU wall clock.

## What It Measures

- GPU kernels from `nsys` SQLite exports
- vLLM server metrics such as preemptions, queue depth, and KV cache usage
- RL-shaped traffic statistics: request count, completion length distribution, and errors
- Microbenchmarks for `silu_and_mul`, `fused_add_rms_norm`, and `rotary_embedding`

`rlprof` does not interpret the numbers. It prints and exports them.

## Requirements

System:

- NVIDIA GPU with CUDA
- `nsys`
- `vllm`
- `cmake >= 3.28`
- C++20 compiler
- SQLite development headers

Python environment for bench and live profiling:

- Python `>= 3.10`
- `aiohttp`
- `torch`
- `triton`
- `vllm`

## Build

```bash
python3 -m venv .venv
.venv/bin/pip install pytest torch triton vllm

cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Install

```bash
python3 -m pip install .
```

This installs the `rlprof` command into the active Python environment. For GPU benchmarking support:

```bash
python3 -m pip install '.[bench]'
```

## Commands

If you run `rlprof`, `rlprof profile`, `rlprof bench`, `rlprof report`,
`rlprof diff`, or `rlprof export` without their required arguments,
`rlprof` now falls back to a sequential interactive prompt flow instead
of a fullscreen UI. Passing the usual flags still skips prompts and runs
non-interactively.

### Profile

```bash
./build/rlprof profile \
  --model Qwen/Qwen3-8B \
  --prompts 8 \
  --rollouts 4 \
  --min-tokens 128 \
  --max-tokens 256 \
  --input-len 128 \
  --port 8000 \
  --output .rlprof/qwen3_8b_moderate_clean
```

For models that require remote code:

```bash
./build/rlprof profile \
  --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --trust-remote-code \
  --prompts 8 \
  --rollouts 4 \
  --min-tokens 128 \
  --max-tokens 256 \
  --input-len 128 \
  --port 8000 \
  --output .rlprof/nemotron_4b_moderate_clean
```

### Report

```bash
./build/rlprof report .rlprof/qwen3_8b_moderate_clean.db
```

Sample category totals from a clean `Qwen/Qwen3-8B` A10G run:

```text
gemm        672.2 ms   57.1%
other       298.0 ms   25.3%
sampling     82.4 ms    7.0%
activation   71.0 ms    6.0%
attention    48.1 ms    4.1%
```

Sample category totals from a clean `NVIDIA-Nemotron-3-Nano-4B-BF16` A10G run:

```text
activation  6353.1 ms   65.2%
other       2836.9 ms   29.1%
gemm         450.1 ms    4.6%
mamba          8.5 ms    0.1%
```

### Diff

```bash
./build/rlprof diff \
  .rlprof/qwen3_8b_moderate_clean.db \
  .rlprof/nemotron_4b_moderate_clean.db
```

Excerpt from the clean Qwen vs Nemotron diff:

```text
activation   71.0 -> 6353.1 ms
gemm        672.2 ->  450.1 ms
attention    48.1 ->   29.7 ms
mamba         0.0 ->    8.5 ms
```

### Bench

`bench` uses real CUDA kernels on GPU machines:

- vLLM CUDA custom ops
- `torch.compile`
- custom Triton kernels

```bash
./build/rlprof bench \
  --kernel silu_and_mul \
  --shapes 1x4096,64x4096,256x4096 \
  --warmup 10 \
  --n-iter 50
```

Example A10G output:

```text
kernel              implementation      shape           avg ms    min ms    p50 ms    p99 ms         GB/s
---------------------------------------------------------------------------------------------------------
silu_and_mul        vllm-cuda           256x4096         0.034     0.030     0.032     0.076      187.043
silu_and_mul        torch-compile       256x4096         0.080     0.074     0.076     0.112       79.084
silu_and_mul        triton-custom       256x4096         0.059     0.055     0.057     0.078      107.373
```

Additional benchmark examples:

```bash
./build/rlprof bench --kernel fused_add_rms_norm --shapes 1x4096,64x4096,256x4096 --warmup 10 --n-iter 50
./build/rlprof bench --kernel rotary_embedding --shapes 1x1024,64x1024,256x1024 --warmup 10 --n-iter 50
```

### Export

```bash
./build/rlprof export .rlprof/qwen3_8b_moderate_clean.db --format csv
./build/rlprof export .rlprof/qwen3_8b_moderate_clean.db --format json
```

CSV export writes:

- `_meta.csv`
- `_kernels.csv`
- `_vllm_metrics.csv`
- `_vllm_metrics_summary.csv`
- `_traffic_stats.csv`

JSON export writes one file with the same sections.

## Live A10G Notes

The current implementation has already been exercised on:

- `Qwen/Qwen3-8B`
- `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`

Observed live behavior:

- clean delayed-capture profiling avoids model load and compile noise
- real `nsys` SQLite exports require resolving `CUPTI_ACTIVITY_KIND_KERNEL.shortName` through `StringIds`
- high-concurrency `Qwen/Qwen3-8B` runs on A10G can produce nonzero `vllm:num_preemptions_total`
- Nemotron shows a very different kernel mix with scan/state kernels and nonzero `mamba`

## Test Matrix

Native unit tests:

- parser
- categorizer
- report
- store
- metrics parser
- diff
- bench math/path glue
- export

Run:

```bash
ctest --test-dir build --output-on-failure
```
