# rlprof

`rlprof` is a measurement tool for profiling vLLM inference under RL-style rollout workloads. It is designed for cases where request-level metrics are not enough and you need direct visibility into kernel execution, server-side pressure, and traffic shape to explain throughput or latency behavior.

It records:
- CUDA kernel timing from `nsys`
- vLLM `/metrics` samples
- rollout traffic shape statistics
- optional benchmark results for selected GPU kernels

`rlprof` stores collected data in SQLite and reports raw numbers. It is not a training framework, dashboard, or optimization layer.

## Example Use

Profile a local vLLM server lifecycle:

```bash
rlprof profile \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts 1 \
  --rollouts 1 \
  --min-tokens 8 \
  --max-tokens 8 \
  --input-len 16 \
  --output .rlprof/local_smoke
```

Inspect the resulting profile:

```bash
rlprof report .rlprof/local_smoke.db
```

## Install

Install the published package from PyPI:

```bash
pip install rlprof
```

## System Requirements

- Linux
- NVIDIA GPU
- CUDA driver
- `nsys`
- `vllm`

## Build From Source

For local development or source builds, install the required system packages first:

- CMake 3.28 or newer
- C++20 compiler
- SQLite development headers

Then build the project:

```bash
cmake -S . -B build
cmake --build build --parallel
```

Run the test suite:

```bash
ctest --test-dir build --output-on-failure
```

Install from the local source tree:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install .
```

## Release Notes

See [CHANGELOG.md](CHANGELOG.md).
