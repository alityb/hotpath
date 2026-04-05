# hotpath

Profiler for LLM inference. Kernel timing, request lifecycle tracing, and disaggregation analysis for vLLM and SGLang.

## What it does

**Profile** a live vLLM or SGLang endpoint with real traffic: capture CUDA kernel timing, Prometheus server metrics, and per-request latency breakdowns.

**Analyze** the results: prefill vs decode phase breakdown, KV cache efficiency, prefix sharing patterns, queue depth over time, TTFT and decode-per-token distributions.

**Advise** on disaggregation: an analytical M/G/1 queueing model estimates whether splitting prefill and decode onto separate GPU pools improves throughput. If recommended, hotpath generates ready-to-use deployment configs for vLLM, llm-d, and Dynamo.

## Install

```bash
pip install hotpath
```

## Quick start

Profile a live vLLM server:

```bash
hotpath serve-profile \
  --endpoint http://localhost:8000 \
  --traffic prompts.jsonl \
  --concurrency 4 \
  --duration 300 \
  --output .hotpath/run
```

View results:

```bash
hotpath serve-report .hotpath/run/serve_profile.db
```

Generate disaggregation deployment configs:

```bash
hotpath disagg-config .hotpath/run/serve_profile.db --format all
```

For full server-side timing (queue wait, prefill, decode phases), start vLLM with debug logging and pass the log file:

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve <model> 2>vllm.log &

hotpath serve-profile \
  --endpoint http://localhost:8000 \
  --traffic prompts.jsonl \
  --server-log vllm.log \
  --concurrency 4 \
  --duration 300
```

For kernel-level GPU phase breakdown, add `--nsys`:

```bash
hotpath serve-profile --endpoint http://localhost:8000 --traffic prompts.jsonl --nsys
```

## Traffic file format

JSONL, one request per line:

```json
{"prompt": "Explain KV cache eviction policy.", "max_tokens": 256}
{"prompt": "Write a Python retry decorator with exponential backoff.", "max_tokens": 400}
```

ShareGPT format is also accepted.

## Commands

| Command | Description |
|---------|-------------|
| `serve-profile` | Profile a live vLLM/SGLang server with traffic replay |
| `serve-report` | Print a serving analysis report |
| `disagg-config` | Generate deployment configs for disaggregated serving |
| `profile` | GPU kernel profiling under RL-style rollout workloads |
| `report` | View a saved kernel profile |
| `diff` | Compare two kernel profiles |
| `bench` | Benchmark individual GPU kernel implementations |
| `export` | Export profile data to JSON, CSV, or OTLP |
| `doctor` | Check local profiling environment |
| `lock-clocks` | Lock GPU clocks for reproducible measurements |

## System requirements

- Linux
- NVIDIA GPU with CUDA driver
- `nsys` (for kernel profiling; not required for serving analysis)
- vLLM or SGLang (for serving analysis)

## Build from source

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Install from source:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install .
```

Requirements: CMake 3.28+, C++20 compiler, SQLite3.

## How it works

hotpath is a single C++ binary with no runtime dependencies beyond SQLite3.

Data is collected from three sources:

1. **Kernel traces** -- nsys captures GPU kernel execution. hotpath parses the output, categorizes kernels (GEMM, attention, MoE, etc.), and classifies them as prefill or decode phase by timing correlation with server events.

2. **Server metrics** -- Prometheus metrics from vLLM or SGLang `/metrics` endpoints are polled at 1 Hz. Batch size, queue depth, KV cache utilization, and preemption counts are tracked over the profiling window.

3. **Request lifecycle** -- vLLM debug logs are parsed to extract per-request timestamps: arrival, queue wait, prefill start, decode start, completion. These are stored as structured traces and can be exported as OpenTelemetry spans.

The disaggregation advisor uses a simplified M/G/1 queueing model to estimate whether splitting prefill and decode onto separate GPU pools would improve throughput. It searches over P:D ratios and accounts for KV transfer overhead to produce a concrete recommendation with estimated throughput improvement.

All data is stored in SQLite databases for offline analysis and comparison across runs.

## Release notes

See [CHANGELOG.md](CHANGELOG.md).
