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
python3 -m venv .venv-install
. .venv-install/bin/activate
pip install .
```

Or install the CLI into an isolated user environment with `pipx`:

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install .
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

For a remote GPU host with a checked-out and built `rlprof` tree:

```bash
./build/rlprof profile \
  --target ubuntu@a10g-box \
  --target-workdir /srv/rlprof \
  --model Qwen/Qwen3-8B \
  --prompts 64 \
  --rollouts 4 \
  --min-tokens 256 \
  --max-tokens 1024 \
  --input-len 512 \
  --output .rlprof/qwen3_8b_remote
```

This runs `vllm`, `nsys`, and the profiler on the remote host over SSH, then copies the core artifacts back locally and rewrites the saved artifact paths in the fetched `.db`.

For repeatable controller-driven remote runs, save a target that uses an SSH alias or any other non-interactive SSH setup that plain `ssh` and `scp` can use directly:

```sshconfig
Host a10g-box
  HostName gpu-host.example.com
  User ubuntu
  IdentityFile ~/.ssh/a10g.pem
  IdentitiesOnly yes
```

```bash
./build/rlprof target add a10g \
  --host a10g-box \
  --workdir /srv/rlprof \
  --python /srv/rlprof/.venv/bin/python \
  --vllm /srv/rlprof/.venv/bin/vllm

./build/rlprof doctor --target a10g
./build/rlprof target bootstrap a10g
./build/rlprof bench \
  --target a10g \
  --kernel silu_and_mul \
  --shapes 64x4096 \
  --warmup 1 \
  --n-iter 5 \
  --repeats 1 \
  --output .rlprof/bench_remote_smoke.json
./build/rlprof profile \
  --target a10g \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts 1 \
  --rollouts 1 \
  --min-tokens 8 \
  --max-tokens 8 \
  --input-len 16 \
  --output .rlprof/qwen_remote_smoke
./build/rlprof recover \
  --target a10g \
  --remote-db /srv/rlprof/.rlprof/qwen_remote_smoke.db \
  --output .rlprof/qwen_remote_smoke_recovered.db
```

If the remote `python3` and `vllm` are already on `PATH`, you can omit `--python` and `--vllm` when saving the target.

For faster remote iteration, `rlprof` now skips copying the heavy remote `.nsys-rep` by default and only fetches the `.db`, `.sqlite`, telemetry XML, and server log. If you want the full report artifact locally too:

```bash
./build/rlprof profile \
  --target ubuntu@a10g-box \
  --target-workdir /srv/rlprof \
  --model Qwen/Qwen3-8B \
  --fetch-nsys-rep \
  --output .rlprof/qwen3_8b_remote_full
```

Without `--fetch-nsys-rep`, `trace --open` will point you at the stored remote report path and tell you to rerun with `--fetch-nsys-rep` or use `recover`.

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

### Aggregate

```bash
./build/rlprof aggregate \
  .rlprof/qwen3_8b_prod_256_stability_r2.db \
  .rlprof/qwen3_8b_prod_256_stability_r3.db \
  --output .rlprof/qwen_aggregate.db
```

This combines per-host or per-run profiles into one stored `.db`. Kernel totals and metric samples are merged, then metric summaries are recomputed from the merged samples.

### Cluster Profile

Run the same profile config across multiple known SSH targets, then write one per-target profile plus an aggregate rollup:

```bash
./build/rlprof cluster-profile \
  --targets a10g-1,a10g-2 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts 8 \
  --rollouts 4 \
  --min-tokens 128 \
  --max-tokens 256 \
  --input-len 128 \
  --output .rlprof/qwen_cluster
```

This writes:

- `.rlprof/qwen_cluster_<target>.db`
- `.rlprof/qwen_cluster_aggregate.db`

`cluster-profile` now launches the per-target profiles concurrently and assigns one synchronized future start epoch to every target so the measured `nsys` windows open at the same time on known hosts. If the same host appears more than once, `rlprof` automatically offsets the port per occurrence. In practice, concurrent profiles on the same single GPU still need enough free memory for multiple `vllm serve` processes.

### Soak Profile

Run the same profile repeatedly for long-run stability or regression checks:

```bash
./build/rlprof soak-profile \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts 1 \
  --rollouts 1 \
  --min-tokens 8 \
  --max-tokens 8 \
  --input-len 16 \
  --iterations 10 \
  --pause-sec 5 \
  --validate-each \
  --output .rlprof/qwen_soak
```

This writes `.rlprof/qwen_soak_i1.db`, `.rlprof/qwen_soak_i2.db`, and so on.

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
  --shapes 64x4096,256x4096 \
  --warmup 20 \
  --n-iter 200 \
  --repeats 5 \
  --output .rlprof/bench_silu.json
```

Example A10G output:

```text
saved: .rlprof/bench_silu.json
gpu: NVIDIA A10G | driver: 580.126.16 | sm clock: 1710 mhz | mem clock: 6251 mhz | temp: 28 c | power: 60.9/300.0 w

kernel              implementation      shape           avg us    stddev     cv %    min us    p50 us    p99 us         GB/s  valid    det  timing     env  unstable
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
silu_and_mul        vllm-cuda           64x4096         33.652     0.785    2.332    29.504    31.936    64.896       46.739    yes    yes      no      no        no
silu_and_mul        vllm-cuda           256x4096        33.738     0.401    1.187    30.048    31.808    69.760      186.479    yes    yes      no      no        no
```

Additional benchmark examples:

```bash
./build/rlprof bench --kernel fused_add_rms_norm --shapes 64x4096,256x4096 --warmup 20 --n-iter 200 --repeats 5 --output auto
./build/rlprof bench --kernel rotary_embedding --shapes 64x1024,256x1024 --warmup 20 --n-iter 200 --repeats 5 --output auto
```

Remote GPU benching uses the same SSH target model:

```bash
./build/rlprof bench \
  --target ubuntu@a10g-box \
  --target-workdir /srv/rlprof \
  --kernel silu_and_mul \
  --shapes 64x4096,256x4096 \
  --warmup 20 \
  --n-iter 200 \
  --repeats 5 \
  --output .rlprof/bench_remote_silu.json
```

The GPU bench helper now does three things before treating a number as usable:

- reference validation against the kernel family reference implementation
- determinism validation on identical cloned inputs
- unmeasured priming before timed repeats so first-use compile/setup cost is kept out of the timed window

`bench` output now separates:

- correctness failures
- timing warnings
- environment warnings

Generic throttle-reason activity alone no longer marks a row unstable. Environment warnings are reserved for material SM clock movement, power-cap throttling, thermal throttling, or high GPU temperature.

Use `--output auto` to archive each run in `.rlprof/bench_<kernel>_<timestamp>.json`, or `--output none` to disable archival.

Compare archived runs:

```bash
./build/rlprof bench-compare \
  .rlprof/bench_release_silu.json \
  .rlprof/bench_release_silu.json
```

### Attach

If you already have a server running, you can attach `rlprof` to its HTTP endpoint:

```bash
./build/rlprof profile \
  --attach http://127.0.0.1:8000 \
  --prompts 64 \
  --rollouts 4 \
  --min-tokens 256 \
  --max-tokens 1024 \
  --input-len 512 \
  --output .rlprof/attached_metrics_only
```

This is a fast path because it skips server launch, but it is currently metrics-only: no kernel trace is captured unless `rlprof` launched the server under `nsys`.

`--attach-pid` is currently a fail-fast boundary. If the installed `nsys` does not advertise PID attach support, `rlprof` exits before sending traffic and prints the `nsys` capability error directly.

### Doctor

```bash
./build/rlprof doctor
```

Remote environment checks:

```bash
./build/rlprof doctor --target ubuntu@a10g-box --target-workdir /srv/rlprof
```

### Targets

Save reusable SSH targets:

```bash
./build/rlprof target add a10g --host a10g-box --workdir /srv/rlprof --python /srv/rlprof/.venv/bin/python --vllm /srv/rlprof/.venv/bin/vllm
./build/rlprof target list
./build/rlprof target show a10g
./build/rlprof target remove a10g
```

Bootstrap a remote host from the current local checkout:

```bash
./build/rlprof target bootstrap a10g
```

This streams the current repo to the remote workdir, then runs `cmake -S . -B build` and `cmake --build build --target rlprof` on the target.

### Recover

If a remote run completed but the local session died before artifacts were copied back:

```bash
./build/rlprof recover \
  --target a10g \
  --remote-db /srv/rlprof/.rlprof/qwen3_8b_remote.db \
  --output .rlprof/qwen3_8b_remote.db
```

`recover` re-fetches the remote `.db`, `.sqlite`, `.nsys-rep`, and any saved telemetry/log artifacts, then rewrites the local artifact pointers in the copied database.

`doctor` checks the local profiling environment: GPU visibility, `nsys`, `vllm`, bench helper imports, clock policy, and `nsys` environment support.

### Trace And Artifacts

```bash
./build/rlprof artifacts .rlprof/qwen3_8b_moderate_clean.db
./build/rlprof trace .rlprof/qwen3_8b_moderate_clean.db
```

`artifacts` lists the stored sibling files for a profile. `trace` narrows that to raw trace-related artifacts such as `.nsys-rep`, `.sqlite`, and the saved `nvidia-smi` XML snapshot.

### Validate

```bash
./build/rlprof validate .rlprof/qwen3_8b_prod_256_stability_r3.db
```

`validate` cross-checks:

- raw `nsys` SQLite kernel totals vs stored `.db` totals
- raw `/metrics` samples vs stored summary rows
- required sibling artifact presence

### Manifest And Cleanup

Write a manifest for one stored profile:

```bash
./build/rlprof manifest .rlprof/qwen3_8b_prod_256_stability_r3.db
```

Clean older artifacts under `.rlprof` while keeping the newest profiles:

```bash
./build/rlprof cleanup --dir .rlprof --keep 10 --compress --apply
```

Without `--apply`, `cleanup` is a dry run.

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

### Completion

```bash
./build/rlprof completion bash > ~/.local/share/bash-completion/completions/rlprof
./build/rlprof completion zsh > ~/.zfunc/_rlprof
./build/rlprof completion fish > ~/.config/fish/completions/rlprof.fish
```

### Version And Defaults

```bash
./build/rlprof version
./build/rlprof reset-defaults
```

`reset-defaults` clears saved interactive prompt defaults.

## Compatibility Matrix

The current validated development environment is:

| Component | Version |
| --- | --- |
| rlprof | `0.1.0` |
| GPU | `NVIDIA A10G` |
| Driver | `580.126.16` |
| Nsight Systems | `2025.3.2.474-253236389321v0` |
| Python | `3.12` |
| torch | `2.10.0+cu128` |
| triton | `3.6.0` |
| vllm | `0.18.0` |

The remote target path assumes the target has compatible CUDA driver support, `nsys`, and a Python environment with the benchmark/profile helper dependencies available.

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

For a heavier non-GPU stress pass:

```bash
ctest --test-dir build --output-on-failure --schedule-random --repeat until-fail:25
```

## Current Boundaries

- `profile --attach URL` is metrics-only unless `rlprof` owns the server lifecycle under `nsys`.
- `profile --target HOST --model ...` gives full remote profiling because `rlprof` can launch `vllm`, `nsys`, and artifact capture on the known host over SSH.
- `profile --attach URL` without a target does not imply kernel visibility. It only has the HTTP endpoint, so it can scrape metrics and fire traffic, not trace kernels.
- `profile --attach URL --attach-pid PID` is still a boundary. If the installed `nsys` does not advertise PID attach support, `rlprof` fails fast with a direct capability error instead of falling back to an ambiguous attach attempt.
- `aggregate` provides per-node/profile rollup after the fact, but it does not synchronize trace windows across multiple remote nodes by itself.
- `cluster-profile` coordinates synchronized start times across known hosts, but a real multi-host validation still requires more than one GPU host. Running multiple traced servers on the same single GPU host can still fail from normal VRAM limits.
- Raw kernel names are authoritative. Category labels use NVTX overlap first, then runtime metadata, then conservative kernel-name matching.

## Release Notes

See [CHANGELOG.md](CHANGELOG.md) for the current release history.
