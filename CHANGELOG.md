# Changelog

## v0.2.2 - 2026-04-05

Release workflow safety rollback.

Highlights:

- Linux wheel publishing is temporarily disabled because the manylinux packaged binary still crashes during wheel smoke tests, even though source builds are healthy
- PyPI release flow now publishes the source distribution only, which keeps installs on the safe path that compiles against the target system rather than shipping a broken prebuilt ELF
- the wheel smoke command remains in `pyproject.toml` so future wheel work still fails fast until the packaging root cause is fixed

## v0.2.1 - 2026-04-05

Packaging and interactive TUI hardening.

Highlights:

- release workflow now installs the built wheel artifacts on fresh Python 3.10, 3.11, and 3.12 runners and blocks PyPI publish unless `hotpath --help`, `hotpath`, and `hotpath version` all succeed from the installed entrypoint
- `cibuildwheel` now smoke-tests the installed `hotpath` CLI instead of only importing the Python package, which would have caught the broken packaged ELF before release
- interactive arrow-key menus now clip rendered lines to terminal width, avoiding wrapped redraw corruption in Ghostty and other narrow terminals

## v0.2.0 - 2026-04-05

Serving analysis, interactive TUI, and numerical hardening.

Highlights:

- `serve-profile` -- live dashboard with in-place redraws during traffic replay, `--concurrency N` for parallel in-flight requests, Prometheus metrics polled at 1 Hz with batch size / queue depth / KV cache tracking
- `serve-report` -- latency percentile table (TTFB, TTFT, decode per token, e2e), KV cache hit rate and eviction counts, prefix sharing analysis, disaggregation recommendation with estimated throughput improvement
- `disagg-config` -- deployment configs for disaggregated prefill/decode targeting vLLM, llm-d, and Dynamo
- Interactive arrow-key menus using DEC cursor save/restore (ESC 7/8) for reliable in-place redraws across all terminal types
- KV bytes auto-detection from HuggingFace `config.json` with full GQA support (`num_key_value_heads`, `head_dim`, dtype)
- Clock detection fallback for GPUs where current SM clock equals hardware max (A10G and similar cloud instances)
- Numerical fixes: cache hit rate clamped to [0, 1], eviction and preemption counts floored at 0, disaggregation throughput percentage uses `(improvement - 1) * 100` with rounding
- JSON injection protection in traffic replayer, temp file open/write failure detection
- `.gitignore` updated to exclude `.hotpath/` run artifacts, `*.log`, and `targets.cfg`
- README rewritten for production use

## v0.1.2 - 2026-04-01

Packaging and release automation update.

Highlights:

- PyPI source distribution cleanup so local virtualenvs, build outputs, and macOS metadata files do not leak into published artifacts
- Linux wheel build configuration via `cibuildwheel` for CPython 3.10, 3.11, and 3.12 on `manylinux_2_28`
- GitHub Actions release workflow for building `sdist` plus Linux wheels and publishing to PyPI via trusted publishing on GitHub Releases

## v0.1.0 - 2026-03-31

Initial public release.

Highlights:

- local and remote `profile`, `report`, `diff`, `export`, `validate`, `artifacts`, `trace`, and `doctor`
- interactive prompt flows alongside the existing flag-based CLI
- real GPU `bench` backed by `torch.cuda.Event`, archived JSON output, and `bench-compare`
- SSH target registry, bootstrap, recover, and remote single-host profiling
- profile stability mode, clock helpers, manifest generation, cleanup, soak runs, and cluster rollup
- non-GPU CI and self-hosted GPU smoke workflow
- controller-verified remote workflow using saved SSH targets, `target bootstrap`, `bench --target`, `profile --target`, and `recover`
- `profile --attach ... --attach-pid ...` support that now uses the best available host-local tracing path: native PID attach when `nsys` exposes it, clone-under-trace when a second local copy fits, and replace-and-restore when a single GPU cannot hold both copies
- attach-by-process fast reuse for `profile --repeat N` and `soak-profile`, so repeated attach runs reuse one traced replacement lifecycle instead of relaunching the fallback per iteration
- stricter `cluster-profile` host handling with distinct-host enforcement by default and explicit `--allow-duplicate-hosts` loopback testing
- fast reuse path for `profile --repeat N` and `soak-profile` when `hotpath` launches the server, with one server startup and one `nsys` session reused across iterations
- local managed warm-server workflow: `server start/list/show/stop` and `profile --server NAME`, reusing one loaded `vllm serve` and one `nsys` session across separate commands
- managed-server hardening: explicit `--max-model-len`, stale-state pruning, stale-lock recovery, and listener PID fallback from `lsof` to `ss`
- publish-ready package metadata for the `pip install hotpath` release surface, plus a rewritten README aligned to the controller-verified local/remote workflows
