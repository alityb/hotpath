# Changelog

## v0.3.1 - 2026-04-06

Metrics-only serve-profile messaging fix and release cleanup.

Highlights:

- metrics-only `serve-profile` runs now say explicitly that no traffic file was provided and that per-request queue, prefill, and decode timing require requests during the capture window
- when a server log is found but a run observed zero requests, hotpath now reports that no requests were observed during the profiling run instead of blaming the vLLM log format
- `logs.md` was removed from the tracked release contents and is now local-only again

## v0.3.0 - 2026-04-06

Serving timing, log discovery, and report accuracy hardening.

Highlights:

- `serve-profile` now honors configured concurrency, stops dispatching new requests when the requested duration window closes, and measures client TTFT from the first streamed token chunk instead of HTTP first-byte timing
- local vLLM DEBUG log autodiscovery now resolves the actual live listener stdout and stderr file paths from `/proc/<pid>/fd/*`, which prevents stale `.hotpath/video-server/*` files from being mistaken for the active server log
- `serve-report` now treats missing latency as unavailable instead of rendering fake `0.0 ms`, uses `TTFT (client)` consistently, and explicitly discloses when vLLM v1 server timings are refined from Prometheus because the raw DEBUG timestamps are only second-resolution
- vLLM v1 parsing now merges exact request IDs from `Added request ...` lines with anonymous `Running batch ... BatchDescriptor(...)` execution lines, which allows exact-ID-matched requests to retain server queue, prefill, and decode timing
- live dashboard redraw was hardened for Ghostty-style terminals so `serve-profile` updates in place instead of stacking duplicate frames

## v0.2.9 - 2026-04-05

serve-profile timing correlation and startup hardening.

Highlights:

- `serve-profile` now injects stable hotpath request IDs into OpenAI-compatible requests and canonicalizes vLLM v1 internal randomized IDs back to those external IDs, which removes the main source of order-only request matching
- vLLM log parsing now scopes itself to the current profiling run instead of parsing the entire historical log file, which prevents stale requests from earlier runs from contaminating timing correlation
- external endpoint validation now waits through a short startup grace window before failing, which makes localhost demo runs less brittle during server warmup
- the Qwen video server script now starts vLLM with `--enable-log-requests` and refuses to start if port `8000` is already owned by another process, avoiding false “server ready” states caused by stale listeners

## v0.2.8 - 2026-04-05

serve-profile terminal redraw fix.

Highlights:

- replaced the live serve-profile dashboard DEC cursor save and restore path with explicit cursor-up redraws
- fixes duplicated dashboard frames in Ghostty and similar terminals during live profiling
- keeps the existing live dashboard behavior while using a more predictable ANSI redraw path

## v0.2.7 - 2026-04-05

vLLM v1 timing correctness and reporting fixes.

Highlights:

- `serve-profile` now parses and uses vLLM 0.19 request histograms for queue, prefill, and decode timing, which removes the worst second-resolution artifacts from `ORDER`-matched v1 logs
- reused `--output` directories now start from a fresh `serve_profile.db`, so stale request traces from older runs do not leak into new reports
- `serve-report` now shows sub-millisecond latency values when needed, so tiny queue waits like `0.013 ms` do not round down to `0.0`
- fixed `server_timing_match_method` persistence and updated the report note to explain when queue, prefill, and decode are refined from Prometheus metrics
- expanded regression coverage for vLLM 0.19 histogram parsing and serve-report formatting

## v0.2.6 - 2026-04-05

serve-profile UX fixes for the local video flow.

Highlights:

- `serve-profile` now auto-discovers a local vLLM debug log under `.hotpath/video-server/` when users do not pass `--server-log`
- interactive `serve-profile` now prompts for `Concurrency`
- interactive runs now forward the auto-discovered vLLM log path so queue, prefill, and decode timing work in the common localhost demo setup
- added a regression test covering local server log autodiscovery

## v0.2.5 - 2026-04-05

Video demo and local startup fixes.

Highlights:

- added a known-good Qwen3.5 video demo traffic file and runbook under `examples/`
- added `examples/start_qwen35_video_server.sh` and `examples/stop_qwen35_video_server.sh`
- demo startup now uses the working local flags for this environment: `--enforce-eager` and `--language-model-only`
- the example flow now waits for `/health` before telling users the server is ready

## v0.2.4 - 2026-04-05

README cleanup and install update.

Highlights:

- README rewritten to be shorter and more direct
- install instructions now use `uv tool install hotpath`
- source install example now uses `uv tool install .`

## v0.2.3 - 2026-04-05

Clock-lock privilege escalation fix.

Highlights:

- `hotpath lock-clocks` and `hotpath unlock-clocks` now retry through `sudo` when direct `nvidia-smi` access fails for a non-root interactive user
- manual fallback messages now suggest the correct `sudo nvidia-smi ...` command when privilege escalation is required
- keeps the existing direct path for environments where clock locking already works without `sudo`

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
