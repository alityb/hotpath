# Changelog

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
- explicit documented boundary for `profile --attach ... --attach-pid ...` when installed `nsys` does not expose PID attach support
