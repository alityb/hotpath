#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rlprof/bench/registry.h"

namespace rlprof::bench {

struct BenchResult {
  std::string kernel;
  std::string implementation;
  Shape shape;
  std::string dtype;
  double avg_ms;
  double stddev_ms = 0.0;
  double repeat_cv_pct = 0.0;
  double min_ms;
  double p50_ms;
  double p99_ms;
  double bandwidth_gb_s;
  bool validation_passed = true;
  double validation_max_abs_error = 0.0;
  bool deterministic_passed = true;
  double determinism_max_abs_error = 0.0;
  bool has_timing_warning = false;
  bool has_environment_warning = false;
  bool unstable = false;
};

struct BenchGpuInfo {
  std::string name;
  std::string driver_version;
  double sm_clock_mhz = 0.0;
  double mem_clock_mhz = 0.0;
  double temp_c = 0.0;
  double power_draw_w = 0.0;
  double power_limit_w = 0.0;
};

struct BenchRunOutput {
  std::optional<BenchGpuInfo> gpu;
  std::vector<BenchResult> results;
  std::vector<std::string> correctness_failures;
  std::vector<std::string> timing_warnings;
  std::vector<std::string> environment_warnings;
};

class BenchmarkBackend {
 public:
  virtual ~BenchmarkBackend() = default;
  virtual double measure_ms(const std::function<void()>& fn) = 0;
  virtual void synchronize() {}
};

std::vector<Shape> parse_shapes(const std::string& spec);

std::vector<BenchResult> benchmark_impl(
    const std::string& category,
    const KernelImpl& implementation,
    const std::vector<Shape>& shapes,
    const std::string& dtype,
    std::int64_t warmup,
    std::int64_t n_iter,
    BenchmarkBackend* backend = nullptr);

std::vector<BenchResult> benchmark_category(
    const std::string& category,
    const std::vector<Shape>& shapes,
    const std::string& dtype,
    std::int64_t warmup,
    std::int64_t n_iter,
    BenchmarkBackend* backend = nullptr);

BenchRunOutput parse_bench_json(const std::string& json_text);
std::string serialize_bench_output_json(const BenchRunOutput& output);
std::string render_bench_comparison(
    const BenchRunOutput& left,
    const BenchRunOutput& right);
std::filesystem::path resolve_bench_output_path(
    const std::string& kernel,
    const std::string& output_spec);
std::string render_bench_results(const std::vector<BenchResult>& results);
std::string render_bench_output(const BenchRunOutput& output);

}  // namespace rlprof::bench
