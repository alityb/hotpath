#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>

#include "rlprof/store.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path temp_dir = fs::temp_directory_path() / "rlprof_cpp_tests";
  fs::create_directories(temp_dir);
  const fs::path db_path = temp_dir / "profile.db";
  fs::remove(db_path);

  const rlprof::ProfileData profile = {
      .meta = {
          {"gpu_name", "NVIDIA A10G"},
          {"model_name", "Qwen/Qwen3-8B"},
          {"prompts", "128"},
          {"vllm_version", "0.8.2"},
      },
      .kernels = {
          {
              .name = "sm80_xmma_gemm_bf16",
              .category = "gemm",
              .total_ns = 200,
              .calls = 2,
              .avg_ns = 100,
              .min_ns = 80,
              .max_ns = 120,
              .registers = 64,
              .shared_mem = 128,
          },
      },
      .metrics = {
          {
              .sample_time = 0.0,
              .source = "cluster",
              .metric = "vllm:num_preemptions_total",
              .value = 1.0,
          },
      },
      .metrics_summary = {
          {
              .metric = "vllm:num_preemptions_total",
              .avg = std::nullopt,
              .peak = 4.0,
              .min = std::nullopt,
          },
      },
      .traffic_stats = {
          .total_requests = 1024,
          .completion_length_mean = std::nullopt,
          .completion_length_p50 = std::nullopt,
          .completion_length_p99 = std::nullopt,
          .max_median_ratio = std::nullopt,
          .errors = 0,
      },
  };

  rlprof::save_profile(db_path, profile);
  const rlprof::ProfileData loaded = rlprof::load_profile(db_path);

  expect_true(loaded.meta == profile.meta, "meta round-trip mismatch");
  expect_true(loaded.kernels.size() == 1, "expected one kernel row");
  expect_true(loaded.kernels[0].name == "sm80_xmma_gemm_bf16", "unexpected kernel name");
  expect_true(loaded.metrics.size() == 1, "expected one metric sample");
  expect_true(loaded.metrics[0].source == "cluster", "unexpected metric source");
  expect_true(loaded.metrics[0].metric == "vllm:num_preemptions_total", "unexpected metric name");
  expect_true(loaded.metrics_summary.size() == 1, "expected one metric summary");
  expect_true(
      loaded.metrics_summary[0].peak.has_value() && *loaded.metrics_summary[0].peak == 4.0,
      "unexpected metric summary peak");
  expect_true(loaded.traffic_stats.total_requests == 1024, "unexpected traffic total");
  expect_true(loaded.traffic_stats.errors == 0, "unexpected traffic errors");

  fs::remove(db_path);
  return 0;
}
