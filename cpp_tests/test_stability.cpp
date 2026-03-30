#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "rlprof/stability.h"
#include "rlprof/store.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

const rlprof::StabilityRow& find_row(
    const std::vector<rlprof::StabilityRow>& rows,
    const std::string& label) {
  for (const auto& row : rows) {
    if (row.label == label) {
      return row;
    }
  }
  throw std::runtime_error("missing stability row: " + label);
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path temp_dir = fs::temp_directory_path() / "rlprof_test_stability";
  fs::create_directories(temp_dir);

  const std::vector<rlprof::ProfileData> profiles_to_save = {
      {
          .meta = {{"model_name", "mock-model"}},
          .kernels = {
              {.name = "gemm_a", .category = "gemm", .total_ns = 100'000'000, .calls = 10, .avg_ns = 10'000'000, .min_ns = 8'000'000, .max_ns = 12'000'000, .registers = 32, .shared_mem = 0},
              {.name = "attention_a", .category = "attention", .total_ns = 50'000'000, .calls = 5, .avg_ns = 10'000'000, .min_ns = 9'000'000, .max_ns = 11'000'000, .registers = 32, .shared_mem = 0},
          },
          .metrics = {},
          .metrics_summary = {
              {.metric = "vllm:num_preemptions_total", .avg = 10.0, .peak = 15.0, .min = 0.0},
              {.metric = "vllm:num_requests_waiting", .avg = 8.0, .peak = 20.0, .min = 0.0},
              {.metric = "vllm:gpu_cache_usage_perc", .avg = 0.75, .peak = 0.90, .min = 0.0},
          },
          .traffic_stats = {.total_requests = 1, .completion_length_mean = 1.0, .completion_length_p50 = 1.0, .completion_length_p99 = 1.0, .max_median_ratio = 1.0, .errors = 0},
      },
      {
          .meta = {{"model_name", "mock-model"}},
          .kernels = {
              {.name = "gemm_a", .category = "gemm", .total_ns = 102'000'000, .calls = 10, .avg_ns = 10'200'000, .min_ns = 8'200'000, .max_ns = 12'200'000, .registers = 32, .shared_mem = 0},
              {.name = "attention_a", .category = "attention", .total_ns = 60'000'000, .calls = 5, .avg_ns = 12'000'000, .min_ns = 10'000'000, .max_ns = 14'000'000, .registers = 32, .shared_mem = 0},
          },
          .metrics = {},
          .metrics_summary = {
              {.metric = "vllm:num_preemptions_total", .avg = 11.0, .peak = 16.0, .min = 0.0},
              {.metric = "vllm:num_requests_waiting", .avg = 9.0, .peak = 25.0, .min = 0.0},
              {.metric = "vllm:gpu_cache_usage_perc", .avg = 0.77, .peak = 0.92, .min = 0.0},
          },
          .traffic_stats = {.total_requests = 1, .completion_length_mean = 1.0, .completion_length_p50 = 1.0, .completion_length_p99 = 1.0, .max_median_ratio = 1.0, .errors = 0},
      },
      {
          .meta = {{"model_name", "mock-model"}},
          .kernels = {
              {.name = "gemm_a", .category = "gemm", .total_ns = 98'000'000, .calls = 10, .avg_ns = 9'800'000, .min_ns = 7'800'000, .max_ns = 11'800'000, .registers = 32, .shared_mem = 0},
              {.name = "attention_a", .category = "attention", .total_ns = 55'000'000, .calls = 5, .avg_ns = 11'000'000, .min_ns = 9'500'000, .max_ns = 12'500'000, .registers = 32, .shared_mem = 0},
          },
          .metrics = {},
          .metrics_summary = {
              {.metric = "vllm:num_preemptions_total", .avg = 10.0, .peak = 14.0, .min = 0.0},
              {.metric = "vllm:num_requests_waiting", .avg = 8.0, .peak = 22.0, .min = 0.0},
              {.metric = "vllm:gpu_cache_usage_perc", .avg = 0.76, .peak = 0.91, .min = 0.0},
          },
          .traffic_stats = {.total_requests = 1, .completion_length_mean = 1.0, .completion_length_p50 = 1.0, .completion_length_p99 = 1.0, .max_median_ratio = 1.0, .errors = 0},
      },
  };

  std::vector<rlprof::ProfileData> profiles;
  for (std::size_t i = 0; i < profiles_to_save.size(); ++i) {
    const fs::path db_path = temp_dir / ("stability_r" + std::to_string(i + 1) + ".db");
    rlprof::save_profile(db_path, profiles_to_save[i]);
    profiles.push_back(rlprof::load_profile(db_path));
  }

  const rlprof::StabilityReport report = rlprof::compute_stability_report(profiles);
  expect_true(report.run_count == 3, "expected 3 runs in stability report");

  expect_true(std::abs(report.total_kernel_time.mean - 155.0) < 1e-9, "unexpected total mean");
  expect_true(std::abs(report.total_kernel_time.min - 150.0) < 1e-9, "unexpected total min");
  expect_true(std::abs(report.total_kernel_time.max - 162.0) < 1e-9, "unexpected total max");
  expect_true(report.total_kernel_time.max_min_ratio.has_value(), "expected total ratio");
  expect_true(std::abs(*report.total_kernel_time.max_min_ratio - 1.08) < 1e-9, "unexpected total ratio");
  expect_true(!report.total_kernel_time.pass, "total kernel time should warn");

  const auto& gemm = find_row(report.category_rows, "gemm");
  expect_true(std::abs(gemm.mean - 100.0) < 1e-9, "unexpected gemm mean");
  expect_true(std::abs(gemm.min - 98.0) < 1e-9, "unexpected gemm min");
  expect_true(std::abs(gemm.max - 102.0) < 1e-9, "unexpected gemm max");
  expect_true(gemm.max_min_ratio.has_value(), "expected gemm ratio");
  expect_true(std::abs(*gemm.max_min_ratio - (102.0 / 98.0)) < 1e-9, "unexpected gemm ratio");
  expect_true(gemm.pass, "gemm should pass");

  const auto& attention = find_row(report.category_rows, "attention");
  expect_true(std::abs(attention.mean - 55.0) < 1e-9, "unexpected attention mean");
  expect_true(attention.max_min_ratio.has_value(), "expected attention ratio");
  expect_true(std::abs(*attention.max_min_ratio - 1.2) < 1e-9, "unexpected attention ratio");
  expect_true(!attention.pass, "attention should warn");

  const auto& preemptions = find_row(report.metric_rows, "preemptions (avg)");
  expect_true(std::abs(preemptions.mean - (31.0 / 3.0)) < 1e-9, "unexpected preemptions mean");
  expect_true(preemptions.max_min_ratio.has_value(), "expected preemptions ratio");
  expect_true(std::abs(*preemptions.max_min_ratio - 1.1) < 1e-9, "unexpected preemptions ratio");
  expect_true(preemptions.pass, "preemptions should pass at threshold");

  const auto& waiting = find_row(report.metric_rows, "requests_waiting (peak)");
  expect_true(waiting.max_min_ratio.has_value(), "expected waiting ratio");
  expect_true(std::abs(*waiting.max_min_ratio - 1.25) < 1e-9, "unexpected waiting ratio");
  expect_true(!waiting.pass, "requests_waiting should warn");

  const auto& kv_cache = find_row(report.metric_rows, "kv_cache_usage (peak)");
  expect_true(kv_cache.max_min_ratio.has_value(), "expected kv cache ratio");
  expect_true(std::abs(*kv_cache.max_min_ratio - (0.92 / 0.90)) < 1e-9, "unexpected kv cache ratio");
  expect_true(kv_cache.pass, "kv cache should pass");

  const std::string rendered = rlprof::render_stability_report(report);
  expect_true(rendered.find("STABILITY REPORT (3 runs)") != std::string::npos, "missing stability heading");
  expect_true(rendered.find("total kernel time") != std::string::npos, "missing total row");
  expect_true(rendered.find("attention") != std::string::npos, "missing attention row");
  expect_true(rendered.find("WARN") != std::string::npos, "missing warn label");
  expect_true(rendered.find("PASS") != std::string::npos, "missing pass label");

  return 0;
}
