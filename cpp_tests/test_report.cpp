#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "rlprof/profiler/kernel_record.h"
#include "rlprof/report.h"

namespace {

void expect_contains(const std::string& haystack, const std::string& needle) {
  if (haystack.find(needle) == std::string::npos) {
    std::cerr << "expected report to contain [" << needle << "]\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  const std::string report = rlprof::render_report(
      rlprof::ReportMeta{
          .model_name = "Qwen/Qwen3-8B",
          .gpu_name = "NVIDIA A10G",
          .vllm_version = "v0.1.0",
          .prompts = 128,
          .rollouts = 8,
          .max_tokens = 4096,
      },
      std::map<std::string, std::string>{
          {"measurement_driver_version", "570.12"},
          {"measurement_gpu_clock_policy", "unlocked"},
          {"measurement_gpu_max_sm_clock_mhz", "1710"},
          {"measurement_persistence_mode", "Enabled"},
          {"measurement_pstates", "P0"},
          {"measurement_samples", "5"},
          {"measurement_sm_clock_min_mhz", "1200"},
          {"measurement_sm_clock_avg_mhz", "1350"},
          {"measurement_sm_clock_max_mhz", "1500"},
          {"measurement_temp_min_c", "61"},
          {"measurement_temp_max_c", "83"},
          {"measurement_power_draw_avg_w", "180"},
          {"measurement_power_draw_peak_w", "220"},
          {"measurement_power_limit_w", "300"},
          {"warning_gpu_clocks_unlocked", "true"},
          {"warning_temp_high", "true"},
      },
      std::vector<rlprof::profiler::KernelRecord>{
          {
              .name = "sm80_xmma_gemm_bf16",
              .category = "gemm",
              .total_ns = 200'000'000,
              .calls = 400,
              .avg_ns = 500'000,
              .min_ns = 400'000,
              .max_ns = 600'000,
              .registers = 64,
              .shared_mem = 128,
          },
          {
              .name = "flash_fwd_splitkv_bf16_sm80",
              .category = "attention",
              .total_ns = 50'000'000,
              .calls = 200,
              .avg_ns = 250'000,
              .min_ns = 200'000,
              .max_ns = 300'000,
              .registers = 48,
              .shared_mem = 64,
          },
          {
              .name = "Kernel2",
              .category = "other",
              .total_ns = 25'000'000,
              .calls = 10,
              .avg_ns = 2'500'000,
              .min_ns = 2'000'000,
              .max_ns = 3'000'000,
              .registers = 32,
              .shared_mem = 0,
          },
      },
      std::vector<rlprof::MetricSummary>{
          {
              .metric = "vllm:num_preemptions_total",
              .avg = std::nullopt,
              .peak = 47.0,
              .min = std::nullopt,
          },
          {
              .metric = "vllm:gpu_cache_usage_perc",
              .avg = 0.72,
              .peak = 0.97,
              .min = std::nullopt,
          },
          {
              .metric = "vllm:time_to_first_token_seconds_p99",
              .avg = 0.8921,
              .peak = std::nullopt,
              .min = std::nullopt,
          },
      },
      rlprof::TrafficStats{
          .total_requests = 1024,
          .completion_length_mean = 1847.0,
          .completion_length_p50 = 1204.0,
          .completion_length_p99 = 3891.0,
          .max_median_ratio = 3.23,
          .errors = 0,
      });

  expect_contains(report, "rlprof | Qwen/Qwen3-8B | NVIDIA A10G | v0.1.0");
  expect_contains(report, "conservative substring matching");
  expect_contains(report, "MEASUREMENT WARNINGS");
  expect_contains(report, "GPU clocks are not locked. Run `rlprof lock-clocks`");
  expect_contains(report, "gpu temperature reached high operating range");
  expect_contains(report, "MEASUREMENT CONTEXT");
  expect_contains(report, "gpu clock policy");
  expect_contains(report, "unlocked");
  expect_contains(report, "max supported sm clock mhz");
  expect_contains(report, "1350");
  expect_contains(report, "KERNEL BREAKDOWN BY CATEGORY");
  expect_contains(report, "gemm");
  expect_contains(report, "attention");
  expect_contains(report, "TOP 10 KERNELS BY TOTAL TIME");
  expect_contains(report, "sm80_xmma_gemm_bf16");
  expect_contains(report, "TOP UNCATEGORIZED KERNELS");
  expect_contains(report, "Kernel2");
  expect_contains(report, "VLLM SERVER METRICS");
  expect_contains(report, "preemptions");
  expect_contains(report, "892.1");
  expect_contains(report, "TRAFFIC SHAPE");
  expect_contains(report, "1,024");
  expect_contains(report, "3.23x");

  return 0;
}
