#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "rlprof/export.h"
#include "rlprof/store.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path);
  return std::string(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
}

}  // namespace

int main() {
  const auto temp_dir = std::filesystem::temp_directory_path() / "rlprof_test_export";
  std::filesystem::create_directories(temp_dir);
  const auto db_path = temp_dir / "profile.db";

  rlprof::ProfileData profile{
      .meta = {
          {"model_name", "Qwen/Qwen3-8B"},
          {"gpu_name", "NVIDIA A10G, 24GB"},
          {"warning_gpu_clocks_unlocked", "true"},
          {"warning_temp_high", "true"},
      },
      .kernels = {
          {
              .name = "vllm::act_and_mul_kernel<bf16,silu,true>",
              .category = "activation",
              .total_ns = 100,
              .calls = 2,
              .avg_ns = 50,
              .min_ns = 40,
              .max_ns = 60,
              .registers = 32,
              .shared_mem = 0,
          },
      },
      .metrics = {
          {.sample_time = 1.25, .metric = "vllm:num_preemptions_total", .value = 4.0},
      },
      .metrics_summary = {
          {.metric = "vllm:num_preemptions_total", .avg = 2.0, .peak = 4.0, .min = 0.0},
      },
      .traffic_stats = {
          .total_requests = 32,
          .completion_length_mean = 200.0,
          .completion_length_p50 = 192.0,
          .completion_length_p99 = 255.0,
          .max_median_ratio = 1.328125,
          .errors = 0,
      },
  };

  rlprof::save_profile(db_path, profile);

  const auto csv_outputs = rlprof::export_profile(db_path, "csv");
  expect_true(csv_outputs.size() == 6, "csv export should write all profile tables and warnings");
  const auto json_outputs = rlprof::export_profile(db_path, "json");
  expect_true(json_outputs.size() == 1, "json export should write one file");

  const std::string kernels_csv = read_text(temp_dir / "profile_kernels.csv");
  expect_true(
      kernels_csv.find("\"vllm::act_and_mul_kernel<bf16,silu,true>\",activation") !=
          std::string::npos,
      "kernel csv should quote names containing commas");

  const std::string metrics_csv = read_text(temp_dir / "profile_vllm_metrics.csv");
  expect_true(
      metrics_csv.find("vllm:num_preemptions_total") != std::string::npos,
      "metrics csv should contain time-series rows");

  const std::string traffic_csv = read_text(temp_dir / "profile_traffic_stats.csv");
  expect_true(
      traffic_csv.find("total_requests,32") != std::string::npos,
      "traffic csv should contain traffic stats");

  const std::string warnings_csv = read_text(temp_dir / "profile_warnings.csv");
  expect_true(
      warnings_csv.find("warning_gpu_clocks_unlocked") != std::string::npos,
      "warnings csv should contain unlocked clock warnings");
  expect_true(
      warnings_csv.find("warning_temp_high") != std::string::npos,
      "warnings csv should contain exported warnings");

  const std::string json_text = read_text(temp_dir / "profile.json");
  expect_true(
      json_text.find("\"warnings\"") != std::string::npos,
      "json export should contain warnings");
  expect_true(
      json_text.find("\"vllm_metrics_summary\"") != std::string::npos,
      "json export should contain metrics summary");
  expect_true(
      json_text.find("\"traffic_stats\"") != std::string::npos,
      "json export should contain traffic stats");

  return 0;
}
