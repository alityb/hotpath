#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "rlprof/profiler/kernel_record.h"
#include "rlprof/store.h"

namespace rlprof::profiler {

struct ProfileConfig {
  std::string model;
  std::int64_t prompts = 128;
  std::int64_t rollouts = 8;
  std::int64_t max_tokens = 4096;
  std::int64_t min_tokens = 256;
  std::int64_t input_len = 512;
  std::int64_t port = 8000;
  std::int64_t tp = 1;
  bool trust_remote_code = false;
  std::filesystem::path output;
  std::int64_t startup_timeout_s = 300;
  std::int64_t metrics_interval_ms = 1000;
};

struct ProfileRunResult {
  std::filesystem::path db_path;
  std::filesystem::path nsys_sqlite_path;
  std::map<std::string, std::string> meta;
  std::vector<KernelRecord> kernels;
  std::vector<MetricSample> metrics;
  TrafficStats traffic_stats;
};

using ProgressCallback = std::function<void(const std::string&)>;

ProfileRunResult run_profile(
    const ProfileConfig& config,
    ProgressCallback progress = {});

}  // namespace rlprof::profiler
