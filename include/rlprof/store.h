#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "rlprof/profiler/kernel_record.h"
#include "rlprof/report.h"

namespace rlprof {

struct MetricSample {
  double sample_time;
  std::string source;
  std::string metric;
  double value;
};

struct ProfileData {
  std::map<std::string, std::string> meta;
  std::vector<profiler::KernelRecord> kernels;
  std::vector<MetricSample> metrics;
  std::vector<MetricSummary> metrics_summary;
  TrafficStats traffic_stats;
};

std::filesystem::path init_db(const std::filesystem::path& path);

std::filesystem::path save_profile(
    const std::filesystem::path& path,
    const ProfileData& profile);

ProfileData load_profile(const std::filesystem::path& path);

}  // namespace rlprof
