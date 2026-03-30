#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rlprof/profiler/kernel_record.h"

namespace rlprof {

struct ReportMeta {
  std::string model_name;
  std::string gpu_name;
  std::string vllm_version;
  std::int64_t prompts;
  std::int64_t rollouts;
  std::int64_t max_tokens;
};

struct MetricSummary {
  std::string metric;
  std::optional<double> avg;
  std::optional<double> peak;
  std::optional<double> min;
};

struct TrafficStats {
  std::int64_t total_requests;
  std::optional<double> completion_length_mean;
  std::optional<double> completion_length_p50;
  std::optional<double> completion_length_p99;
  std::optional<double> max_median_ratio;
  std::int64_t errors;
};

std::string render_report(
    const ReportMeta& meta,
    const std::map<std::string, std::string>& metadata,
    const std::vector<profiler::KernelRecord>& kernels,
    const std::vector<MetricSummary>& metrics_summary,
    const TrafficStats& traffic_stats);

}  // namespace rlprof
