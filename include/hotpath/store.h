#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "hotpath/profiler/kernel_record.h"
#include "hotpath/report.h"
#include "hotpath/request_trace.h"

namespace hotpath {

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

// Request trace storage
int64_t insert_request_trace(const std::filesystem::path& db_path,
                             int64_t profile_id,
                             const RequestTrace& trace);

std::vector<RequestTrace> load_request_traces(const std::filesystem::path& db_path,
                                              int64_t profile_id);

std::vector<RequestTrace> query_traces_prefill_gt(const std::filesystem::path& db_path,
                                                  int64_t profile_id,
                                                  int64_t min_prefill_us);

std::vector<RequestTrace> query_traces_cached_gt(const std::filesystem::path& db_path,
                                                 int64_t profile_id,
                                                 int min_cached_tokens);

// Serve analysis key-value storage (persists analysis results for serve-report / disagg-config)
void save_serve_analysis(const std::filesystem::path& db_path,
                         const std::map<std::string, std::string>& kv);

std::map<std::string, std::string> load_serve_analysis(const std::filesystem::path& db_path);

}  // namespace hotpath
