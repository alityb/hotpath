#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "hotpath/profiler/kernel_record.h"

namespace hotpath {

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
  std::int64_t completion_length_samples = 0;
};

std::string render_report(
    const ReportMeta& meta,
    const std::map<std::string, std::string>& metadata,
    const std::vector<profiler::KernelRecord>& kernels,
    const std::vector<MetricSummary>& metrics_summary,
    const TrafficStats& traffic_stats,
    bool color = false);

struct ServeReportData {
  std::string model_name;
  std::string engine;
  std::string gpu_info;
  int total_requests = 0;
  double duration_seconds = 0.0;
  double throughput_rps = 0.0;
  bool queue_wait_available = false;
  bool server_timing_available = false;
  std::string server_timing_match_method;
  double server_timing_max_offset_ms = 0.0;
  bool server_timing_remote_correlation = false;
  // Latency percentiles (ms)
  double server_ttft_mean_ms = -1.0;  // from Prometheus histogram; -1 = not available
  double queue_p50 = 0, queue_p90 = 0, queue_p99 = 0;
  double server_prefill_p50 = 0, server_prefill_p90 = 0, server_prefill_p99 = 0;
  double server_decode_p50 = 0, server_decode_p90 = 0, server_decode_p99 = 0;
  double prefill_p50 = 0, prefill_p90 = 0, prefill_p99 = 0;
  double decode_total_p50 = 0, decode_total_p90 = 0, decode_total_p99 = 0;
  double decode_per_token_p50 = 0, decode_per_token_p90 = 0, decode_per_token_p99 = 0;
  double e2e_p50 = 0, e2e_p90 = 0, e2e_p99 = 0;
  // GPU phase breakdown
  bool gpu_phase_available = false;
  double prefill_compute_pct = 0, decode_compute_pct = 0, other_idle_pct = 0;
  // KV Cache
  bool cache_hit_rate_available = false;
  bool cache_usage_available = false;
  bool cache_hit_rate_aggregate_only = false;
  bool cache_histogram_available = false;
  double cache_hit_rate = 0;
  double avg_cache_usage = 0;
  double peak_cache_usage = 0;
  int evictions = 0;
  std::array<int, 5> cache_hit_rate_histogram = {};
  // Prefix sharing
  bool prefix_sharing_available = false;
  int unique_prefixes = 0;
  double cacheable_tokens_pct = 0;
  // Disaggregation
  bool should_disaggregate = false;
  int optimal_p = 0, optimal_d = 0;
  double projected_throughput_pct = 0;
  double projected_throughput_rps = 0;
  double mono_p99_ttft = 0, disagg_p99_ttft = 0;
  double min_bandwidth_gbps = 0;
  std::string advisor_caveat;
};

std::string render_serve_report(const ServeReportData& data);

}  // namespace hotpath
