#include "rlprof/aggregate.h"

#include <algorithm>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rlprof/profiler/vllm_metrics.h"

namespace rlprof {

ProfileData aggregate_profiles(const std::vector<std::filesystem::path>& paths) {
  if (paths.empty()) {
    throw std::runtime_error("aggregate requires at least one profile path");
  }

  ProfileData aggregate;
  aggregate.meta["aggregate_profile_count"] = std::to_string(paths.size());
  aggregate.meta["aggregation_scope"] = "multi_profile";
  aggregate.meta["aggregate_sources"] = std::to_string(paths.size());

  std::map<std::string, profiler::KernelRecord> kernels_by_name;
  std::int64_t total_requests = 0;
  std::int64_t total_errors = 0;
  double weighted_completion_mean = 0.0;
  std::int64_t weighted_mean_count = 0;
  std::optional<double> completion_p50_upper_bound;
  std::optional<double> completion_p99_upper_bound;
  std::optional<double> max_median_ratio_observed_max;

  for (const auto& path : paths) {
    const auto profile = load_profile(path);
    if (aggregate.meta.contains("model_name") == false && profile.meta.contains("model_name")) {
      aggregate.meta["model_name"] = profile.meta.at("model_name");
    }
    if (aggregate.meta.contains("gpu_name") == false && profile.meta.contains("gpu_name")) {
      aggregate.meta["gpu_name"] = profile.meta.at("gpu_name");
    }
    if (aggregate.meta.contains("vllm_version") == false &&
        profile.meta.contains("vllm_version")) {
      aggregate.meta["vllm_version"] = profile.meta.at("vllm_version");
    }
    for (auto sample : profile.metrics) {
      if (sample.source.empty()) {
        sample.source = path.filename().string();
      }
      aggregate.metrics.push_back(sample);
    }
    for (const auto& kernel : profile.kernels) {
      auto& merged = kernels_by_name[kernel.name];
      if (merged.name.empty()) {
        merged = kernel;
        continue;
      }
      merged.category = kernel.category;
      merged.total_ns += kernel.total_ns;
      merged.calls += kernel.calls;
      merged.avg_ns = merged.calls == 0 ? 0 : merged.total_ns / merged.calls;
      merged.min_ns = std::min(merged.min_ns, kernel.min_ns);
      merged.max_ns = std::max(merged.max_ns, kernel.max_ns);
      merged.registers = std::max(merged.registers, kernel.registers);
      merged.shared_mem = std::max(merged.shared_mem, kernel.shared_mem);
    }

    total_requests += profile.traffic_stats.total_requests;
    total_errors += profile.traffic_stats.errors;
    if (profile.traffic_stats.completion_length_mean.has_value() &&
        profile.traffic_stats.completion_length_samples > 0) {
      weighted_completion_mean +=
          *profile.traffic_stats.completion_length_mean *
          static_cast<double>(profile.traffic_stats.completion_length_samples);
      weighted_mean_count += profile.traffic_stats.completion_length_samples;
    }
    if (profile.traffic_stats.completion_length_p50.has_value()) {
      completion_p50_upper_bound =
          std::max(completion_p50_upper_bound.value_or(0.0),
                   *profile.traffic_stats.completion_length_p50);
    }
    if (profile.traffic_stats.completion_length_p99.has_value()) {
      completion_p99_upper_bound =
          std::max(completion_p99_upper_bound.value_or(0.0),
                   *profile.traffic_stats.completion_length_p99);
    }
    if (profile.traffic_stats.max_median_ratio.has_value()) {
      max_median_ratio_observed_max =
          std::max(max_median_ratio_observed_max.value_or(0.0),
                   *profile.traffic_stats.max_median_ratio);
    }
  }

  for (const auto& [_, kernel] : kernels_by_name) {
    aggregate.kernels.push_back(kernel);
  }
  std::sort(
      aggregate.kernels.begin(),
      aggregate.kernels.end(),
      [](const auto& left, const auto& right) {
        return left.total_ns > right.total_ns;
      });

  aggregate.metrics_summary = profiler::summarize_samples(aggregate.metrics);
  aggregate.traffic_stats.total_requests = total_requests;
  aggregate.traffic_stats.errors = total_errors;
  aggregate.traffic_stats.completion_length_samples = weighted_mean_count;
  if (weighted_mean_count > 0) {
    aggregate.traffic_stats.completion_length_mean =
        weighted_completion_mean / static_cast<double>(weighted_mean_count);
  }
  aggregate.traffic_stats.completion_length_p50 = std::nullopt;
  aggregate.traffic_stats.completion_length_p99 = std::nullopt;
  aggregate.traffic_stats.max_median_ratio = std::nullopt;
  if (completion_p50_upper_bound.has_value()) {
    aggregate.meta["aggregate_completion_length_p50_upper_bound"] =
        std::to_string(*completion_p50_upper_bound);
  }
  if (completion_p99_upper_bound.has_value()) {
    aggregate.meta["aggregate_completion_length_p99_upper_bound"] =
        std::to_string(*completion_p99_upper_bound);
  }
  if (max_median_ratio_observed_max.has_value()) {
    aggregate.meta["aggregate_max_median_ratio_observed_max"] =
        std::to_string(*max_median_ratio_observed_max);
  }
  if (completion_p50_upper_bound.has_value() || completion_p99_upper_bound.has_value() ||
      max_median_ratio_observed_max.has_value()) {
    aggregate.meta["warning_aggregate_traffic_percentiles"] = "true";
  }
  return aggregate;
}

}  // namespace rlprof
