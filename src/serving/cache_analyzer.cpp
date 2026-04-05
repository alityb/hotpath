#include "hotpath/cache_analyzer.h"

#include <algorithm>
#include <numeric>

namespace hotpath {

CacheAnalysis analyze_cache(const std::vector<RequestTrace>& traces,
                            const std::vector<MetricSnapshot>& snapshots,
                            std::optional<double> aggregate_cache_hit_rate) {
  CacheAnalysis result;

  // Cache hit rate from request traces
  int64_t total_cached = 0;
  int64_t total_prompt = 0;
  const bool have_per_request_cache_data = !traces.empty();
  for (const auto& t : traces) {
    total_cached += t.cached_tokens;
    total_prompt += t.prompt_tokens;

    // Histogram bucket assignment
    double hit_frac = 0.0;
    if (t.prompt_tokens > 0) {
      hit_frac = static_cast<double>(t.cached_tokens) / t.prompt_tokens;
    }
    // Half-open buckets: [0,0], (0,0.25), [0.25,0.50), [0.50,0.75), [0.75,1.0]
    // Using < for upper bounds so exactly 25%, 50%, 75% go into the higher bucket,
    // matching the display labels "1-25%", "25-50%", "50-75%", "75-100%".
    if (hit_frac <= 0.0) {
      result.hit_rate_histogram[0]++;
    } else if (hit_frac < 0.25) {
      result.hit_rate_histogram[1]++;
    } else if (hit_frac < 0.50) {
      result.hit_rate_histogram[2]++;
    } else if (hit_frac < 0.75) {
      result.hit_rate_histogram[3]++;
    } else {
      result.hit_rate_histogram[4]++;
    }
  }
  if (have_per_request_cache_data) {
    if (total_prompt > 0) {
      result.cache_hit_rate = std::clamp(
          static_cast<double>(total_cached) / total_prompt, 0.0, 1.0);
    } else {
      result.cache_hit_rate = 0.0;
    }
    result.cache_hit_rate_available = true;
    result.cache_hit_rate_aggregate_only = false;
    result.hit_rate_histogram_available = true;
  } else if (aggregate_cache_hit_rate.has_value()) {
    result.cache_hit_rate = *aggregate_cache_hit_rate;
    result.cache_hit_rate_available = true;
    result.cache_hit_rate_aggregate_only = true;
    result.hit_rate_histogram = {};
    result.hit_rate_histogram_available = false;
  } else {
    result.hit_rate_histogram = {};
    result.hit_rate_histogram_available = false;
  }

  // Cache usage from metric snapshots
  if (!snapshots.empty()) {
    double sum_usage = 0.0;
    double max_usage = 0.0;
    double max_preemption = 0.0;
    double min_preemption = snapshots[0].preemption_total;
    double pressure_samples = 0.0;

    for (const auto& s : snapshots) {
      sum_usage += s.cache_usage;
      max_usage = std::max(max_usage, s.cache_usage);
      max_preemption = std::max(max_preemption, s.preemption_total);
      min_preemption = std::min(min_preemption, s.preemption_total);
      if (s.cache_usage > 90.0) {
        pressure_samples += 1.0;
      }
    }

    result.avg_cache_usage = sum_usage / static_cast<double>(snapshots.size());
    result.peak_cache_usage = max_usage;
    // Each snapshot is ~1 second apart
    result.cache_pressure_seconds = pressure_samples;
    result.eviction_count = static_cast<int>(std::max(0.0, max_preemption - min_preemption));
  }

  return result;
}

}  // namespace hotpath
