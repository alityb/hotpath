#include "hotpath/batch_analyzer.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace hotpath {
namespace {

double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double idx = p / 100.0 * static_cast<double>(values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = std::min(lo + 1, values.size() - 1);
  const double frac = idx - static_cast<double>(lo);
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

}  // namespace

BatchAnalysis analyze_batches(const std::vector<MetricSnapshot>& snapshots) {
  BatchAnalysis result;
  if (snapshots.empty()) return result;

  std::vector<double> batch_sizes;
  std::vector<double> queue_depths;
  std::vector<double> cache_usages;
  batch_sizes.reserve(snapshots.size());
  queue_depths.reserve(snapshots.size());
  cache_usages.reserve(snapshots.size());

  double max_preemption = 0.0;
  double min_preemption = snapshots[0].preemption_total;

  for (const auto& s : snapshots) {
    batch_sizes.push_back(s.batch_size);
    queue_depths.push_back(s.queue_depth);
    cache_usages.push_back(s.cache_usage);
    max_preemption = std::max(max_preemption, s.preemption_total);
    min_preemption = std::min(min_preemption, s.preemption_total);

    result.batch_size_series.emplace_back(s.timestamp_us, s.batch_size);
    result.queue_depth_series.emplace_back(s.timestamp_us, s.queue_depth);
  }

  const double batch_sum = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 0.0);
  result.avg_batch_size = batch_sum / static_cast<double>(batch_sizes.size());
  result.p50_batch_size = percentile(batch_sizes, 50.0);
  result.p99_batch_size = percentile(batch_sizes, 99.0);

  const double queue_sum = std::accumulate(queue_depths.begin(), queue_depths.end(), 0.0);
  result.avg_queue_depth = queue_sum / static_cast<double>(queue_depths.size());
  result.p99_queue_depth = percentile(queue_depths, 99.0);

  result.total_preemptions = static_cast<int>(std::max(0.0, max_preemption - min_preemption));

  const double cache_sum = std::accumulate(cache_usages.begin(), cache_usages.end(), 0.0);
  result.avg_cache_usage = cache_sum / static_cast<double>(cache_usages.size());
  result.peak_cache_usage = *std::max_element(cache_usages.begin(), cache_usages.end());

  return result;
}

}  // namespace hotpath
