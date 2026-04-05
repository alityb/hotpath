#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "hotpath/cache_analyzer.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  // Create 50 request traces with varying cached_tokens
  std::vector<hotpath::RequestTrace> traces;
  traces.reserve(50);
  for (int i = 0; i < 50; ++i) {
    hotpath::RequestTrace t;
    t.request_id = "req_" + std::to_string(i);
    t.prompt_tokens = 1000;
    // 10 requests: 0 cached, 10: 100, 10: 300, 10: 600, 10: 900
    if (i < 10) t.cached_tokens = 0;
    else if (i < 20) t.cached_tokens = 100;
    else if (i < 30) t.cached_tokens = 300;
    else if (i < 40) t.cached_tokens = 600;
    else t.cached_tokens = 900;
    traces.push_back(t);
  }

  // Create 60 metric snapshots
  std::vector<hotpath::MetricSnapshot> snapshots;
  snapshots.reserve(60);
  for (int i = 0; i < 60; ++i) {
    hotpath::MetricSnapshot s;
    s.timestamp_us = static_cast<int64_t>(i) * 1000000;
    s.batch_size = 8;
    s.queue_depth = 2;
    // Cache usage: 80% for first 40, 95% for last 20
    s.cache_usage = (i < 40) ? 80.0 : 95.0;
    s.preemption_total = static_cast<double>(i < 40 ? 0 : (i - 40) / 4);
    snapshots.push_back(s);
  }

  const auto result = hotpath::analyze_cache(traces, snapshots);

  // Expected hit rate: sum(cached) / sum(prompt)
  // = (10*0 + 10*100 + 10*300 + 10*600 + 10*900) / (50*1000) = 19000/50000 = 0.38
  expect_true(std::abs(result.cache_hit_rate - 0.38) < 0.01,
              "cache_hit_rate: " + std::to_string(result.cache_hit_rate));
  expect_true(result.cache_hit_rate_available, "cache_hit_rate should be available");
  expect_true(!result.cache_hit_rate_aggregate_only, "per-request hit rate should not be aggregate-only");
  expect_true(result.hit_rate_histogram_available, "histogram should be available");

  // Histogram check
  // 0%: 10, 1-25%: 10 (100/1000=10%), 25-50%: 10 (300/1000=30%), 50-75%: 10 (600/1000=60%), 75-100%: 10 (900/1000=90%)
  expect_true(result.hit_rate_histogram[0] == 10, "histogram[0]: " + std::to_string(result.hit_rate_histogram[0]));
  expect_true(result.hit_rate_histogram[1] == 10, "histogram[1]: " + std::to_string(result.hit_rate_histogram[1]));
  expect_true(result.hit_rate_histogram[2] == 10, "histogram[2]: " + std::to_string(result.hit_rate_histogram[2]));
  expect_true(result.hit_rate_histogram[3] == 10, "histogram[3]: " + std::to_string(result.hit_rate_histogram[3]));
  expect_true(result.hit_rate_histogram[4] == 10, "histogram[4]: " + std::to_string(result.hit_rate_histogram[4]));

  // Cache usage
  expect_true(result.avg_cache_usage > 80.0 && result.avg_cache_usage < 95.0,
              "avg_cache_usage: " + std::to_string(result.avg_cache_usage));
  expect_true(result.peak_cache_usage == 95.0,
              "peak_cache_usage: " + std::to_string(result.peak_cache_usage));

  // Pressure seconds: last 20 snapshots have cache > 90%
  expect_true(result.cache_pressure_seconds == 20.0,
              "pressure_seconds: " + std::to_string(result.cache_pressure_seconds));

  // Evictions: snapshots use preemption_total = 0 for i<40, then (i-40)/4 for i>=40.
  // max = (59-40)/4 = 4, min = 0, so eviction_count = 4
  expect_true(result.eviction_count == 4,
              "eviction_count should be 4 (delta of preemption_total), got " +
                  std::to_string(result.eviction_count));

  const auto aggregate_only = hotpath::analyze_cache({}, snapshots, 0.42);
  expect_true(aggregate_only.cache_hit_rate_available, "aggregate hit rate should be available");
  expect_true(aggregate_only.cache_hit_rate_aggregate_only, "aggregate hit rate should be marked aggregate-only");
  expect_true(!aggregate_only.hit_rate_histogram_available, "aggregate-only histogram should be unavailable");
  expect_true(std::abs(aggregate_only.cache_hit_rate - 0.42) < 1e-9,
              "aggregate cache_hit_rate: " + std::to_string(aggregate_only.cache_hit_rate));

  std::cerr << "test_cache_analyzer: all tests passed\n";
  return 0;
}
