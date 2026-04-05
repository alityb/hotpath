// Comprehensive audit tests for all analyzer, model, and report modules.
// Ground-truth values are hand-computed in the test — modules must match.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "hotpath/batch_analyzer.h"
#include "hotpath/cache_analyzer.h"
#include "hotpath/disagg_model.h"
#include "hotpath/kv_config.h"
#include "hotpath/otlp_export.h"
#include "hotpath/phase_analyzer.h"
#include "hotpath/prefix_analyzer.h"
#include "hotpath/recommender.h"
#include "hotpath/report.h"
#include "hotpath/request_trace.h"
#include "hotpath/workload_classifier.h"

namespace {

int g_fail_count = 0;
int g_pass_count = 0;

void check(bool condition, const std::string& msg) {
  if (!condition) {
    std::cerr << "  FAIL: " << msg << "\n";
    ++g_fail_count;
  } else {
    ++g_pass_count;
  }
}

void check_near(double actual, double expected, double tol, const std::string& msg) {
  if (std::abs(actual - expected) > tol) {
    std::cerr << "  FAIL: " << msg << " (got " << actual << ", want " << expected << ")\n";
    ++g_fail_count;
  } else {
    ++g_pass_count;
  }
}

bool contains(const std::string& h, const std::string& n) {
  return h.find(n) != std::string::npos;
}

void section(const std::string& name) {
  std::cerr << "\n=== " << name << " ===\n";
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 1: Phase Analyzer
// ─────────────────────────────────────────────────────────────────────

void test_phase_analyzer() {
  using hotpath::KernelEntry;
  using hotpath::profiler::KernelPhase;

  section("Phase Analyzer — empty input");
  {
    auto r = hotpath::analyze_phases({});
    check(r.breakdown.total_us == 0, "empty total == 0");
    check(r.breakdown.prefill_us == 0, "empty prefill == 0");
    check(r.breakdown.decode_us == 0, "empty decode == 0");
    check(r.breakdown.unknown_us == 0, "empty unknown == 0");
    check(r.breakdown.prefill_fraction == 0.0, "empty prefill_frac == 0");
    check(r.breakdown.decode_fraction == 0.0, "empty decode_frac == 0");
    check(r.time_series.empty(), "empty time_series");
  }

  section("Phase Analyzer — single kernel");
  {
    std::vector<KernelEntry> k = {{"k1", KernelPhase::PREFILL, 0, 500}};
    auto r = hotpath::analyze_phases(k);
    check(r.breakdown.prefill_us == 500, "single prefill_us");
    check(r.breakdown.decode_us == 0, "single decode_us");
    check(r.breakdown.total_us == 500, "single total_us");
    check_near(r.breakdown.prefill_fraction, 1.0, 1e-9, "single prefill_frac");
    check(r.breakdown.prefill_kernel_count == 1, "single prefill_count");
    check(r.time_series.size() == 1, "single time_series has 1 window");
  }

  section("Phase Analyzer — all identical timestamps");
  {
    std::vector<KernelEntry> k;
    for (int i = 0; i < 10; ++i)
      k.push_back({"k", KernelPhase::DECODE, 1000, 100});
    auto r = hotpath::analyze_phases(k);
    check(r.breakdown.decode_us == 1000, "identical decode_us = 10*100");
    check(r.breakdown.total_us == 1000, "identical total_us");
    check(r.breakdown.decode_kernel_count == 10, "identical count");
  }

  section("Phase Analyzer — zero duration kernels");
  {
    std::vector<KernelEntry> k = {
        {"a", KernelPhase::PREFILL, 0, 0},
        {"b", KernelPhase::DECODE, 100, 0},
    };
    auto r = hotpath::analyze_phases(k);
    check(r.breakdown.total_us == 0, "zero dur total == 0");
    check(r.breakdown.prefill_fraction == 0.0, "zero dur no div-by-zero");
    check(r.breakdown.prefill_kernel_count == 1, "zero dur counts");
    check(r.breakdown.decode_kernel_count == 1, "zero dur counts");
  }

  section("Phase Analyzer — extreme duration");
  {
    const int64_t big = 1000000000000LL; // 10^12
    std::vector<KernelEntry> k = {{"k", KernelPhase::PREFILL, 0, big}};
    auto r = hotpath::analyze_phases(k);
    check(r.breakdown.prefill_us == big, "extreme big");
    check(r.breakdown.total_us == big, "extreme total");
  }

  section("Phase Analyzer — large duration (10 seconds)");
  {
    // Large but not so large that time series iteration explodes
    const int64_t big_dur = 10000000LL;  // 10 seconds
    std::vector<KernelEntry> k = {{"k", KernelPhase::DECODE, 0, big_dur}};
    auto r = hotpath::analyze_phases(k);
    check(r.breakdown.decode_us == big_dur, "large duration stored");
    check(r.breakdown.total_us == big_dur, "large total");
    check_near(r.breakdown.decode_fraction, 1.0, 1e-9, "large fraction");
    check(r.time_series.size() == 10, "large 10s → 10 windows");
  }

  section("Phase Analyzer — extreme accumulation (no time series)");
  {
    // Test that very large values accumulate correctly in breakdown
    // Use two kernels at same start to avoid huge time series
    const int64_t v1 = 1000000000LL;  // 10^9
    const int64_t v2 = 2000000000LL;  // 2*10^9
    std::vector<KernelEntry> k = {
        {"a", KernelPhase::PREFILL, 0, v1},
        {"b", KernelPhase::DECODE, 0, v2},
    };
    auto r = hotpath::analyze_phases(k);
    check(r.breakdown.prefill_us == v1, "extreme accum prefill");
    check(r.breakdown.decode_us == v2, "extreme accum decode");
    check(r.breakdown.total_us == v1 + v2, "extreme accum total");
    check_near(r.breakdown.prefill_fraction,
               static_cast<double>(v1) / (v1 + v2), 1e-9, "extreme accum frac");
  }

  section("Phase Analyzer — conservation law and numerical verification");
  {
    // 4 prefill kernels: 100, 200, 300, 400 us = 1000 us
    // 3 decode kernels: 150, 250, 350 us = 750 us
    // 2 unknown kernels: 50, 100 us = 150 us
    // total = 1900 us
    std::vector<KernelEntry> k = {
        {"p1", KernelPhase::PREFILL, 0, 100},
        {"p2", KernelPhase::PREFILL, 100, 200},
        {"d1", KernelPhase::DECODE, 300, 150},
        {"p3", KernelPhase::PREFILL, 450, 300},
        {"d2", KernelPhase::DECODE, 750, 250},
        {"u1", KernelPhase::UNKNOWN, 1000, 50},
        {"p4", KernelPhase::PREFILL, 1050, 400},
        {"d3", KernelPhase::DECODE, 1450, 350},
        {"u2", KernelPhase::UNKNOWN, 1800, 100},
    };
    auto r = hotpath::analyze_phases(k);

    check(r.breakdown.prefill_us == 1000, "exact prefill_us");
    check(r.breakdown.decode_us == 750, "exact decode_us");
    check(r.breakdown.unknown_us == 150, "exact unknown_us");
    check(r.breakdown.total_us == 1900, "exact total_us");

    // Conservation: parts sum to whole
    check(r.breakdown.prefill_us + r.breakdown.decode_us + r.breakdown.unknown_us
          == r.breakdown.total_us,
          "conservation: prefill + decode + unknown == total");

    // Fractions
    check_near(r.breakdown.prefill_fraction, 1000.0 / 1900.0, 1e-9, "exact prefill_frac");
    check_near(r.breakdown.decode_fraction, 750.0 / 1900.0, 1e-9, "exact decode_frac");
    check(r.breakdown.prefill_fraction + r.breakdown.decode_fraction <= 1.0 + 1e-9,
          "fractions <= 1.0");
    check(r.breakdown.prefill_fraction >= 0.0, "prefill_frac >= 0");
    check(r.breakdown.decode_fraction >= 0.0, "decode_frac >= 0");

    // Counts
    check(r.breakdown.prefill_kernel_count == 4, "prefill_count == 4");
    check(r.breakdown.decode_kernel_count == 3, "decode_count == 3");

    // Time series: kernels span 0..1900 us, < 1 second, so 1 window
    check(r.time_series.size() == 1, "1 window for <1s span");
    if (!r.time_series.empty()) {
      check(r.time_series[0].prefill_fraction >= 0.0, "ts prefill >= 0");
      check(r.time_series[0].decode_fraction >= 0.0, "ts decode >= 0");
    }
  }

  section("Phase Analyzer — overlapping kernels in same window");
  {
    // Two kernels overlapping in time: both run 0..500000 us (0.5s each)
    // One PREFILL, one DECODE — simulates concurrent GPU streams
    std::vector<KernelEntry> k = {
        {"p", KernelPhase::PREFILL, 0, 500000},
        {"d", KernelPhase::DECODE, 0, 500000},
    };
    auto r = hotpath::analyze_phases(k);
    // Breakdown should still sum correctly (overlaps don't merge)
    check(r.breakdown.prefill_us == 500000, "overlap: prefill_us correct");
    check(r.breakdown.decode_us == 500000, "overlap: decode_us correct");
    check(r.breakdown.total_us == 1000000, "overlap: total = sum of durations");
    // Conservation still holds — total is sum of durations, not wall clock
    check(r.breakdown.prefill_us + r.breakdown.decode_us + r.breakdown.unknown_us
          == r.breakdown.total_us,
          "overlap: conservation holds");
    // Fractions: each is 0.5
    check_near(r.breakdown.prefill_fraction, 0.5, 1e-9, "overlap: prefill_frac = 0.5");
    check_near(r.breakdown.decode_fraction, 0.5, 1e-9, "overlap: decode_frac = 0.5");

    // Time series: spans 0..500000, all within 1 second → 1 window
    check(r.time_series.size() == 1, "overlap: 1 window");
    if (!r.time_series.empty()) {
      // In the window, prefill contributes 500000 and decode 500000
      // total_in_window = 1000000. prefill_frac = 0.5, decode_frac = 0.5
      check_near(r.time_series[0].prefill_fraction, 0.5, 1e-9,
                 "overlap: ts prefill_frac = 0.5");
      check_near(r.time_series[0].decode_fraction, 0.5, 1e-9,
                 "overlap: ts decode_frac = 0.5");
    }
  }

  section("Phase Analyzer — time series exact numerical");
  {
    // 2-second span: 1 prefill kernel 0..1500000 (1.5s), 1 decode 1500000..2000000 (0.5s)
    // Window 0 [0, 1000000): prefill overlap = 1000000, total = 1000000
    //   → prefill_frac = 1.0, decode_frac = 0.0
    // Window 1 [1000000, 2000000): prefill overlap = 500000, decode = 500000, total = 1000000
    //   → prefill_frac = 0.5, decode_frac = 0.5
    std::vector<KernelEntry> k = {
        {"p", KernelPhase::PREFILL, 0, 1500000},
        {"d", KernelPhase::DECODE, 1500000, 500000},
    };
    auto r = hotpath::analyze_phases(k);
    check(r.time_series.size() == 2, "ts_exact: 2 windows");
    if (r.time_series.size() == 2) {
      check_near(r.time_series[0].prefill_fraction, 1.0, 1e-9,
                 "ts_exact: w0 prefill=1.0");
      check_near(r.time_series[0].decode_fraction, 0.0, 1e-9,
                 "ts_exact: w0 decode=0.0");
      check_near(r.time_series[1].prefill_fraction, 0.5, 1e-9,
                 "ts_exact: w1 prefill=0.5");
      check_near(r.time_series[1].decode_fraction, 0.5, 1e-9,
                 "ts_exact: w1 decode=0.5");
    }
  }

  section("Phase Analyzer — determinism");
  {
    std::vector<KernelEntry> k = {
        {"a", KernelPhase::PREFILL, 0, 500},
        {"b", KernelPhase::DECODE, 500, 300},
        {"c", KernelPhase::UNKNOWN, 800, 200},
    };
    auto r1 = hotpath::analyze_phases(k);
    auto r2 = hotpath::analyze_phases(k);
    check(r1.breakdown.prefill_us == r2.breakdown.prefill_us, "det prefill");
    check(r1.breakdown.decode_us == r2.breakdown.decode_us, "det decode");
    check(r1.breakdown.total_us == r2.breakdown.total_us, "det total");
    check(r1.breakdown.prefill_fraction == r2.breakdown.prefill_fraction, "det frac");
    check(r1.time_series.size() == r2.time_series.size(), "det ts size");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 2: Batch Analyzer
// ─────────────────────────────────────────────────────────────────────

void test_batch_analyzer() {
  section("Batch Analyzer — empty input");
  {
    auto r = hotpath::analyze_batches({});
    check(r.avg_batch_size == 0.0, "empty avg == 0");
    check(r.p50_batch_size == 0.0, "empty p50 == 0");
    check(r.p99_batch_size == 0.0, "empty p99 == 0");
    check(r.batch_size_series.empty(), "empty series");
  }

  section("Batch Analyzer — single snapshot");
  {
    hotpath::MetricSnapshot s{.timestamp_us = 0, .batch_size = 7.0,
                              .queue_depth = 3.0, .preemption_total = 5.0,
                              .cache_usage = 42.0};
    auto r = hotpath::analyze_batches({s});
    check_near(r.avg_batch_size, 7.0, 1e-9, "single avg");
    check_near(r.p50_batch_size, 7.0, 1e-9, "single p50");
    check_near(r.p99_batch_size, 7.0, 1e-9, "single p99");
    check_near(r.avg_queue_depth, 3.0, 1e-9, "single avg_queue");
    check(r.total_preemptions == 0, "single preemptions == 0 (no delta)");
    check_near(r.avg_cache_usage, 42.0, 1e-9, "single avg_cache");
    check_near(r.peak_cache_usage, 42.0, 1e-9, "single peak_cache");
  }

  section("Batch Analyzer — all identical");
  {
    std::vector<hotpath::MetricSnapshot> snaps(20);
    for (auto& s : snaps) {
      s = {.timestamp_us = 0, .batch_size = 10.0, .queue_depth = 2.0,
           .preemption_total = 3.0, .cache_usage = 50.0};
    }
    auto r = hotpath::analyze_batches(snaps);
    check_near(r.avg_batch_size, 10.0, 1e-9, "identical avg");
    check_near(r.p50_batch_size, 10.0, 1e-9, "identical p50");
    check_near(r.p99_batch_size, 10.0, 1e-9, "identical p99");
    check(r.total_preemptions == 0, "identical preemptions == 0");
  }

  section("Batch Analyzer — precise percentile computation");
  {
    // 4 snapshots with batch_size = [1, 2, 3, 4]
    // p50: idx = 0.5 * 3 = 1.5 → interpolate values[1] and values[2]
    //   = 2 * 0.5 + 3 * 0.5 = 2.5
    // p99: idx = 0.99 * 3 = 2.97 → interpolate values[2] and values[3]
    //   = 3 * 0.03 + 4 * 0.97 = 3.97
    std::vector<hotpath::MetricSnapshot> snaps;
    for (int i = 1; i <= 4; ++i) {
      snaps.push_back({.timestamp_us = static_cast<int64_t>(i) * 1000000,
                        .batch_size = static_cast<double>(i),
                        .queue_depth = static_cast<double>(i * 2),
                        .preemption_total = 0.0, .cache_usage = 0.0});
    }
    auto r = hotpath::analyze_batches(snaps);
    check_near(r.avg_batch_size, 2.5, 1e-9, "4-elem avg = 2.5");
    check_near(r.p50_batch_size, 2.5, 1e-6, "4-elem p50 = 2.5");
    check_near(r.p99_batch_size, 3.97, 1e-6, "4-elem p99 = 3.97");

    // avg queue = (2+4+6+8)/4 = 5.0
    check_near(r.avg_queue_depth, 5.0, 1e-9, "4-elem avg_queue = 5.0");
  }

  section("Batch Analyzer — conservation: p50 <= p99, avg in range");
  {
    std::vector<hotpath::MetricSnapshot> snaps;
    for (int i = 0; i < 100; ++i) {
      snaps.push_back({.timestamp_us = static_cast<int64_t>(i) * 1000000,
                        .batch_size = static_cast<double>(i + 1),
                        .queue_depth = 0.0, .preemption_total = 0.0,
                        .cache_usage = 0.0});
    }
    auto r = hotpath::analyze_batches(snaps);
    check(r.p50_batch_size <= r.p99_batch_size, "p50 <= p99");
    check(r.avg_batch_size >= 1.0, "avg >= min");
    check(r.avg_batch_size <= 100.0, "avg <= max");
    // Note: avg can exceed p99 with extreme right-skew, so we only check
    // avg is between min and max of the raw values (0..99 → min 0).
    check(r.avg_queue_depth >= 0.0, "avg_queue >= 0");
    check(r.p99_queue_depth >= 0.0, "p99_queue >= 0");
    check(r.total_preemptions >= 0, "preemptions >= 0");
    check(r.avg_cache_usage >= 0.0, "avg_cache >= 0");
    check(r.peak_cache_usage >= r.avg_cache_usage - 1e-9, "peak_cache >= avg_cache");
  }

  section("Batch Analyzer — preemption delta");
  {
    // preemption counter goes 5 → 8 → 12
    std::vector<hotpath::MetricSnapshot> snaps = {
        {0, 1, 0, 5.0, 0},
        {1000000, 1, 0, 8.0, 0},
        {2000000, 1, 0, 12.0, 0},
    };
    auto r = hotpath::analyze_batches(snaps);
    check(r.total_preemptions == 7, "preemption delta = 12 - 5 = 7");
  }

  section("Batch Analyzer — 2-element percentile exact");
  {
    // 2 elements [10, 100]
    // p50: idx=0.5*1=0.5. lo=0,hi=1,frac=0.5. result=10*0.5+100*0.5=55
    // p99: idx=0.99*1=0.99. lo=0,hi=1,frac=0.99. result=10*0.01+100*0.99=0.1+99.0=99.1
    std::vector<hotpath::MetricSnapshot> snaps = {
        {0, 10.0, 0, 0, 0},
        {1000000, 100.0, 0, 0, 0},
    };
    auto r = hotpath::analyze_batches(snaps);
    check_near(r.p50_batch_size, 55.0, 1e-6, "2-elem p50 = 55.0");
    check_near(r.p99_batch_size, 99.1, 1e-6, "2-elem p99 = 99.1");
    check_near(r.avg_batch_size, 55.0, 1e-9, "2-elem avg = 55.0");
  }

  section("Batch Analyzer — avg can exceed p99 (extreme right-skew)");
  {
    // 100 elements: 99 zeros and one 1000000
    // avg = 1000000/100 = 10000
    // p99: idx=0.99*99=98.01. sorted[98]=0, sorted[99]=1000000.
    //   = 0*0.99 + 1000000*0.01 = 10000. So avg==p99 here.
    // But 4 elements: [0,0,0,1000000]
    // avg = 250000. p99: idx=0.99*3=2.97. val[2]=0, val[3]=1000000.
    //   = 0*0.03+1000000*0.97 = 970000. avg < p99 here.
    // The point: avg vs p99 relationship is NOT guaranteed.
    std::vector<hotpath::MetricSnapshot> snaps;
    for (int i = 0; i < 4; ++i) {
      snaps.push_back({static_cast<int64_t>(i)*1000000,
                        (i == 3) ? 1000000.0 : 0.0, 0, 0, 0});
    }
    auto r = hotpath::analyze_batches(snaps);
    check_near(r.avg_batch_size, 250000.0, 1e-6, "skew: avg = 250000");
    check_near(r.p99_batch_size, 970000.0, 1e-3, "skew: p99 = 970000");
    // avg < p99 in this case, but we don't assert ordering because it's input-dependent
  }

  section("Batch Analyzer — series preserves time order");
  {
    std::vector<hotpath::MetricSnapshot> snaps = {
        {100, 5.0, 0, 0, 0},
        {200, 10.0, 0, 0, 0},
        {300, 15.0, 0, 0, 0},
    };
    auto r = hotpath::analyze_batches(snaps);
    check(r.batch_size_series.size() == 3, "series len = 3");
    check(r.batch_size_series[0].first == 100, "series[0].ts = 100");
    check(r.batch_size_series[1].first == 200, "series[1].ts = 200");
    check(r.batch_size_series[2].first == 300, "series[2].ts = 300");
    check_near(r.batch_size_series[0].second, 5.0, 1e-9, "series[0].val = 5");
    check_near(r.batch_size_series[1].second, 10.0, 1e-9, "series[1].val = 10");
    check_near(r.batch_size_series[2].second, 15.0, 1e-9, "series[2].val = 15");
  }

  section("Batch Analyzer — determinism");
  {
    std::vector<hotpath::MetricSnapshot> snaps = {
        {0, 5, 2, 0, 40}, {1000000, 10, 3, 1, 60},
    };
    auto r1 = hotpath::analyze_batches(snaps);
    auto r2 = hotpath::analyze_batches(snaps);
    check(r1.avg_batch_size == r2.avg_batch_size, "det avg");
    check(r1.p50_batch_size == r2.p50_batch_size, "det p50");
    check(r1.p99_batch_size == r2.p99_batch_size, "det p99");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 3: Cache Analyzer
// ─────────────────────────────────────────────────────────────────────

void test_cache_analyzer() {
  section("Cache Analyzer — empty input");
  {
    auto r = hotpath::analyze_cache({}, {});
    check_near(r.cache_hit_rate, 0.0, 1e-9, "empty hit_rate");
    check_near(r.avg_cache_usage, 0.0, 1e-9, "empty avg_cache");
    check_near(r.peak_cache_usage, 0.0, 1e-9, "empty peak_cache");
    check(r.eviction_count == 0, "empty evictions");
    check(!r.cache_hit_rate_available, "empty cache_hit_rate unavailable");
    check(!r.hit_rate_histogram_available, "empty histogram unavailable");
    int sum = 0;
    for (int b : r.hit_rate_histogram) sum += b;
    check(sum == 0, "empty histogram sum == 0");
  }

  section("Cache Analyzer — prompt_tokens = 0");
  {
    hotpath::RequestTrace t;
    t.prompt_tokens = 0;
    t.cached_tokens = 0;
    auto r = hotpath::analyze_cache({t}, {});
    check_near(r.cache_hit_rate, 0.0, 1e-9, "zero prompt hit_rate");
    check(r.hit_rate_histogram[0] == 1, "zero prompt in bucket 0");
  }

  section("Cache Analyzer — cached > prompt (nonsensical)");
  {
    hotpath::RequestTrace t;
    t.prompt_tokens = 100;
    t.cached_tokens = 200;
    auto r = hotpath::analyze_cache({t}, {});
    // hit_frac = 200/100 = 2.0 → clamped to 1.0. Bucket [4] (>= 0.75).
    check(r.hit_rate_histogram[4] == 1, "cached>prompt → bucket 4");
    check_near(r.cache_hit_rate, 1.0, 1e-9, "cached>prompt rate clamped to 1.0");
  }

  section("Cache Analyzer — prompt_tokens = 1000000 (large)");
  {
    hotpath::RequestTrace t;
    t.prompt_tokens = 1000000;
    t.cached_tokens = 500000;
    auto r = hotpath::analyze_cache({t}, {});
    check_near(r.cache_hit_rate, 0.5, 1e-9, "large prompt hit_rate = 0.5");
    // 0.50 is not < 0.50, so goes to bucket[3] (50-75%), not bucket[2] (25-50%)
    check(r.hit_rate_histogram[3] == 1, "large prompt 50% → bucket 3 ([0.50,0.75))");
  }

  section("Cache Analyzer — cache_usage > 100 (nonsensical percentage)");
  {
    hotpath::MetricSnapshot s{0, 1, 0, 0, 150.0};
    auto r = hotpath::analyze_cache({}, {s});
    check_near(r.peak_cache_usage, 150.0, 1e-9, "oversized cache_usage preserved");
    check_near(r.avg_cache_usage, 150.0, 1e-9, "oversized cache avg preserved");
    // >90% threshold still applies
    check_near(r.cache_pressure_seconds, 1.0, 1e-9, "oversized counts as pressure");
  }

  section("Cache Analyzer — precise numerical verification");
  {
    // 5 requests: cached = [0, 100, 300, 600, 1000], prompt = 1000 each
    // hit_rate = (0+100+300+600+1000) / (5*1000) = 2000/5000 = 0.4
    // buckets: 0/1000=0→[0], 100/1000=0.1→[1], 300/1000=0.3→[2],
    //          600/1000=0.6→[3], 1000/1000=1.0→[4]
    std::vector<hotpath::RequestTrace> traces;
    int cached_vals[] = {0, 100, 300, 600, 1000};
    for (int c : cached_vals) {
      hotpath::RequestTrace t;
      t.prompt_tokens = 1000;
      t.cached_tokens = c;
      traces.push_back(t);
    }
    auto r = hotpath::analyze_cache(traces, {});
    check_near(r.cache_hit_rate, 0.4, 1e-9, "exact hit_rate = 0.4");
    check(r.hit_rate_histogram[0] == 1, "bucket[0] = 1");
    check(r.hit_rate_histogram[1] == 1, "bucket[1] = 1");
    check(r.hit_rate_histogram[2] == 1, "bucket[2] = 1");
    check(r.hit_rate_histogram[3] == 1, "bucket[3] = 1");
    check(r.hit_rate_histogram[4] == 1, "bucket[4] = 1");

    // Conservation: histogram sum == total requests
    int hsum = 0;
    for (int b : r.hit_rate_histogram) hsum += b;
    check(hsum == 5, "histogram sum == 5 requests");

    // Independent verification of cache_hit_rate
    int64_t sum_cached = 0 + 100 + 300 + 600 + 1000;
    int64_t sum_prompt = 5 * 1000;
    double expected_rate = static_cast<double>(sum_cached) / sum_prompt;
    check_near(r.cache_hit_rate, expected_rate, 1e-9, "independent hit_rate");
    check(r.cache_hit_rate_available, "per-request hit_rate available");
    check(!r.cache_hit_rate_aggregate_only, "per-request hit_rate not aggregate-only");
    check(r.hit_rate_histogram_available, "per-request histogram available");
  }

  section("Cache Analyzer — aggregate-only cache hit rate");
  {
    auto r = hotpath::analyze_cache({}, {}, 0.33);
    check(r.cache_hit_rate_available, "aggregate-only hit_rate available");
    check(r.cache_hit_rate_aggregate_only, "aggregate-only flag set");
    check(!r.hit_rate_histogram_available, "aggregate-only histogram unavailable");
    check_near(r.cache_hit_rate, 0.33, 1e-9, "aggregate-only hit_rate = 0.33");
    int hsum = 0;
    for (int b : r.hit_rate_histogram) hsum += b;
    check(hsum == 0, "aggregate-only histogram empty");
  }

  section("Cache Analyzer — pressure seconds and evictions");
  {
    // 10 snapshots: first 4 at 80% cache, last 6 at 95% (>90%)
    // preemption counter: 0,0,0,0, 1,1,2,2,3,3
    std::vector<hotpath::MetricSnapshot> snaps;
    double preemptions[] = {0,0,0,0, 1,1,2,2,3,3};
    for (int i = 0; i < 10; ++i) {
      snaps.push_back({static_cast<int64_t>(i)*1000000,
                        8.0, 2.0, preemptions[i],
                        (i < 4) ? 80.0 : 95.0});
    }
    auto r = hotpath::analyze_cache({}, snaps);
    check_near(r.cache_pressure_seconds, 6.0, 1e-9, "pressure = 6 seconds");
    check(r.eviction_count == 3, "evictions = 3-0 = 3");
    check_near(r.peak_cache_usage, 95.0, 1e-9, "peak = 95");
    // avg = (4*80 + 6*95) / 10 = (320+570)/10 = 89.0
    check_near(r.avg_cache_usage, 89.0, 1e-9, "avg_cache = 89.0");
  }

  section("Cache Analyzer — histogram exact boundaries");
  {
    // Test exact boundary values: 0.0, 0.25, 0.50, 0.75
    // hit_frac = cached / prompt
    std::vector<hotpath::RequestTrace> traces;
    auto make = [&](int cached, int prompt) {
      hotpath::RequestTrace t;
      t.prompt_tokens = prompt;
      t.cached_tokens = cached;
      traces.push_back(t);
    };
    // Buckets use half-open intervals: [0,0], (0,0.25), [0.25,0.50), [0.50,0.75), [0.75,1.0]
    make(0, 400);      // 0.0    → bucket[0] (== 0.0)
    make(100, 400);     // 0.25   → bucket[2] ([0.25,0.50): not < 0.25)
    make(200, 400);     // 0.50   → bucket[3] ([0.50,0.75): not < 0.50)
    make(300, 400);     // 0.75   → bucket[4] ([0.75,1.0]: not < 0.75)
    make(400, 400);     // 1.0    → bucket[4]
    make(1, 400);       // 0.0025 → bucket[1] ((0,0.25): < 0.25)
    make(101, 400);     // 0.2525 → bucket[2] ([0.25,0.50): < 0.50)
    make(201, 400);     // 0.5025 → bucket[3] ([0.50,0.75): < 0.75)
    make(301, 400);     // 0.7525 → bucket[4] (not < 0.75)

    auto r = hotpath::analyze_cache(traces, {});
    check(r.hit_rate_histogram[0] == 1, "boundary: bucket[0] = 1 (exact 0.0)");
    check(r.hit_rate_histogram[1] == 1, "boundary: bucket[1] = 1 (0.0025 only; 0.25 moved to bucket[2])");
    check(r.hit_rate_histogram[2] == 2, "boundary: bucket[2] = 2 (0.25 and 0.2525)");
    check(r.hit_rate_histogram[3] == 2, "boundary: bucket[3] = 2 (0.50 and 0.5025)");
    check(r.hit_rate_histogram[4] == 3, "boundary: bucket[4] = 3 (0.75, 1.0, 0.7525)");
    int hsum = 0;
    for (int b : r.hit_rate_histogram) hsum += b;
    check(hsum == 9, "boundary: histogram sum == 9");
  }

  section("Cache Analyzer — single request exact");
  {
    hotpath::RequestTrace t;
    t.prompt_tokens = 800;
    t.cached_tokens = 200;
    // hit_frac = 200/800 = 0.25 → bucket[2] ([0.25,0.50): not < 0.25, but < 0.50)
    auto r = hotpath::analyze_cache({t}, {});
    check_near(r.cache_hit_rate, 0.25, 1e-9, "single exact hit_rate = 0.25");
    check(r.hit_rate_histogram[2] == 1, "single exact 25% → bucket[2] (25-50%)");
  }

  section("Cache Analyzer — determinism");
  {
    hotpath::RequestTrace t;
    t.prompt_tokens = 500;
    t.cached_tokens = 200;
    auto r1 = hotpath::analyze_cache({t}, {});
    auto r2 = hotpath::analyze_cache({t}, {});
    check(r1.cache_hit_rate == r2.cache_hit_rate, "det hit_rate");
    for (int i = 0; i < 5; ++i)
      check(r1.hit_rate_histogram[i] == r2.hit_rate_histogram[i], "det histogram");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 4: Prefix Analyzer
// ─────────────────────────────────────────────────────────────────────

void test_prefix_analyzer() {
  section("Prefix Analyzer — empty input");
  {
    auto r = hotpath::analyze_prefixes({});
    check(r.total_requests == 0, "empty total");
    check(r.unique_prefixes == 0, "empty unique");
    check(r.top_prefixes.empty(), "empty top");
    check_near(r.cacheable_token_fraction, 0.0, 1e-9, "empty cacheable");
  }

  section("Prefix Analyzer — single prompt");
  {
    std::vector<std::vector<int>> prompts = {{1, 2, 3, 4, 5}};
    auto r = hotpath::analyze_prefixes(prompts);
    check(r.total_requests == 1, "single total == 1");
    check(r.unique_prefixes == 1, "single unique == 1 (singleton)");
    check_near(r.cacheable_token_fraction, 0.0, 1e-9, "single no sharing");
  }

  section("Prefix Analyzer — all identical prompts");
  {
    // 10 identical [1,2,3,4,5] → 1 shared prefix of length 5, 10 requests
    std::vector<std::vector<int>> prompts(10, {1, 2, 3, 4, 5});
    auto r = hotpath::analyze_prefixes(prompts);
    check(r.total_requests == 10, "all-same total");
    // Should have 1 shared prefix group
    check(!r.top_prefixes.empty(), "all-same has top prefixes");
    if (!r.top_prefixes.empty()) {
      check(r.top_prefixes[0].request_count == 10, "all-same group = 10");
      check(r.top_prefixes[0].prefix_length == 5, "all-same length = 5");
    }
    // cacheable = 5*10 / 5*10 = 1.0
    check_near(r.cacheable_token_fraction, 1.0, 1e-9, "all-same cacheable = 1.0");
  }

  section("Prefix Analyzer — precise computation");
  {
    // 6 prompts: 4 share prefix [10,20,30,40] + unique suffix, 2 unique
    std::vector<std::vector<int>> prompts;
    for (int i = 0; i < 4; ++i) {
      prompts.push_back({10, 20, 30, 40, 100 + i * 10, 110 + i * 10});
    }
    prompts.push_back({50, 60, 70, 80, 90, 100});
    prompts.push_back({200, 210, 220, 230, 240, 250});

    auto r = hotpath::analyze_prefixes(prompts);
    check(r.total_requests == 6, "precise total == 6");
    // 1 shared prefix group (4 requests), 2 singletons = 3 unique_prefixes
    check(r.unique_prefixes == 3, "precise unique == 3");
    if (!r.top_prefixes.empty()) {
      check(r.top_prefixes[0].request_count == 4, "top group = 4 requests");
      check(r.top_prefixes[0].prefix_length == 4, "top group length = 4");
      // fraction = 4/6
      check_near(r.top_prefixes[0].fraction, 4.0 / 6.0, 1e-9, "top fraction");
    }
    // cacheable_tokens = 4 * 4 = 16
    // total_tokens = 4*6 + 2*6 = 36
    check_near(r.cacheable_token_fraction, 16.0 / 36.0, 1e-9, "cacheable = 16/36");
  }

  section("Prefix Analyzer — nested shared prefixes (branching depth)");
  {
    // 3 prompts: [1,2,3,4,5], [1,2,3,4,6], [1,2,3,4,6]
    // Trie branches at depth 4 (after token 4). The two [..,6] prompts share
    // a depth-5 prefix but the algorithm stops at the first branch point.
    // Reported: shared prefix of length 4, count 3
    // The depth-5 sub-prefix (2 requests sharing [1,2,3,4,6]) is NOT reported.
    // cacheable = 4*3 = 12. total = 3*5 = 15. fraction = 12/15 = 0.8
    std::vector<std::vector<int>> prompts = {
        {1, 2, 3, 4, 5},
        {1, 2, 3, 4, 6},
        {1, 2, 3, 4, 6},
    };
    auto r = hotpath::analyze_prefixes(prompts);
    check(r.total_requests == 3, "nested: total = 3");
    check(!r.top_prefixes.empty(), "nested: has a group");
    if (!r.top_prefixes.empty()) {
      check(r.top_prefixes[0].request_count == 3, "nested: group count = 3");
      check(r.top_prefixes[0].prefix_length == 4, "nested: stops at branch depth 4");
    }
    // Algorithm gives a lower bound on cacheability — the 2 identical prompts
    // sharing 5 tokens are under-reported. This is by design.
    check_near(r.cacheable_token_fraction, 12.0 / 15.0, 1e-9,
               "nested: cacheable = 12/15 (conservative)");
  }

  section("Prefix Analyzer — conservation: top_prefix counts");
  {
    // 8 prompts: 5 share [1,2,3,4]+unique, 3 share [10,20,30,40]+unique
    std::vector<std::vector<int>> prompts;
    for (int i = 0; i < 5; ++i)
      prompts.push_back({1, 2, 3, 4, 500 + i, 600 + i});
    for (int i = 0; i < 3; ++i)
      prompts.push_back({10, 20, 30, 40, 700 + i, 800 + i});
    auto r = hotpath::analyze_prefixes(prompts);
    // top_prefixes should account for all shared requests
    int top_sum = 0;
    for (const auto& g : r.top_prefixes) {
      top_sum += g.request_count;
      check(g.request_count >= 2, "prefix group has >= 2 requests");
      check(g.prefix_length >= 4, "prefix length >= min_prefix_len");
      check(g.fraction >= 0.0 && g.fraction <= 1.0, "prefix fraction in [0,1]");
    }
    // 5 + 3 = 8 requests in shared groups
    check(top_sum == 8, "top_prefix sum == 8 shared requests");
    check(r.cacheable_token_fraction >= 0.0 && r.cacheable_token_fraction <= 1.0,
          "cacheable_token_fraction in [0,1]");
  }

  section("Prefix Analyzer — determinism");
  {
    std::vector<std::vector<int>> prompts = {{1, 2, 3, 4, 5}, {1, 2, 3, 4, 6}};
    auto r1 = hotpath::analyze_prefixes(prompts);
    auto r2 = hotpath::analyze_prefixes(prompts);
    check(r1.total_requests == r2.total_requests, "det total");
    check(r1.unique_prefixes == r2.unique_prefixes, "det unique");
    check(r1.cacheable_token_fraction == r2.cacheable_token_fraction, "det cacheable");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 5: Workload Classifier
// ─────────────────────────────────────────────────────────────────────

void test_workload_classifier() {
  section("Workload Classifier — PREFILL_HEAVY invariant");
  {
    hotpath::WorkloadClassifierInput input;
    input.median_prompt_tokens = 4096;
    input.median_output_tokens = 256;
    input.phase.prefill_fraction = 0.65;
    input.phase.decode_fraction = 0.25;
    input.prefix.cacheable_token_fraction = 0.1;
    input.request_rate = 10.0;
    auto r = hotpath::classify_workload(input);
    check(r.primary_class == hotpath::WorkloadClass::PREFILL_HEAVY,
          "classified PREFILL_HEAVY");
    check(r.prefill_fraction > 0.5,
          "PREFILL_HEAVY requires prefill_fraction > 0.5");
  }

  section("Workload Classifier — PREFILL_HEAVY rejected when prefill_frac low");
  {
    hotpath::WorkloadClassifierInput input;
    input.median_prompt_tokens = 4096;
    input.median_output_tokens = 256;
    input.phase.prefill_fraction = 0.3;  // too low
    input.phase.decode_fraction = 0.5;
    input.prefix.cacheable_token_fraction = 0.1;
    auto r = hotpath::classify_workload(input);
    check(r.primary_class != hotpath::WorkloadClass::PREFILL_HEAVY,
          "low prefill_frac → not PREFILL_HEAVY");
  }

  section("Workload Classifier — fractions in [0, 1]");
  {
    hotpath::WorkloadClassifierInput input;
    input.median_prompt_tokens = 1024;
    input.median_output_tokens = 512;
    input.phase.prefill_fraction = 0.4;
    input.cache.cache_hit_rate = 0.3;
    input.prefix.cacheable_token_fraction = 0.2;
    auto r = hotpath::classify_workload(input);
    check(r.prefill_fraction >= 0.0 && r.prefill_fraction <= 1.0, "prefill_frac in [0,1]");
    check(r.prefix_sharing_rate >= 0.0 && r.prefix_sharing_rate <= 1.0, "sharing in [0,1]");
    check(r.cache_hit_rate >= 0.0 && r.cache_hit_rate <= 1.0, "hit_rate in [0,1]");
  }

  section("Workload Classifier — contention exact value");
  {
    hotpath::WorkloadClassifierInput input;
    input.median_prompt_tokens = 1024;
    input.median_output_tokens = 512;
    input.median_decode_latency_us = 4000;
    input.p99_decode_latency_us = 12000;
    auto r = hotpath::classify_workload(input);
    // contention = (12000 - 4000) / 4000 = 2.0
    check_near(r.prefill_contention, 2.0, 1e-9, "contention = 2.0");
  }

  section("Workload Classifier — zero decode latency");
  {
    hotpath::WorkloadClassifierInput input;
    input.median_prompt_tokens = 1024;
    input.median_decode_latency_us = 0;
    input.p99_decode_latency_us = 0;
    auto r = hotpath::classify_workload(input);
    check_near(r.prefill_contention, 0.0, 1e-9, "zero latency → 0 contention");
  }

  section("Workload Classifier — SHORT_CONTEXT beats CACHE_FRIENDLY");
  {
    // Both conditions met: prompt < 256 AND prefix sharing > 60%
    // SHORT_CONTEXT is checked first and should win
    hotpath::WorkloadClassifierInput input;
    input.median_prompt_tokens = 128;
    input.median_output_tokens = 32;
    input.prefix.cacheable_token_fraction = 0.9;  // very cache-friendly
    input.phase.prefill_fraction = 0.8;
    auto r = hotpath::classify_workload(input);
    check(r.primary_class == hotpath::WorkloadClass::SHORT_CONTEXT,
          "SHORT_CONTEXT wins over CACHE_FRIENDLY");
  }

  section("Workload Classifier — CACHE_FRIENDLY beats PREFILL_HEAVY");
  {
    // Both conditions met: prompt > 2048 AND prefix sharing > 60%
    // CACHE_FRIENDLY is checked first and should win
    hotpath::WorkloadClassifierInput input;
    input.median_prompt_tokens = 4096;
    input.median_output_tokens = 256;
    input.prefix.cacheable_token_fraction = 0.8;
    input.phase.prefill_fraction = 0.7;
    auto r = hotpath::classify_workload(input);
    check(r.primary_class == hotpath::WorkloadClass::CACHE_FRIENDLY,
          "CACHE_FRIENDLY wins over PREFILL_HEAVY");
  }

  section("Workload Classifier — all 5 classes reachable");
  {
    // SHORT_CONTEXT
    {
      hotpath::WorkloadClassifierInput i;
      i.median_prompt_tokens = 100; i.median_output_tokens = 50;
      check(hotpath::classify_workload(i).primary_class == hotpath::WorkloadClass::SHORT_CONTEXT,
            "SHORT_CONTEXT reachable");
    }
    // CACHE_FRIENDLY
    {
      hotpath::WorkloadClassifierInput i;
      i.median_prompt_tokens = 1024; i.median_output_tokens = 256;
      i.prefix.cacheable_token_fraction = 0.8;
      check(hotpath::classify_workload(i).primary_class == hotpath::WorkloadClass::CACHE_FRIENDLY,
            "CACHE_FRIENDLY reachable");
    }
    // PREFILL_HEAVY
    {
      hotpath::WorkloadClassifierInput i;
      i.median_prompt_tokens = 4096; i.median_output_tokens = 256;
      i.phase.prefill_fraction = 0.7;
      check(hotpath::classify_workload(i).primary_class == hotpath::WorkloadClass::PREFILL_HEAVY,
            "PREFILL_HEAVY reachable");
    }
    // DECODE_HEAVY
    {
      hotpath::WorkloadClassifierInput i;
      i.median_prompt_tokens = 512; i.median_output_tokens = 2048;
      check(hotpath::classify_workload(i).primary_class == hotpath::WorkloadClass::DECODE_HEAVY,
            "DECODE_HEAVY reachable");
    }
    // BALANCED
    {
      hotpath::WorkloadClassifierInput i;
      i.median_prompt_tokens = 1024; i.median_output_tokens = 512;
      check(hotpath::classify_workload(i).primary_class == hotpath::WorkloadClass::BALANCED,
            "BALANCED reachable");
    }
  }

  section("Workload Classifier — determinism");
  {
    hotpath::WorkloadClassifierInput input;
    input.median_prompt_tokens = 2000;
    input.median_output_tokens = 1000;
    input.phase.prefill_fraction = 0.4;
    auto r1 = hotpath::classify_workload(input);
    auto r2 = hotpath::classify_workload(input);
    check(r1.primary_class == r2.primary_class, "det class");
    check(r1.prefill_fraction == r2.prefill_fraction, "det frac");
    check(r1.prefill_contention == r2.prefill_contention, "det contention");
  }

  section("Workload Classifier — contention never negative (p99 < median from noise)");
  {
    // With small samples, p99 can come out below median due to measurement noise.
    // Negative contention would make blocking_factor > 1.0 and disagg throughput
    // exceed physical max. Must be clamped to 0.
    hotpath::WorkloadClassifierInput input;
    input.median_prompt_tokens = 1024;
    input.median_decode_latency_us = 5000;
    input.p99_decode_latency_us = 4000;  // p99 < median — impossible in theory, real with noise
    auto r = hotpath::classify_workload(input);
    check(r.prefill_contention >= 0.0,
          "contention must be clamped to >= 0 even when p99 < median");
    check_near(r.prefill_contention, 0.0, 1e-9,
               "noisy p99 < median → contention clamped to exactly 0.0");
  }

  section("Workload Classifier — zero contention does not break disagg model");
  {
    // Zero contention (normal for light load) should produce valid disagg outputs,
    // not divide-by-zero or negative values.
    hotpath::WorkloadClassifierInput input;
    input.median_prompt_tokens = 2048;
    input.median_output_tokens = 256;
    input.median_decode_latency_us = 5000;
    input.p99_decode_latency_us = 5000;  // exactly equal → contention = 0
    input.request_rate = 5.0;
    auto profile = hotpath::classify_workload(input);
    check_near(profile.prefill_contention, 0.0, 1e-9, "equal p99/median → zero contention");

    hotpath::DisaggModelInput di;
    di.profile = profile;
    di.total_gpus = 4;
    di.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(di);
    check(r.mono_throughput_rps > 0.0, "zero contention: positive mono throughput");
    check(r.mono_p99_ttft_ms > 0.0, "zero contention: positive TTFT");
    check(!std::isnan(r.mono_throughput_rps), "zero contention: no NaN");
    check(!std::isnan(r.disagg_throughput_rps), "zero contention: no disagg NaN");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 6: Disagg Model
// ─────────────────────────────────────────────────────────────────────

void test_disagg_model() {
  section("Disagg Model — GPU split sum");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 2048;
    input.profile.median_output_tokens = 256;
    input.profile.request_rate = 10.0;
    input.profile.prefill_contention = 1.0;
    input.total_gpus = 8;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);
    check(r.optimal_prefill_gpus + r.optimal_decode_gpus == 8,
          "GPU split sums to total");
    check(r.optimal_prefill_gpus >= 1, "prefill >= 1");
    check(r.optimal_decode_gpus >= 1, "decode >= 1");
  }

  section("Disagg Model — kv_transfer > 0 always");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 4096;
    input.profile.median_output_tokens = 64;
    input.profile.request_rate = 10.0;
    input.profile.prefill_contention = 1.5;
    input.total_gpus = 8;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);
    check(r.kv_transfer_overhead_ms > 0.0, "kv_transfer > 0");
  }

  section("Disagg Model — disagg throughput > mono when should_disaggregate");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 4096;
    input.profile.median_output_tokens = 256;
    input.profile.request_rate = 10.0;
    input.profile.prefill_contention = 1.5;
    input.total_gpus = 8;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);
    if (r.should_disaggregate) {
      check(r.disagg_throughput_rps > r.mono_throughput_rps,
            "disagg throughput > mono when recommending disagg");
      check(r.throughput_improvement > 1.0,
            "throughput_improvement > 1.0 when disagg recommended");
    }
  }

  section("Disagg Model — all outputs non-negative");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 1024;
    input.profile.median_output_tokens = 128;
    input.profile.request_rate = 20.0;
    input.profile.prefill_contention = 0.8;
    input.total_gpus = 8;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);
    check(r.mono_throughput_rps >= 0.0, "mono_throughput >= 0");
    check(r.mono_p99_ttft_ms >= 0.0, "mono_p99_ttft >= 0");
    check(r.mono_p99_itl_ms >= 0.0, "mono_p99_itl >= 0");
    check(r.disagg_throughput_rps >= 0.0, "disagg_throughput >= 0");
    check(r.disagg_p99_ttft_ms >= 0.0, "disagg_p99_ttft >= 0");
    check(r.disagg_p99_itl_ms >= 0.0, "disagg_p99_itl >= 0");
    check(r.kv_transfer_overhead_ms >= 0.0, "kv_transfer >= 0");
    check(r.min_bandwidth_gbps >= 0.0, "min_bandwidth >= 0");
    check(!r.reason.empty(), "reason not empty");
  }

  section("Disagg Model — zero prompt tokens (no crash)");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 0;
    input.profile.median_output_tokens = 100;
    input.profile.request_rate = 10.0;
    input.total_gpus = 4;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);
    // Should not crash or produce NaN/Inf
    check(!std::isnan(r.mono_throughput_rps), "zero prompt: no NaN throughput");
    check(!std::isinf(r.mono_throughput_rps) || r.mono_throughput_rps > 0,
          "zero prompt: throughput finite or +inf");
    check(r.optimal_prefill_gpus + r.optimal_decode_gpus == 4,
          "zero prompt: GPU split still valid");
    check(!r.should_disaggregate, "zero prompt: should not disaggregate");
  }

  section("Disagg Model — zero output tokens (no crash)");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 2048;
    input.profile.median_output_tokens = 0;
    input.profile.request_rate = 10.0;
    input.profile.prefill_contention = 1.0;
    input.total_gpus = 4;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);
    check(!std::isnan(r.mono_throughput_rps), "zero output: no NaN");
    check(!std::isnan(r.disagg_throughput_rps), "zero output: no NaN disagg");
    check(r.optimal_prefill_gpus + r.optimal_decode_gpus == 4,
          "zero output: GPU split valid");
  }

  section("Disagg Model — both zero tokens");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 0;
    input.profile.median_output_tokens = 0;
    input.profile.request_rate = 10.0;
    input.total_gpus = 4;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);
    check(!std::isnan(r.mono_throughput_rps), "both zero: no NaN");
    check(!std::isnan(r.disagg_throughput_rps), "both zero: no NaN disagg");
    check(!r.should_disaggregate, "both zero: no disagg");
  }

  section("Disagg Model — 1 GPU (cannot disaggregate)");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 4096;
    input.profile.median_output_tokens = 256;
    input.profile.request_rate = 10.0;
    input.profile.prefill_contention = 2.0;
    input.total_gpus = 1;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);
    check(!r.should_disaggregate, "1 GPU: cannot disaggregate");
    check(r.optimal_prefill_gpus == 0, "1 GPU: prefill gpus = 0");
    check(r.optimal_decode_gpus == 0, "1 GPU: decode gpus = 0");
    check(contains(r.reason, "2 GPUs"), "1 GPU: reason mentions 2 GPUs");
    check(r.mono_throughput_rps > 0, "1 GPU: mono throughput still computed");
    check(!std::isnan(r.mono_throughput_rps), "1 GPU: no NaN");
  }

  section("Disagg Model — 0 GPUs (degenerate)");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 1024;
    input.profile.median_output_tokens = 128;
    input.profile.request_rate = 5.0;
    input.total_gpus = 0;
    auto r = hotpath::estimate_disaggregation(input);
    check(!r.should_disaggregate, "0 GPU: cannot disaggregate");
    check(r.optimal_prefill_gpus == 0, "0 GPU: prefill = 0");
    check(r.optimal_decode_gpus == 0, "0 GPU: decode = 0");
    check(!std::isnan(r.mono_throughput_rps), "0 GPU: no NaN");
  }

  section("Disagg Model — multi-node cluster (cross-node bandwidth)");
  {
    // 8 GPUs across 2 nodes. Intra-node NVLink is 400Gbps but cross-node
    // is only 25Gbps (typical Infiniband). KV transfer crosses nodes.
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 4096;
    input.profile.median_output_tokens = 256;
    input.profile.request_rate = 10.0;
    input.profile.prefill_contention = 1.5;
    input.total_gpus = 8;
    input.network_bandwidth_gbps = 400.0;  // intra-node
    input.num_nodes = 2;
    input.cross_node_bandwidth_gbps = 25.0;  // inter-node

    auto r = hotpath::estimate_disaggregation(input);
    // KV transfer should use the cross-node bandwidth (25Gbps), not 400Gbps
    // kv_bytes = 4096 * 256 * 32 = 33554432
    // overhead = 33554432 / (25 * 1e9/8) * 1000 = 33554432 / 3125000000 * 1000 = 10.74ms
    double expected_kv = (4096.0 * 256.0 * 32.0) / (25.0 * 1e9 / 8.0) * 1000.0;
    check_near(r.kv_transfer_overhead_ms, expected_kv, 1e-3,
               "multi-node: kv uses cross-node bandwidth");

    // Compare: same cluster but single-node (400Gbps)
    hotpath::DisaggModelInput input_fast = input;
    input_fast.num_nodes = 1;
    input_fast.cross_node_bandwidth_gbps = 0.0;
    auto r_fast = hotpath::estimate_disaggregation(input_fast);
    check(r.kv_transfer_overhead_ms > r_fast.kv_transfer_overhead_ms,
          "multi-node: cross-node is slower than intra-node");
  }

  section("Disagg Model — 2 GPUs (minimum viable split is 1:1)");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 4096;
    input.profile.median_output_tokens = 256;
    input.profile.request_rate = 5.0;
    input.profile.prefill_contention = 1.0;
    input.total_gpus = 2;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);
    check(r.optimal_prefill_gpus == 1, "2 GPU: prefill = 1");
    check(r.optimal_decode_gpus == 1, "2 GPU: decode = 1");
  }

  section("Disagg Model — exact mono throughput with blocking");
  {
    // Hand-compute: prompt=2000 → prefill=20ms, output=100 → decode=500ms
    // mono_service = 520ms. 4 GPUs. mu = 4/520*1000 = 7.692...
    // contention=1.0. blocking = 1/(1+1.0*0.3) = 1/1.3 = 0.76923...
    // mono_throughput = 7.6923 * 0.76923 = 5.917...
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 2000;
    input.profile.median_output_tokens = 100;
    input.profile.request_rate = 2.0;
    input.profile.prefill_contention = 1.0;
    input.total_gpus = 4;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);

    const double prefill_ms = 2000 * 0.01;  // 20
    const double decode_ms = 100 * 5.0;      // 500
    const double service = prefill_ms + decode_ms;  // 520
    const double mu = 4.0 / service * 1000.0;  // 7.6923...
    const double bf = 1.0 / (1.0 + 1.0 * 0.3);  // 0.76923...
    const double expected_mono = mu * bf;
    check_near(r.mono_throughput_rps, expected_mono, 1e-6,
               "exact mono throughput with blocking");

    // ttft = prefill * (1 + contention) = 20 * 2 = 40
    check_near(r.mono_p99_ttft_ms, 40.0, 1e-6, "exact mono p99 ttft = 40");
    // itl = 5 * (1 + 1.0*0.5) = 5 * 1.5 = 7.5
    check_near(r.mono_p99_itl_ms, 7.5, 1e-6, "exact mono p99 itl = 7.5");
  }

  section("Disagg Model — exact kv_transfer calculation");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 1000;
    input.profile.median_output_tokens = 100;
    input.profile.request_rate = 5.0;
    input.total_gpus = 4;
    input.network_bandwidth_gbps = 100.0;
    input.avg_kv_transfer_bytes = 1000000.0;  // 1 MB
    auto r = hotpath::estimate_disaggregation(input);
    // kv_overhead = 1000000 / (100 * 1e9/8) * 1000
    //            = 1000000 / 12500000000 * 1000
    //            = 0.00008 * 1000 = 0.08 ms
    double expected_kv = (1000000.0 / (100.0 * 1e9 / 8.0)) * 1000.0;
    check_near(r.kv_transfer_overhead_ms, expected_kv, 1e-6, "exact kv_transfer");
  }

  section("Disagg Model — determinism");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 2048;
    input.profile.median_output_tokens = 256;
    input.profile.request_rate = 10.0;
    input.profile.prefill_contention = 1.0;
    input.total_gpus = 8;
    input.network_bandwidth_gbps = 100.0;
    auto r1 = hotpath::estimate_disaggregation(input);
    auto r2 = hotpath::estimate_disaggregation(input);
    check(r1.mono_throughput_rps == r2.mono_throughput_rps, "det mono");
    check(r1.disagg_throughput_rps == r2.disagg_throughput_rps, "det disagg");
    check(r1.optimal_prefill_gpus == r2.optimal_prefill_gpus, "det p_gpus");
    check(r1.optimal_decode_gpus == r2.optimal_decode_gpus, "det d_gpus");
    check(r1.should_disaggregate == r2.should_disaggregate, "det should");
    check(r1.kv_transfer_overhead_ms == r2.kv_transfer_overhead_ms, "det kv");
  }

  section("Disagg Model — measured_prefill_p99 overrides token-count model for TTFT");
  {
    // Without measured: mono_p99_ttft = prefill_time * (1 + contention) = 1000*0.01*(1+1) = 20ms
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 1000;
    input.profile.median_output_tokens = 128;
    input.profile.prefill_contention = 1.0;
    input.total_gpus = 4;
    input.network_bandwidth_gbps = 100.0;
    auto r_no_measured = hotpath::estimate_disaggregation(input);
    check_near(r_no_measured.mono_p99_ttft_ms, 1000 * 0.01 * (1.0 + 1.0), 1e-6,
               "without measured: mono_p99_ttft = prefill_time*(1+contention)");

    // With measured_prefill_p99=75ms: must override model
    input.measured_prefill_p99_ms = 75.0;
    auto r_measured = hotpath::estimate_disaggregation(input);
    check_near(r_measured.mono_p99_ttft_ms, 75.0, 1e-6,
               "measured_prefill_p99 overrides mono_p99_ttft exactly");
    // disagg_p99_ttft = 75ms + kv_overhead
    check_near(r_measured.disagg_p99_ttft_ms,
               75.0 + r_measured.kv_transfer_overhead_ms, 1e-6,
               "measured_prefill_p99 is basis for disagg_p99_ttft");
    // throughput model is UNCHANGED (p99 not used there)
    check_near(r_measured.mono_throughput_rps, r_no_measured.mono_throughput_rps, 1e-6,
               "measured_prefill_p99 does not change throughput model");
    check_near(r_measured.disagg_throughput_rps, r_no_measured.disagg_throughput_rps, 1e-6,
               "measured_prefill_p99 does not change disagg throughput");
  }

  section("Disagg Model — measured_prefill_p99 = -1 (default) uses model estimate");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 2000;
    input.profile.prefill_contention = 0.5;
    input.total_gpus = 4;
    input.network_bandwidth_gbps = 100.0;
    // measured_prefill_p99_ms defaults to -1
    auto r = hotpath::estimate_disaggregation(input);
    const double expected = 2000 * 0.01 * (1.0 + 0.5);
    check_near(r.mono_p99_ttft_ms, expected, 1e-6,
               "default measured=-1 → model estimate");
  }

  section("Disagg Model — measured_prefill_p99 = 0.0 is a valid measurement (fully cached)");
  {
    // 0.0ms is a legitimate measured value (e.g. fully cached workload with near-instant prefill).
    // Only -1.0 is the sentinel for "no measurement available".
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 1000;
    input.profile.prefill_contention = 0.5;
    input.total_gpus = 4;
    input.network_bandwidth_gbps = 100.0;
    input.measured_prefill_p99_ms = 0.0;  // valid measured value: near-instant prefill
    auto r = hotpath::estimate_disaggregation(input);
    check_near(r.mono_p99_ttft_ms, 0.0, 1e-6,
               "measured_prefill_p99=0.0 is valid → used directly, not overridden by model");
    // disagg_p99_ttft = 0.0 + kv_overhead
    check_near(r.disagg_p99_ttft_ms, 0.0 + r.kv_transfer_overhead_ms, 1e-6,
               "measured_prefill_p99=0.0 → disagg_ttft = 0 + kv_overhead");
  }

  section("Disagg Model — measured_prefill_p99 very small: disagg_p99 = measured + kv");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 4096;
    input.profile.median_output_tokens = 64;
    input.profile.request_rate = 5.0;
    input.total_gpus = 8;
    input.network_bandwidth_gbps = 100.0;
    input.measured_prefill_p99_ms = 0.5;  // very fast prefill (cached?)
    auto r = hotpath::estimate_disaggregation(input);
    check_near(r.mono_p99_ttft_ms, 0.5, 1e-6,
               "very small measured_prefill_p99 propagates correctly");
    check(r.disagg_p99_ttft_ms >= 0.5,
          "disagg_p99_ttft >= measured_prefill (kv overhead is non-negative)");
    check_near(r.disagg_p99_ttft_ms, 0.5 + r.kv_transfer_overhead_ms, 1e-6,
               "disagg_p99_ttft = 0.5 + kv_overhead");
  }

  section("Disagg Model — measured overrides but kv_overhead still uses token model");
  {
    // kv_transfer_overhead depends only on kv_bytes and bandwidth, not on measured_prefill
    hotpath::DisaggModelInput base;
    base.profile.median_prompt_tokens = 2048;
    base.total_gpus = 4;
    base.network_bandwidth_gbps = 100.0;
    auto r_base = hotpath::estimate_disaggregation(base);

    hotpath::DisaggModelInput with_m = base;
    with_m.measured_prefill_p99_ms = 999.0;
    auto r_m = hotpath::estimate_disaggregation(with_m);

    check_near(r_base.kv_transfer_overhead_ms, r_m.kv_transfer_overhead_ms, 1e-9,
               "kv_transfer unchanged when measured_prefill_p99 is set");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 7: Recommender
// ─────────────────────────────────────────────────────────────────────

void test_recommender() {
  section("Recommender — no kv-connector when monolithic");
  {
    hotpath::DisaggEstimate est;
    est.should_disaggregate = false;
    est.optimal_prefill_gpus = 2;
    est.optimal_decode_gpus = 6;
    est.reason = "test";
    auto cfg = hotpath::generate_vllm_config(est, "test-model");
    check(!contains(cfg, "kv-connector"), "mono: no kv-connector");
    check(!contains(cfg, "kv_producer"), "mono: no kv_producer");
    check(!contains(cfg, "kv_consumer"), "mono: no kv_consumer");
    check(contains(cfg, "Monolithic"), "mono: says monolithic");
  }

  section("Recommender — disagg config has connectors");
  {
    hotpath::DisaggEstimate est;
    est.should_disaggregate = true;
    est.optimal_prefill_gpus = 2;
    est.optimal_decode_gpus = 6;
    est.throughput_improvement = 1.3;
    est.min_bandwidth_gbps = 50.0;
    est.reason = "test";
    auto cfg = hotpath::generate_vllm_config(est, "test-model");
    check(contains(cfg, "kv_producer"), "disagg: has kv_producer");
    check(contains(cfg, "kv_consumer"), "disagg: has kv_consumer");
  }

  section("Recommender — model name with special chars");
  {
    hotpath::DisaggEstimate est;
    est.should_disaggregate = true;
    est.optimal_prefill_gpus = 1;
    est.optimal_decode_gpus = 3;
    est.throughput_improvement = 1.2;
    est.min_bandwidth_gbps = 30.0;
    est.reason = "r";
    // Model names in practice contain slashes
    auto cfg = hotpath::generate_vllm_config(est, "meta-llama/Llama-3.1-70B-Instruct");
    check(contains(cfg, "meta-llama/Llama-3.1-70B-Instruct"),
          "special chars: model name preserved in vllm config");
    auto llmd = hotpath::generate_llmd_config(est, "meta-llama/Llama-3.1-70B-Instruct");
    check(contains(llmd, "meta-llama/Llama-3.1-70B-Instruct"),
          "special chars: model name in llmd config");
  }

  section("Recommender — 1-GPU config");
  {
    // Simulates what a 1-GPU user gets after disagg model says no
    hotpath::DisaggEstimate est;
    est.should_disaggregate = false;
    est.optimal_prefill_gpus = 0;
    est.optimal_decode_gpus = 0;
    est.reason = "Disaggregation requires at least 2 GPUs (have 1).";
    auto cfg = hotpath::generate_vllm_config(est, "Qwen/Qwen3-8B");
    check(contains(cfg, "tensor-parallel-size 1"),
          "1-GPU: config uses TP=1 not TP=0");
    check(!contains(cfg, "kv_producer"),
          "1-GPU: no disagg arguments");
    auto llmd = hotpath::generate_llmd_config(est, "Qwen/Qwen3-8B");
    check(contains(llmd, "replicas: 1"),
          "1-GPU: llmd uses 1 replica not 0");
    auto dynamo = hotpath::generate_dynamo_config(est, "Qwen/Qwen3-8B");
    check(contains(dynamo, "replicas: 1"),
          "1-GPU: dynamo uses 1 replica not 0");
  }

  section("Recommender — monolithic: GPU count in config");
  {
    hotpath::DisaggEstimate est;
    est.should_disaggregate = false;
    est.optimal_prefill_gpus = 3;
    est.optimal_decode_gpus = 5;
    est.reason = "test";
    // Monolithic should use total GPUs (3+5=8)
    auto cfg = hotpath::generate_vllm_config(est, "m");
    check(contains(cfg, "8"), "mono: total GPU count in config");
    auto llmd = hotpath::generate_llmd_config(est, "m");
    check(contains(llmd, "8"), "mono: total GPU count in llmd");
  }

  section("Recommender — determinism");
  {
    hotpath::DisaggEstimate est;
    est.should_disaggregate = true;
    est.optimal_prefill_gpus = 2;
    est.optimal_decode_gpus = 6;
    est.throughput_improvement = 1.3;
    est.min_bandwidth_gbps = 50.0;
    est.reason = "r";
    auto a = hotpath::generate_vllm_config(est, "m");
    auto b = hotpath::generate_vllm_config(est, "m");
    check(a == b, "det vllm config");
    auto c = hotpath::generate_llmd_config(est, "m");
    auto d = hotpath::generate_llmd_config(est, "m");
    check(c == d, "det llmd config");
    auto e = hotpath::generate_dynamo_config(est, "m");
    auto f = hotpath::generate_dynamo_config(est, "m");
    check(e == f, "det dynamo config");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 8: OTLP Export
// ─────────────────────────────────────────────────────────────────────

void test_otlp_export() {
  section("OTLP Export — empty");
  {
    auto json = hotpath::export_otlp_json({});
    check(contains(json, "resourceSpans"), "empty has structure");
  }

  section("OTLP Export — single trace span count");
  {
    hotpath::RequestTrace t;
    t.request_id = "r1";
    t.arrival_us = 1000;
    t.prefill_start_us = 2000;
    t.prefill_end_us = 3000;
    t.first_token_us = 3500;
    t.last_token_us = 5000;
    t.completion_us = 5500;
    t.prompt_tokens = 100;
    t.output_tokens = 50;
    auto json = hotpath::export_otlp_json({t});
    // 1 root + 1 queue + 1 prefill + 1 decode = 4 spans
    int spans = 0;
    size_t pos = 0;
    while ((pos = json.find("\"spanId\"", pos)) != std::string::npos) {
      ++spans; ++pos;
    }
    check(spans == 4, "single trace: 4 spans");
  }

  section("OTLP Export — determinism (bitwise identical)");
  {
    hotpath::RequestTrace t;
    t.request_id = "r1";
    t.arrival_us = 1000;
    t.prefill_start_us = 2000;
    t.prefill_end_us = 3000;
    t.first_token_us = 3500;
    t.last_token_us = 5000;
    t.completion_us = 5500;
    auto json1 = hotpath::export_otlp_json({t});
    auto json2 = hotpath::export_otlp_json({t});
    check(json1 == json2, "OTLP export is deterministic (bitwise)");
  }

  section("OTLP Export — completion before arrival (nonsensical)");
  {
    hotpath::RequestTrace t;
    t.request_id = "nonsense";
    t.arrival_us = 5000;
    t.completion_us = 1000;  // before arrival
    t.prefill_start_us = 3000;
    t.prefill_end_us = 2000;  // end before start
    t.first_token_us = 4000;
    t.last_token_us = 3000;   // end before start
    // Module should not crash — just produce spans with inverted times
    auto json = hotpath::export_otlp_json({t});
    check(!json.empty(), "nonsensical timestamps: no crash");
    check(contains(json, "\"spanId\""), "nonsensical: still produces spans");
  }

  section("OTLP Export — missing fields produce fewer spans");
  {
    hotpath::RequestTrace t;
    t.request_id = "r1";
    t.arrival_us = 1000;
    t.completion_us = 5000;
    // All other timestamps are 0 → no queue, prefill, decode spans
    auto json = hotpath::export_otlp_json({t});
    int spans = 0;
    size_t pos = 0;
    while ((pos = json.find("\"spanId\"", pos)) != std::string::npos) {
      ++spans; ++pos;
    }
    check(spans == 1, "missing fields: only root span");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 9: Serve Report
// ─────────────────────────────────────────────────────────────────────

void test_serve_report() {
  section("Serve Report — all sections present");
  {
    hotpath::ServeReportData d;
    d.model_name = "test-model";
    d.engine = "vllm";
    d.gpu_info = "1x A100";
    d.total_requests = 100;
    d.duration_seconds = 10.0;
    d.throughput_rps = 10.0;
    d.queue_wait_available = true;
    d.server_timing_available = true;
    d.queue_p50 = 1.0; d.queue_p90 = 5.0; d.queue_p99 = 20.0;
    d.server_prefill_p50 = 10.0; d.server_prefill_p90 = 30.0; d.server_prefill_p99 = 80.0;
    d.server_decode_p50 = 100.0; d.server_decode_p90 = 200.0; d.server_decode_p99 = 500.0;
    d.prefill_p50 = 10.0; d.prefill_p90 = 30.0; d.prefill_p99 = 80.0;
    d.decode_total_p50 = 100.0; d.decode_total_p90 = 200.0; d.decode_total_p99 = 500.0;
    d.decode_per_token_p50 = 3.0; d.decode_per_token_p90 = 4.0; d.decode_per_token_p99 = 6.0;
    d.e2e_p50 = 120.0; d.e2e_p90 = 240.0; d.e2e_p99 = 600.0;
    d.prefill_compute_pct = 40.0;
    d.decode_compute_pct = 45.0;
    d.other_idle_pct = 15.0;
    d.cache_hit_rate_available = true;
    d.cache_histogram_available = true;
    d.cache_hit_rate = 0.35;
    d.avg_cache_usage = 60.0;
    d.evictions = 5;
    d.cache_hit_rate_histogram = {1, 2, 3, 4, 5};
    d.unique_prefixes = 10;
    d.cacheable_tokens_pct = 50.0;
    d.should_disaggregate = true;
    d.optimal_p = 2; d.optimal_d = 6;
    d.projected_throughput_pct = 30.0;
    d.projected_throughput_rps = 13.0;
    d.mono_p99_ttft = 80.0;
    d.disagg_p99_ttft = 55.0;
    d.min_bandwidth_gbps = 40.0;

    auto report = hotpath::render_serve_report(d);

    check(contains(report, "Latency"), "report has Latency section");
    check(contains(report, "GPU Phase"), "report has GPU Phase section");
    check(contains(report, "KV Cache"), "report has KV Cache section");
    check(contains(report, "Prefix Sharing"), "report has Prefix Sharing section");
    check(contains(report, "Disaggregation"), "report has Disaggregation section");
    check(contains(report, "test-model"), "report has model name");
    check(contains(report, "vllm"), "report has engine");
    check(contains(report, "100"), "report has request count");
    check(contains(report, "Prefill (server)"), "report has server prefill row");
    check(contains(report, "Decode (server)"), "report has server decode row");
    check(contains(report, "Hit histogram"), "report has cache histogram");
    check(contains(report, "DISAGGREGATE"), "report has recommendation");
    check(contains(report, "2:6"), "report has P:D ratio");
  }

  section("Serve Report — monolithic recommendation");
  {
    hotpath::ServeReportData d;
    d.model_name = "m";
    d.engine = "e";
    d.gpu_info = "g";
    d.should_disaggregate = false;
    auto report = hotpath::render_serve_report(d);
    check(contains(report, "MONOLITHIC"), "mono report says MONOLITHIC");
    check(!contains(report, "DISAGGREGATE"), "mono report does not say DISAGGREGATE");
  }

  section("Serve Report — determinism");
  {
    hotpath::ServeReportData d;
    d.model_name = "det";
    d.engine = "e";
    d.gpu_info = "g";
    d.total_requests = 42;
    d.throughput_rps = 4.2;
    d.prefill_compute_pct = 30.0;
    d.decode_compute_pct = 50.0;
    d.other_idle_pct = 20.0;
    d.should_disaggregate = false;
    auto r1 = hotpath::render_serve_report(d);
    auto r2 = hotpath::render_serve_report(d);
    check(r1 == r2, "serve report determinism (bitwise)");
  }

  section("Serve Report — zero values don't crash");
  {
    hotpath::ServeReportData d{};
    auto report = hotpath::render_serve_report(d);
    check(!report.empty(), "zero-valued report is not empty");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 10: Cross-Module Integration
// ─────────────────────────────────────────────────────────────────────

void test_integration() {
  section("Cross-Module Integration — full pipeline");

  // ── Build synthetic workload ──
  // 50 requests, 4096 prompt tokens, 64 output tokens
  // 20 have 200 cached tokens, 30 have 0
  std::vector<hotpath::RequestTrace> traces;
  for (int i = 0; i < 50; ++i) {
    hotpath::RequestTrace t;
    t.request_id = "req_" + std::to_string(i);
    t.prompt_tokens = 4096;
    t.output_tokens = 64;
    t.cached_tokens = (i < 20) ? 200 : 0;
    traces.push_back(t);
  }

  // Kernel timings: 60% prefill, 30% decode, 10% unknown over 5 seconds
  std::vector<hotpath::KernelEntry> kernels;
  // 30 prefill kernels of 10000 us each = 300000 us
  for (int i = 0; i < 30; ++i) {
    kernels.push_back({"flash_fwd_" + std::to_string(i),
                        hotpath::profiler::KernelPhase::PREFILL,
                        static_cast<int64_t>(i) * 100000,
                        10000});
  }
  // 15 decode kernels of 10000 us each = 150000 us
  for (int i = 0; i < 15; ++i) {
    kernels.push_back({"paged_attn_" + std::to_string(i),
                        hotpath::profiler::KernelPhase::DECODE,
                        3000000 + static_cast<int64_t>(i) * 100000,
                        10000});
  }
  // 5 unknown kernels of 10000 us each = 50000 us
  for (int i = 0; i < 5; ++i) {
    kernels.push_back({"other_" + std::to_string(i),
                        hotpath::profiler::KernelPhase::UNKNOWN,
                        4500000 + static_cast<int64_t>(i) * 100000,
                        10000});
  }

  // Metric snapshots: 10 seconds, batch_size=16, cache_usage=70%
  std::vector<hotpath::MetricSnapshot> snapshots;
  for (int i = 0; i < 10; ++i) {
    snapshots.push_back({static_cast<int64_t>(i) * 1000000,
                          16.0, 2.0, 0.0, 70.0});
  }

  // Prefix: 10 share a 20-token prefix, 40 are unique (low sharing → not CACHE_FRIENDLY)
  std::vector<std::vector<int>> prompts;
  for (int i = 0; i < 10; ++i) {
    std::vector<int> p;
    for (int t = 0; t < 20; ++t) p.push_back(t + 1);  // shared prefix
    for (int t = 0; t < 100; ++t) p.push_back(1000 + i * 200 + t);  // unique
    prompts.push_back(std::move(p));
  }
  for (int i = 0; i < 40; ++i) {
    std::vector<int> p;
    for (int t = 0; t < 120; ++t) p.push_back(5000 + i * 200 + t);  // fully unique
    prompts.push_back(std::move(p));
  }

  // ── Run all modules ──

  auto phase = hotpath::analyze_phases(kernels);
  check(phase.breakdown.prefill_us == 300000, "integration: prefill_us");
  check(phase.breakdown.decode_us == 150000, "integration: decode_us");
  check(phase.breakdown.unknown_us == 50000, "integration: unknown_us");
  check(phase.breakdown.total_us == 500000, "integration: total_us");
  check_near(phase.breakdown.prefill_fraction, 0.6, 1e-9, "integration: prefill_frac = 0.6");
  check_near(phase.breakdown.decode_fraction, 0.3, 1e-9, "integration: decode_frac = 0.3");

  auto batch = hotpath::analyze_batches(snapshots);
  check_near(batch.avg_batch_size, 16.0, 1e-9, "integration: avg_batch = 16");

  auto cache = hotpath::analyze_cache(traces, snapshots);
  // hit_rate = 20*200 / 50*4096 = 4000/204800
  double expected_hit_rate = 4000.0 / 204800.0;
  check_near(cache.cache_hit_rate, expected_hit_rate, 1e-9, "integration: cache_hit_rate");
  int hsum = 0;
  for (int b : cache.hit_rate_histogram) hsum += b;
  check(hsum == 50, "integration: histogram sums to 50");

  auto prefix = hotpath::analyze_prefixes(prompts);
  check(prefix.total_requests == 50, "integration: prefix total");
  // Low sharing: 10*20 / (10*120 + 40*120) = 200/6000 = 0.033
  check(prefix.cacheable_token_fraction < 0.6, "integration: low cacheability (not CACHE_FRIENDLY)");

  // Build classifier input
  hotpath::WorkloadClassifierInput ci;
  ci.phase = phase.breakdown;
  ci.batch = batch;
  ci.cache = cache;
  ci.prefix = prefix;
  ci.median_prompt_tokens = 4096;
  ci.median_output_tokens = 64;
  ci.request_rate = 50.0 / 10.0;  // 50 requests in 10 seconds
  ci.median_decode_latency_us = 5000;
  ci.p99_decode_latency_us = 15000;

  auto profile = hotpath::classify_workload(ci);
  check(profile.primary_class == hotpath::WorkloadClass::PREFILL_HEAVY,
        "integration: PREFILL_HEAVY (4096 prompt, 64 output, prefill_frac=0.6)");
  check(profile.prefill_fraction > 0.5,
        "integration: PREFILL_HEAVY has prefill_fraction > 0.5");

  // Verify numbers flow correctly: classifier → model
  check_near(profile.cache_hit_rate, cache.cache_hit_rate, 1e-9,
             "integration: cache_hit flows to profile");
  check_near(profile.prefix_sharing_rate, prefix.cacheable_token_fraction, 1e-9,
             "integration: prefix sharing flows to profile");

  // Disagg model
  hotpath::DisaggModelInput di;
  di.profile = profile;
  di.total_gpus = 8;
  di.network_bandwidth_gbps = 100.0;

  auto disagg = hotpath::estimate_disaggregation(di);
  check(disagg.optimal_prefill_gpus + disagg.optimal_decode_gpus == 8,
        "integration: GPU split sums to 8");
  check(disagg.mono_throughput_rps > 0, "integration: mono throughput > 0");
  check(disagg.disagg_throughput_rps > 0, "integration: disagg throughput > 0");
  if (disagg.should_disaggregate) {
    check(disagg.disagg_throughput_rps > disagg.mono_throughput_rps,
          "integration: disagg > mono when recommended");
  }
  check(!disagg.reason.empty(), "integration: reason not empty");

  // Recommender
  auto vllm_cfg = hotpath::generate_vllm_config(disagg, "meta-llama/Llama-3-70B");
  check(!vllm_cfg.empty(), "integration: vllm config not empty");
  auto summary = hotpath::generate_summary(disagg, "meta-llama/Llama-3-70B");
  check(!summary.empty(), "integration: summary not empty");

  if (disagg.should_disaggregate) {
    check(contains(vllm_cfg, "kv_producer"), "integration: disagg config has producer");
    check(contains(summary, "DISAGGREGATE"), "integration: summary says disaggregate");
  } else {
    check(!contains(vllm_cfg, "kv_producer"), "integration: mono config no producer");
    check(contains(summary, "MONOLITHIC"), "integration: summary says monolithic");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 11: Phase Analyzer — more scenarios
// ─────────────────────────────────────────────────────────────────────

void test_phase_analyzer_extra() {
  using hotpath::KernelEntry;
  using hotpath::profiler::KernelPhase;

  section("Phase Analyzer — only UNKNOWN kernels");
  {
    std::vector<KernelEntry> k = {
        {"u1", KernelPhase::UNKNOWN, 0, 1000},
        {"u2", KernelPhase::UNKNOWN, 1000, 2000},
    };
    auto r = hotpath::analyze_phases(k);
    check(r.breakdown.unknown_us == 3000, "all-unknown: unknown=3000");
    check(r.breakdown.prefill_us == 0, "all-unknown: prefill=0");
    check(r.breakdown.decode_us == 0, "all-unknown: decode=0");
    check_near(r.breakdown.prefill_fraction, 0.0, 1e-9, "all-unknown: pfrac=0");
    check_near(r.breakdown.decode_fraction, 0.0, 1e-9, "all-unknown: dfrac=0");
    check(r.breakdown.prefill_kernel_count == 0, "all-unknown: pcount=0");
    check(r.breakdown.decode_kernel_count == 0, "all-unknown: dcount=0");
  }

  section("Phase Analyzer — out-of-order kernel timestamps");
  {
    // Kernels not sorted by start_us — breakdown should still be correct
    std::vector<KernelEntry> k = {
        {"d", KernelPhase::DECODE, 500, 200},
        {"p", KernelPhase::PREFILL, 0, 300},
        {"u", KernelPhase::UNKNOWN, 250, 100},
    };
    auto r = hotpath::analyze_phases(k);
    check(r.breakdown.prefill_us == 300, "unordered: prefill=300");
    check(r.breakdown.decode_us == 200, "unordered: decode=200");
    check(r.breakdown.unknown_us == 100, "unordered: unknown=100");
    check(r.breakdown.total_us == 600, "unordered: total=600");
  }

  section("Phase Analyzer — sparse kernels with gap windows");
  {
    // Two kernels 3 seconds apart: [0, 100000] and [3000000, 3100000]
    // Windows 0,1,2,3 → windows 1 and 2 have zero GPU activity
    std::vector<KernelEntry> k = {
        {"p", KernelPhase::PREFILL, 0, 100000},
        {"d", KernelPhase::DECODE, 3000000, 100000},
    };
    auto r = hotpath::analyze_phases(k);
    // Spans from 0 to 3100000 → 4 windows
    check(r.time_series.size() == 4, "sparse: 4 windows");
    if (r.time_series.size() == 4) {
      // Window 0: prefill in [0,100000], total=100000
      check_near(r.time_series[0].prefill_fraction, 1.0, 1e-9, "sparse: w0 all prefill");
      // Windows 1,2: no activity → fractions 0/0
      check_near(r.time_series[1].prefill_fraction, 0.0, 1e-9, "sparse: w1 empty");
      check_near(r.time_series[1].decode_fraction, 0.0, 1e-9, "sparse: w1 empty dec");
      check_near(r.time_series[2].prefill_fraction, 0.0, 1e-9, "sparse: w2 empty");
      // Window 3: decode in [3000000,3100000], total=100000
      check_near(r.time_series[3].decode_fraction, 1.0, 1e-9, "sparse: w3 all decode");
    }
  }

  section("Phase Analyzer — kernel spanning window boundary");
  {
    // One prefill kernel from 0 to 1500000 (spans two 1-second windows)
    // min_start=0, max_end=1500000 → windows [0,1000000) and [1000000,2000000)
    std::vector<KernelEntry> k = {{"p", KernelPhase::PREFILL, 0, 1500000}};
    auto r = hotpath::analyze_phases(k);
    check(r.time_series.size() == 2, "boundary: 2 windows");
    if (r.time_series.size() == 2) {
      // Window 0: overlap [0, 1000000) = 1000000 us → prefill=1.0
      // Window 1: overlap [1000000, 1500000) = 500000 us → prefill=1.0
      check_near(r.time_series[0].prefill_fraction, 1.0, 1e-9,
                 "boundary: w0 prefill=1.0");
      check_near(r.time_series[1].prefill_fraction, 1.0, 1e-9,
                 "boundary: w1 prefill=1.0");
    }
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 12: Batch Analyzer — more scenarios
// ─────────────────────────────────────────────────────────────────────

void test_batch_analyzer_extra() {
  section("Batch Analyzer — 3-element exact percentile");
  {
    // [5, 10, 20]
    // p50: idx=0.5*2=1.0 → val[1]=10 exactly
    // p99: idx=0.99*2=1.98 → val[1]*0.02 + val[2]*0.98 = 10*0.02+20*0.98 = 19.8
    std::vector<hotpath::MetricSnapshot> snaps = {
        {0, 5.0, 0, 0, 0}, {1000000, 10.0, 0, 0, 0}, {2000000, 20.0, 0, 0, 0},
    };
    auto r = hotpath::analyze_batches(snaps);
    check_near(r.p50_batch_size, 10.0, 1e-6, "3-elem p50=10");
    check_near(r.p99_batch_size, 19.8, 1e-6, "3-elem p99=19.8");
    check_near(r.avg_batch_size, 35.0/3.0, 1e-9, "3-elem avg=11.667");
  }

  section("Batch Analyzer — 10-element uniform");
  {
    // [1,2,3,...,10]. avg=5.5
    // p50: idx=0.5*9=4.5 → val[4]*0.5+val[5]*0.5 = 5*0.5+6*0.5=5.5
    // p99: idx=0.99*9=8.91 → val[8]*0.09+val[9]*0.91 = 9*0.09+10*0.91=0.81+9.1=9.91
    std::vector<hotpath::MetricSnapshot> snaps;
    for (int i = 1; i <= 10; ++i)
      snaps.push_back({static_cast<int64_t>(i)*1000000, static_cast<double>(i), 0, 0, 0});
    auto r = hotpath::analyze_batches(snaps);
    check_near(r.avg_batch_size, 5.5, 1e-9, "10-elem avg=5.5");
    check_near(r.p50_batch_size, 5.5, 1e-6, "10-elem p50=5.5");
    check_near(r.p99_batch_size, 9.91, 1e-6, "10-elem p99=9.91");
  }

  section("Batch Analyzer — 100-element: p50 is median, p99 near max");
  {
    // [1,2,...,100]. avg=50.5
    // p50: idx=0.5*99=49.5 → val[49]*0.5+val[50]*0.5 = 50*0.5+51*0.5=50.5
    // p99: idx=0.99*99=98.01 → val[98]*0.99+val[99]*0.01 = 99*0.99+100*0.01=98.01+1.0=99.01
    std::vector<hotpath::MetricSnapshot> snaps;
    for (int i = 1; i <= 100; ++i)
      snaps.push_back({static_cast<int64_t>(i)*1000000, static_cast<double>(i), 0, 0, 0});
    auto r = hotpath::analyze_batches(snaps);
    check_near(r.avg_batch_size, 50.5, 1e-9, "100-elem avg=50.5");
    check_near(r.p50_batch_size, 50.5, 1e-6, "100-elem p50=50.5");
    check_near(r.p99_batch_size, 99.01, 1e-6, "100-elem p99=99.01");
  }

  section("Batch Analyzer — all-zero cache usage");
  {
    std::vector<hotpath::MetricSnapshot> snaps(5, {0, 8, 1, 0, 0.0});
    auto r = hotpath::analyze_batches(snaps);
    check_near(r.avg_cache_usage, 0.0, 1e-9, "zero cache avg");
    check_near(r.peak_cache_usage, 0.0, 1e-9, "zero cache peak");
  }

  section("Batch Analyzer — decreasing preemption counter (anomalous)");
  {
    // Preemption goes 10 → 5 → 3 (shouldn't happen but don't crash)
    std::vector<hotpath::MetricSnapshot> snaps = {
        {0, 1, 0, 10.0, 0}, {1000000, 1, 0, 5.0, 0}, {2000000, 1, 0, 3.0, 0},
    };
    auto r = hotpath::analyze_batches(snaps);
    // max=10, min=3, delta=7 (negative preemptions reported as positive delta)
    check(r.total_preemptions == 7, "decreasing preemption: delta still positive");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 13: Cache Analyzer — more scenarios
// ─────────────────────────────────────────────────────────────────────

void test_cache_analyzer_extra() {
  section("Cache Analyzer — all 100% cache hit");
  {
    std::vector<hotpath::RequestTrace> traces;
    for (int i = 0; i < 10; ++i) {
      hotpath::RequestTrace t;
      t.prompt_tokens = 500;
      t.cached_tokens = 500;
      traces.push_back(t);
    }
    auto r = hotpath::analyze_cache(traces, {});
    check_near(r.cache_hit_rate, 1.0, 1e-9, "all-cached: hit_rate=1.0");
    check(r.hit_rate_histogram[4] == 10, "all-cached: all in bucket[4]");
    int s = 0; for (int b : r.hit_rate_histogram) s += b;
    check(s == 10, "all-cached: histogram sum");
  }

  section("Cache Analyzer — all 0% cache hit");
  {
    std::vector<hotpath::RequestTrace> traces;
    for (int i = 0; i < 10; ++i) {
      hotpath::RequestTrace t;
      t.prompt_tokens = 500;
      t.cached_tokens = 0;
      traces.push_back(t);
    }
    auto r = hotpath::analyze_cache(traces, {});
    check_near(r.cache_hit_rate, 0.0, 1e-9, "no-cache: hit_rate=0");
    check(r.hit_rate_histogram[0] == 10, "no-cache: all in bucket[0]");
  }

  section("Cache Analyzer — cache at exactly 90% is NOT pressure");
  {
    // 90.0 is NOT > 90.0
    hotpath::MetricSnapshot s{0, 1, 0, 0, 90.0};
    auto r = hotpath::analyze_cache({}, {s});
    check_near(r.cache_pressure_seconds, 0.0, 1e-9, "exactly 90% = no pressure");
  }

  section("Cache Analyzer — cache at 90.01% IS pressure");
  {
    hotpath::MetricSnapshot s{0, 1, 0, 0, 90.01};
    auto r = hotpath::analyze_cache({}, {s});
    check_near(r.cache_pressure_seconds, 1.0, 1e-9, "90.01% = pressure");
  }

  section("Cache Analyzer — 1000 traces bulk");
  {
    std::vector<hotpath::RequestTrace> traces;
    int64_t total_cached = 0, total_prompt = 0;
    int buckets[5] = {};
    for (int i = 0; i < 1000; ++i) {
      hotpath::RequestTrace t;
      t.prompt_tokens = 1000;
      t.cached_tokens = (i * 7) % 1001;  // deterministic pseudo-random 0..1000
      traces.push_back(t);
      total_cached += t.cached_tokens;
      total_prompt += t.prompt_tokens;
      double frac = static_cast<double>(t.cached_tokens) / 1000.0;
      if (frac <= 0.0) buckets[0]++;
      else if (frac < 0.25) buckets[1]++;
      else if (frac < 0.50) buckets[2]++;
      else if (frac < 0.75) buckets[3]++;
      else buckets[4]++;
    }
    auto r = hotpath::analyze_cache(traces, {});
    check_near(r.cache_hit_rate, static_cast<double>(total_cached) / total_prompt,
               1e-9, "bulk: exact hit rate");
    for (int i = 0; i < 5; ++i)
      check(r.hit_rate_histogram[i] == buckets[i],
            "bulk: bucket[" + std::to_string(i) + "]=" + std::to_string(buckets[i]));
    int s = 0; for (int b : r.hit_rate_histogram) s += b;
    check(s == 1000, "bulk: histogram sum=1000");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 14: Prefix Analyzer — more scenarios
// ─────────────────────────────────────────────────────────────────────

void test_prefix_analyzer_extra() {
  section("Prefix Analyzer — prompts shorter than min_prefix_len");
  {
    // All prompts are 3 tokens, min_prefix_len=4 → no shared prefixes
    std::vector<std::vector<int>> prompts = {
        {1, 2, 3}, {1, 2, 3}, {1, 2, 3},
    };
    auto r = hotpath::analyze_prefixes(prompts, 4);
    // Even though all 3 share [1,2,3], it's below min_prefix_len
    // The trie leaf at depth 3 has count=3 but depth < 4
    check(r.top_prefixes.empty() || r.top_prefixes[0].prefix_length < 4,
          "short prompts: no prefix >= min_prefix_len shared");
  }

  section("Prefix Analyzer — custom min_prefix_len=1");
  {
    std::vector<std::vector<int>> prompts = {
        {1, 100}, {1, 200}, {2, 300},
    };
    // With min_prefix_len=1: [1] is shared by 2 requests
    auto r = hotpath::analyze_prefixes(prompts, 1);
    check(!r.top_prefixes.empty(), "min1: has shared prefix");
    if (!r.top_prefixes.empty()) {
      check(r.top_prefixes[0].prefix_length == 1, "min1: prefix len=1");
      check(r.top_prefixes[0].request_count == 2, "min1: 2 requests");
    }
  }

  section("Prefix Analyzer — three distinct groups");
  {
    std::vector<std::vector<int>> prompts;
    // Group A: 5 prompts sharing [1,2,3,4]
    for (int i = 0; i < 5; ++i) prompts.push_back({1,2,3,4, 100+i, 200+i});
    // Group B: 4 prompts sharing [10,20,30,40]
    for (int i = 0; i < 4; ++i) prompts.push_back({10,20,30,40, 300+i, 400+i});
    // Group C: 3 prompts sharing [50,60,70,80]
    for (int i = 0; i < 3; ++i) prompts.push_back({50,60,70,80, 500+i, 600+i});

    auto r = hotpath::analyze_prefixes(prompts);
    check(r.total_requests == 12, "3groups: total=12");
    check(r.top_prefixes.size() == 3, "3groups: 3 prefix groups");
    // Sorted by count desc: 5, 4, 3
    if (r.top_prefixes.size() == 3) {
      check(r.top_prefixes[0].request_count == 5, "3groups: top=5");
      check(r.top_prefixes[1].request_count == 4, "3groups: second=4");
      check(r.top_prefixes[2].request_count == 3, "3groups: third=3");
    }
    // cacheable = (5+4+3)*4 = 48. total = 12*6 = 72. fraction = 48/72 = 2/3
    check_near(r.cacheable_token_fraction, 48.0/72.0, 1e-9, "3groups: cacheable=2/3");
    check(r.unique_prefixes == 3, "3groups: 3 unique (no singletons)");
  }

  section("Prefix Analyzer — empty prompts (0 tokens)");
  {
    std::vector<std::vector<int>> prompts = {{}, {}, {}};
    auto r = hotpath::analyze_prefixes(prompts);
    check(r.total_requests == 3, "empty prompts: total=3");
    check_near(r.cacheable_token_fraction, 0.0, 1e-9, "empty prompts: no tokens");
  }

  section("Prefix Analyzer — top-10 limit");
  {
    // 15 distinct groups → top_prefixes should be capped at 10
    std::vector<std::vector<int>> prompts;
    for (int g = 0; g < 15; ++g) {
      for (int i = 0; i < 2; ++i) {
        prompts.push_back({g*100+1, g*100+2, g*100+3, g*100+4, g*100+10+i});
      }
    }
    auto r = hotpath::analyze_prefixes(prompts);
    check(r.top_prefixes.size() <= 10, "top-10: at most 10 groups");
    check(r.total_requests == 30, "top-10: total=30");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 15: Workload Classifier — boundary conditions
// ─────────────────────────────────────────────────────────────────────

void test_classifier_boundaries() {
  section("Classifier — SHORT_CONTEXT boundary: prompt=256 is NOT short");
  {
    hotpath::WorkloadClassifierInput i;
    i.median_prompt_tokens = 256;
    i.median_output_tokens = 128;
    auto r = hotpath::classify_workload(i);
    check(r.primary_class != hotpath::WorkloadClass::SHORT_CONTEXT,
          "256 tokens is not SHORT_CONTEXT (< 256 required)");
  }

  section("Classifier — SHORT_CONTEXT boundary: prompt=255 IS short");
  {
    hotpath::WorkloadClassifierInput i;
    i.median_prompt_tokens = 255;
    i.median_output_tokens = 128;
    auto r = hotpath::classify_workload(i);
    check(r.primary_class == hotpath::WorkloadClass::SHORT_CONTEXT,
          "255 tokens IS SHORT_CONTEXT");
  }

  section("Classifier — DECODE_HEAVY boundary: output=prompt*2 is NOT decode-heavy");
  {
    hotpath::WorkloadClassifierInput i;
    i.median_prompt_tokens = 500;
    i.median_output_tokens = 1000;  // exactly 2x
    auto r = hotpath::classify_workload(i);
    check(r.primary_class != hotpath::WorkloadClass::DECODE_HEAVY,
          "output==2*prompt is not DECODE_HEAVY (> required)");
  }

  section("Classifier — DECODE_HEAVY boundary: output=prompt*2+1 IS decode-heavy");
  {
    hotpath::WorkloadClassifierInput i;
    i.median_prompt_tokens = 500;
    i.median_output_tokens = 1001;
    auto r = hotpath::classify_workload(i);
    check(r.primary_class == hotpath::WorkloadClass::DECODE_HEAVY,
          "output>2*prompt IS DECODE_HEAVY");
  }

  section("Classifier — PREFILL_HEAVY boundary: output=prompt/4 is NOT prefill-heavy");
  {
    hotpath::WorkloadClassifierInput i;
    i.median_prompt_tokens = 4096;
    i.median_output_tokens = 1024;  // exactly prompt/4
    i.phase.prefill_fraction = 0.7;
    auto r = hotpath::classify_workload(i);
    check(r.primary_class != hotpath::WorkloadClass::PREFILL_HEAVY,
          "output==prompt/4 is not PREFILL_HEAVY (< required)");
  }

  section("Classifier — all fields pass through");
  {
    hotpath::WorkloadClassifierInput i;
    i.median_prompt_tokens = 1024;
    i.median_output_tokens = 512;
    i.request_rate = 42.0;
    i.phase.prefill_fraction = 0.35;
    i.cache.cache_hit_rate = 0.22;
    i.prefix.cacheable_token_fraction = 0.15;
    i.median_decode_latency_us = 3000;
    i.p99_decode_latency_us = 9000;
    auto r = hotpath::classify_workload(i);
    check_near(r.median_prompt_tokens, 1024.0, 1e-9, "passthrough: prompt");
    check_near(r.median_output_tokens, 512.0, 1e-9, "passthrough: output");
    check_near(r.request_rate, 42.0, 1e-9, "passthrough: rate");
    check_near(r.prefill_fraction, 0.35, 1e-9, "passthrough: pfrac");
    check_near(r.cache_hit_rate, 0.22, 1e-9, "passthrough: hit_rate");
    check_near(r.prefix_sharing_rate, 0.15, 1e-9, "passthrough: sharing");
    // contention = (9000-3000)/3000 = 2.0
    check_near(r.prefill_contention, 2.0, 1e-9, "passthrough: contention");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 16: Disagg Model — more scenarios
// ─────────────────────────────────────────────────────────────────────

void test_disagg_model_extra() {
  section("Disagg Model — high request rate (saturated mono)");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 2048;
    input.profile.median_output_tokens = 256;
    input.profile.request_rate = 1000.0;  // very high
    input.profile.prefill_contention = 1.0;
    input.total_gpus = 8;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);
    check(r.mono_throughput_rps > 0, "saturated: mono > 0");
    check(r.disagg_throughput_rps > 0, "saturated: disagg > 0");
    check(!std::isnan(r.mono_throughput_rps), "saturated: no NaN");
  }

  section("Disagg Model — very low request rate");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 2048;
    input.profile.median_output_tokens = 256;
    input.profile.request_rate = 0.001;  // 1 req per 1000 seconds
    input.profile.prefill_contention = 0.1;
    input.total_gpus = 8;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);
    check(!std::isnan(r.mono_throughput_rps), "low rate: no NaN");
    // Low rate → no saturation, disagg probably not needed
    check(!r.should_disaggregate || r.disagg_throughput_rps > r.mono_throughput_rps,
          "low rate: if disagg, then faster");
  }

  section("Disagg Model — 64 GPUs");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 4096;
    input.profile.median_output_tokens = 256;
    input.profile.request_rate = 100.0;
    input.profile.prefill_contention = 1.5;
    input.total_gpus = 64;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);
    check(r.optimal_prefill_gpus + r.optimal_decode_gpus == 64,
          "64 GPU: split sums to 64");
    check(r.optimal_prefill_gpus >= 1, "64 GPU: prefill >= 1");
    check(r.optimal_decode_gpus >= 1, "64 GPU: decode >= 1");
  }

  section("Disagg Model — disagg_p99_ttft = prefill + kv_overhead");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 1000;
    input.profile.median_output_tokens = 100;
    input.profile.request_rate = 5.0;
    input.total_gpus = 4;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);
    // prefill_time = 1000*0.01 = 10ms
    // disagg_p99_ttft should equal prefill_time + kv_overhead
    double expected_ttft = 10.0 + r.kv_transfer_overhead_ms;
    check_near(r.disagg_p99_ttft_ms, expected_ttft, 1e-6,
               "disagg ttft = prefill + kv_overhead");
    // disagg_p99_itl = decode_step = 5.0 (no contention in disagg)
    check_near(r.disagg_p99_itl_ms, 5.0, 1e-6, "disagg itl = 5.0ms");
  }

  section("Disagg Model — zero contention → no blocking penalty");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 2048;
    input.profile.median_output_tokens = 256;
    input.profile.request_rate = 5.0;
    input.profile.prefill_contention = 0.0;
    input.total_gpus = 8;
    input.network_bandwidth_gbps = 100.0;
    auto r = hotpath::estimate_disaggregation(input);
    // blocking_factor = 1/(1+0*0.3) = 1.0 → no penalty
    double service = 2048*0.01 + 256*5.0;
    double expected = 8.0 / service * 1000.0 * 1.0;
    check_near(r.mono_throughput_rps, expected, 1e-6, "zero contention: no blocking");
    // ttft = prefill * (1+0) = prefill
    check_near(r.mono_p99_ttft_ms, 2048*0.01, 1e-6, "zero contention: ttft=prefill");
    // itl = 5 * (1+0*0.5) = 5
    check_near(r.mono_p99_itl_ms, 5.0, 1e-6, "zero contention: itl=5");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 17: OTLP Export — more scenarios
// ─────────────────────────────────────────────────────────────────────

void test_otlp_extra() {
  section("OTLP — timestamps in nanoseconds");
  {
    hotpath::RequestTrace t;
    t.arrival_us = 1000000;  // 1 second in microseconds
    t.completion_us = 2000000;
    t.prefill_start_us = 1100000;
    t.prefill_end_us = 1200000;
    t.first_token_us = 1300000;
    t.last_token_us = 1900000;
    auto json = hotpath::export_otlp_json({t});
    // 1000000 us * 1000 = 1000000000 ns = 10^9
    check(contains(json, "1000000000"), "ns: arrival_us=1s → 10^9 ns");
    // 2000000 us * 1000 = 2000000000 ns
    check(contains(json, "2000000000"), "ns: completion_us=2s → 2*10^9 ns");
  }

  section("OTLP — multiple traces have distinct trace IDs");
  {
    std::vector<hotpath::RequestTrace> traces(5);
    for (int i = 0; i < 5; ++i) {
      traces[i].arrival_us = i * 1000;
      traces[i].completion_us = (i + 1) * 1000;
    }
    auto json = hotpath::export_otlp_json(traces);
    // Count unique traceId values
    std::vector<std::string> ids;
    size_t pos = 0;
    while ((pos = json.find("\"traceId\":\"", pos)) != std::string::npos) {
      pos += 11;
      auto end = json.find('"', pos);
      ids.push_back(json.substr(pos, end - pos));
      pos = end;
    }
    // 5 traces * (1 root + 0 children since no prefill/decode) = 5 span entries
    // but all spans in same trace share the same traceId, so we need unique traceIds
    std::sort(ids.begin(), ids.end());
    auto last = std::unique(ids.begin(), ids.end());
    int unique_traces = static_cast<int>(last - ids.begin());
    check(unique_traces == 5, "5 traces → 5 distinct traceIds");
  }

  section("OTLP — 100 traces span count");
  {
    std::vector<hotpath::RequestTrace> traces(100);
    for (int i = 0; i < 100; ++i) {
      traces[i].arrival_us = i * 10000;
      traces[i].prefill_start_us = i * 10000 + 1000;
      traces[i].prefill_end_us = i * 10000 + 3000;
      traces[i].first_token_us = i * 10000 + 3500;
      traces[i].last_token_us = i * 10000 + 8000;
      traces[i].completion_us = i * 10000 + 9000;
    }
    auto json = hotpath::export_otlp_json(traces);
    int spans = 0;
    size_t pos = 0;
    while ((pos = json.find("\"spanId\"", pos)) != std::string::npos) { ++spans; ++pos; }
    check(spans == 400, "100 traces × 4 spans = 400");
  }

  section("OTLP — attributes contain correct values");
  {
    hotpath::RequestTrace t;
    t.arrival_us = 100;
    t.prefill_start_us = 200;
    t.prefill_end_us = 300;
    t.first_token_us = 400;
    t.last_token_us = 500;
    t.completion_us = 600;
    t.prompt_tokens = 42;
    t.output_tokens = 17;
    t.cached_tokens = 5;
    auto json = hotpath::export_otlp_json({t});
    check(contains(json, "\"intValue\":\"42\""), "attr: prompt_tokens=42");
    check(contains(json, "\"intValue\":\"17\""), "attr: output_tokens=17");
    check(contains(json, "\"intValue\":\"5\""), "attr: cached_tokens=5");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 18: Serve Report — value verification
// ─────────────────────────────────────────────────────────────────────

void test_serve_report_extra() {
  section("Serve Report — specific latency values appear");
  {
    hotpath::ServeReportData d{};
    d.model_name = "m"; d.engine = "e"; d.gpu_info = "g";
    d.queue_wait_available = true;
    d.server_timing_available = true;
    d.queue_p50 = 2.1; d.queue_p90 = 8.4; d.queue_p99 = 41.2;
    d.server_prefill_p50 = 5.5;
    d.prefill_p50 = 12.3;
    auto report = hotpath::render_serve_report(d);
    check(contains(report, "2.1"), "report shows queue_p50=2.1");
    check(contains(report, "8.4"), "report shows queue_p90=8.4");
    check(contains(report, "41.2"), "report shows queue_p99=41.2");
    check(contains(report, "12.3"), "report shows prefill_p50=12.3");
    check(contains(report, "Prefill (server)"), "report shows server prefill label");
  }

  section("Serve Report — GPU phase bars use block chars");
  {
    hotpath::ServeReportData d{};
    d.model_name = "m"; d.engine = "e"; d.gpu_info = "g";
    d.gpu_phase_available = true;  // explicitly flag real GPU phase data
    d.prefill_compute_pct = 50.0;
    d.decode_compute_pct = 30.0;
    d.other_idle_pct = 20.0;
    auto report = hotpath::render_serve_report(d);
    // Should contain block characters ░ or █
    check(contains(report, "\xe2\x96\x88") || contains(report, "\xe2\x96\x91"),
          "report has bar chart characters");
    check(contains(report, "50.0"), "report shows prefill_pct=50.0");
    check(contains(report, "30.0"), "report shows decode_pct=30.0");
  }

  section("Serve Report — disagg shows bandwidth");
  {
    hotpath::ServeReportData d{};
    d.model_name = "m"; d.engine = "e"; d.gpu_info = "g";
    d.should_disaggregate = true;
    d.optimal_p = 3; d.optimal_d = 5;
    d.projected_throughput_pct = 25.0;
    d.projected_throughput_rps = 12.5;
    d.mono_p99_ttft = 100.0; d.disagg_p99_ttft = 60.0;
    d.min_bandwidth_gbps = 75.0;
    auto report = hotpath::render_serve_report(d);
    check(contains(report, "3:5"), "report shows P:D 3:5");
    check(contains(report, "75"), "report shows bandwidth=75");
    check(contains(report, "12.5"), "report shows throughput=12.5");
  }

  section("Serve Report — TTFT label: both (est.) when no server timing");
  {
    // server_timing_available=false → both mono and disagg sides are model estimates
    hotpath::ServeReportData d{};
    d.model_name = "m"; d.engine = "e"; d.gpu_info = "g";
    d.should_disaggregate = true;
    d.optimal_p = 1; d.optimal_d = 3;
    d.mono_p99_ttft = 88.0;
    d.disagg_p99_ttft = 55.0;
    d.min_bandwidth_gbps = 50.0;
    // server_timing_available defaults to false, server_prefill_p99 defaults to 0
    auto report = hotpath::render_serve_report(d);
    check(contains(report, "88ms (est.)"), "no server timing: mono side labeled (est.)");
    check(contains(report, "55ms (est.)"), "no server timing: disagg side labeled (est.)");
    check(!contains(report, "(measured)"), "no server timing: no (measured) label");
  }

  section("Serve Report — TTFT label: (measured) when server_prefill_p99 available");
  {
    hotpath::ServeReportData d{};
    d.model_name = "m"; d.engine = "e"; d.gpu_info = "g";
    d.should_disaggregate = true;
    d.optimal_p = 1; d.optimal_d = 3;
    d.server_timing_available = true;
    d.server_prefill_p99 = 112.0;  // actual measured server prefill p99
    d.mono_p99_ttft = 88.0;        // model estimate (should NOT appear for mono)
    d.disagg_p99_ttft = 55.0;
    d.min_bandwidth_gbps = 50.0;
    auto report = hotpath::render_serve_report(d);
    check(contains(report, "112ms (measured)"), "server_prefill_p99 used as measured mono baseline");
    check(!contains(report, "88ms"), "model estimate not shown when measured available");
    check(contains(report, "55ms (est.)"), "disagg side always (est.)");
  }

  section("Serve Report — TTFT label: server_timing_available but prefill_p99=0 → (est.)");
  {
    // server_timing_available=true but server_prefill_p99=0 (e.g. no prefill events in log)
    // → should fall back to model estimate
    hotpath::ServeReportData d{};
    d.model_name = "m"; d.engine = "e"; d.gpu_info = "g";
    d.should_disaggregate = true;
    d.optimal_p = 2; d.optimal_d = 6;
    d.server_timing_available = true;
    d.server_prefill_p99 = 0.0;  // no prefill data
    d.mono_p99_ttft = 70.0;
    d.disagg_p99_ttft = 45.0;
    d.min_bandwidth_gbps = 50.0;
    auto report = hotpath::render_serve_report(d);
    check(contains(report, "70ms (est.)"),
          "server_prefill_p99=0 → model estimate labeled (est.)");
    check(!contains(report, "(measured)"), "no (measured) when server_prefill_p99=0");
  }

  section("Serve Report — TTFT does not appear when should_disaggregate=false");
  {
    hotpath::ServeReportData d{};
    d.model_name = "m"; d.engine = "e"; d.gpu_info = "g";
    d.should_disaggregate = false;
    d.mono_p99_ttft = 50.0; d.disagg_p99_ttft = 30.0;
    auto report = hotpath::render_serve_report(d);
    check(!contains(report, "(est.)"), "MONOLITHIC: no TTFT estimates shown");
    check(!contains(report, "(measured)"), "MONOLITHIC: no measured label");
    check(contains(report, "MONOLITHIC"), "shows MONOLITHIC recommendation");
  }

  section("Serve Report — TTFT integer rounding (rounds, does not truncate)");
  {
    // 99.9ms → should display as 100ms (std::lround), not 99ms (truncation)
    // 49.4ms → should display as 49ms (rounds down)
    hotpath::ServeReportData d{};
    d.model_name = "m"; d.engine = "e"; d.gpu_info = "g";
    d.should_disaggregate = true;
    d.optimal_p = 1; d.optimal_d = 3;
    d.mono_p99_ttft = 99.9;
    d.disagg_p99_ttft = 49.4;
    d.min_bandwidth_gbps = 50.0;
    auto report = hotpath::render_serve_report(d);
    check(contains(report, "100ms"), "99.9 rounds up to 100 (not truncated to 99)");
    check(!contains(report, "99ms (est.)"), "99.9 must not truncate to 99");
    check(contains(report, "49ms"), "49.4 rounds down to 49");
    check(!contains(report, "50ms"), "49.4 does not round up to 50");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 19: More integration — other workload types
// ─────────────────────────────────────────────────────────────────────

void test_integration_extra() {
  section("Integration — DECODE_HEAVY workload on 1 GPU (this machine)");
  {
    // Simulates: reasoning model, short prompt, very long output, single A10G
    std::vector<hotpath::RequestTrace> traces;
    for (int i = 0; i < 20; ++i) {
      hotpath::RequestTrace t;
      t.prompt_tokens = 256;
      t.output_tokens = 2048;
      t.cached_tokens = 0;
      traces.push_back(t);
    }

    std::vector<hotpath::KernelEntry> kernels;
    for (int i = 0; i < 5; ++i)
      kernels.push_back({"prefill_k", hotpath::profiler::KernelPhase::PREFILL,
                          static_cast<int64_t>(i)*50000, 10000});
    for (int i = 0; i < 40; ++i)
      kernels.push_back({"decode_k", hotpath::profiler::KernelPhase::DECODE,
                          250000 + static_cast<int64_t>(i)*50000, 40000});

    auto phase = hotpath::analyze_phases(kernels);
    // prefill = 5*10000 = 50000, decode = 40*40000 = 1600000
    check(phase.breakdown.prefill_us == 50000, "decode_heavy: prefill_us");
    check(phase.breakdown.decode_us == 1600000, "decode_heavy: decode_us");

    auto cache = hotpath::analyze_cache(traces, {});
    check_near(cache.cache_hit_rate, 0.0, 1e-9, "decode_heavy: no cache hits");

    hotpath::WorkloadClassifierInput ci;
    ci.phase = phase.breakdown;
    ci.cache = cache;
    ci.median_prompt_tokens = 256;
    ci.median_output_tokens = 2048;
    ci.request_rate = 2.0;
    auto profile = hotpath::classify_workload(ci);
    check(profile.primary_class == hotpath::WorkloadClass::DECODE_HEAVY,
          "decode_heavy: classified DECODE_HEAVY");

    hotpath::DisaggModelInput di;
    di.profile = profile;
    di.total_gpus = 1;
    di.network_bandwidth_gbps = 0;
    auto disagg = hotpath::estimate_disaggregation(di);
    check(!disagg.should_disaggregate,
          "decode_heavy 1GPU: cannot disaggregate");
    check(contains(disagg.reason, "2 GPUs"),
          "decode_heavy 1GPU: reason says need 2 GPUs");

    auto cfg = hotpath::generate_vllm_config(disagg, "Qwen/Qwen3-8B");
    check(contains(cfg, "tensor-parallel-size 1"), "1GPU: TP=1 config");
    check(!contains(cfg, "kv_producer"), "1GPU: no disagg");
  }

  section("Integration — SHORT_CONTEXT, cache-friendly, 8 GPUs");
  {
    hotpath::WorkloadClassifierInput ci;
    ci.median_prompt_tokens = 128;
    ci.median_output_tokens = 64;
    ci.request_rate = 100.0;
    ci.prefix.cacheable_token_fraction = 0.9;
    ci.phase.prefill_fraction = 0.1;
    auto profile = hotpath::classify_workload(ci);
    // SHORT_CONTEXT wins (checked before CACHE_FRIENDLY)
    check(profile.primary_class == hotpath::WorkloadClass::SHORT_CONTEXT,
          "short+cache: SHORT_CONTEXT wins");

    hotpath::DisaggModelInput di;
    di.profile = profile;
    di.total_gpus = 8;
    di.network_bandwidth_gbps = 100.0;
    auto disagg = hotpath::estimate_disaggregation(di);
    check(!disagg.should_disaggregate,
          "short context: disagg not recommended (prompt < 1024)");
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 20: Parametric sweep — vary one input, check output monotonicity
// ─────────────────────────────────────────────────────────────────────

void test_parametric() {
  section("Parametric — more prefill kernels → higher prefill_fraction");
  {
    using hotpath::KernelEntry;
    using hotpath::profiler::KernelPhase;
    // Keep decode constant, increase prefill count
    double prev_frac = -1.0;
    for (int np = 1; np <= 10; ++np) {
      std::vector<KernelEntry> k;
      for (int i = 0; i < np; ++i)
        k.push_back({"p", KernelPhase::PREFILL, static_cast<int64_t>(i)*100, 100});
      for (int i = 0; i < 5; ++i)
        k.push_back({"d", KernelPhase::DECODE, static_cast<int64_t>(np*100+i*100), 100});
      auto r = hotpath::analyze_phases(k);
      check(r.breakdown.prefill_fraction >= prev_frac - 1e-9,
            "mono: prefill_frac increases with np=" + std::to_string(np));
      prev_frac = r.breakdown.prefill_fraction;
    }
  }

  section("Parametric — more cache hits → higher cache_hit_rate");
  {
    double prev_rate = -1.0;
    for (int pct = 0; pct <= 100; pct += 10) {
      std::vector<hotpath::RequestTrace> traces;
      for (int i = 0; i < 100; ++i) {
        hotpath::RequestTrace t;
        t.prompt_tokens = 1000;
        t.cached_tokens = (i < pct) ? 500 : 0;
        traces.push_back(t);
      }
      auto r = hotpath::analyze_cache(traces, {});
      check(r.cache_hit_rate >= prev_rate - 1e-9,
            "mono: hit_rate increases with pct=" + std::to_string(pct));
      prev_rate = r.cache_hit_rate;
      // Independent: expected = pct*500 / 100*1000 = pct*0.005
      double expected = static_cast<double>(pct) * 500.0 / 100000.0;
      check_near(r.cache_hit_rate, expected, 1e-9,
                 "exact: hit_rate at pct=" + std::to_string(pct));
    }
  }

  section("Parametric — higher contention → lower mono throughput");
  {
    double prev_throughput = 1e18;
    for (double c = 0.0; c <= 3.0; c += 0.5) {
      hotpath::DisaggModelInput input;
      input.profile.median_prompt_tokens = 2048;
      input.profile.median_output_tokens = 256;
      input.profile.request_rate = 5.0;
      input.profile.prefill_contention = c;
      input.total_gpus = 4;
      input.network_bandwidth_gbps = 100.0;
      auto r = hotpath::estimate_disaggregation(input);
      check(r.mono_throughput_rps <= prev_throughput + 1e-6,
            "mono: throughput decreases with contention=" + std::to_string(c));
      prev_throughput = r.mono_throughput_rps;
    }
  }

  section("Parametric — more GPUs → higher mono throughput");
  {
    double prev_throughput = 0.0;
    for (int g = 1; g <= 16; ++g) {
      hotpath::DisaggModelInput input;
      input.profile.median_prompt_tokens = 2048;
      input.profile.median_output_tokens = 256;
      input.profile.request_rate = 5.0;
      input.profile.prefill_contention = 0.5;
      input.total_gpus = g;
      input.network_bandwidth_gbps = 100.0;
      auto r = hotpath::estimate_disaggregation(input);
      check(r.mono_throughput_rps >= prev_throughput - 1e-6,
            "mono: throughput increases with gpus=" + std::to_string(g));
      prev_throughput = r.mono_throughput_rps;
    }
  }

  section("Parametric — lower bandwidth → higher kv_transfer_overhead");
  {
    double prev_kv = 0.0;
    for (double bw = 200.0; bw >= 10.0; bw -= 20.0) {
      hotpath::DisaggModelInput input;
      input.profile.median_prompt_tokens = 4096;
      input.profile.median_output_tokens = 256;
      input.profile.request_rate = 10.0;
      input.total_gpus = 8;
      input.network_bandwidth_gbps = bw;
      auto r = hotpath::estimate_disaggregation(input);
      check(r.kv_transfer_overhead_ms >= prev_kv - 1e-6,
            "kv_overhead increases as bandwidth drops to " + std::to_string(bw));
      prev_kv = r.kv_transfer_overhead_ms;
    }
  }

  section("Parametric — batch_size_series length == input length");
  {
    for (int n = 1; n <= 20; ++n) {
      std::vector<hotpath::MetricSnapshot> snaps(n, {0, 8.0, 1.0, 0, 50.0});
      auto r = hotpath::analyze_batches(snaps);
      check(static_cast<int>(r.batch_size_series.size()) == n,
            "series len=" + std::to_string(n));
      check(static_cast<int>(r.queue_depth_series.size()) == n,
            "queue series len=" + std::to_string(n));
    }
  }

  section("Parametric — prefix groups sorted by count descending");
  {
    // Groups of size 10, 7, 5, 3, 2
    std::vector<std::vector<int>> prompts;
    int sizes[] = {10, 7, 5, 3, 2};
    int base = 0;
    for (int s : sizes) {
      for (int i = 0; i < s; ++i)
        prompts.push_back({base+1, base+2, base+3, base+4, base+100+i});
      base += 1000;
    }
    auto r = hotpath::analyze_prefixes(prompts);
    check(r.top_prefixes.size() == 5, "5 groups");
    for (size_t i = 1; i < r.top_prefixes.size(); ++i) {
      check(r.top_prefixes[i].request_count <= r.top_prefixes[i-1].request_count,
            "groups sorted desc at i=" + std::to_string(i));
    }
  }

  section("Parametric — OTLP span count = 4 * N for full traces");
  {
    for (int n = 1; n <= 10; ++n) {
      std::vector<hotpath::RequestTrace> traces(n);
      for (int i = 0; i < n; ++i) {
        traces[i].arrival_us = i*10000;
        traces[i].prefill_start_us = i*10000+100;
        traces[i].prefill_end_us = i*10000+500;
        traces[i].first_token_us = i*10000+600;
        traces[i].last_token_us = i*10000+9000;
        traces[i].completion_us = i*10000+9500;
      }
      auto json = hotpath::export_otlp_json(traces);
      int spans = 0;
      size_t pos = 0;
      while ((pos = json.find("\"spanId\"", pos)) != std::string::npos) {++spans;++pos;}
      check(spans == 4*n, "span count=4*" + std::to_string(n) + "=" + std::to_string(4*n));
    }
  }
}

// ─────────────────────────────────────────────────────────────────────
// SECTION 21: Sentinel and guard invariants
// ─────────────────────────────────────────────────────────────────────

void test_sentinel_and_guards() {
  section("Serve Report — latency row shows '-' when p50 < 0 (all-failure sentinel)");
  {
    // When all requests fail, latency vectors are empty and percentile_vec returns -1.0.
    // The report should display '-' rather than '0.0ms'.
    hotpath::ServeReportData d{};
    d.model_name = "m"; d.engine = "e"; d.gpu_info = "g";
    d.server_timing_available = true;
    d.queue_wait_available = true;
    // Simulate all-failure run: p50 = -1.0 (sentinel from empty vector)
    d.queue_p50 = -1.0; d.queue_p90 = -1.0; d.queue_p99 = -1.0;
    d.server_prefill_p50 = -1.0; d.server_prefill_p90 = -1.0; d.server_prefill_p99 = -1.0;
    d.prefill_p50 = -1.0; d.prefill_p90 = -1.0; d.prefill_p99 = -1.0;
    d.decode_total_p50 = -1.0; d.decode_total_p90 = -1.0; d.decode_total_p99 = -1.0;
    d.decode_per_token_p50 = -1.0; d.decode_per_token_p90 = -1.0; d.decode_per_token_p99 = -1.0;
    d.e2e_p50 = -1.0; d.e2e_p90 = -1.0; d.e2e_p99 = -1.0;
    auto report = hotpath::render_serve_report(d);
    // Should show '-' for unavailable rows, not '0.0'
    check(contains(report, "-"), "sentinel -1 shows '-' not '0.0ms'");
    check(!contains(report, "0.0\n") && !contains(report, "   0.0"),
          "no '0.0' latency values shown for all-failure run");
  }

  section("Serve Report — normal latency row when p50 >= 0");
  {
    hotpath::ServeReportData d{};
    d.model_name = "m"; d.engine = "e"; d.gpu_info = "g";
    d.prefill_p50 = 12.5; d.prefill_p90 = 30.0; d.prefill_p99 = 95.0;
    d.e2e_p50 = 100.0; d.e2e_p90 = 200.0; d.e2e_p99 = 400.0;
    auto report = hotpath::render_serve_report(d);
    check(contains(report, "12.5"), "normal row: p50=12.5 shown");
    check(contains(report, "95.0"), "normal row: p99=95.0 shown");
  }

  section("Disagg Model — measured_prefill_p99 = -1 (sentinel) uses token-count model");
  {
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 1500;
    input.profile.prefill_contention = 0.8;
    input.total_gpus = 4;
    input.network_bandwidth_gbps = 100.0;
    // Default sentinel: -1.0 means "no measurement"
    check(input.measured_prefill_p99_ms == -1.0, "default measured_prefill_p99 is -1.0");
    auto r = hotpath::estimate_disaggregation(input);
    const double expected = 1500 * 0.01 * (1.0 + 0.8);
    check_near(r.mono_p99_ttft_ms, expected, 1e-6,
               "sentinel -1 → token-count model: prefill_time*(1+contention)");
  }

  section("Disagg Model — measured_prefill_p99 = 0.0 is a valid measurement");
  {
    // 0.0 is a real (if unusual) measurement — near-instant prefill for cached requests.
    hotpath::DisaggModelInput input;
    input.profile.median_prompt_tokens = 1000;
    input.profile.prefill_contention = 1.0;
    input.total_gpus = 4;
    input.network_bandwidth_gbps = 100.0;
    input.measured_prefill_p99_ms = 0.0;
    auto r = hotpath::estimate_disaggregation(input);
    check_near(r.mono_p99_ttft_ms, 0.0, 1e-6,
               "measured=0.0 is valid, not rejected as sentinel");
    // Without measured, model would give 1000*0.01*(1+1) = 20ms
    hotpath::DisaggModelInput input_no_m = input;
    input_no_m.measured_prefill_p99_ms = -1.0;
    auto r_no_m = hotpath::estimate_disaggregation(input_no_m);
    check(r_no_m.mono_p99_ttft_ms > r.mono_p99_ttft_ms,
          "model estimate > 0.0 measurement (model overestimates here)");
  }
}

void test_kv_config() {
  section("KV Config — Llama 3.1 8B (GQA): correct bytes/token");
  {
    // Llama 3.1 8B: 32 layers, 32 attention heads, 8 KV heads, head_dim=128, bfloat16
    // Expected: 2 * 8 * 128 * 2 * 32 = 131072 bytes/token
    const std::string config = R"({
      "num_hidden_layers": 32,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "torch_dtype": "bfloat16"
    })";
    const int64_t result = hotpath::parse_kv_bytes_per_token_from_config(config);
    check(result == 131072, "Llama 3.1 8B GQA: 131072 bytes/token");
  }

  section("KV Config — GPT-2 (MHA, head_dim from hidden_size): correct bytes/token");
  {
    // GPT-2 small: 12 layers, 12 attention heads, no KV heads (MHA), hidden_size=768
    // head_dim = 768/12 = 64, dtype not specified → 2 bytes
    // Expected: 2 * 12 * 64 * 2 * 12 = 36864 bytes/token
    const std::string config = R"({
      "num_hidden_layers": 12,
      "num_attention_heads": 12,
      "hidden_size": 768
    })";
    const int64_t result = hotpath::parse_kv_bytes_per_token_from_config(config);
    check(result == 36864, "GPT-2 MHA: 36864 bytes/token");
  }

  section("KV Config — Qwen 2.5 0.5B (GQA, explicit head_dim): correct bytes/token");
  {
    // Qwen 2.5 0.5B: 24 layers, 14 attention heads, 2 KV heads, head_dim=64, bfloat16
    // Expected: 2 * 2 * 64 * 2 * 24 = 12288 bytes/token
    const std::string config = R"({
      "num_hidden_layers": 24,
      "num_attention_heads": 14,
      "num_key_value_heads": 2,
      "head_dim": 64,
      "torch_dtype": "bfloat16"
    })";
    const int64_t result = hotpath::parse_kv_bytes_per_token_from_config(config);
    check(result == 12288, "Qwen 2.5 0.5B GQA: 12288 bytes/token");
  }

  section("KV Config — float32 dtype doubles the byte count vs float16");
  {
    // Same as GPT-2 but with float32: should give 4 bytes per element
    // Expected: 2 * 12 * 64 * 4 * 12 = 73728 bytes/token
    const std::string config = R"({
      "num_hidden_layers": 12,
      "num_attention_heads": 12,
      "hidden_size": 768,
      "torch_dtype": "float32"
    })";
    const int64_t result = hotpath::parse_kv_bytes_per_token_from_config(config);
    check(result == 73728, "float32: 73728 bytes/token (2x float16)");
  }

  section("KV Config — missing num_hidden_layers returns 0");
  {
    const std::string config = R"({
      "num_attention_heads": 32,
      "hidden_size": 4096
    })";
    const int64_t result = hotpath::parse_kv_bytes_per_token_from_config(config);
    check(result == 0, "missing layers → returns 0");
  }

  section("KV Config — missing attention heads returns 0");
  {
    const std::string config = R"({
      "num_hidden_layers": 32,
      "hidden_size": 4096
    })";
    const int64_t result = hotpath::parse_kv_bytes_per_token_from_config(config);
    check(result == 0, "missing attention heads → returns 0");
  }

  section("KV Config — empty config returns 0");
  {
    check(hotpath::parse_kv_bytes_per_token_from_config("{}") == 0, "empty config → 0");
    check(hotpath::parse_kv_bytes_per_token_from_config("") == 0, "empty string → 0");
  }

  section("KV Config — detect_kv_bytes_per_token: empty model name returns 0");
  {
    check(hotpath::detect_kv_bytes_per_token("") == 0,
          "empty model_name → 0 without crashing");
  }

  section("KV Config — GQA vs MHA ratio: GQA should be significantly smaller");
  {
    // MHA: 32 KV heads
    const std::string mha_config = R"({
      "num_hidden_layers": 32,
      "num_attention_heads": 32,
      "hidden_size": 4096,
      "torch_dtype": "float16"
    })";
    // GQA: 8 KV heads (same as Llama 3.1 8B)
    const std::string gqa_config = R"({
      "num_hidden_layers": 32,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "hidden_size": 4096,
      "torch_dtype": "float16"
    })";
    const int64_t mha = hotpath::parse_kv_bytes_per_token_from_config(mha_config);
    const int64_t gqa = hotpath::parse_kv_bytes_per_token_from_config(gqa_config);
    check(mha > 0 && gqa > 0, "both configs parse successfully");
    check(mha == gqa * 4, "GQA 8-heads is 4x smaller than MHA 32-heads");
  }

  section("KV Config — old default estimate is wrong for GQA models");
  {
    // The old fallback was: median_prompt_tokens * 256 * 32 = 8192 bytes/token
    // For Llama 3.1 8B with 1024-token prompts:
    //   old: 1024 * 8192 = 8,388,608 bytes ≈ 8 MB
    //   new: 1024 * 131072 = 134,217,728 bytes ≈ 128 MB
    const int64_t old_bytes_per_token = 256 * 32;  // = 8192
    const std::string llama_config = R"({
      "num_hidden_layers": 32,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "torch_dtype": "bfloat16"
    })";
    const int64_t correct = hotpath::parse_kv_bytes_per_token_from_config(llama_config);
    check(correct > old_bytes_per_token * 10,
          "correct GQA estimate is >10x larger than the old default");
    // Verify the exact ratio: 131072 / 8192 = 16
    check(correct == old_bytes_per_token * 16,
          "Llama 3.1 8B: correct estimate is exactly 16x the old default");
  }
}

}  // namespace

int main() {
  test_phase_analyzer();
  test_batch_analyzer();
  test_cache_analyzer();
  test_prefix_analyzer();
  test_workload_classifier();
  test_disagg_model();
  test_recommender();
  test_otlp_export();
  test_serve_report();
  test_integration();
  test_phase_analyzer_extra();
  test_batch_analyzer_extra();
  test_cache_analyzer_extra();
  test_prefix_analyzer_extra();
  test_classifier_boundaries();
  test_disagg_model_extra();
  test_otlp_extra();
  test_serve_report_extra();
  test_integration_extra();
  test_parametric();
  test_sentinel_and_guards();
  test_kv_config();

  std::cerr << "\n════════════════════════════════\n";
  std::cerr << "AUDIT RESULTS: " << g_pass_count << " passed, "
            << g_fail_count << " failed\n";

  if (g_fail_count > 0) {
    std::cerr << "AUDIT FAILED\n";
    return 1;
  }
  std::cerr << "ALL AUDIT CHECKS PASSED\n";
  return 0;
}
