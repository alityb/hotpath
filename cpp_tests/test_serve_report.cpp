#include <cstdlib>
#include <iostream>
#include <string>

#include "hotpath/report.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

int main() {
  hotpath::ServeReportData data;
  data.model_name = "meta-llama/Llama-3-70B";
  data.engine = "vllm";
  data.gpu_info = "8x NVIDIA A100";
  data.total_requests = 847;
  data.duration_seconds = 60.0;
  data.throughput_rps = 14.1;

  data.queue_wait_available = true;
  data.server_timing_available = true;
  data.queue_p50 = 2.1; data.queue_p90 = 8.4; data.queue_p99 = 41.2;
  data.server_prefill_p50 = 12.3; data.server_prefill_p90 = 45.1; data.server_prefill_p99 = 112.8;
  data.server_decode_p50 = 184.2; data.server_decode_p90 = 412.5; data.server_decode_p99 = 891.3;
  data.prefill_p50 = 12.3; data.prefill_p90 = 45.1; data.prefill_p99 = 112.8;
  data.decode_total_p50 = 184.2; data.decode_total_p90 = 412.5; data.decode_total_p99 = 891.3;
  data.decode_per_token_p50 = 4.2; data.decode_per_token_p90 = 5.1; data.decode_per_token_p99 = 8.9;
  data.e2e_p50 = 201.4; data.e2e_p90 = 458.3; data.e2e_p99 = 983.1;

  data.gpu_phase_available = true;
  data.prefill_compute_pct = 31.2;
  data.decode_compute_pct = 48.1;
  data.other_idle_pct = 20.7;

  data.cache_hit_rate_available = true;
  data.cache_usage_available = true;
  data.cache_histogram_available = true;
  data.cache_hit_rate = 0.412;
  data.avg_cache_usage = 73.2;
  data.peak_cache_usage = 88.1;
  data.evictions = 12;
  data.cache_hit_rate_histogram = {5, 8, 4, 3, 3};

  data.prefix_sharing_available = true;
  data.unique_prefixes = 23;
  data.cacheable_tokens_pct = 72.3;

  data.should_disaggregate = true;
  data.optimal_p = 1;
  data.optimal_d = 3;
  data.projected_throughput_pct = 40.0;
  data.projected_throughput_rps = 19.8;
  data.mono_p99_ttft = 89.0;
  data.disagg_p99_ttft = 52.0;
  data.min_bandwidth_gbps = 50.0;

  const std::string report = hotpath::render_serve_report(data);

  // Verify key sections
  expect_true(contains(report, "Latency"), "missing Latency section");
  expect_true(contains(report, "GPU Phase"), "missing GPU Phase section");
  expect_true(contains(report, "KV Cache"), "missing KV Cache section");
  expect_true(contains(report, "Disaggregation"), "missing Disaggregation section");
  expect_true(contains(report, "Prefix Sharing"), "missing Prefix Sharing section");
  expect_true(contains(report, "TTFB (client)"), "missing TTFB label");
  expect_true(contains(report, "Generation (client)"), "missing generation label");
  expect_true(contains(report, "Queue wait"), "queue wait should be rendered");
  expect_true(contains(report, "Prefill (server)"), "missing server prefill label");
  expect_true(contains(report, "Decode (server)"), "missing server decode label");
  expect_true(contains(report, "Hit histogram"), "missing cache histogram");

  // Verify key values
  expect_true(contains(report, "847"), "missing request count");
  expect_true(contains(report, "14.1"), "missing throughput");
  expect_true(contains(report, "DISAGGREGATE"), "missing recommendation");
  expect_true(contains(report, "1:3"), "missing P:D ratio");
  expect_true(contains(report, "meta-llama"), "missing model name");

  // TTFT in disagg section: mono side measured, disagg side estimated
  expect_true(contains(report, "ms (measured)"),
              "disagg TTFT: mono side should be labeled (measured) when server_timing_available");
  expect_true(contains(report, "ms (est.)"),
              "disagg TTFT: disagg side should be labeled (est.)");

  // Test monolithic recommendation
  data.should_disaggregate = false;
  const std::string mono_report = hotpath::render_serve_report(data);
  expect_true(contains(mono_report, "MONOLITHIC"), "should show MONOLITHIC");

  // When server_timing_available=false, disagg TTFT mono side should be labeled (est.)
  {
    hotpath::ServeReportData est_only;
    est_only.model_name = "est-test";
    est_only.engine = "vllm";
    est_only.gpu_info = "1x A10G";
    est_only.should_disaggregate = true;
    est_only.optimal_p = 1; est_only.optimal_d = 3;
    est_only.mono_p99_ttft = 80.0; est_only.disagg_p99_ttft = 50.0;
    // server_timing_available defaults to false
    const std::string est_report = hotpath::render_serve_report(est_only);
    expect_true(contains(est_report, "80ms (est.)"),
                "without server timing, mono TTFT should show model estimate with (est.) label");
    expect_true(!contains(est_report, "(measured)"),
                "without server timing, no (measured) label should appear");
  }

  // Placeholder sections should render as unavailable, not as measured zeroes
  hotpath::ServeReportData unavailable;
  unavailable.model_name = "placeholder";
  unavailable.engine = "vllm";
  unavailable.gpu_info = "1x A10G";
  unavailable.total_requests = 3;
  unavailable.duration_seconds = 5.0;
  unavailable.throughput_rps = 0.6;
  unavailable.prefill_p50 = 4.0;
  unavailable.decode_total_p50 = 20.0;
  unavailable.decode_per_token_p50 = 2.0;
  unavailable.e2e_p50 = 24.0;
  const std::string unavailable_report = hotpath::render_serve_report(unavailable);
  expect_true(contains(unavailable_report,
                       "GPU Phase Breakdown: not available"),
              "gpu phase placeholder should be unavailable");
  expect_true(contains(unavailable_report,
                       "KV Cache: not available"),
              "kv cache placeholder should be unavailable");
  expect_true(contains(unavailable_report,
                       "Prefix Sharing: not available"),
              "prefix placeholder should be unavailable");
  expect_true(!contains(unavailable_report, "Queue wait                   0.0"),
              "queue row should not be rendered as a measured zero");

  hotpath::ServeReportData aggregate_cache;
  aggregate_cache.model_name = "aggregate";
  aggregate_cache.engine = "vllm";
  aggregate_cache.gpu_info = "1x A10G";
  aggregate_cache.cache_hit_rate_available = true;
  aggregate_cache.cache_hit_rate_aggregate_only = true;
  aggregate_cache.cache_hit_rate = 0.25;
  const std::string aggregate_report = hotpath::render_serve_report(aggregate_cache);
  expect_true(contains(aggregate_report, "Hit rate (aggregate)"),
              "aggregate cache report should label aggregate hit rate");
  expect_true(contains(aggregate_report, "aggregate from server logs"),
              "aggregate cache report should explain source");

  // server_ttft_mean_ms > 0 should produce a "TTFT (server, mean)" row with the value
  {
    hotpath::ServeReportData ttft_data;
    ttft_data.model_name = "ttft-test";
    ttft_data.engine = "vllm";
    ttft_data.gpu_info = "1x A10G";
    ttft_data.total_requests = 10;
    ttft_data.duration_seconds = 30.0;
    ttft_data.prefill_p50 = 4.2;
    ttft_data.server_ttft_mean_ms = 13.7;
    const std::string ttft_report = hotpath::render_serve_report(ttft_data);
    expect_true(contains(ttft_report, "TTFT (server, mean)"),
                "server_ttft_mean_ms > 0 should add TTFT (server, mean) row");
    expect_true(contains(ttft_report, "13.7"),
                "server TTFT value 13.7 should appear in report");
    expect_true(contains(ttft_report, "TTFB (client)"),
                "TTFB (client) row should always be present");
    // The two measurements should both appear and show the difference
    expect_true(contains(ttft_report, "4.2"),
                "TTFB client value 4.2 should appear in report");
  }

  // server_ttft_mean_ms = -1 (default) should NOT produce a server TTFT row
  {
    hotpath::ServeReportData no_ttft;
    no_ttft.model_name = "no-ttft";
    no_ttft.engine = "vllm";
    no_ttft.gpu_info = "1x A10G";
    no_ttft.prefill_p50 = 5.0;
    // server_ttft_mean_ms defaults to -1.0
    const std::string no_ttft_report = hotpath::render_serve_report(no_ttft);
    expect_true(!contains(no_ttft_report, "TTFT (server, mean)"),
                "server_ttft_mean_ms = -1 should NOT show TTFT (server, mean) row");
    expect_true(contains(no_ttft_report, "TTFB (client)"),
                "TTFB (client) row must always be present");
  }

  std::cerr << "test_serve_report: all tests passed\n";
  return 0;
}
