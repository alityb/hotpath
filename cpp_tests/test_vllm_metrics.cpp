#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "hotpath/profiler/vllm_metrics.h"

namespace {

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  const std::string text = R"TXT(
# HELP vllm:num_preemptions_total Cumulative number of preemptions
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{model_name="Qwen"} 47.0
vllm:num_preemptions_total{model_name="Qwen",worker="1"} 49.0
vllm:gpu_cache_usage_perc{model_name="Qwen"} 0.723
vllm:num_requests_running{model_name="Qwen"} 84
ignored_metric 123
)TXT";

  const auto parsed = hotpath::profiler::parse_metrics_text(text);
  expect_true(parsed.size() == 4, "expected all labeled series to be preserved");
  std::multimap<std::string, double> values(parsed.begin(), parsed.end());
  expect_true(values.count("vllm:num_preemptions_total") == 2, "expected two preemption series");
  expect_true(values.count("vllm:gpu_cache_usage_perc") == 1, "bad cache parse");
  expect_true(values.count("vllm:num_requests_running") == 1, "bad running parse");
  expect_true(values.find("vllm:gpu_cache_usage_perc")->second == 0.723, "bad cache value");
  expect_true(values.find("vllm:num_requests_running")->second == 84.0, "bad running value");

  const std::vector<hotpath::MetricSample> samples = {
      {.sample_time = 0.0, .source = "cluster", .metric = "vllm:num_preemptions_total", .value = 3.0},
      {.sample_time = 1.0, .source = "cluster", .metric = "vllm:num_preemptions_total", .value = 5.0},
      {.sample_time = 0.0, .source = "cluster", .metric = "vllm:gpu_cache_usage_perc", .value = 0.5},
      {.sample_time = 1.0, .source = "cluster", .metric = "vllm:gpu_cache_usage_perc", .value = 0.75},
      {.sample_time = 0.0, .source = "peer1", .metric = "vllm:num_preemptions_total", .value = 30.0},
      {.sample_time = 1.0, .source = "peer1", .metric = "vllm:num_preemptions_total", .value = 50.0},
  };

  const auto summaries = hotpath::profiler::summarize_samples(samples);
  expect_true(summaries.size() == 2, "expected two summaries");
  expect_true(summaries[0].metric == "vllm:gpu_cache_usage_perc", "unexpected first summary metric");
  expect_true(summaries[0].avg.has_value() && *summaries[0].avg == 0.625, "unexpected first avg");
  expect_true(summaries[0].peak.has_value() && *summaries[0].peak == 0.75, "unexpected first peak");
  expect_true(summaries[0].min.has_value() && *summaries[0].min == 0.5, "unexpected first min");
  expect_true(summaries[1].metric == "vllm:num_preemptions_total", "unexpected second summary metric");
  expect_true(summaries[1].avg.has_value() && *summaries[1].avg == 4.0, "unexpected second avg");
  expect_true(summaries[1].peak.has_value() && *summaries[1].peak == 5.0, "unexpected second peak");
  expect_true(summaries[1].min.has_value() && *summaries[1].min == 3.0, "unexpected second min");

  namespace fs = std::filesystem;
  const fs::path temp_root = fs::temp_directory_path() / "hotpath_test_vllm_metrics";
  fs::remove_all(temp_root);
  fs::create_directories(temp_root);
  const fs::path curl_path = temp_root / "curl";
  {
    std::ofstream script(curl_path);
    script << "#!/bin/sh\n";
    script << "last=\"\"\n";
    script << "for arg in \"$@\"; do\n";
    script << "  last=\"$arg\"\n";
    script << "done\n";
    script << "case \"$last\" in\n";
    script << "  *node0*)\n";
    script << "    cat <<'EOF'\n";
    script << "vllm:num_requests_running 2\n";
    script << "vllm:time_to_first_token_seconds_p99 0.5\n";
    script << "EOF\n";
    script << "    ;;\n";
    script << "  *node1*)\n";
    script << "    cat <<'EOF'\n";
    script << "vllm:num_requests_running 3\n";
    script << "vllm:time_to_first_token_seconds_p99 1.5\n";
    script << "EOF\n";
    script << "    ;;\n";
    script << "esac\n";
  }
  chmod(curl_path.c_str(), 0755);

  const std::string original_path = std::getenv("PATH") == nullptr ? "" : std::getenv("PATH");
  setenv("PATH", (temp_root.string() + ":" + original_path).c_str(), 1);
  const auto fetched = hotpath::profiler::fetch_metrics_once({
      {.source = "local", .server_url = "http://node0"},
      {.source = "peer1", .server_url = "http://node1"},
  });
  setenv("PATH", original_path.c_str(), 1);

  bool saw_cluster_running = false;
  bool saw_cluster_p99 = false;
  for (const auto& sample : fetched) {
    if (sample.source == "cluster" && sample.metric == "vllm:num_requests_running" &&
        sample.value == 5.0) {
      saw_cluster_running = true;
    }
    if (sample.source == "cluster" &&
        sample.metric == "vllm:time_to_first_token_seconds_p99") {
      saw_cluster_p99 = true;
    }
  }
  expect_true(saw_cluster_running, "expected summed cluster running-request metric");
  expect_true(!saw_cluster_p99, "cluster percentile metrics should not be synthesized");

  const auto fetched_summaries = hotpath::profiler::summarize_samples(fetched);
  bool saw_p99_summary = false;
  for (const auto& summary : fetched_summaries) {
    if (summary.metric == "vllm:time_to_first_token_seconds_p99") {
      saw_p99_summary = true;
      expect_true(summary.avg.has_value() && *summary.avg == 1.0,
                  "expected node percentile summaries to remain visible when no cluster percentile is synthesized");
      expect_true(summary.peak.has_value() && *summary.peak == 1.5,
                  "expected peak percentile summary from node samples");
      expect_true(summary.min.has_value() && *summary.min == 0.5,
                  "expected min percentile summary from node samples");
    }
  }
  expect_true(saw_p99_summary,
              "expected percentile summary to remain present for clustered metric fetches");

  fs::remove_all(temp_root);

  // ── vLLM 0.19 metric name changes ──
  // kv_cache_usage_perc replaces gpu_cache_usage_perc
  {
    const std::string text_019 = R"TXT(
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen"} 0.341
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen"} 1200.0
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen"} 2000.0
vllm:time_to_first_token_seconds_sum{engine="0",model_name="Qwen"} 0.3496
vllm:time_to_first_token_seconds_count{engine="0",model_name="Qwen"} 18.0
vllm:num_requests_running{engine="0",model_name="Qwen"} 4
)TXT";
    const auto parsed_019 = hotpath::profiler::parse_metrics_text(text_019);
    std::multimap<std::string, double> v019(parsed_019.begin(), parsed_019.end());
    expect_true(v019.count("vllm:kv_cache_usage_perc") == 1,
                "FAIL: vllm:kv_cache_usage_perc should be recognized (vLLM 0.19)");
    expect_true(std::abs(v019.find("vllm:kv_cache_usage_perc")->second - 0.341) < 1e-9,
                "FAIL: kv_cache_usage_perc value should be 0.341");
    expect_true(v019.count("vllm:prefix_cache_hits_total") == 1,
                "FAIL: prefix_cache_hits_total should be recognized");
    expect_true(v019.find("vllm:prefix_cache_hits_total")->second == 1200.0,
                "FAIL: prefix_cache_hits_total value should be 1200");
    expect_true(v019.count("vllm:prefix_cache_queries_total") == 1,
                "FAIL: prefix_cache_queries_total should be recognized");
    expect_true(v019.count("vllm:time_to_first_token_seconds_sum") == 1,
                "FAIL: time_to_first_token_seconds_sum should be recognized (vLLM 0.19)");
    expect_true(v019.count("vllm:time_to_first_token_seconds_count") == 1,
                "FAIL: time_to_first_token_seconds_count should be recognized (vLLM 0.19)");
    expect_true(std::abs(v019.find("vllm:time_to_first_token_seconds_sum")->second - 0.3496) < 1e-9,
                "FAIL: time_to_first_token_seconds_sum value mismatch");
    expect_true(v019.find("vllm:time_to_first_token_seconds_count")->second == 18.0,
                "FAIL: time_to_first_token_seconds_count value should be 18");
    expect_true(v019.count("vllm:num_requests_running") == 1,
                "FAIL: num_requests_running should still be recognized in vLLM 0.19 format");
  }

  // kv_cache_usage_perc should be averaged (not summed) across cluster nodes
  {
    const std::vector<hotpath::MetricSample> kv_samples = {
        {.sample_time = 0.0, .source = "cluster", .metric = "vllm:kv_cache_usage_perc", .value = 0.4},
        {.sample_time = 1.0, .source = "cluster", .metric = "vllm:kv_cache_usage_perc", .value = 0.6},
    };
    const auto kv_summaries = hotpath::profiler::summarize_samples(kv_samples);
    expect_true(kv_summaries.size() == 1, "FAIL: expected one kv_cache summary");
    expect_true(kv_summaries[0].metric == "vllm:kv_cache_usage_perc",
                "FAIL: expected kv_cache_usage_perc summary");
    expect_true(kv_summaries[0].avg.has_value() &&
                    std::abs(*kv_summaries[0].avg - 0.5) < 1e-9,
                "FAIL: kv_cache_usage_perc should average to 0.5, got " +
                    std::to_string(kv_summaries[0].avg.value_or(-1)));
  }

  // Counter metrics (prefix_cache_hits_total, queries_total, sum, count) should
  // be collected and summed (not averaged) at the cluster level
  {
    const std::vector<hotpath::MetricSample> counter_samples = {
        {.sample_time = 0.0, .source = "cluster", .metric = "vllm:prefix_cache_hits_total", .value = 100.0},
        {.sample_time = 1.0, .source = "cluster", .metric = "vllm:prefix_cache_hits_total", .value = 150.0},
        {.sample_time = 0.0, .source = "cluster", .metric = "vllm:prefix_cache_queries_total", .value = 200.0},
        {.sample_time = 1.0, .source = "cluster", .metric = "vllm:prefix_cache_queries_total", .value = 300.0},
        {.sample_time = 0.0, .source = "cluster", .metric = "vllm:time_to_first_token_seconds_sum", .value = 0.2},
        {.sample_time = 1.0, .source = "cluster", .metric = "vllm:time_to_first_token_seconds_sum", .value = 0.35},
        {.sample_time = 0.0, .source = "cluster", .metric = "vllm:time_to_first_token_seconds_count", .value = 10.0},
        {.sample_time = 1.0, .source = "cluster", .metric = "vllm:time_to_first_token_seconds_count", .value = 18.0},
    };
    const auto counter_summaries = hotpath::profiler::summarize_samples(counter_samples);
    std::multimap<std::string, hotpath::MetricSummary> by_metric;
    for (const auto& s : counter_summaries) by_metric.emplace(s.metric, s);

    // All four counter metrics should appear
    expect_true(by_metric.count("vllm:prefix_cache_hits_total") == 1,
                "FAIL: prefix_cache_hits_total should appear in summaries");
    expect_true(by_metric.count("vllm:prefix_cache_queries_total") == 1,
                "FAIL: prefix_cache_queries_total should appear in summaries");
    expect_true(by_metric.count("vllm:time_to_first_token_seconds_sum") == 1,
                "FAIL: time_to_first_token_seconds_sum should appear in summaries");
    expect_true(by_metric.count("vllm:time_to_first_token_seconds_count") == 1,
                "FAIL: time_to_first_token_seconds_count should appear in summaries");

    // peak (max observed) should be the later counter value
    const auto& hits_summary = by_metric.find("vllm:prefix_cache_hits_total")->second;
    expect_true(hits_summary.peak.has_value() && *hits_summary.peak == 150.0,
                "FAIL: hits_total peak should be 150");
    expect_true(hits_summary.min.has_value() && *hits_summary.min == 100.0,
                "FAIL: hits_total min should be 100");
  }

  std::cerr << "test_vllm_metrics: all tests passed\n";
  return 0;
}
